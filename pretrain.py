import pandas as pd
import torch
#for speed improvements
torch.set_float32_matmul_precision('high')  
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.optim as optim
from transformers import PreTrainedTokenizerFast, DebertaV2ForMaskedLM, DebertaV2Config, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import TQDMProgressBar
import gc
from SynCodonLM import synonymous_codons
from SynCodonLM import species_token_type


#### MODEL ####
class SynCodonLM(DebertaV2ForMaskedLM):
    def __init__(self, config, mask_matrix=None, tokenizer=None):
        super().__init__(config)
        self.mask_matrix = mask_matrix  # [vocab_size, vocab_size]
        self.tokenizer = tokenizer

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = super().forward(
            input_ids=input_ids.to(torch.long),
            attention_mask=attention_mask.to(torch.long),
            token_type_ids=token_type_ids.to(torch.long),
            labels=None  # don't compute loss yet
        )

        logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        if labels is not None and self.mask_matrix is not None:
            if (labels >= logits.size(-1)).any():
                raise ValueError("Label value exceeds vocabulary size.")

            masked_positions = labels != -100
            for b, s in masked_positions.nonzero(as_tuple=False):
                label_id = labels[b, s].item()
                logits[b, s] += self.mask_matrix[label_id].to(logits.device)

            # compute loss manually
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            # compute accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == labels) & masked_positions
            accuracy = correct.sum().float() / masked_positions.sum().float()
        else:
            loss = torch.tensor(0.0, device=logits.device)
            accuracy = torch.tensor(0.0, device=logits.device)

        return loss, accuracy



##### LIGHTNING MODEL ####
class CDSembedLightningModule(pl.LightningModule):
    def __init__(self, model, tokenizer, train_loader, val_loader):
        super().__init__()
        self.model = model
        self.model.gradient_checkpointing_enable()
        self.tokenizer = tokenizer
        self.automatic_optimization = False  # necessary to use manual_backward
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_loss_sum = 0.0
        self.train_accuracy_sum = 0
        self.val_accuracy = 0
        self.lr_sum = 0.0
        self.step_count = 0



    def on_train_epoch_start(self):
        current_epoch = self.current_epoch
        self.saved_halfway = False
        print('saving model!!')
        save_model_and_tokenizer(self.model, self.tokenizer, current_epoch)
        print(f"Starting epoch {current_epoch}")
    

    def on_train_batch_end(self, outputs, batch, batch_idx):
        total_batches = len(self.train_dataloader())
        halfway_point = total_batches // 2

        if not self.saved_halfway and batch_idx >= halfway_point:
            print("Saving model at halfway point of epoch!")
            save_model_and_tokenizer(self.model, self.tokenizer, f"epoch{self.current_epoch}_halfway")
            self.saved_halfway = True



    def training_step(self, batch, batch_idx):
        self.model.train()
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        batch = {k: v.to(self.device) for k, v in batch.items()}

        loss, accuracy = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'], 
            labels=batch['labels']
        )

        self.manual_backward(loss)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # accumulate metrics
        self.train_loss_sum += loss.item()
        self.train_accuracy_sum += accuracy.item()
        self.lr_sum += optimizer.param_groups[0]['lr']
        self.step_count += 1

        if (batch_idx + 1) % 30 == 0:
            self.log('train_loss', self.train_loss_sum / 30, on_step=True, prog_bar=True, sync_dist=True, reduce_fx=torch.mean)
            self.log('lr', self.lr_sum / 30, on_step=True, prog_bar=True, logger=True, sync_dist=True, reduce_fx=torch.mean)
            self.log('accuracy', self.train_accuracy_sum / 30, on_step=True, prog_bar=True, logger=True, sync_dist=True, reduce_fx=torch.mean)

            self.train_loss_sum = 0.0
            self.train_accuracy_sum = 0.0
            self.lr_sum = 0.0
            self.step_count = 0



    def validation_step(self, batch, batch_idx):        
        self.model.eval()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        loss, accuracy = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'], 
            labels=batch['labels']
        )


        self.log('val_loss', loss, sync_dist=True, prog_bar=True, reduce_fx=torch.mean, on_epoch=True)
        self.log('val_accuracy', accuracy, sync_dist=True, prog_bar=True, reduce_fx=torch.mean, on_epoch=True)


    def configure_optimizers(self):
        base_lr = 2e-4
        optimizer = optim.AdamW(self.model.parameters(), lr=base_lr, weight_decay=0.01, fused=True)

        num_batches_per_epoch = len(self.train_dataloader())
        total_devices = self.trainer.world_size if self.trainer else 1
        total_steps = (num_batches_per_epoch * self.trainer.max_epochs) // total_devices
        warmup_steps = int(0.1 * total_steps)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

        scheduler = LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
    

class CDSDataset(Dataset):
    def __init__(self, sequences, species_list, tokenizer):
        self.sequences = sequences
        self.species_list = species_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        species = self.species_list[idx]
        cluster_id = species_token_type.get(species, 500)  # default to 500

        # split into codons
        codons = [sequence[i:i+3] for i in range(0, len(sequence), 3) if len(sequence[i:i+3]) == 3]
        codon_sequence = " ".join(codons)

        # tokenize the codon sequence
        tokenized = self.tokenizer(
            codon_sequence,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=1024,
            return_tensors="pt"
        )
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}

        # assign token_type_ids based on the species cluster
        tokenized["token_type_ids"] = torch.full_like(tokenized["input_ids"], cluster_id)
        tokenized["input_ids"] = tokenized["input_ids"].to(torch.uint8)
        tokenized["token_type_ids"] = tokenized["token_type_ids"].to(torch.uint16)
        tokenized["attention_mask"] = tokenized["attention_mask"].to(torch.uint8)

        return tokenized




class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def torch_mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: torch.Tensor = None):
        labels = inputs.clone()

        # create a probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # mask out special tokens
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # determine which tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # replace all selected tokens with [MASK]
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels



@rank_zero_only
def save_model_and_tokenizer(model, tokenizer, epoch):
    torch.cuda.empty_cache()
    save_path="./SynCodonLM-Epoch-" + str(epoch)
    model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)
    print("Model and tokenizer saved.")

def main():
    num_nodes = 3  # Number of nodes
    gpus_per_node = 4 # Number of GPUs per node
    num_workers = 4
    batch_size = 130

    tokenizer = PreTrainedTokenizerFast.from_pretrained("./SynCodonLM")
    


    # Create the config first
    config = DebertaV2Config(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_act="gelu_new",
        legacy=True,
        hidden_dropout_prob=0.1,
        type_vocab_size=501,
        pad_token_id=tokenizer.pad_token_id,
        max_position_embeddings=1024,
        relative_attention=True,
        pos_att_type="p2c|c2p", # enables both position-to-content and content-to-position attention
    )


    # build codon → amino acid and amino acid → token ID mappings
    codon_to_aa = {codon: aa for aa, codons in synonymous_codons.items() for codon in codons}
    aa_to_token_ids = {
        aa: [tokenizer.convert_tokens_to_ids(codon) for codon in codons if tokenizer.convert_tokens_to_ids(codon) != tokenizer.unk_token_id]
        for aa, codons in synonymous_codons.items()
    }

    # create the logits mask matrix
    vocab_size = tokenizer.vocab_size
    mask_matrix = torch.full((vocab_size, vocab_size), float('-inf'))
    for codon, aa in codon_to_aa.items():
        token_id = tokenizer.convert_tokens_to_ids(codon)
        if token_id != tokenizer.unk_token_id:
            allowed_ids = aa_to_token_ids[aa]
            mask_matrix[token_id, allowed_ids] = 0.0


    model = SynCodonLM(config, mask_matrix=mask_matrix, tokenizer=tokenizer)
    model = torch.compile(model)  # pre-compile backpropogation path for increased speed
    
    

    df = pd.read_csv("cds-dataset.csv", usecols=["CDS", "Species", "Set"])

    # Split into train and test sets based on the 'Set' column
    train_df = df[df["Set"] == "Train"]
    test_df = df[df["Set"] == "Test"]

    # Print the number of rows in each set
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    # Extract sequences and groups
    
    train_sequences = train_df["CDS"].tolist()
    train_species = train_df["Species"].tolist()

    test_sequences = test_df["CDS"].tolist()
    test_species = test_df["Species"].tolist()

    
    del df, train_df, test_df


    # Create datasets
    train_dataset = CDSDataset(train_sequences, train_species, tokenizer)
    test_dataset = CDSDataset(test_sequences, test_species, tokenizer)

    
    del train_sequences, train_species, test_sequences, test_species
    gc.collect()  # save space

    data_collator = CustomDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
        return_tensors="pt"
    )

    def custom_collate_fn(batch):
        batch = data_collator(batch)
        batch['attention_mask'] = (batch['input_ids'] != tokenizer.pad_token_id).long()
        return batch

    # create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )


    lightning_model = CDSembedLightningModule(model, tokenizer, train_loader, test_loader)
    progress_bar = TQDMProgressBar(refresh_rate=30)

    trainer = pl.Trainer(
        precision='bf16-mixed',   # use mixed precision for much faster training
        max_epochs=2,
        devices=gpus_per_node,
        num_nodes=num_nodes,
        strategy=pl.strategies.DDPStrategy(static_graph=False),
        callbacks=[progress_bar],  # add the progress bar callback here
        log_every_n_steps=1,
        enable_checkpointing=False,  # disable checkpointing
        val_check_interval=0.5  
    )

    trainer.fit(lightning_model, train_loader, test_loader)

    save_model_and_tokenizer(lightning_model.model, tokenizer, 'finalmodel')
    print('PROCESS COMPLETED PROPERLY')




if __name__ == "__main__":
    main()