import argparse
import gc
import pandas as pd
import torch
from torch import nn

# speed improvements
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.optim as optim
from transformers import (
    PreTrainedTokenizerFast,
    DebertaV2ForMaskedLM,
    DebertaV2Config,
    BertForMaskedLM,
    BertConfig,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import TQDMProgressBar

# project imports
from SynCodonLM import synonymous_codons
from SynCodonLM import species_token_type

# =========================
# TOKEN TYPE MAPPING UTIL
# =========================
def map_cluster_to_group(cluster_id: int) -> int:
    """
    Map a species cluster id (0-499) to a grouped token-type id using ranges:
      0-26 -> 0 (archaea)
      27-103 -> 1 (bacteria)
      104-180 -> 2 (fungi)
      181-257 -> 3 (invertebrate)
      258-327 -> 4 (plant)
      328-339 -> 5 (protozoa)
      340-416 -> 6 (vertebrate mammalian)
      417-493 -> 7 (vertebrate others)
      494-499 -> 8 (virus)
    """
    if 0 <= cluster_id <= 26:
        return 0
    elif 27 <= cluster_id <= 103:
        return 1
    elif 104 <= cluster_id <= 180:
        return 2
    elif 181 <= cluster_id <= 257:
        return 3
    elif 258 <= cluster_id <= 327:
        return 4
    elif 328 <= cluster_id <= 339:
        return 5
    elif 340 <= cluster_id <= 416:
        return 6
    elif 417 <= cluster_id <= 493:
        return 7
    elif 494 <= cluster_id <= 499:
        return 8
    else:
        return 0

MACRO_GROUP_NAMES = [
    "archaea",
    "bacteria",
    "fungi",
    "invertebrate",
    "plant",
    "protozoa",
    "vertebrate_mammalian",
    "vertebrate_others",
    "virus",
]

# ================
# DATASET / I/O
# ================
class CDSDataset(Dataset):
    def __init__(self, sequences, species_list, tokenizer, token_type_mode: str, groups_list=None):
        self.sequences = sequences
        self.species_list = species_list
        self.tokenizer = tokenizer
        self.token_type_mode = token_type_mode  # 'species' | 'none' | 'grouped'
        self.groups_list = groups_list  # list[int] aligned to sequences (required for per-group logging)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        species = self.species_list[idx]

        # token_type_ids choice
        default_cluster = 500 if self.token_type_mode == 'species' else 0
        cluster_id = species_token_type.get(species, default_cluster)
        if self.token_type_mode == 'none':
            ttype_value = 0
        elif self.token_type_mode == 'grouped':
            ttype_value = map_cluster_to_group(int(cluster_id))
        else:
            ttype_value = int(cluster_id)

        # split into codons and tokenize (space-delimited codons)
        codons = [sequence[i:i+3] for i in range(0, len(sequence), 3) if len(sequence[i:i+3]) == 3]
        codon_sequence = " ".join(codons)
        tokenized = self.tokenizer(
            codon_sequence,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=1024,
            return_tensors="pt"
        )
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}

        # token types
        tokenized["token_type_ids"] = torch.full_like(tokenized["input_ids"], ttype_value)

        # compact dtypes
        tokenized["input_ids"] = tokenized["input_ids"].to(torch.uint8)
        tokenized["token_type_ids"] = tokenized["token_type_ids"].to(torch.uint16)
        tokenized["attention_mask"] = tokenized["attention_mask"].to(torch.uint8)

        # pass group_id through for aggregation
        if self.groups_list is None:
            # Should not happen in this pipeline (we compute in main), but keep a safe default
            group_id = 0
        else:
            group_id = self.groups_list[idx]
        tokenized["group_id"] = torch.tensor(group_id, dtype=torch.long)
        return tokenized


class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def torch_mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: torch.Tensor = None):
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        return inputs, labels


@rank_zero_only
def save_model_and_tokenizer(model, tokenizer, save_path: str):
    torch.cuda.empty_cache()
    model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to: {save_path}")

# ======================
# MASK & MAPPINGS (NEW)
# ======================
def build_mask_matrix(tokenizer):
    """
    Builds the [vocab_size, vocab_size] additive mask so that for a true label
    (row = correct token id), only synonymous codons of that label's amino acid
    remain allowed (0.0), others are -inf.
    """
    codon_to_aa = {codon: aa for aa, codons in synonymous_codons.items() for codon in codons}
    aa_to_token_ids = {
        aa: [
            tokenizer.convert_tokens_to_ids(codon)
            for codon in codons
            if tokenizer.convert_tokens_to_ids(codon) != tokenizer.unk_token_id
        ]
        for aa, codons in synonymous_codons.items()
    }
    vocab_size = tokenizer.vocab_size
    mask_matrix = torch.full((vocab_size, vocab_size), float('-inf'))
    for codon, aa in codon_to_aa.items():
        token_id = tokenizer.convert_tokens_to_ids(codon)
        if token_id != tokenizer.unk_token_id:
            allowed_ids = aa_to_token_ids[aa]
            mask_matrix[token_id, allowed_ids] = 0.0
    return mask_matrix


def build_token_id_to_aa_index(tokenizer):
    """
    Returns a LongTensor of shape [vocab_size] mapping token_id -> aa_index (0..K-1),
    or -1 for non-codon tokens. Used for AA-level accuracy under raw logits.
    """
    aa_list = sorted(list(synonymous_codons.keys()))
    aa_to_index = {aa: i for i, aa in enumerate(aa_list)}

    mapping = torch.full((tokenizer.vocab_size,), -1, dtype=torch.long)
    for aa, codons in synonymous_codons.items():
        idx = aa_to_index[aa]
        for c in codons:
            tid = tokenizer.convert_tokens_to_ids(c)
            if tid != tokenizer.unk_token_id:
                mapping[tid] = idx
    return mapping




# ==========
# MODELS (Mixin does NOT inherit nn.Module; avoids double init)
# ==========

class _SynCodonBase:  # <-- no nn.Module inheritance here
    """
    Shared post-processing:
      - Takes raw_logits from the backbone (labels=None there)
      - Optionally builds constrained logits with synonym mask (self._syn_mask_matrix)
      - Uses training-time toggle (apply_syn_mask_train) for the loss path
      - Computes diagnostics (masked positions + 3 accuracies) when requested
    """
    def __init__(self, mask_matrix=None, tokenizer=None, apply_syn_mask_train=True, token_id_to_aa=None):
        # IMPORTANT: do NOT call super().__init__() here
        self._syn_mask_matrix = mask_matrix  # Tensor[V, V] or None
        self._aa_map = token_id_to_aa        # LongTensor[V] or None
        self.tokenizer = tokenizer
        self.apply_syn_mask_train = apply_syn_mask_train

    def _postprocess_from_raw(self, raw_logits, labels=None, return_logits=False):
        device = raw_logits.device
        dtype = raw_logits.dtype

        # Ensure helper tensors are on correct device/dtype *when used*
        syn_mask = None
        if self._syn_mask_matrix is not None:
            syn_mask = self._syn_mask_matrix
            if syn_mask.device != device or syn_mask.dtype != dtype:
                syn_mask = syn_mask.to(device=device, dtype=dtype)

        aa_map = None
        if self._aa_map is not None:
            aa_map = self._aa_map
            if aa_map.device != device:
                aa_map = aa_map.to(device)

        constrained_logits = raw_logits
        masked_positions = None

        if labels is not None:
            masked_positions = labels != -100

            # Constrain logits (only masked positions)
            if syn_mask is not None:
                bsz, seqlen, vocab = raw_logits.shape
                flat_logits = raw_logits.view(-1, vocab)           # [B*S, V]
                flat_labels = labels.view(-1)                      # [B*S]
                flat_masked = masked_positions.view(-1)            # [B*S]
                if flat_masked.any():
                    # Gather the rows of syn_mask for the *masked* true labels.
                    # Use embedding() instead of index_select() (often nicer with torch.compile)
                    masked_labels = flat_labels[flat_masked]       # contains only >= 0 ids
                    add = torch.nn.functional.embedding(masked_labels, syn_mask)  # [M, V]
                    flat_logits = flat_logits.clone()
                    flat_logits[flat_masked] = flat_logits[flat_masked] + add
                constrained_logits = flat_logits.view(bsz, seqlen, vocab)


        # Loss path (train-time toggle) + "train_acc"
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            if self.apply_syn_mask_train and (syn_mask is not None):
                loss = loss_fct(constrained_logits.view(-1, constrained_logits.size(-1)), labels.view(-1))
                preds_train = torch.argmax(constrained_logits, dim=-1)
            else:
                loss = loss_fct(raw_logits.view(-1, raw_logits.size(-1)), labels.view(-1))
                preds_train = torch.argmax(raw_logits, dim=-1)

            if masked_positions is not None and masked_positions.any():
                correct_train = ((preds_train == labels) & masked_positions).sum().float()
                denom = masked_positions.sum().float().clamp_min(1.0)
                train_acc = correct_train / denom
            else:
                train_acc = torch.tensor(0.0, device=device)
        else:
            loss = torch.tensor(0.0, device=device)
            train_acc = torch.tensor(0.0, device=device)

        # Diagnostics on masked tokens
        acc_syn = torch.tensor(0.0, device=device)
        acc_nosyn = torch.tensor(0.0, device=device)
        acc_aa = torch.tensor(0.0, device=device)
        if (labels is not None) and (masked_positions is not None) and masked_positions.any():
            preds_constrained = torch.argmax(constrained_logits, dim=-1)
            preds_raw = torch.argmax(raw_logits, dim=-1)
            denom = masked_positions.sum().float().clamp_min(1.0)

            correct_syn = ((preds_constrained == labels) & masked_positions).sum().float()
            acc_syn = correct_syn / denom

            correct_nosyn = ((preds_raw == labels) & masked_positions).sum().float()
            acc_nosyn = correct_nosyn / denom

            if aa_map is not None:
                # AA-level (raw) computed only on masked tokens → no -100 indexing
                aa_pred_masked = aa_map[preds_raw[masked_positions]]
                aa_true_masked = aa_map[labels[masked_positions]]
                correct_aa = (aa_pred_masked == aa_true_masked).sum().float()
                acc_aa = correct_aa / denom

        if return_logits:
            return (loss, train_acc, constrained_logits, raw_logits, labels, masked_positions,
                    acc_syn, acc_nosyn, acc_aa)
        else:
            return loss, train_acc


class SynCodonDebertaLM(DebertaV2ForMaskedLM, _SynCodonBase):
    """DeBERTa-v2 masked LM with optional synonymous-codon masking (train-time toggle)."""
    def __init__(self, config, mask_matrix=None, tokenizer=None,
                 apply_syn_mask_train=True, token_id_to_aa=None):
        DebertaV2ForMaskedLM.__init__(self, config)
        _SynCodonBase.__init__(self, mask_matrix=mask_matrix, tokenizer=tokenizer,
                               apply_syn_mask_train=apply_syn_mask_train, token_id_to_aa=token_id_to_aa)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, return_logits=False):
        outputs = super().forward(
            input_ids=input_ids.to(torch.long),
            attention_mask=attention_mask.to(torch.long),
            token_type_ids=token_type_ids.to(torch.long),
            labels=None
        )
        raw_logits = outputs.logits
        return self._postprocess_from_raw(raw_logits, labels=labels, return_logits=return_logits)


class SynCodonBertLM(BertForMaskedLM, _SynCodonBase):
    """BERT masked LM with optional synonymous-codon masking (train-time toggle)."""
    def __init__(self, config, mask_matrix=None, tokenizer=None,
                 apply_syn_mask_train=True, token_id_to_aa=None):
        BertForMaskedLM.__init__(self, config)
        _SynCodonBase.__init__(self, mask_matrix=mask_matrix, tokenizer=tokenizer,
                               apply_syn_mask_train=apply_syn_mask_train, token_id_to_aa=token_id_to_aa)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, return_logits=False):
        outputs = super().forward(
            input_ids=input_ids.to(torch.long),
            attention_mask=attention_mask.to(torch.long),
            token_type_ids=token_type_ids.to(torch.long),
            labels=None
        )
        raw_logits = outputs.logits
        return self._postprocess_from_raw(raw_logits, labels=labels, return_logits=return_logits)

# ==================
# LIGHTNING MODULE
# ==================
class CDSembedLightningModule(pl.LightningModule):
    def __init__(self, model, tokenizer, train_loader, val_loader, save_prefix, all_groups, token_id_to_aa):
        super().__init__()
        self.model = model
        # In CDSembedLightningModule.__init__(...)
        # Try to enable gradient checkpointing if supported; fall back gracefully.
        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                # (optional) set config flag to hint HF
                if hasattr(self.model, "config"):
                    setattr(self.model.config, "gradient_checkpointing", True)
                # HF sometimes requires inputs to require grads for checkpointing
                if hasattr(self.model, "enable_input_require_grads"):
                    self.model.enable_input_require_grads()
                self.model.gradient_checkpointing_enable()
                print("[INFO] Gradient checkpointing enabled.")
            except ValueError as e:
                # BERT path may not support it; just continue without GC
                print(f"[WARN] Gradient checkpointing not supported for this model: {e}")
            except Exception as e:
                print(f"[WARN] Failed to enable gradient checkpointing, continuing without it: {e}")
        self.tokenizer = tokenizer
        self.automatic_optimization = False
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_prefix = save_prefix

        self.all_groups = all_groups
        self.num_groups = len(all_groups)
        self.token_id_to_aa = token_id_to_aa  # not used here directly, stored in model

        self.train_loss_sum = 0.0
        self.train_accuracy_sum = 0.0
        self.lr_sum = 0.0
        self.step_count = 0

    def on_train_epoch_start(self):
        current_epoch = self.current_epoch
        self.saved_halfway = False
        print(f"Starting epoch {current_epoch}")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        total_batches = len(self.train_dataloader())
        halfway_point = total_batches // 2
        # (optional snapshots disabled)

    def training_step(self, batch, batch_idx):
        self.model.train()
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        batch = {k: v.to(self.device) for k, v in batch.items()}
        loss, train_acc = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'],
            labels=batch['labels']
        )
        
        # --- quick MLM sanity check ---
        assert batch["input_ids"].dtype == torch.long, "input_ids must be long"
        assert batch["labels"].dtype    == torch.long, "labels must be long"

        neg100 = (batch["labels"] == -100).sum().item()
        total  = batch["labels"].numel()
        print(f"[MLM sanity] -100 count: {neg100} ({100.0*neg100/max(1,total):.2f}% of tokens)")

        self.manual_backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        self.train_loss_sum += loss.item()
        self.train_accuracy_sum += train_acc.item()
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

    # ---- NEW: epoch-level accumulators for groupwise & overall diagnostics
    def on_validation_epoch_start(self):
        device = self.device
        G = self.num_groups
        # groupwise
        self._grp_loss_sums = torch.zeros(G, dtype=torch.float32, device=device)
        self._grp_example_counts = torch.zeros(G, dtype=torch.float32, device=device)
        self._grp_mask_counts = torch.zeros(G, dtype=torch.float32, device=device)
        self._grp_correct_syn = torch.zeros(G, dtype=torch.float32, device=device)
        self._grp_correct_nosyn = torch.zeros(G, dtype=torch.float32, device=device)
        self._grp_correct_aa = torch.zeros(G, dtype=torch.float32, device=device)
        # overall
        self._tot_loss_sum = torch.tensor(0.0, device=device)
        self._tot_examples = torch.tensor(0.0, device=device)
        self._tot_masked = torch.tensor(0.0, device=device)
        self._tot_correct_syn = torch.tensor(0.0, device=device)
        self._tot_correct_nosyn = torch.tensor(0.0, device=device)
        self._tot_correct_aa = torch.tensor(0.0, device=device)

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # get both constrained and raw logits with diagnostics
        (loss, _train_acc, logits_con, logits_raw, labels, masked_positions,
         acc_syn, acc_nosyn, acc_aa) = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'],
            labels=batch['labels'],
            return_logits=True
        )

        # Log classic scalar loss for compatibility
        self.log('val_loss', loss, sync_dist=True, prog_bar=True, reduce_fx=torch.mean, on_epoch=True)

        # per-example token-level CE (based on the training loss path; to log per-group loss we recompute with constrained logits)
        ce = torch.nn.functional.cross_entropy(
            logits_con.view(-1, logits_con.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction='none'
        ).view_as(labels)  # [B, S]
        masked = masked_positions  # [B, S] bool

        # per-example loss = mean over masked tokens in the example
        per_ex_loss = (ce * masked).sum(dim=1) / masked.sum(dim=1).clamp_min(1)
        per_ex_masked = masked.sum(dim=1).float().clamp_min(0.0)

        # per-batch overall accumulators
        self._tot_loss_sum += per_ex_loss.sum()
        self._tot_examples += torch.tensor(float(per_ex_loss.size(0)), device=self.device)
        self._tot_masked += per_ex_masked.sum()

        # From returned accuracies, compute numerators via counts
        # Convert acc * masked_count back to count of correct
        correct_syn = acc_syn * per_ex_masked.sum()  # (scalar per batch) but acc_syn is averaged across all masked tokens in the batch
        correct_nosyn = acc_nosyn * per_ex_masked.sum()
        correct_aa = acc_aa * per_ex_masked.sum()

        # NOTE: acc_* as computed above are batch-wise; for exact group numerators, recompute by token below
        preds_constrained = torch.argmax(logits_con, dim=-1)
        preds_raw = torch.argmax(logits_raw, dim=-1)

        per_ex_correct_syn = (((preds_constrained == labels) & masked).sum(dim=1)).float()
        per_ex_correct_nosyn = (((preds_raw == labels) & masked).sum(dim=1)).float()

        # AA-level (raw) -- SAFE, masked-only, device-correct
        aa_map = self.model._aa_map  # [V]
        # Ensure aa_map is on the same device as preds_raw (typically GPU)
        if aa_map.device != preds_raw.device:
            aa_map = aa_map.to(preds_raw.device)

        # Compute AA-level correctness only on masked tokens (so we never touch -100)
        # masked is a [B, S] bool tensor you already have
        preds_raw_masked = preds_raw[masked]     # [N_masked_tokens]
        labels_masked    = labels[masked]        # [N_masked_tokens], all >= 0 here

        aa_pred_masked = aa_map[preds_raw_masked]
        aa_true_masked = aa_map[labels_masked]
        correct_masked = (aa_pred_masked == aa_true_masked).to(torch.float32)  # [N_masked_tokens]

        # Re-aggregate per example
        B = labels.size(0)
        ex_idx = torch.arange(B, device=labels.device).unsqueeze(1).expand_as(labels)  # [B, S]
        ex_idx_masked = ex_idx[masked]                                                 # [N_masked_tokens]

        per_ex_correct_aa = torch.zeros(B, device=labels.device, dtype=torch.float32)
        per_ex_correct_aa.index_add_(0, ex_idx_masked, correct_masked)

        # group ids for accumulation
        group_ids = batch['group_id']  # [B] long
        ones = torch.ones_like(per_ex_loss, dtype=torch.float32, device=self.device)

        # groupwise accumulators
        self._grp_loss_sums.index_add_(0, group_ids, per_ex_loss.float())
        self._grp_example_counts.index_add_(0, group_ids, ones)
        self._grp_mask_counts.index_add_(0, group_ids, per_ex_masked)
        self._grp_correct_syn.index_add_(0, group_ids, per_ex_correct_syn)
        self._grp_correct_nosyn.index_add_(0, group_ids, per_ex_correct_nosyn)
        self._grp_correct_aa.index_add_(0, group_ids, per_ex_correct_aa)

    def on_validation_epoch_end(self):
        # overall (micro averages)
        total_mask = self._grp_mask_counts.sum().clamp_min(1.0)
        val_acc_syn = self._grp_correct_syn.sum() / total_mask
        val_acc_nosyn = self._grp_correct_nosyn.sum() / total_mask
        val_acc_aa = self._grp_correct_aa.sum() / total_mask

        self.log('val_acc_syn', val_acc_syn, sync_dist=True, prog_bar=True, reduce_fx=torch.mean, on_epoch=True)
        self.log('val_acc_nosyn', val_acc_nosyn, sync_dist=True, prog_bar=False, reduce_fx=torch.mean, on_epoch=True)
        self.log('val_acc_aa', val_acc_aa, sync_dist=True, prog_bar=False, reduce_fx=torch.mean, on_epoch=True)

        # per-group logs
        avg_loss = self._grp_loss_sums / (self._grp_example_counts.clamp_min(1))
        acc_syn = self._grp_correct_syn / (self._grp_mask_counts.clamp_min(1))
        acc_nosyn = self._grp_correct_nosyn / (self._grp_mask_counts.clamp_min(1))
        acc_aa = self._grp_correct_aa / (self._grp_mask_counts.clamp_min(1))

        for g in range(self.num_groups):
            name = str(self.all_groups[g])
            self.log(f'val_loss_group_{name}', avg_loss[g], prog_bar=False, sync_dist=True, reduce_fx=torch.mean, on_epoch=True)
            self.log(f'val_acc_syn_group_{name}', acc_syn[g], prog_bar=False, sync_dist=True, reduce_fx=torch.mean, on_epoch=True)
            self.log(f'val_acc_nosyn_group_{name}', acc_nosyn[g], prog_bar=False, sync_dist=True, reduce_fx=torch.mean, on_epoch=True)
            self.log(f'val_acc_aa_group_{name}', acc_aa[g], prog_bar=False, sync_dist=True, reduce_fx=torch.mean, on_epoch=True)

    def configure_optimizers(self):
        base_lr = 1e-4
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


# =============
# ARG PARSING
# =============
def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain masked LM on CDS with optional synonymous codon masking, token-type controls, selectable arch, and per-group diagnostics.")
    # Architecture
    parser.add_argument("--model-arch", type=str, choices=["deberta", "bert"], default="deberta")
    # Syn mask toggle (train-time)
    syn_mask = parser.add_mutually_exclusive_group()
    syn_mask.add_argument("--use-syn-mask", dest="use_syn_mask", action="store_true",
                          help="Enable synonymous codon masking for the training loss (default).")
    syn_mask.add_argument("--no-syn-mask", dest="use_syn_mask", action="store_false",
                          help="Disable synonymous codon masking for the training loss.")
    parser.set_defaults(use_syn_mask=True)
    # Token type usage mode
    parser.add_argument("--token-type-mode", type=str, choices=["species", "none", "grouped"], default="species")
    # Training/runtime knobs
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--tokenizer-path", type=str, default="./SynCodonLM")
    parser.add_argument("--dataset-csv", type=str, default="cds-dataset-new.csv.gz")
    # Naming for saves
    parser.add_argument("--save-prefix", type=str, default="SynCodonLM", help="Prefix for periodic snapshot dirs.")
    parser.add_argument("--final-save-name", type=str, default="finalmodel", help="Dir name for final save.")
    return parser.parse_args()


# =========
# MAIN
# =========
def main():
    args = parse_args()
    num_nodes = args.nodes
    gpus_per_node = args.gpus_per_node
    num_workers = args.num_workers
    batch_size = args.batch_size

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)

    # type_vocab_size according to token-type mode
    if args.token_type_mode == "none":
        type_vocab_size = 1
    elif args.token_type_mode == "grouped":
        type_vocab_size = 9
    else:
        type_vocab_size = 501  # original species granularity

    # Build configs by architecture (Option B applied to both)

    if args.model_arch == "deberta":
        config = DebertaV2Config(
            vocab_size=tokenizer.vocab_size,
            hidden_size=320,                 # was 512 -> 320
            intermediate_size=1280,          # keep ~4x hidden size
            num_hidden_layers=6,             # was 8/12 -> 6
            num_attention_heads=5,           # 320 / 5 = 64 dims/head
            hidden_act="gelu_new",
            legacy=True,
            hidden_dropout_prob=0.1,
            type_vocab_size=type_vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            max_position_embeddings=1024,
            relative_attention=True,
            pos_att_type="p2c|c2p",
            position_biased_input=False,
        )
    else:  # 'bert'
        config = BertConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=320,                 # was 512 -> 320
            intermediate_size=1280,          # keep ~4x hidden size
            num_hidden_layers=6,             # was 8 -> 6
            num_attention_heads=5,           # 320 / 5 = 64 dims/head
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            type_vocab_size=type_vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            max_position_embeddings=1024,
        )

    # --- ALWAYS build a mask matrix and aa mapping for diagnostics (NEW)
    mask_matrix = build_mask_matrix(tokenizer)  # used for diagnostics and (optionally) training loss
    token_id_to_aa = build_token_id_to_aa_index(tokenizer)

    # Instantiate model with a training-time toggle
    if args.model_arch == "deberta":
        model = SynCodonDebertaLM(
            config, mask_matrix=mask_matrix, tokenizer=tokenizer,
            apply_syn_mask_train=args.use_syn_mask, token_id_to_aa=token_id_to_aa
        )
    else:
        model = SynCodonBertLM(
            config, mask_matrix=mask_matrix, tokenizer=tokenizer,
            apply_syn_mask_train=args.use_syn_mask, token_id_to_aa=token_id_to_aa
        )
    #model = torch.compile(model)

    # Load dataset (auto-detect Group column) (NEW)
    # Try to include 'Group' if present; otherwise fall back gracefully.
    try:
        df = pd.read_csv(args.dataset_csv, usecols=["CDS", "Species", "Set", "Group"], nrows=100000)
        has_group_col = True
    except Exception:
        df = pd.read_csv(args.dataset_csv, usecols=["CDS", "Species", "Set"], nrows=100000)
        has_group_col = False

    train_df = df[df["Set"] == "Train"]
    test_df = df[df["Set"] == "Test"]

    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    # Build group ids and names
    if has_group_col:
        all_groups = sorted(df["Group"].dropna().unique().tolist())
        group_to_id = {g: i for i, g in enumerate(all_groups)}
        train_groups_list = [group_to_id[g] for g in train_df["Group"].tolist()]
        test_groups_list = [group_to_id[g] for g in test_df["Group"].tolist()]
    else:
        # Derive macro groups from species -> cluster -> macro id
        # Keep names fixed to MACRO_GROUP_NAMES
        all_groups = MACRO_GROUP_NAMES
        def species_to_macro(species):
            cl = species_token_type.get(species, 500)
            return map_cluster_to_group(int(cl))
        train_groups_list = [species_to_macro(s) for s in train_df["Species"].tolist()]
        test_groups_list = [species_to_macro(s) for s in test_df["Species"].tolist()]

    train_sequences = train_df["CDS"].tolist()
    train_species = train_df["Species"].tolist()
    test_sequences = test_df["CDS"].tolist()
    test_species = test_df["Species"].tolist()
    del df, train_df, test_df
    gc.collect()

    train_dataset = CDSDataset(train_sequences, train_species, tokenizer, token_type_mode=args.token_type_mode,
                               groups_list=train_groups_list)
    test_dataset = CDSDataset(test_sequences, test_species, tokenizer, token_type_mode=args.token_type_mode,
                              groups_list=test_groups_list)
    del train_sequences, train_species, test_sequences, test_species
    gc.collect()

    data_collator = CustomDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
        return_tensors="pt"
    )

    def custom_collate_fn(batch):
        # Preserve group_id across the collator (NEW)
        group_ids = torch.tensor([item["group_id"] for item in batch], dtype=torch.long)
        for item in batch:
            item.pop("group_id")
        batch_out = data_collator(batch)
        batch_out['attention_mask'] = (batch_out['input_ids'] != tokenizer.pad_token_id).long()
        batch_out['group_id'] = group_ids
        return batch_out

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

    lightning_model = CDSembedLightningModule(
        model, tokenizer, train_loader, test_loader,
        save_prefix=args.save_prefix,
        all_groups=all_groups,
        token_id_to_aa=token_id_to_aa
    )

    progress_bar = TQDMProgressBar(refresh_rate=30)
    trainer = pl.Trainer(
        precision='16-mixed',
        max_epochs=args.epochs,
        devices=gpus_per_node,
        num_nodes=num_nodes,
        strategy=pl.strategies.DDPStrategy(static_graph=False),
        callbacks=[progress_bar],
        log_every_n_steps=1,
        enable_checkpointing=False,
        val_check_interval=0.1
    )

    trainer.fit(lightning_model, train_loader, test_loader)

    final_dir = f"./{args.final_save_name}"
    save_model_and_tokenizer(lightning_model.model, tokenizer, final_dir)
    print('PROCESS COMPLETED PROPERLY')


if __name__ == "__main__":
    main()