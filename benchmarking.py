import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import numpy as np
from SynCodonLM import clean_split_sequence

dataset_token_type_map = {
    1: 67,
    2: 67,
    3: 67,
    4: 67,
    5: 108,
    6: 108,
    7: 108
}  #assigns token type based off species grouping to input embeddings




# Load data
data = pd.read_csv('./benchmarking-datasets/evaluation-datasets.csv')

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, outputs):
        self.embeddings = embeddings
        self.outputs = outputs

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.outputs[idx]


# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("jheuschkel/SynCodonLM")
config = AutoConfig.from_pretrained("jheuschkel/SynCodonLM")
base_model = AutoModelForMaskedLM.from_pretrained("jheuschkel/SynCodonLM", config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)
base_model.eval()



# freeze all parameters
for param in base_model.parameters():
    param.requires_grad = False

def compute_embeddings(sequences, token_type_id=0):
    embeddings = []
    for seq in tqdm(sequences, desc="Computing embeddings"):
        seq = clean_split_sequence(seq)
        inputs = tokenizer(seq, return_tensors="pt").to(device)

        # manually set token_type_ids
        inputs['token_type_ids'] = torch.full_like(inputs['input_ids'], token_type_id)
        with torch.no_grad():
            outputs = base_model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            mean_embedding = torch.mean(last_hidden, dim=1).squeeze(0)
            embeddings.append(mean_embedding)
    return torch.stack(embeddings)



class LinearHeadModel(nn.Module):
    def __init__(self):
        super(LinearHeadModel, self).__init__()
        self.linear = nn.Linear(768, 1)

    def forward(self, embeddings):
        output = self.linear(embeddings).squeeze()
        return output.view(-1)


# evaluate function
def evaluate(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for embeddings, targets in loader:
            embeddings = embeddings.to(device)
            targets = targets.to(device)
            preds = model(embeddings)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    return np.array(all_preds), np.array(all_targets)


# regularization strengths
l1_lambda = 0.001
l2_lambda = 0.001

all_results = []

for seed in range(20):  # 20 different seeds for statistical significance
    print(f"Starting Seed: {seed}")
    for dataset_id, group in data.groupby('Dataset'):
        print(f'Processing Dataset: {dataset_id} with Seed {seed}')
        
        token_type_id = dataset_token_type_map.get(dataset_id, 0)

        sequences = group['Sequence'].tolist()
        outputs = group['Output'].tolist()


        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        fold_metrics = []

        all_embeddings = compute_embeddings(sequences, token_type_id=token_type_id)


        for fold, (train_idx, val_idx) in enumerate(kf.split(sequences)):
            train_embeddings = all_embeddings[train_idx]
            val_embeddings = all_embeddings[val_idx]
            train_outputs = torch.tensor([outputs[i] for i in train_idx])
            val_outputs = torch.tensor([outputs[i] for i in val_idx])

            train_dataset = EmbeddingDataset(train_embeddings, train_outputs)
            val_dataset = EmbeddingDataset(val_embeddings, val_outputs)

            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16)

            model = LinearHeadModel().to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=l2_lambda)

            best_val_loss = float('inf')
            
            best_model_state = None

            for epoch in range(100):
                model.train()
                total_loss = 0
                for batch_sequences, batch_outputs in train_loader:
                    batch_outputs = batch_outputs.float().to(device)
                    predictions = model(batch_sequences)

                    l1_norm = sum(p.abs().sum() for p in model.linear.parameters())
                    loss = criterion(predictions, batch_outputs) + l1_lambda * l1_norm

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += loss.item() * len(batch_outputs)

                val_preds, val_targets = evaluate(model, val_loader)
                val_loss = mean_squared_error(val_targets, val_preds)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()

            model.load_state_dict(best_model_state)

            val_preds, val_targets = evaluate(model, val_loader)
            val_loss = mean_squared_error(val_targets, val_preds)
            val_r2 = r2_score(val_targets, val_preds)
            val_pearson = pearsonr(val_targets, val_preds)[0]
            val_spearman = spearmanr(val_targets, val_preds)[0]

            fold_metrics.append((val_loss, val_r2, val_pearson, val_spearman))
            all_results.append({
                'Seed': seed,
                'Dataset': dataset_id,
                'Fold': fold + 1,
                'MSE_Loss': val_loss,
                'R2': val_r2,
                'Pearson': val_pearson,
                'Spearman': val_spearman
            })

        avg_metrics = np.mean(fold_metrics, axis=0)
        all_results.append({
            'Seed': seed,
            'Dataset': dataset_id,
            'Fold': 'Mean',
            'MSE_Loss': avg_metrics[0],
            'R2': avg_metrics[1],
            'Pearson': avg_metrics[2],
            'Spearman': avg_metrics[3]
        })


results_df = pd.DataFrame(all_results)


mean_df = results_df[results_df['Fold'] == 'Mean']


with pd.ExcelWriter('cross_validation_metrics_SynCodonLM.xlsx', engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='All_Results', index=False)
    mean_df.to_excel(writer, sheet_name='Mean_Only', index=False)

print("Saved results")


