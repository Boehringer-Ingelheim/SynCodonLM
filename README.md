![SynCodonLM Logo](SynCodonLM/logo/logo.jpg)


# Advancing Codon Language Modeling with Synonymous Codon Constrained Masking



- This repository contains code to utilize the model, and reproduce results of the preprint [**Advancing Codon Language Modeling with Synonymous Codon Constrained Masking**](link), by **James Heuschkel**, **Laura Kingsley**, **Noah Pefaur**, **Andrew Nixon**, and **Steven Cramer**.
- Unlike other Codon Language Models, SynCodonLM was trained with logit-level control, masking logits for non-synonymous codons. This allowed the model to learn codon-specific patterns disentangled from protein-level semantics.
- [Pre-training dataset of 66 Million CDS is available on Hugging Face here.](https://huggingface.co/datasets/jheuschkel/cds-dataset)
---
## Installation

```python
git clone https://github.com/Boehringer-Ingelheim/SynCodonLM.git
pip install -r requirements.txt
```
---
# Usage
## Prepare Sequence

```python
from SynCodonLM.utils import clean_split_sequence
seq = 'ATGTCCACCGGGCGGTGA'
seq = clean_split_sequence(seq)  # Returns: 'ATG TCC ACC GGG CGG TGA'
```

## Load Model & Tokenizer from Hugging Face
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
import torch

tokenizer = AutoTokenizer.from_pretrained("jheuschkel/SynCodonLM")
config = AutoConfig.from_pretrained("jheuschkel/SynCodonLM")
model = AutoModelForMaskedLM.from_pretrained("jheuschkel/SynCodonLM", config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```
### If there are networking issues, you can manually [download the model from Hugging Face](https://huggingface.co/jheuschkel/SynCodonLM/resolve/main/model.safetensors?download=true) & place it in the /SynCodonLM directory
```python
tokenizer = AutoTokenizer.from_pretrained("./SynCodonLM", trust_remote_code=True)
config = AutoConfig.from_pretrained("./SynCodonLM", trust_remote_code=True)
model = AutoModel.from_pretrained("./SynCodonLM", trust_remote_code=True, config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

```

## Tokenize Input Sequences, Set Token Type ID Based on Species ID found in [linktospeciestokentype]

```python
token_type_id = 67  #E. coli
inputs = tokenizer(seq, return_tensors="pt").to(device)
inputs['token_type_ids'] = torch.full_like(inputs['input_ids'], token_type_id) # manually set token_type_ids
```

## Gather Model Outputs
```python
outputs = model(**inputs, output_hidden_states=True)
```

## Get Mean Embedding from Final Layer
```python
embedding = outputs.hidden_states[-1] #this can also index any layer (0-11)
mean_embedding = torch.mean(embedding, dim=1).squeeze(0)
```

## You Can Also View Language Head Output
```python
logits = outputs.logits  # shape: [batch_size, sequence_length, vocab_size]
```

## Citation
-If you use this model, please cite our preprint:

-----

## Usage With Batches
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
import torch
from SynCodonLM.utils import clean_split_sequence

tokenizer = AutoTokenizer.from_pretrained("jheuschkel/SynCodonLM")
config = AutoConfig.from_pretrained("jheuschkel/SynCodonLM")
model = AutoModelForMaskedLM.from_pretrained("jheuschkel/SynCodonLM", config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# List of sequences
seqs = [
    'ATGTCCACCGGGCGGTGA',
    'ATGCGTACCGGGTAGTGA',
    'ATGTTTACCGGGTGGTGA'
]

# List of token type ids (species)
species_token_type_ids = [
    67,   # E. coli
    394,  # C. griseus
    317   # H. sapiens
]

# Prepare list
seqs = [clean_split_sequence(seq) for seq in seqs]

# Tokenize batch with padding
inputs = tokenizer(seqs, return_tensors="pt", padding=True).to(device)

# Create token_type_ids tensor
batch_size, seq_len = inputs['input_ids'].shape
token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long).to(device)

# Fill each row with the species-specific token_type_id
for i, species_id in enumerate(species_token_type_ids):
    token_type_ids[i, :] = species_id  # Fill entire row with the species ID

# Add to inputs
inputs['token_type_ids'] = token_type_ids

# Run model
outputs = model(**inputs)
```


