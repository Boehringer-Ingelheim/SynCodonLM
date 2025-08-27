import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from typing import Optional
from .utils import clean_split_sequence


class CodonEmbeddings:
    """Class to simplify model usage :)"""
    def __init__(self, model_name: str = "jheuschkel/SynCodonLM", device: Optional[str] = None):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, config=self.config).to(self.device).eval()

    def get_mean_embedding(self, sequence, species_token_type=500, layer=-1):
        sequence = clean_split_sequence(sequence)
        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        inputs['token_type_ids'] = torch.full_like(inputs['input_ids'], species_token_type) # manually set token_type_ids
        outputs = self.model(**inputs, output_hidden_states=True)
        embedding = outputs.hidden_states[layer] #this can also index any layer (0-11)
        mean_embedding = torch.mean(embedding, dim=1).squeeze(0)
        return mean_embedding
    
    def get_raw_embeddings(self, sequence, species_token_type=500):
        sequence = clean_split_sequence(sequence)
        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        inputs['token_type_ids'] = torch.full_like(inputs['input_ids'], species_token_type) # manually set token_type_ids
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs