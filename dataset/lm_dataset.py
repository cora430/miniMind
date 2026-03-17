from sympy import false
from torch.utils.data import Dataset
import os
import torch
import random
from datasets import load_dataset
from transformers.activations import FastGELUActivation
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    def __init__(self, datapath, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=datapath, split="train")
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample["text"]), add_special_tokens = False, truncation = True, max_length = self.max_length - 2).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eof_token_id]
        input_ids = tokens + (self.max_length - len(tokens)) * [self.tokenizer.pad_token_id]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels