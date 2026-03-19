from sympy import false
from torch.utils.data import Dataset
import os
import torch
import random
from datasets import load_dataset
from transformers.activations import FastGELUActivation
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    if conversations and conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [{"role": "system", "content": random.choice(SYSTEM_PROMPTS)}] + conversations    
    return conversations
def normalize_conversations(conversations):
    # 防止多套一层 list
    while isinstance(conversations, list) and len(conversations) > 0 and isinstance(conversations[0], list):
        conversations = conversations[0]
    return conversations
def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', "")
    return prompt_content
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
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + (self.max_length - len(tokens)) * [self.tokenizer.pad_token_id]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels

class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
    def __len__(self):
        return len(self.samples)
    def create_chat_prompt(self, conversions):
        tools = conversions[0].get("functions", None) if (conversions and conversions[0]["role"] == "system") else None
        return self.tokenizer.apply_chat_template(conversions, tokenize=False, add_generation_prompt=False, tools=tools)
    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i+len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end+len(self.eos_id), len(input_ids))):
                    labels[j] = input_ids[j]
                i = min(end+len(self.eos_id), len(input_ids))
            else:
                i += 1

        return labels
    def __getitem__(self, index):
        conversations = self.samples[index]["conversations"]
        # {"conversations": [
        # [ {...}, {...} ],
        # [ {...}, {...} ]
        # ]}  异常样本
        conversations = normalize_conversations(conversations)
        conversations = pre_processing_chat(conversations)
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        print("labels:" + str(sum([1 for x in labels if x != -100])))
        for i in range(len(input_ids)):
            if input_ids[i] == self.tokenizer.pad_token_id:
                labels[i] = -100
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    

    




        
