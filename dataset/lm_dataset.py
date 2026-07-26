from gc import enable

from torch.utils.data import Dataset
import os
import torch
import random
from datasets import load_dataset
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    has_tools = conversations and conversations[0].get("role") == "system" and (conversations[0].get("functions") or conversations[0].get("tools"))
    if has_tools or any(conv.get("role") == "tool" for conv in conversations): return conversations
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
        import json
        
        self.samples = []
        print(f"📦 正在加载并过滤数据: {jsonl_path}")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                try:
                    data = json.loads(line)
                    # 🔥 核心过滤逻辑：确保 conversations 存在且每个 content 都不为空
                    convs = data.get("conversations", [])
                    if convs and all(msg.get("content", "").strip() for msg in convs):
                        self.samples.append(data)
                except Exception:
                    continue # 跳过彻底损坏的 JSON 行

        print(f"✅ 加载完成！原始行数约 90万，过滤后有效样本: {len(self.samples)}")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
    def __len__(self):
        return len(self.samples)
    def create_chat_prompt(self, conversions):
        tools = conversions[0].get("functions", None) if (conversions and conversions[0]["role"] == "system") else None
        return self.tokenizer.apply_chat_template(conversions, tokenize=False, add_generation_prompt=False, tools=tools)
    def generate_labels(self, prompt, input_ids):
        labels = [-100] * len(input_ids)

        # 🔥 用字符串找 assistant 起点（稳定）
        assistant_tag = "<|im_start|>assistant\n"
        end_tag = "<|im_end|>\n"

        start_pos = 0

        while True:
            # 找 assistant 起点
            start_idx = prompt.find(assistant_tag, start_pos)
            if start_idx == -1:
                break

            # 找结束
            end_idx = prompt.find(end_tag, start_idx)
            if end_idx == -1:
                break

            end_idx += len(end_tag)

            # 🔥 转成 token 位置
            prefix_ids = self.tokenizer(prompt[:start_idx]).input_ids
            target_ids = self.tokenizer(prompt[start_idx:end_idx]).input_ids

            start_token = len(prefix_ids)
            end_token = min(start_token + len(target_ids), len(input_ids))

            # 写 labels
            for i in range(start_token, end_token):
                labels[i] = input_ids[i]

            start_pos = end_idx

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
        labels = self.generate_labels(prompt, input_ids)
        # print("labels:" + str(sum([1 for x in labels if x != -100])))
        for i in range(len(input_ids)):
            if input_ids[i] == self.tokenizer.pad_token_id:
                labels[i] = -100
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    
class DPODataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=4096):
        super().__init__()
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.samples)
    def generate_loss_mask(self, input_ids, prompt):
        assistant_tag = "<|im_start|>assistant\n"
        end_tag = "<|im_end|>\n"
        mask = [0] * len(input_ids)
        start_pos = 0
        while True:
            start_idx = prompt.find(assistant_tag, start_pos)
            if start_idx == -1:
                break
            end_idx = prompt.find(end_tag, start_pos)
            if end_idx == -1:
                break
            end_idx += len(end_tag)
            prefix_ids = self.tokenizer(prompt[:start_idx]).input_ids
            target_ids = self.tokenizer(prompt[start_idx:end_idx]).input_ids
            start_token = len(prefix_ids)
            end_token = min(start_token + len(target_ids), len(input_ids))
            mask[start_token:end_token] = [1] * (end_token - start_token)
            start_pos = end_idx
        return mask
    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  
        chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
        rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)
        chosen_prompt = post_processing_chat(chosen_prompt)
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(chosen_prompt, truncation=True, max_length=self.max_length, padding="max_length")
        rejected_encoding = self.tokenizer(rejected_prompt, truncation=True, max_length=self.max_length, padding="max_length")
        chosen_ids = chosen_encoding.input_ids
        rejected_ids = rejected_encoding.input_ids
        chosen_mask = self.generate_loss_mask(chosen_ids, chosen_prompt)
        rejected_mask = self.generate_loss_mask(rejected_ids, rejected_prompt)
        x_chosen = torch.tensor(chosen_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_ids[1:], dtype=torch.long)
        chosen_mask = torch.tensor(chosen_mask[1:], dtype=torch.long)
        rejected_mask = torch.tensor(rejected_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_ids[1:], dtype=torch.long)
        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': chosen_mask,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': rejected_mask
        }
class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.samples = load_dataset('json', data_files=jsonl_path, split="train")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio
    def __len__(self):
        return len(self.samples)
    def create_chat_template(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        prompt = self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )
        return prompt
    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = sample["conversations"]
        prompt = self.create_chat_template(conversations)
        answer = conversations[-1]["content"] if conversations else ""
        return {
            "prompt": prompt,
            "answer": ""
        }
    
class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))
    def __len__(self):
        return len(self.samples)
    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for m in conversations:
            m = dict(m)
            if m.get("role") == "system" and m.get("tools"):
                tools = json.loads(m["tools"]) if isinstance(m["tools"], str) else m["tools"]
            messages.append(m)
        return messages[:-1], tools
    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample["conversations"])
        return{
            "messages": messages,
            "gt": sample["gt"],
            "tools": tools
        }

        

    




        
