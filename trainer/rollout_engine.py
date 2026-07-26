import torch
from torch import Tensor
from typing import Optional, List, Tuple
from torch.nn.parallel import DistributedDataParallel
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch.nn as nn
from contextlib import nullcontext
from transformers import AutoTokenizer
import requests
import os
def compute_per_token_logps(model, input_ids: Tensor, logits_to_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:
    if logits_to_keep <= 0:
        return input_ids.new_empty((input_ids.shape[0], 0), dtype=torch.float32)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    logits = raw_model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits[:, :-1, :]
    # logits -> B, logits_to_keep, vocab_size
    # input_ids -> B, seq_len
    logp = []
    for logit, ids in zip(logits, input_ids[:, -logits_to_keep:]):
        # logit -> logits_to_keep, vocab_size
        # ids -> logits_to_keep
        ids = ids.detach().clone() if ids.is_inference() else ids
        logp.append(torch.gather(input=logit.log_softmax(-1), dim=1, index=ids.unsqueeze(1)).squeeze())
    return torch.stack(logp, dim=0) # B, logits_to_keep 这里用stack是为了不丢失梯度
@dataclass
class RolloutResult:
    output_ids: Tensor
    completion_ids: Tensor
    per_token_logps: Tensor
    completions: List[str]
class RolloutEngine(ABC):
    tokenizer = None
    @abstractmethod
    def rollout(self, input_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        pass
    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        pass
class TorchRolloutEngine(RolloutEngine):
    def __init__(self, policy_model: nn.Module, tokenizer, device: str = "cuda", autocast_ctx = None):
        
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx
    def rollout(self, input_ids, attention_mask, num_generations, max_new_tokens, temperature = 0.8) -> RolloutResult:
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
        with torch.no_grad(), ctx:
            output_ids = model.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                max_new_tokens = max_new_tokens,
                do_sample = True,
                temperature = temperature,
                num_return_sequences = num_generations,
                pad_token_id = self.tokenizer.pad_token_id,
                eos_token_id = self.tokenizer.eos_token_id
            )
            # output_ids -> B*num_generations, P + R
            prompt_len = input_ids.shape[1]
            completions_ids = output_ids[:, prompt_len:] # completions_ids -> B*num_generations, R
            full_attention_mask = (output_ids != self.tokenizer.pad_token_id).long()
            per_token_logps = compute_per_token_logps(self.policy_model, input_ids=output_ids, logits_to_keep=completions_ids.shape[1], attention_mask=full_attention_mask)
        completions = self.tokenizer.batch_decode(completions_ids, skip_special_tokens=True)
        return RolloutResult(output_ids, completions_ids, per_token_logps, completions)
    def update_policy(self, model):
        self.policy_model = model
class SGLangRolloutEngine(RolloutEngine):
    def __init__(self, base_url: str, model_path: str, shared_ckpt_path = "/root/autodl-tmp/miniMind/trainer/sglang_ckpt", timeout=120):
        
        self.base_url = base_url.rstrip('/')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code= True)
        self.shared_ckpt_path = shared_ckpt_path
        self.timeout = timeout
        self.http = requests

    def rollout(self, input_ids, attention_mask, num_generations, max_new_tokens, temperature = 0.8) -> RolloutResult:
        all_input_ids = []
        input_ids_list = []
        for ids, mask in zip(input_ids, attention_mask):
            valid_ids = ids[mask.bool()].tolist()
            input_ids_list.append(valid_ids)
            for _ in range(num_generations):
                all_input_ids.append(valid_ids)
        payload = {
            "input_ids": all_input_ids,
            "sampling_params":{
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer else []
            },
            "return_logprob": True
        }
        resp = self.http.post(f"{self.base_url}/generate", timeout=self.timeout, json=payload)
        resp.raise_for_status()
        results = resp.json()
        if not isinstance(results, List):
            results = [results]
        all_output_ids, all_completions_ids, all_logprobs, all_completions = [], [], [], []
        for i, result in enumerate(results):
            meta = result.get("meta_info", {})
            completion_ids = meta.get("output_ids", [])
            raw_logprobs = meta.get("output_logprobs", [])
            logprobs = []
            for logprob in raw_logprobs:
                if isinstance(logprob, (List, Tuple)):
                    logprobs.append(logprob[0])
                elif isinstance(logprob, (int, float)):
                    logprobs.append(logprob)
            all_logprobs.append(logprobs)
            prompt_ids = all_input_ids[i]
            all_output_ids.append(prompt_ids + completion_ids)
            all_completions_ids.append(completion_ids)
            all_completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))
        max_out_len = max(len(o) for o in all_output_ids)
        max_comp_len = max(len(o) for o in all_completions_ids)
        max_logp_len = max(len(o) for o in all_logprobs)
        def pad_to_tensor(seqs, max_len, pad_val=0):
            return torch.tensor([(s + (max_len - len(s))*[pad_val]) for s in seqs], device=input_ids.device)
        all_output_ids = pad_to_tensor(all_output_ids, max_out_len, )
        all_completions_ids = pad_to_tensor(all_completions_ids, max_comp_len, )
        all_logprobs = pad_to_tensor(all_logprobs, max_logp_len, 0.0)
        return RolloutResult(all_output_ids, all_completions_ids, all_logprobs, all_completions)
    def update_policy(self, model: torch.nn.Module) -> bool:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        abs_path = os.path.abspath(self.shared_ckpt_path)
        raw_model.lm_head.weight = nn.Parameter(raw_model.lm_head.weight.clone())
        state_dict = {k: v.detach().half().cpu() for k, v in raw_model.state_dict().items()}
        raw_model.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
        resp = self.http.post(
            f"{self.base_url}/update_weights_from_disk",
            json={"model_path": abs_path},
            timeout=self.timeout
        )
        if resp.status_code != 200: print(f"[SGLANG WARNING] update_weights 失败: {resp.status_code}, {resp.text}")
        return resp.status_code == 200
    
    def flush_cache(self) -> bool:
        resp = self.http.post(f"{self.base_url}/flush_cache", timeout=30)
        return resp.status_code == 200
    
    def health(self) -> bool:
        try:
            resp = self.http.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False
def create_rollout_engine(
    engine_type: str = "torch",
    policy_model: torch.nn.Module = None,
    tokenizer = None,
    device: str = "cuda",
    autocast_ctx = None,
    sglang_base_url: str = None,
    sglang_model_path: str = None,
    sglang_shared_path: str = None,
) -> RolloutEngine:
    if engine_type == "torch":
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
    elif engine_type == "sglang":
        return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")
                    



        
        

        


