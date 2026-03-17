import math
import os
from sympy import is_increasing
import torch 
import random
import numpy as np
from pathlib import Path
import torch.distributed as dist
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
def get_lr(current_steps, total_steps, lr):
    return lr * ((1 + math.cos(math.pi * current_steps / total_steps)) * 0.45 + 0.1)

def setup_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed_experts = getattr(config, "n_routed_experts", getattr(config, "num_experts", 0))
    nun_experts_per_tokens = getattr(config, "num_experts_per_tokens", 0)
    n_shared_experts = getattr(config, "n_shared_experts", 0)
    experts_params = sum(p.numel() for n, p in model.named_parameters() if "mlp.experts.0" in n) / 1e6
    base = total - experts_params * n_routed_experts - experts_params * n_shared_experts
    active = base + n_shared_experts * experts_params + nun_experts_per_tokens * experts_params
    if active < base : Logger(f"moe model params: {total:.2f}M - active params: {active:.2f}M")
    else: Logger(f"model params:{total:.2f}M")

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def Logger(content):
    if is_main_process():
        print(content)

def init_model(lm_config, from_weight='pretrain', tokenizer_path='/root/autodl-tmp/miniMind/model', save_dir='/root/autodl-tmp/miniMind/checkpoints', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)
    if from_weight != "none":
        moe_suffix = "_moe" if lm_config.use_moe else ""
        weight_path = f"{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False) 
    Logger(f"trainer params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    get_model_params(model, lm_config)
    return model.to(device), tokenizer

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='/root/autodl-tmp/miniMind/checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_suffix = "_moe" if lm_config.use_moe else ""
    ckp_path  = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_suffix}.pth"
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_suffix}_resume.pth"
    ckp_file = Path(ckp_path)
    resume_file = Path(resume_path)
    if model is not None:
        # 模型参数拿出来，仅存储state_dict到ckp，再加上别的，存到resume
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + ".tmp"
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        wandb_id = None
        if wandb is not None:
            if hasattr(wandb, "get_run"):
                run = wandb.get_run
                wandb_id = getattr(run, "id", None)
            else:
                wandb_id = getattr(wandb, "id", None)
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for k, v in kwargs.items():
            if v is not None:
                if hasattr(v, "state_dict"):
                    v = v.module if isinstance(v, DistributedDataParallel) else v
                    v = getattr(v, "_orig_mod", v)
                    resume_data[k] = v.state_dict()
        resume_tmp = resume_path + "tmp"
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:
        if not resume_file.exists() or not resume_file.is_file():
            Logger(f"没找到{resume_path}. Initializing from scratch.")
            return None
        # 根据resume_path把resume_data拿出来，注意step
        resume_data = torch.load(resume_path, map_location="cpu")
        old_world_size = resume_data["world_size"]
        current_world_size = dist.get_world_size() if dist.is_initialized() else 1
        if old_world_size != current_world_size:
            resume_data["step"] = resume_data["step"] * old_world_size // current_world_size
            Logger(f"gpu从{old_world_size} -> {current_world_size}")
        return resume_data
    return None

def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK"))
    torch.cuda.set_device(local_rank)
    return local_rank

class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches):
        super().__init__()
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches
    def __iter__(self):
        skip = 0
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skip < self.skip_batches:
                    batch = []
                    skip += 1
                    continue
                
                yield batch
                batch = []
        if skip >= self.skip_batches and len(batch) > 0 :
            yield batch 
    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1 )// self.batch_size
        return (total_batches - self.skip_batches)
            




            






