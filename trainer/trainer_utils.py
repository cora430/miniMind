import math
import os
from sympy import is_increasing
import torch
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import random
import numpy as np
from pathlib import Path
import torch.distributed as dist
from transformers import AutoModel, AutoTokenizer
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

def arch_suffix(lm_config):
    # 结构开关会改变 state_dict 形状，不能和默认结构共用同一份 checkpoint 文件名，
    # 否则 strict=False 会静默丢参数/装错权重
    suffix = "_moe" if getattr(lm_config, "use_moe", False) else ""
    suffix += "_ga" if getattr(lm_config, "use_gated_attention", False) else ""
    suffix += "_eng" if getattr(lm_config, "use_engram", False) else ""
    suffix += "_mtp" if getattr(lm_config, "use_mtp", False) else ""
    return suffix

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def Logger(content):
    if is_main_process():
        print(content)

def init_model(lm_config, from_weight='pretrain', tokenizer_path=None, save_dir=None, device='cuda'):
    if tokenizer_path is None:
        tokenizer_path = os.path.join(_PROJECT_ROOT, 'model')
    if save_dir is None:
        save_dir = os.path.join(_PROJECT_ROOT, 'out')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)
    if from_weight != "none":
        weight_path = f"{save_dir}/{from_weight}_{lm_config.hidden_size}{arch_suffix(lm_config)}.pth"
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False) 
        Logger(f"init model from {weight_path}")
    Logger(f"trainer params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    get_model_params(model, lm_config)
    
    return model.to(device), tokenizer

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir=None, tokenizer=None, **kwargs):
    if save_dir is None:
        save_dir = os.path.join(_PROJECT_ROOT, 'out')
    os.makedirs(save_dir, exist_ok=True)
    suffix = arch_suffix(lm_config)
    ckp_path  = f"{save_dir}/{weight}_{lm_config.hidden_size}{suffix}.pth"
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{suffix}_resume.pth"
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
        if tokenizer is not None:
            import shutil
            hf_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{suffix}_hf"
            os.makedirs(hf_path, exist_ok=True)
            # 保存时用局部 auto_map，使 web_demo 无需依赖 model 包即可加载
            orig_auto_map = lm_config.auto_map
            lm_config.auto_map = {
                "AutoConfig": "model_minimind.MiniMindConfig",
                "AutoModelForCausalLM": "model_minimind.MiniMindForCausalLM",
            }
            raw_model.save_pretrained(hf_path, state_dict=state_dict, safe_serialization=False)
            lm_config.auto_map = orig_auto_map
            tokenizer.save_pretrained(hf_path)
            shutil.copy2(
                os.path.join(_PROJECT_ROOT, 'model', 'model_minimind.py'),
                os.path.join(hf_path, 'model_minimind.py')
            )

        wandb_id = None
        if wandb is not None:
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
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
        Logger(f"从 {resume_path} 加载数据")
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

class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype)
        self.model = self.model.to(device).eval()
        self.dtype = dtype
    @torch.no_grad()
    def get_score(self, messages, responses):
        history_content = "\n".join([f"{m['role']}:{m['content']}" for m in messages[:-1]])
        querry = messages[-1]["content"] if messages else ""
        messages_content = f"{history_content}\n以上是对话历史。我的新问题是:\n{querry}" if history_content else querry
        eval_messages = [
            {"role":"user", "content": messages_content},
            {"role":"assistant", "content": responses},
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)




              




            






