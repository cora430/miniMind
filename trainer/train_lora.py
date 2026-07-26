import os
import sys

from numpy import save
from sympy import total_degree

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from model.model_lora import save_lora, apply_lora
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')

def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    model.train()
    start_time = time.time()
    steps_done_this_run = 0
    for step, (input_ids, labels) in enumerate(loader, start_step):
        steps_done_this_run += 1
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(step+1+epoch*iters, args.epochs*iters, lr=args.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        with autocast_ctx:
            res = model(input_ids=input_ids, labels=labels)
            loss = res.loss 
            loss = loss / args.accumulation_steps
        scaler.scale(loss).backward()
        if (step+1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if (step + 1) % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            eta_min = spend_time / steps_done_this_run * (iters - step - 1) // 60
            # 1. 当前总损失 (已经包含 aux 和 mtp)
            current_total_loss = res.loss.item()
            # 2. 辅助损失
            current_aux_loss = res.aux_loss.item()
            # 3. MTP 损失 (按权重折算，与 total_loss 中实际占比一致)
            current_mtp_loss = lm_config.mtp_loss_weight * res.mtp_loss.item()
            # 4. 纯 Logits 损失 (总减去辅助、减去 mtp)
            current_logits_loss = current_total_loss - current_aux_loss - current_mtp_loss
            
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step + 1}/{iters}),"
                f"Total Loss: {current_total_loss:.4f}," # 改名区分，更清晰
                f"Logits Loss:{current_logits_loss:.4f},"
                f"Aux Loss:{current_aux_loss:.4f},"
                f"MTP Loss:{current_mtp_loss:.4f},"
                f"lr:{lr:.8f},"
                f"eta min:{eta_min:.1f}"
            )
            if wandb:
                wandb.log({
                    "loss":current_total_loss,
                    "logits_loss":current_logits_loss,
                    "aux_loss":current_aux_loss,
                    "mtp_loss":current_mtp_loss,
                    "learning_rate":lr,
                    "epoch_time":eta_min
                })
        if ((step+1) % args.save_interval == 0 or step == iters - 1 ) and is_main_process():
            model.eval()
            os.makedirs(args.save_dir, exist_ok=True)
            moe_suffix = "_moe" if lm_config.use_moe else ""
            lora_path  = f"{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}{moe_suffix}.pth"
            save_lora(model, lora_path)
            lm_checkpoint(lm_config, args.lora_name, model, optimizer, epoch, step, wandb, args.save_dir, scaler=scaler)
            model.train()
        del input_ids, labels, res, loss




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    parser.add_argument("--save_dir", type=str, default=None, help="模型保存目录（默认 <项目根>/out/lora）")
    parser.add_argument("--lora_name", type=str, default="lora_identity", help="LoRA权重名称(如lora_identity/lora_medical等)")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', action="store_true", help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/miniMind/dataset/lora_identity.jsonl", help="LoRA训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练，默认full_sft")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="wandb项目名")
    parser.add_argument("--use_compile", action="store_true", help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'out', 'lora')
    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + dist.get_rank() if dist.is_initialized() else 0)
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        use_moe=args.use_moe,
        num_hidden_layers=args.num_hidden_layers
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir=args.save_dir) if args.from_resume == 1 else None
    # ========== 3. 设置混合精度 ==========
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    device_type =  "cuda" if "cuda" in args.device else "cpu"
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=(device_type=="cuda"))
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb:
        import swanlab as wandb
        wandb_id = ckp_data["wandb_id"] if ckp_data is not None else None
        run_name = f"MiniMind lora epoches : {args.epochs} - batch_size : {args.batch_size} - learning_rate : {args.learning_rate} - use_moe: {args.use_moe}"
        resume = "must" if args.from_resume == 1 else None
        wandb.init(project=args.wandb_project, experiment_name=run_name, id=wandb_id, resume=resume, config=lm_config.to_dict())
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile :
        model = torch.compile(model)
        Logger("torch.compile enabled")
    apply_lora(model=model)
    total_p = sum(p.numel()for n, p in model.named_parameters())
    lora_p = sum(p.numel()for n, p in model.named_parameters() if "lora" in n)
    Logger(f"LLM总参数： {total_p/1e6:.2f}M")
    Logger(f"lora参数： {lora_p/1e6:.2f}M")
    Logger(f"占比：{lora_p/total_p*100:.2f}M")
    # 冻结非LoRA参数，收集LoRA参数
    lora_params = []
    for n, p in model.named_parameters():
        if "lora" in n:
            p.requires_grad = True
            lora_params.append(p)
        else:
            p.requires_grad = False
    train_ds = SFTDataset(args.data_path, tokenizer, args.max_seq_len)
    sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler(device=device_type, enabled=(args.dtype == 'float16'))
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate, weight_decay=0.1)
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data is not None:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data["step"]
        if "scaler" in ckp_data and scaler is not None: scaler.load_state_dict(ckp_data["scaler"])  
        Logger(f"成功从 checkpoint 恢复: Epoch {start_epoch}, Step {start_step}")
    del ckp_data
        # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
        
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch,args.epochs):
        sampler and sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler  = SkipBatchSampler(sampler or indices, args.batch_size, skip_batches=skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler , num_workers=args.num_workers, pin_memory=True)
        if epoch == start_epoch: Logger(f"{epoch + 1}/{args.epochs}  跳过前{skip}个batch, 从{skip}开始")
        train_epoch(epoch, loader, len(loader)+skip, lora_params, skip, wandb=wandb)
    if dist.is_initialized(): dist.destroy_process_group()