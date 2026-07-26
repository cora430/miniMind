import os
import re
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')
def logits_to_log_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # log_probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    labels_for_gather = labels.clone()
    # 这一步是为了防止超出索引的边界
    labels_for_gather[labels_for_gather == -100] = 0 # 随便给个索引，反正后面 mask 会乘 0
    # 这里结果的形状和 index 是完全一样的
    log_probs_per_tokens = torch.gather(log_probs, dim=2, index=labels_for_gather.unsqueeze(2)).squeeze(-1)
    return log_probs_per_tokens
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # ref_log_probs 和 policy_log_probs 都是 shape: (batch_size, seq_len)
    ref_log_probs = (ref_log_probs * mask).sum(1)
    policy_log_probs = (policy_log_probs * mask).sum(1)
    half = ref_log_probs.shape[0] // 2
    chosen_ref_log_probs = ref_log_probs[:half]
    reject_ref_log_probs = ref_log_probs[half:]
    chosen_policy_log_probs = policy_log_probs[:half]
    reject_policy_log_probs = policy_log_probs[half:]
    logits = (chosen_policy_log_probs - reject_policy_log_probs) - (chosen_ref_log_probs - reject_ref_log_probs)
    loss = -F.logsigmoid(logits * beta)
    return loss.mean()
def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    model.train()
    start_time = time.time()
    steps_done_this_run = 0
    for step, batch in enumerate(loader, start_step):
        steps_done_this_run += 1
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        x = torch.cat((x_chosen, x_rejected), dim=0)
        y = torch.cat((y_chosen, y_rejected), dim=0)
        mask = torch.cat((mask_chosen, mask_rejected), dim=0)
        lr = get_lr(step+1+epoch*iters, args.epochs*iters, lr=args.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        with autocast_ctx:
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)
            policy_outputs = model(x)
            policy_logits = policy_outputs.logits
            policy_log_probs = logits_to_log_probs(policy_logits, y)
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta)
            loss = policy_outputs.aux_loss + dpo_loss_val
            loss = loss / args.accumulation_steps
        scaler.scale(loss).backward()
        if (step+1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if (step + 1) % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            eta_min = spend_time / steps_done_this_run * (iters - step - 1) // 60
            current_total_loss = loss.item() 
            current_aux_loss = policy_outputs.aux_loss.item()
            current_dpo_loss = current_total_loss - current_aux_loss
            
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step + 1}/{iters}),"
                f"Total Loss: {current_total_loss:.4f}," # 改名区分，更清晰
                f"dpo Loss:{current_dpo_loss:.4f},"
                f"Aux Loss:{current_aux_loss:.4f},"
                f"lr:{lr:.8f},"
                f"eta min:{eta_min:.1f}"
            )
            if wandb:
                wandb.log({
                    "loss":current_total_loss,
                    "dpo_loss":current_dpo_loss,
                    "aux_loss":current_aux_loss,
                    "learning_rate":lr,
                    "epoch_time":eta_min
                })
        if ((step + 1) % args.save_interval == 0 or step == iters - 1 ) and is_main_process() :
            model.eval()
            lm_checkpoint(
                lm_config=lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir=args.save_dir,
                tokenizer=tokenizer,
                scaler=scaler
            )
            model.train() 
        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, policy_outputs, policy_logits, policy_log_probs, loss

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    parser.add_argument("--save_dir", type=str, default=None, help="模型保存目录（默认 <项目根>/out）")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率（建议<=5e-8避免遗忘）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', action="store_true", help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/miniMind/dataset/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument('--from_weight', default='full_sft_t2t', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--beta', default=0.15, type=float, help="DPO中的beta参数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb项目名")
    parser.add_argument("--use_compile", action="store_true", help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'out')
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
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=args.save_dir) if args.from_resume == 1 else None
    # ========== 3. 设置混合精度 ==========
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    device_type =  "cuda" if "cuda" in args.device else "cpu"
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=(device_type=="cuda"))
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb:
        import swanlab as wandb
        wandb_id = ckp_data["wandb_id"] if ckp_data is not None else None
        run_name = f"MiniMind dpo epoches : {args.epochs} - batch_size : {args.batch_size} - learning_rate : {args.learning_rate} - use_moe: { args.use_moe }"
        resume = "must" if args.from_resume == 1 else None
        wandb.init(project=args.wandb_project, experiment_name=run_name, id=wandb_id, resume=resume, config=lm_config.to_dict())
    # ========== 5. 定义模型和参考模型 ==========
    model, tokenizer = init_model(lm_config, from_weight=args.from_weight, device=args.device)
    Logger(f"policy model 的总参数量为 {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    ref_model, _ = init_model(lm_config, from_weight=args.from_weight, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    Logger(f"ref model 的总参数量为 {sum(p.numel() for p in ref_model.parameters()) / 1e6:.2f} M")
    train_ds = DPODataset(args.data_path, tokenizer, args.max_seq_len)
    sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler(device=device_type, enabled=(args.dtype == 'float16'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
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
    # ========== 7. compile and DDP包模型 ==========
    if args.use_compile :
        model = torch.compile(model)
        Logger("torch.compile enabled")
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
        train_epoch(epoch, loader, iters=len(loader)+skip, ref_model=ref_model, lm_config=lm_config, start_step=skip, wandb=wandb, beta=args.beta)
    if dist.is_initialized(): dist.destroy_process_group()
    

    


