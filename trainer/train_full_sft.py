from contextlib import nullcontext
import sys
import os 
import argparse
import time
from h11 import Data
from sentry_sdk import is_initialized
import torch
import warnings

warnings.filterwarnings("ignore")
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from trainer.trainer_utils import get_lr, init_model, Logger, is_main_process,lm_checkpoint,init_distributed_mode, setup_seed, SkipBatchSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel

def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
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
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps
        scaler.scale(loss).backward()
        if (step+1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if (step+1) % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            eta_min = spend_time / steps_done_this_run * (iters - step - 1) // 60
            current_loss = loss.item() * args.accumulation_steps
            current_logits_loss = res.loss.item()
            current_aux_loss = res.aux_loss.item()
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step + 1}/{iters}), "
                f"loss: {current_loss:.4f}, "
                f"logits loss:{current_logits_loss:.4f}, "
                f"aux loss:{current_aux_loss:.4f}, "
                f"lr:{lr:.8f}, "
                f"eta min:{eta_min:.1f}"
            )
            if wandb:
                wandb.log({
                    "loss":current_loss,
                    "logits_loss":current_logits_loss,
                    "aux_loss":current_aux_loss,
                    "learning_rate":lr,
                    "epoch_time":eta_min
                })
        if ((step+1) % args.save_interval == 0 or step == iters - 1 ) and is_main_process():
            model.eval()
            lm_checkpoint(lm_config, args.save_weight, model, optimizer, epoch, step, wandb, '/root/autodl-tmp/miniMind/checkpoints', scaler=scaler)
            model.train()
        del input_ids, labels, res, loss




    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="/root/autodl-tmp/miniMind/out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/miniMind/dataset/sft_mini_512.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--wandb_id", type=str,  help="wandb_id")
    args = parser.parse_args()
    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + dist.get_rank() if dist.is_initialized() else 0)
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        use_moe=bool(args.use_moe),
        num_hidden_layers=args.num_hidden_layers
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='/root/autodl-tmp/miniMind/checkpoints') if args.from_resume==1 else None
    # ========== 3. 设置混合精度 ==========
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    device_type =  "cuda" if "cuda" in args.device else "cpu"
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=(device_type=="cuda"))
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb:
        import swanlab as wandb
        wandb_id = ckp_data["wandb_id"] if ckp_data is not None else None
        run_name = f"MiniMind Pretrained epoches : {args.epochs} - batch_size : {args.batch_size} - learning_rate : {args.learning_rate} - use_moe: {args.use_moe}"
        resume = "must" if args.from_resume == 1 else None
        wandb.init(project=args.wandb_project, experiment_name=run_name, id=args.wandb_id or wandb_id, resume=resume, config=lm_config.to_dict())
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile ==1:
        model = torch.compile(model)
        Logger("torch.compile enabled")
    train_ds = SFTDataset(args.data_path, tokenizer, args.max_seq_len)
    sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler(device=device_type, enabled=(args.dtype == 'float16'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
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
        loader = DataLoader(train_ds, sampler=batch_sampler , num_workers=args.num_workers, pin_memory=True)
        Logger(f"{epoch + 1}/{args.epochs}  跳过前{skip}个batch, 从{skip}开始")
        train_epoch(epoch, loader, len(loader)+skip, skip, wandb=wandb)
    if dist.is_initialized(): dist.destroy_process_group()




    
    

