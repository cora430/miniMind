from contextlib import nullcontext
import sys
import os 
import argparse
import time
from h11 import Data
import torch
import warnings

warnings.filterwarnings("ignore")
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from trainer.trainer_utils import get_lr, init_model, Logger, is_main_process,lm_checkpoint,init_distributed_mode, setup_seed, SkipBatchSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel

def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    # 训练一个epoch的函数
    # epoch: 当前epoch编号
    # loader: 数据加载器
    # iters: 本epoch总step数
    # start_step: 恢复训练时从哪个step开始
    # wandb: 日志系统（可选）
    model.train()
    start_time = time.time()
    steps_done_this_run = 0
    for step, (input_ids, labels) in enumerate(loader, start=start_step):
        steps_done_this_run += 1
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr((step + 1) + epoch * iters, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        with autocast_ctx:
            res = model(input_ids=input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps
        scaler.scale(loss).backward()
        if (step + 1) % args.accumulation_steps == 0:
            # 更新参数
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if (step + 1) % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            eta_min = spend_time / steps_done_this_run * (iters - step - 1) // 60
            current_loss = loss.item() * args.accumulation_steps
            current_logits_loss = res.loss.item()
            current_aux_loss = res.aux_loss.item()
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step + 1}/{iters}),"
                f"loss: {current_loss:.4f},"
                f"logits loss:{current_logits_loss:.4f},"
                f"aux loss:{current_aux_loss:.4f},"
                f"lr:{lr:.8f},"
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
                save_dir="/root/autodl-tmp/miniMind/checkpoints",
                scaler=scaler
            )
            model.train() 
        del input_ids, labels, loss, res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=" MiniMind Pretrain")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累计步数")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--save_dir", type=str, default="/root/autodl-tmp/miniMind/out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重前缀")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument('--from_weight', default='none', type=str, help="从哪个权重开始训练")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Transformer层数")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])

    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/miniMind/dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")

    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    # local_rank 如果是Python .py的形式返回0
    args.device = f"cuda:{local_rank}" 
    setup_seed(42 + local_rank)

     # ========== 2. 配置目录、模型参数、检查ckp ==========
     # 这里全都装进checkpoints里了,不需要用到svae_dir
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe= bool(args.use_moe)
    )
    ckp_data = lm_checkpoint(lm_config=lm_config, weight=args.save_weight, ) if args.from_resume == 1 else None

     # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16 # float16 容易溢出，必须配合前面提到的 GradScaler 使用。 bfloat16大模型首选
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=(device_type == "cuda"))

    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data["wandb_id"] if ckp_data is not None else None
        run_name = f"MiniMind Pretrained epoches : {args.epochs} - batch_size : {args.batch_size} - learning_rate : {args.learning_rate} - use_moe: {args.use_moe}"
        resume = "must" if wandb_id is not None else None
        wandb.init(project=args.wandb_project, experiment_name=run_name, id=wandb_id, resume=resume, config=lm_config.to_dict())

    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config=lm_config, from_weight=args.from_weight, device=args.device) # 这里model就已经加载了在checkpoint里的参数了

    train_ds = PretrainDataset(args.data_path, tokenizer=tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler(device=device_type, enabled=(args.dtype == "float16"))
    optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate)
    
    if bool(args.use_compile):
        model = torch.compile(model)
        Logger("torch.compile enabled")

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data is not None:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data["optimizer"])
        if "scaler" in ckp_data and scaler is not None: scaler.load_state_dict(ckp_data["scaler"]) 
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data["step"] 
        Logger(f"成功从 checkpoint 恢复: Epoch {start_epoch}, Step {start_step}")
    del ckp_data

    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip_batches=skip)
        loader = DataLoader(dataset=train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        Logger(f"epoch : {epoch}/{args.epochs}   跳过前{skip}个， 从{skip}开始")
        train_epoch(epoch, loader, len(loader) + skip, skip, wandb)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()

    
     



