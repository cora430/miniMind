import torch
import re
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch.distributed as dist
from model.model_minimind import MiniMindConfig
from torch.utils.data import DistributedSampler, DataLoader
from trainer.rollout_engine import create_rollout_engine
from dataset.lm_dataset import RLAIFDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel
from dataclasses import dataclass
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import time
from trainer.trainer_utils import init_model, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, LMForRewardModel

"""
GRPO 训练循环（Group Relative Policy Optimization）。

与 PPO 的核心区别：
  - 无 Critic / Value head，彻底去掉 value loss
  - 每条 prompt 生成 G 条回复，组内归一化奖励作为优势函数
  - 优势 A_i = (r_i - mean(r_1..G)) / (std(r_1..G) + ε)，广播到 response 所有 token

逻辑划分：
  1. prepare_batch       —— prompt 编码、rollout 生成 G 条、reward 计算
  2. build_masks         —— 构造 response 区域的有效位置 mask
  3. collect_rollout     —— 无梯度推理，拿 old_logp / ref_logp（无 Critic）
  4. compute_grpo_adv    —— 组内相对归一化优势
  5. grpo_update         —— 多轮更新（含早停）
  6. logging / checkpoint
"""

# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------
@dataclass
class RolloutBatch:
    """一次 rollout 产生的 pre-batch 数据，全部无梯度。维度均为 B*G。"""
    gen_out: torch.Tensor         # [B*G, P+R]
    full_mask: torch.Tensor       # [B*G, P+R]
    labels: torch.Tensor          # [B*G, P+R-1]
    resp_start: int
    resp_length: torch.Tensor     # [B*G]
    resp_policy_mask: torch.Tensor # [B*G, R]
    rewards: torch.Tensor         # [B*G]
    old_logp: torch.Tensor        # [B*G, R]
    ref_logp: torch.Tensor        # [B*G, R]

@dataclass
class AdvantageBatch:
    """GRPO 组内归一化优势。"""
    advantages: torch.Tensor      # [B*G, R]

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0

def _unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model

def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(1)

def _gather_logp(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits: [B, T, V], labels: [B, T] → [B, T]"""
    logp = F.log_softmax(logits, dim=-1)
    return torch.gather(logp, dim=2, index=labels.unsqueeze(2)).squeeze(2)

# ---------------------------------------------------------------------------
# Reward 计算
# ---------------------------------------------------------------------------
def calculate_rewards(prompts, responses, reward_model, device):
    """
    prompts:   长度 B*G 的 str list（每条 prompt 重复了 G 次）
    responses: 长度 B*G 的 str list
    返回 [B*G] rewards tensor
    """
    reward_model_scores = []
    rewards = torch.zeros(len(responses), device=device)
    with torch.no_grad():
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
            answer = response
            if "</think>" in response:
                thinking_content, answer_content = response.split("</think>", 1)
                answer = answer_content.strip()
                rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                rewards[i] += 0.25 if response.count("</think>") == 1 else -0.25
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]
            reward_model_scores.append(reward_model(messages, answer))
        reward_model_scores = torch.tensor(reward_model_scores, device=device)
        rewards = rewards + reward_model_scores
    return rewards

# ---------------------------------------------------------------------------
# 1. 准备 batch：编码 → rollout G 条 → reward
# ---------------------------------------------------------------------------
def prepare_batch(prompts, tokenizer, rollout_engine, reward_model, args):
    """
    每条 prompt 生成 G=num_generations 条回复。
    返回的 gen_out / completions 维度均为 B*G。
    prompts_repeated 也是 B*G，方便 reward 计算。
    """
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        truncation=True,
        max_length=args.max_seq_len
    ).to(args.device)

    rollout_result = rollout_engine.rollout(
        enc.input_ids,
        enc.attention_mask,
        num_generations=args.num_generations,
        max_new_tokens=args.max_gen_len,
        temperature=0.8
    )
    # rollout_result.output_ids: [B*G, P+R]
    # rollout_result.completions: list of B*G str
    # 将 prompt 重复 G 次，与 completions 对齐
    prompts_repeated = [p for p in prompts for _ in range(args.num_generations)]
    rewards = calculate_rewards(prompts_repeated, rollout_result.completions, reward_model, args.device)  # [B*G]
    return {
        "prompt_length": enc.input_ids.shape[1],
        "prompt_attention_mask": enc.attention_mask,
        "gen_out": rollout_result.output_ids,        # [B*G, P+R]
        "responses_text": rollout_result.completions, # list B*G
        "prompts_repeated": prompts_repeated,
        "rewards": rewards                            # [B*G]
    }

# ---------------------------------------------------------------------------
# 2. 构造 mask
# ---------------------------------------------------------------------------
def build_masks(gen_out: torch.Tensor, prompt_length: int, tokenizer, prompt_attention_mask: torch.Tensor = None):
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    B = gen_out.shape[0]
    labels = gen_out[:, 1:]                        # [B*G, P+R-1]
    resp_start = prompt_length - 1
    resp_labels = labels[:, resp_start:]           # [B*G, R]

    if pad_id != eos_id:
        full_mask = (gen_out != pad_id).float()
        resp_lengths = (resp_labels != pad_id).sum(1)
    else:
        # pad_id == eos_id: 第一个 EOS 是真实结尾，后续 EOS 是 padding
        resp_gen = gen_out[:, prompt_length:]      # [B*G, R]
        is_eos = (resp_gen == eos_id)
        prev_cumsum = torch.cat([
            torch.zeros(B, 1, dtype=torch.long, device=gen_out.device),
            is_eos[:, :-1].long().cumsum(dim=1)
        ], dim=1)                                  # [B*G, R]
        resp_valid = (prev_cumsum == 0)            # 包含第一个 EOS，排除其后
        resp_lengths = resp_valid.sum(1)
        prompt_mask = prompt_attention_mask.float() if prompt_attention_mask is not None \
            else torch.ones(B, prompt_length, device=gen_out.device)
        full_mask = torch.cat([prompt_mask, resp_valid.float()], dim=1)

    resp_policy_mask = (
        torch.arange(resp_labels.shape[1], device=gen_out.device).unsqueeze(0) < resp_lengths.unsqueeze(1)
    ).float()
    return {
        "labels": labels,
        "full_mask": full_mask,
        "resp_start": resp_start,
        "resp_policy_mask": resp_policy_mask,
        "resp_lengths": resp_lengths,
    }

# ---------------------------------------------------------------------------
# 3. rollout 阶段无梯度推理（GRPO 无 Critic）
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_rollout(actor_model, ref_model, gen_out, full_mask, labels, resp_start, autocast_ctx):
    """
    只需要 Actor 快照 log π_old 和 Ref log π_ref，无 Critic。
    """
    with autocast_ctx:
        old_logits = actor_model(input_ids=gen_out, attention_mask=full_mask).logits
    old_logp = _gather_logp(old_logits[:, :-1], labels)[:, resp_start:]  # [B*G, R]
    ref_logits = ref_model(input_ids=gen_out, attention_mask=full_mask).logits
    ref_logp = _gather_logp(ref_logits[:, :-1], labels=labels)[:, resp_start:]  # [B*G, R]
    return old_logp, ref_logp

# ---------------------------------------------------------------------------
# 4. GRPO 组内相对优势估计
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_grpo_adv(rewards: torch.Tensor, resp_policy_mask: torch.Tensor,
                     num_generations: int) -> AdvantageBatch:
    """
    rewards: [B*G]
    对每组 G 条回复做 z-score 归一化，广播到 token 级别。

    A_i = (r_i - mean_g) / (std_g + ε)
    """
    BG = rewards.shape[0]
    B = BG // num_generations
    G = num_generations

    r = rewards.view(B, G)                              # [B, G]
    mean_r = r.mean(dim=1, keepdim=True)                # [B, 1]
    std_r = r.std(dim=1, keepdim=True)                  # [B, 1]
    adv_scalar = ((r - mean_r) / (std_r + 1e-8)).view(BG)  # [B*G]

    R = resp_policy_mask.shape[1]
    advantages = adv_scalar.unsqueeze(1).expand(BG, R) * resp_policy_mask  # [B*G, R]
    return AdvantageBatch(advantages)

# ---------------------------------------------------------------------------
# 5. 一次 mini-batch forward + loss（GRPO 无 value loss）
# ---------------------------------------------------------------------------
def forward_minibatch(actor_model, batch: RolloutBatch, adv: AdvantageBatch,
                      inds: torch.Tensor, args, autocast_ctx, lm_config):
    """
    支持两种 loss_type：
      grpo:  L = -min(ratio*A, clip(ratio, 1-ε, 1+ε)*A) + β*KL
      cispo: L = -(clamp(ratio, max=ε_high).detach() * A * logp) + β*KL
             CISPO 只做上界裁剪，用 detach 阻止梯度经 ratio 路径回传，
             改为直接对 logp 求导，避免 PPO clip 的梯度截断问题。
    KL_t = exp(log π_ref - log π_new) - (log π_ref - log π_new) - 1
    """
    actor = _unwrap(actor_model)
    with autocast_ctx:
        mb_result = actor(input_ids=batch.gen_out[inds], attention_mask=batch.full_mask[inds])
        aux_loss = mb_result.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
        mb_logits = mb_result.logits[:, :-1]
        mb_logp = _gather_logp(mb_logits, batch.labels[inds])[:, batch.resp_start:]  # [mb, R]

    log_ratio = mb_logp - batch.old_logp[inds]
    ratio = torch.exp(log_ratio)
    mask = batch.resp_policy_mask[inds]

    clipfrac = _masked_mean(((ratio - 1.0).abs() > args.clip_epsilon).float(), mask)
    approx_kl = _masked_mean(0.5 * log_ratio ** 2, mask)

    A = adv.advantages[inds]
    x = batch.ref_logp[inds] - mb_logp
    KL = torch.exp(x) - x - 1

    if args.loss_type == "cispo":
        # 上界裁剪；detach 让梯度只走 logp，不走 ratio
        clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
        per_token_loss = -(clamped_ratio * A * mb_logp - args.kl_coef * KL)
    else:
        clipped_ratio = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon)
        per_token_loss = -(torch.min(ratio * A, clipped_ratio * A) - args.kl_coef * KL)

    # 先序列内平均，再序列间平均，使每条序列权重相等
    seq_len = mask.sum(dim=1).clamp(1)
    per_seq_loss = (per_token_loss * mask).sum(dim=1) / seq_len
    policy_loss = per_seq_loss.mean()
    kl_ref = ((KL * mask).sum(dim=1) / seq_len).mean().detach()

    return {
        "policy_loss": policy_loss,
        "aux_loss": aux_loss,
        "approx_kl": approx_kl,
        "kl_ref": kl_ref,
        "clipfrac": clipfrac,
    }

def _sync_kl(kl: torch.Tensor) -> float:
    """跨卡同步 kl，避免早停在不同 rank 上分叉导致 DDP 死锁。"""
    v = kl.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(v, op=dist.ReduceOp.AVG)
    return v.item()

def _optimizer_step(actor_optimizer, actor_scheduler, actor_model, args):
    """GRPO 只有一个 Actor optimizer。"""
    clip_grad_norm_(actor_model.parameters(), args.grad_clip)
    actor_optimizer.step()
    actor_scheduler.step()
    actor_optimizer.zero_grad()

# ---------------------------------------------------------------------------
# 6. GRPO 多轮更新
# ---------------------------------------------------------------------------
def grpo_update(actor_model, batch: RolloutBatch, adv: AdvantageBatch,
                actor_optimizer, actor_scheduler,
                args, autocast_ctx, lm_config, grad_accum_step: int):
    total_size = batch.gen_out.shape[0]  # B*G
    inds = torch.arange(total_size, device=args.device)
    stats = {"policy_loss": 0.0, "aux_loss": 0.0, "approx_kl": 0.0, "kl_ref": 0.0, "clipfrac": 0.0, "count": 0}

    stop_grpo = False
    for _ in range(args.grpo_update_iters):
        if stop_grpo:
            break
        out = forward_minibatch(actor_model, batch, adv, inds, args, autocast_ctx, lm_config)
        kl_ref = _sync_kl(out["kl_ref"])
        if kl_ref > args.early_stop_kl:
            stop_grpo = True
        loss = (out["policy_loss"] + out["aux_loss"]) * 0.0 if stop_grpo else \
               (out["policy_loss"] + out["aux_loss"]).div(args.accumulation_steps)
        loss.backward()
        grad_accum_step += 1
        if grad_accum_step % args.accumulation_steps == 0:
            _optimizer_step(actor_optimizer, actor_scheduler, actor_model, args)
        stats["policy_loss"] += out["policy_loss"].item()
        stats["aux_loss"] += out["aux_loss"].item()
        stats["approx_kl"] += out["approx_kl"].item()
        stats["kl_ref"] += kl_ref
        stats["clipfrac"] += out["clipfrac"].item()
        stats["count"] += 1
    return stats, grad_accum_step

# ---------------------------------------------------------------------------
# 7. 日志 & checkpoint
# ---------------------------------------------------------------------------
def log_step(epoch, step, iters, stats: dict,
             rewards: torch.Tensor, resp_lengths: torch.Tensor,
             actor_optimizer, args, start_time, wandb=None):
    if not is_main_process():
        return
    avg_rewards = rewards.mean().item()
    avg_resp_length = resp_lengths.float().mean().item()
    actor_lr = actor_optimizer.param_groups[0]["lr"]
    eta_min = (time.time() - start_time) / step * (iters - step + (args.epochs - (epoch + 1)) * iters) // 60
    n = max(stats["count"], 1)
    matrix = {
        "policy_loss": stats["policy_loss"] / n,
        "aux_loss": stats["aux_loss"] / n,
        "approx_kl": stats["approx_kl"] / n,
        "kl_ref": stats["kl_ref"] / n,
        "clipfrac": stats["clipfrac"] / n,
        "avg_rewards": avg_rewards,
        "avg_resp_length": avg_resp_length,
        "actor_lr": actor_lr,
        "eta_min": eta_min,
    }
    if wandb is not None:
        wandb.log(matrix)
    Logger(
        f"epoch: [{epoch} / {args.epochs}] ({step} / {iters}), "
        f"policy_loss: {matrix['policy_loss']:.4f}, "
        f"aux_loss: {matrix['aux_loss']:.4f}, "
        f"approx_kl: {matrix['approx_kl']:.4f}, "
        f"kl_ref: {matrix['kl_ref']:.4f}, "
        f"clipfrac: {matrix['clipfrac']:.4f}, "
        f"avg_rewards: {matrix['avg_rewards']:.4f}, "
        f"avg_resp_length: {matrix['avg_resp_length']:.2f}, "
        f"actor_lr: {matrix['actor_lr']:.8f}, "
        f"eta_min: {matrix['eta_min']:.2f}"
    )


def save_checkpoint(epoch, step, actor_model, actor_optimizer, actor_scheduler,
                    args, lm_config, wandb=None):
    if not is_main_process():
        return
    actor_model.eval()
    lm_checkpoint(
        lm_config=lm_config,
        weight=args.save_weight,
        model=actor_model,
        optimizer=actor_optimizer,
        scheduler=actor_scheduler,
        epoch=epoch,
        step=step,
        wandb=wandb,
        save_dir=args.save_dir,
        tokenizer=tokenizer,
    )
    actor_model.train()


def debug_log_samples(step, prompts, responses_text, rewards, prompt_length, args):
    if not (args.debug_mode and is_main_process() and step % args.debug_interval == 0):
        return
    for i, prompt in enumerate(prompts):
        Logger(f"[Debug]: step = {step}, sample[{i}]")
        Logger(f"[Debug]: prompt_length = {prompt_length}, response_length = {len(responses_text[i])}")
        Logger(f"{'-' * 100}")
        Logger(f"Context begin {'=' * 30}")
        Logger(prompt)
        Logger(f"Context end {'=' * 30}")
        Logger(f"Response begin {'=' * 30}")
        Logger(responses_text[i])
        Logger(f"Response end {'=' * 30}")
        Logger(f"{'-' * 100}")
        Logger(f"reward: {rewards[i].item():.4f}")

# ---------------------------------------------------------------------------
# 主训练循环
# ---------------------------------------------------------------------------
def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model,
                     actor_scheduler, reward_model,
                     start_step=0, wandb=None):
    """
    单个 epoch 的 GRPO 训练。
    每个 step 的生命周期：
        prepare(G 条) → masks → rollout 快照 → 组内优势 → GRPO 更新 → log → checkpoint
    """
    actor_model.train()
    start_time = time.time()
    grad_accum_step = 0

    for step, raw_batch in enumerate(loader, start_step + 1):
        p = prepare_batch(raw_batch["prompt"], tokenizer, rollout_engine, reward_model, args)
        debug_log_samples(step, p["prompts_repeated"], p["responses_text"], p["rewards"], p["prompt_length"], args)
        m = build_masks(p["gen_out"], p["prompt_length"], tokenizer, p["prompt_attention_mask"])
        old_logp, ref_logp = collect_rollout(
            actor_model, ref_model,
            p["gen_out"], m["full_mask"], m["labels"], m["resp_start"], autocast_ctx
        )
        rollout_batch = RolloutBatch(
            gen_out=p["gen_out"],
            full_mask=m["full_mask"],
            labels=m["labels"],
            resp_start=m["resp_start"],
            resp_length=m["resp_lengths"],
            resp_policy_mask=m["resp_policy_mask"],
            rewards=p["rewards"],
            old_logp=old_logp,
            ref_logp=ref_logp
        )

        adv = compute_grpo_adv(
            rollout_batch.rewards, rollout_batch.resp_policy_mask,
            args.num_generations
        )
        stats, grad_accum_step = grpo_update(
            actor_model=actor_model,
            batch=rollout_batch,
            adv=adv,
            actor_optimizer=actor_optimizer,
            actor_scheduler=actor_scheduler,
            args=args,
            autocast_ctx=autocast_ctx,
            lm_config=lm_config,
            grad_accum_step=grad_accum_step
        )
        if grad_accum_step % args.accumulation_steps != 0:
            _optimizer_step(actor_optimizer, actor_scheduler, actor_model, args)
            if is_main_process() and step % args.save_interval == 0: rollout_engine.update_policy(actor_model)


        if step % args.log_interval == 0 or step == iters:
            log_step(epoch, step, iters, stats, rollout_batch.rewards, rollout_batch.resp_length,
                     actor_optimizer, args, start_time, wandb)
        if is_main_process() & step % args.save_interval == 0:
            rollout_engine.update_policy(actor_model)
        if step % args.save_interval == 0 or step == iters:
            save_checkpoint(epoch, step, actor_model, actor_optimizer, actor_scheduler, args, lm_config, wandb)

        del p, m, old_logp, ref_logp, rollout_batch, adv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default=None, help="模型保存目录（默认 <项目根>/out）")
    parser.add_argument('--save_weight', default='grpo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="每步 prompt 数量 B（实际序列数为 B*G）")
    parser.add_argument("--num_generations", type=int, default=8, help="每条 prompt 生成的回复数 G")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="Actor 学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', action="store_true", help="是否使用 MoE 架构")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt 最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/miniMind/dataset/rlaif.jsonl", help="RLAIF 数据路径")
    parser.add_argument("--loss_type", type=str, default="grpo", choices=["grpo", "cispo"], help="loss 类型：grpo=双边clip，cispo=上界clip")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="GRPO 双边裁剪参数 ε")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="CISPO 上界裁剪参数 ε_high")
    parser.add_argument("--kl_coef", type=float, default=0.1, help="KL 散度惩罚系数 β")
    parser.add_argument("--grpo_update_iters", type=int, default=2, help="同一批 rollout 重复更新次数")
    parser.add_argument("--early_stop_kl", type=float, default=0.25, help="KL 早停阈值")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="/root/autodl-tmp/internlm2-1_8b-reward", help="Reward 模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用 wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb 项目名")
    parser.add_argument("--use_compile", action="store_true", help="是否使用 torch.compile 加速")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug 模式下每隔多少 step 打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启 thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout 引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang 服务器 URL")
    parser.add_argument("--sglang_model_path", type=str, default="/root/autodl-tmp/miniMind/model", help="SGLang tokenizer 路径")
    parser.add_argument("--sglang_shared_path", type=str, default="/root/autodl-tmp/miniMind/trainer/sglang_ckpt_grpo", help="SGLang 共享存储路径")
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'out')

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + dist.get_rank() if dist.is_initialized() else 0)

    # ========== 2. 配置目录、模型参数、检查 ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        use_moe=args.use_moe,
        num_hidden_layers=args.num_hidden_layers
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=args.save_dir) if args.from_resume == 1 else None

    # ========== 3. 设置混合精度 ==========
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    device_type = "cuda" if "cuda" in args.device else "cpu"
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=(device_type == "cuda"))

    # ========== 4. 配 wandb ==========
    wandb = None
    if args.use_wandb:
        import swanlab as wandb
        wandb_id = ckp_data["wandb_id"] if ckp_data is not None else None
        run_name = f"MiniMind GRPO epochs:{args.epochs} bs:{args.batch_size} G:{args.num_generations} lr:{args.learning_rate} moe:{args.use_moe}"
        resume = "must" if args.from_resume == 1 else None
        wandb.init(project=args.wandb_project, experiment_name=run_name, id=wandb_id, resume=resume, config=lm_config.to_dict())

    # ========== 5. 初始化模型（GRPO 无 Critic）==========
    actor_model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval().requires_grad_(False)
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=dtype)

    roll_out = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=actor_model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path
    )

    # ========== 6. 数据集与 optimizer ==========
    ds = RLAIFDataset(args.data_path, tokenizer, max_length=args.max_seq_len + args.max_gen_len, thinking_ratio=args.thinking_ratio)
    sampler = DistributedSampler(ds) if dist.is_initialized() else None
    loader_for_count = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,)
    iters = len(loader_for_count)
    total_optimizer_steps = (
        iters * args.epochs
        * args.grpo_update_iters
        // args.accumulation_steps
    )
    actor_optimizer = torch.optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    actor_scheduler = CosineAnnealingLR(actor_optimizer, total_optimizer_steps, eta_min=args.learning_rate / 10)

    start_epoch, start_step = 0, 0
    if ckp_data is not None:
        actor_model.load_state_dict(ckp_data["model"])
        actor_optimizer.load_state_dict(ckp_data["optimizer"])
        actor_scheduler.load_state_dict(ckp_data["scheduler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data["step"]
        Logger(f"成功从 checkpoint 恢复: Epoch {start_epoch}, Step {start_step}")

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile:
        actor_model = torch.compile(actor_model)

    if dist.is_initialized():
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])

    if is_main_process(): roll_out.update_policy(actor_model)

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        sampler and sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(sampler or indices, args.batch_size, skip_batches=skip)
        loader = DataLoader(ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        Logger(f"{epoch + 1}/{args.epochs}  跳过前{skip}个batch, 从{skip}开始")
        grpo_train_epoch(epoch, loader, len(loader), roll_out, ref_model,
                         actor_scheduler, reward_model, skip, wandb)

    if dist.is_initialized(): dist.destroy_process_group()
