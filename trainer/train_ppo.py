import torch
import torch.nn as nn
import re
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch.distributed as dist
from model.model_minimind import Attention, MiniMindForCausalLM, MiniMindConfig
from torch.utils.data import DistributedSampler, DataLoader
from trainer.rollout_engine import create_rollout_engine, RolloutResult
from dataset.lm_dataset import RLAIFDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel
from dataclasses import dataclass
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import time
from trainer.trainer_utils import get_lr, init_model, Logger, is_main_process,lm_checkpoint,init_distributed_mode, setup_seed, SkipBatchSampler, LMForRewardModel
def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower()) # 0, len -1 -> 0, len - n
    grams= [tuple(toks[i:i+n])for i in range(len(toks)-n+1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        self.value_head = nn.Linear(params.hidden_size, 1)
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        values = self.value_head(hidden_states).squeeze(-1)
        return values # b, seq,
def calculate_rewards(prompts, responses, reward_model):
    reward_model_scores = []
    rewards = torch.zeros(len(responses), device=args.device)
    with torch.no_grad():
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
            answer = response
            if "</think>" in response:
                thinking_content, answer_content = response.split("</think>", 1)
                answer = answer_content.strip()
                rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                rewards[i] += 0.25 if response.count("</think>") == 1 else -0.25  # fix: was `rewards +=` (whole tensor)
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip() }for role, content in matches]
            reward_model_scores.append(reward_model(messages, answer))
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards = rewards + reward_model_scores
    return rewards
"""
PPO 训练循环。

逻辑划分：
  1. prepare_batch       —— prompt 编码、rollout 生成、reward 计算
  2. build_masks         —— 构造 response 区域的有效位置 mask
  3. collect_rollout     —— 无梯度推理，拿 old_values / old_logp / ref_logp
  4. compute_gae         —— GAE 优势估计 + 归一化
  5. ppo_update          —— 多轮 mini-batch 更新（含早停）
  6. logging / checkpoint 与其它副作用
"""
# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------
@dataclass
class RolloutBatch:
    """一次rollout产生的pre-batch数据，全部没有梯度"""
    gen_out: torch.Tensor # [B, P+R]
    full_mask: torch.Tensor # [B, P+R] 非 pad mask
    labels: torch.Tensor # [B, P+R-1] 右移一位
    resp_start: int # resp在labels的起始位置
    resp_length: torch.Tensor # [B] resp的真实长度
    resp_policy_mask: torch.Tensor # 在policy loss里的mask
    resp_value_mask: torch.Tensor # 在value loss里的mask
    rewards: torch.Tensor # [B] 外部奖励，末尾
    old_logp: torch.Tensor # [B, R] Actor 模型快照 log
    old_value: torch.Tensor # [B, R] Critic 模型快照
    ref_logp: torch.Tensor # [B, R] Ref 模型 logp

@dataclass
class AdvantageBatch:
    """GAE计算结果：归一化a， returns"""
    advantages: torch.Tensor # [B, R] 归一化后的a
    returns: torch.Tensor # [B, R] returns_t = A_t（原始，未归一化） + V_old(s_t) Critic目标
@dataclass
class MiniBatchStats:
    """一次mini batch更新的指标累加器"""
    policy_loss: float = 0.0
    value_loss: float = 0.0
    aux_loss: float = 0.0
    approx_kl: float = 0.0 # E[ 0.5 × (log π_new(a_t) − log π_old(a_t))² ]
    kl_ref: float = 0.0 # x = log π_ref(a_t) − log π_new(a_t) kl_ref ≈ E[ exp(x) − x − 1 ] （≥ 0 恒成立）
    clipfrac: float = 0.0
    # ratio_t = π_new(a_t) / π_old(a_t)
    # clipfrac = E[ 1 if |ratio − 1| > ε else 0 ]
    count: int = 0

    def add(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, getattr(self, k) + v)
        self.count += 1
    def mean(self, key: str) -> float:
        return getattr(self, key) / max(self.count, 1)

# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _unwrap(model):
    """脱掉 DDP 外壳，拿到原始模型。"""
    return model.module if isinstance(model, DistributedDataParallel) else model
def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """mask 加权平均，防除零。"""
    return (x * mask).sum() / mask.sum().clamp(1)
def _gather_logp(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    从 logits 中取出每个实际 token 的 log 概率。
    logits: [B, T, V]，labels: [B, T] → 返回 [B, T]
    """
    logp = F.log_softmax(logits, dim=-1)
    return torch.gather(logp, dim=2, index=labels.unsqueeze(2)).squeeze(2)
# ---------------------------------------------------------------------------
# 1. 准备 batch：编码 → rollout → reward
# ---------------------------------------------------------------------------
def prepare_batch(prompts, tokenizer, rollout_engine, reward_model, args):
    enc = tokenizer(
        prompts,
        return_tensors = "pt",
        padding = True,
        padding_side = "left",
        truncation = True,  # fix: was `trunction`
        max_length = args.max_seq_len
    ).to(args.device)
    rollout_result = rollout_engine.rollout(
        enc.input_ids,
        enc.attention_mask,
        num_generations = 1,
        max_new_tokens = args.max_gen_len,
        temperature = 0.8
    )
    rewards = calculate_rewards(prompts, responses=rollout_result.completions, reward_model=reward_model) # [B]
    return {
        "prompt_length": enc.input_ids.shape[1],
        "prompt_attention_mask": enc.attention_mask,
        "gen_out": rollout_result.output_ids,
        "responses_text": rollout_result.completions,
        "rewards": rewards
    }
def build_masks(gen_out: torch.Tensor, prompt_length: int, tokenizer, prompt_attention_mask: torch.Tensor = None):
    """
    返回 response 区域的 policy mask 和有效长度。
      - 非 pad 区域
      - 非 prompt 区域
    当 pad_id == eos_id 时，用 first-EOS cumsum 区分真实 EOS 和填充 EOS。
    """
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    B = gen_out.shape[0]
    labels = gen_out[:, 1:]                          # [B, P+R-1]
    resp_start = prompt_length - 1
    resp_labels = labels[:, resp_start:]             # [B, R]

    if pad_id != eos_id:
        full_mask = (gen_out != pad_id).float()
        resp_lengths = (resp_labels != pad_id).sum(1)
    else:
        # pad_id == eos_id: 第一个 EOS 是真实结尾，后续 EOS 是 padding
        resp_gen = gen_out[:, prompt_length:]        # [B, R]
        is_eos = (resp_gen == eos_id)
        # prev_eos_cumsum[b, t] = t 之前出现过的 EOS 数量
        prev_cumsum = torch.cat([
            torch.zeros(B, 1, dtype=torch.long, device=gen_out.device),
            is_eos[:, :-1].long().cumsum(dim=1)
        ], dim=1)                                    # [B, R]
        resp_valid = (prev_cumsum == 0)              # 包含第一个 EOS，排除其后 [B, R]
        resp_lengths = resp_valid.sum(1)             # [B]
        if prompt_attention_mask is not None:
            prompt_mask = prompt_attention_mask.float()
        else:
            prompt_mask = torch.ones(B, prompt_length, device=gen_out.device)
        full_mask = torch.cat([prompt_mask, resp_valid.float()], dim=1)

    resp_policy_mask = (
        torch.arange(resp_labels.shape[1], device=gen_out.device).unsqueeze(0) < resp_lengths.unsqueeze(1)
    ).float()
    return {
        "labels": labels,
        "full_mask": full_mask,
        "resp_start": resp_start,
        "resp_policy_mask": resp_policy_mask,
        "resp_value_mask": resp_policy_mask.clone(),
        "resp_lengths": resp_lengths,
    }
# ---------------------------------------------------------------------------
# 3. rollout 阶段的无梯度推理
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_rollout(actor_model, critic_model, ref_model,
                    gen_out, full_mask, labels, resp_start, autocast_ctx):
    """
    一次 no_grad 推理，拿到三个网络的快照：
      - Critic  → V_old
      - Actor   → log π_old
      - Ref     → log π_ref
    """
    old_values = critic_model(input_ids=gen_out, attention_mask=full_mask)[:, resp_start:-1] # [B, R]
    with autocast_ctx:
        old_logits = actor_model(input_ids=gen_out, attention_mask=full_mask).logits
    old_logp = _gather_logp(old_logits[:, :-1], labels)[:, resp_start:] # [B, R]
    ref_logits = ref_model(input_ids=gen_out, attention_mask=full_mask).logits
    ref_logp = _gather_logp(ref_logits[:, :-1], labels=labels)[:, resp_start:] # [B, R]
    return old_values, old_logp, ref_logp
@torch.no_grad()
def compute_gae(old_values: torch.Tensor, rewards: torch.Tensor,
                resp_lengths: torch.Tensor, resp_policy_mask: torch.Tensor,
                gamma: float, lam: float, device) -> AdvantageBatch:
    """
    返回归一化后的 advantages 和未归一化的 returns。
    算advantages没有梯度
    GAE 递推：
        δ_t = r_t + γ · V(t+1) − V(t)
        A_t = δ_t + γλ · A_{t+1}
    returns_t = A_t（原始，未归一化） + V_old(s_t) Critic目标
    外部奖励只加在每条回复的最后一个有效 token 上。
    """
    B, R = old_values.shape
    advantages = torch.zeros_like(old_values, device=device)
    # rewards: [B] resp_length: [B]
    token_rewards = torch.zeros_like(old_values, device=device)
    token_rewards[torch.arange(B, device=device), (resp_lengths - 1).long()] = rewards
    a_t = torch.zeros(B, device=device)  # fix: was torch.zeros_like(B, ...) — B is int, not Tensor
    for t in reversed(range(R)):
        next_value = old_values[:, t + 1] if t + 1 < R else 0.0
        delta = token_rewards[:, t] + gamma * next_value - old_values[:, t]
        a_t = delta + lam * gamma * a_t
        advantages[:, t] = a_t
    returns = (advantages + old_values)
    adv_mean = _masked_mean(advantages, resp_policy_mask)
    adv_var = _masked_mean((advantages - adv_mean)**2, resp_policy_mask)
    advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) *resp_policy_mask

    return AdvantageBatch(advantages, returns)
# ---------------------------------------------------------------------------
# 5. 一次 mini-batch 的 forward + loss
# ---------------------------------------------------------------------------

def forward_minibatch(actor_model, critic_model, batch: RolloutBatch,
                      adv: AdvantageBatch, inds: torch.Tensor,
                      args, autocast_ctx, lm_config):
    """
    对一个 mini-batch 跑一次前向，返回所有 loss 组件和指标。
    ratio_t = π_θ_new(a_t) / π_θ_old(a_t)= exp( log π_new(a_t) − log π_old(a_t) )
    approx_kl = _masked_mean(0.5 * log_ratio ** 2, mask)
    L_clip_t = −min( ratio_t · A_t, clip(ratio_t, 1−ε, 1+ε) · A_t )
    x_t = log π_ref(a_t) − log π_new(a_t)
    KL_t = exp(x_t) − x_t − 1
    L_policy = mean L_clip + β · mean KL
    V_pred_t = Critic 当前输出（新的一次前向传播）
    V_clipped_t = clip( V_pred_t, V_old_t − ε_v, V_old_t + ε_v )
    L_value_t = 0.5 × max( (V_pred_t − returns_t)²,(V_clipped_t − returns_t)² )

    """
    actor = _unwrap(actor_model)
    critic = _unwrap(critic_model)

    mb_values = critic(input_ids=batch.gen_out[inds], attention_mask=batch.full_mask[inds])[:, batch.resp_start:-1] # len(inds), R
    with autocast_ctx:
        mb_result = actor(input_ids=batch.gen_out[inds], attention_mask=batch.full_mask[inds])
        aux_loss = mb_result.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
        mb_logits = mb_result.logits[:, :-1]  # fix: was logits[:-1] (wrong dim, cut batch not seq)
        mb_logp = _gather_logp(mb_logits, batch.labels[inds])[:, batch.resp_start:]  # fix: was [:, batch.resp_start] (single col)

    log_ratio = mb_logp - batch.old_logp[inds]
    ratio = torch.exp(log_ratio) # len(inds), R
    clipfrac = _masked_mean(((ratio - 1.0).abs() > args.clip_epsilon).float(), mask=batch.resp_policy_mask[inds])
    approx_kl = _masked_mean(0.5 * log_ratio ** 2, batch.resp_policy_mask[inds])
    # fix: correct PPO clip formula — min(r*A, clip(r)*A) ≠ min(r, clip(r))*A when A < 0
    A = adv.advantages[inds]
    L_clip = -torch.min(ratio * A, torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon) * A)
    mean_L_clip = _masked_mean(L_clip, mask=batch.resp_policy_mask[inds])
    x = batch.ref_logp[inds] - mb_logp
    KL = torch.exp(x) - x - 1
    kl_ref = _masked_mean(KL, mask=batch.resp_policy_mask[inds])
    policy_loss = mean_L_clip + kl_ref * args.kl_coef
    v_clip = torch.clamp(mb_values, batch.old_value[inds] - args.cliprange_value, batch.old_value[inds] + args.cliprange_value)
    L_value = 0.5 * torch.max((mb_values - adv.returns[inds]) ** 2, (v_clip - adv.returns[inds]) ** 2)
    value_loss = _masked_mean(L_value, mask=batch.resp_policy_mask[inds])

    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "aux_loss": aux_loss,
        "approx_kl": approx_kl,
        "kl_ref": kl_ref.detach(),
        "clipfrac": clipfrac,
        }

def _sync_kl(approx_kl: torch.Tensor) -> torch.Tensor:
    """跨卡同步 approx_kl，避免早停在不同 rank 上分叉导致 DDP 死锁。"""
    kl = approx_kl.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(kl, op=dist.ReduceOp.AVG)
    return kl
def _optimizer_step(actor_optimizer, critic_optimizer,
                    actor_scheduler, critic_scheduler,
                    actor_model, critic_model, args):
    """梯度裁剪 → step → scheduler → zero_grad。"""
    clip_grad_norm_(actor_model.parameters(), args.grad_clip)
    clip_grad_norm_(critic_model.parameters(), args.grad_clip)
    actor_optimizer.step()
    critic_optimizer.step()
    actor_scheduler.step()
    critic_scheduler.step()
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()

# ---------------------------------------------------------------------------
# 6. PPO 多轮 mini-batch 更新
# ---------------------------------------------------------------------------

def ppo_update(actor_model, critic_model, batch: RolloutBatch, adv: AdvantageBatch,
               actor_optimizer, critic_optimizer,
               actor_scheduler, critic_scheduler,
               args, autocast_ctx, lm_config, grad_accum_step: int):
    """
    跑 ppo_update_iters 轮、每轮切若干 mini-batch 做更新。
    支持早停、梯度累积，并返回更新后的 grad_accum_step 和累计指标。
    """
    mini_batch = min(args.batch_size, args.mini_batch_size)
    stop_ppo = False
    stats = MiniBatchStats()
    for _ in range (args.ppo_update_iters):
        perm = torch.randperm(args.batch_size, device=args.device)
        if stop_ppo:
            break
        for i in range(0, args.batch_size, mini_batch):
            inds = perm[i:i+mini_batch]
            mb_out = forward_minibatch(actor_model, critic_model, batch, adv, inds, args, autocast_ctx, lm_config)
            kl_ref = _sync_kl(mb_out["kl_ref"])
            if kl_ref > args.early_stop_kl:
                stop_ppo = True
            total = mb_out["policy_loss"] + args.vf_coef * mb_out["value_loss"] + mb_out["aux_loss"]
            loss = total * 0.0 if stop_ppo else total / args.accumulation_steps
            loss.backward()
            grad_accum_step += 1
            if grad_accum_step % args.accumulation_steps == 0:
                _optimizer_step(actor_optimizer, critic_optimizer, actor_scheduler, critic_scheduler, actor_model, critic_model, args)
            stats.add(
                policy_loss = mb_out["policy_loss"].item(),
                value_loss = mb_out["value_loss"].item(),
                aux_loss = mb_out["aux_loss"].item(),
                approx_kl = mb_out["approx_kl"].item(),
                kl_ref = kl_ref.item(),
                clipfrac = mb_out["clipfrac"].item()
            )
    return stats, grad_accum_step
# ---------------------------------------------------------------------------
# 7. 日志 & checkpoint（副作用集中在这里）
# ---------------------------------------------------------------------------

def log_step(epoch, step, iters, stats: MiniBatchStats,
             rewards: torch.Tensor, resp_lengths: torch.Tensor,
             actor_optimizer, critic_optimizer, args, start_time, wandb=None, ):
    if not is_main_process():  # fix: was `is_main_process` (missing call parens)
        return
    avg_rewards = rewards.mean().item()
    avg_resp_length = resp_lengths.mean().item()
    actor_lr = actor_optimizer.param_groups[0]["lr"]
    critic_lr = critic_optimizer.param_groups[0]["lr"]
    # TODO: 增加剩下时间
    eta_min = (time.time() - start_time) / step * (iters - step + (args.epochs - (epoch + 1)) * iters) // 60
    matrix = {
        "policy_loss": stats.policy_loss,
        "value_loss": stats.value_loss,
        "aux_loss": stats.aux_loss,
        "approx_kl": stats.approx_kl,
        "rl_ref": stats.kl_ref,
        "clipfrac": stats.clipfrac,
        "avg_rewards": avg_rewards,
        "avg_resp_length": avg_resp_length,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "eta_min": eta_min,
    }
    if wandb is not None:
        wandb.log(matrix)
    Logger(
        f"epoch: [{epoch} / {args.epochs}] ({step} / {iters}), "
        f"policy loss: {matrix['policy_loss']:.4f}, "
        f"aux_loss: {matrix['aux_loss']:.4f}"
        f"approx_kl: {matrix['approx_kl']:.4f}"
        f"rl_ref: {matrix['rl_ref']:.4f}"
        f"clipfrac: {matrix['clipfrac']:.4f}"
        f"avg_rewards: {matrix['avg_rewards']:.4f}"
        f"avg_resp_length: {matrix['avg_resp_length']:.2f}"
        f"actor_lr: {matrix['actor_lr']:.8f}"
        f"critic_lr: {matrix['critic_lr']:.8f}"
        f"eta_min: {matrix['eta_min']:.2f}"  # fix: was matrix["criteta_minic_lr"] (wrong key)

    )


def save_checkpoint(epoch, step, actor_model, critic_model,
                    actor_optimizer, critic_optimizer,
                    actor_scheduler, critic_scheduler,
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
        critic_model=critic_model,
        critic_optimizer=critic_optimizer,
        critic_scheduler=critic_scheduler,
    )
    actor_model.train()
def debug_log_samples(step, prompts, responses_text, rewards, prompt_length, args):
    if not (args.debug_mode and is_main_process() and step % args.debug_interval == 0):  # fix: & → and (precedence bug)
        return
    for i, prompt in enumerate(prompts):
        Logger(f"[Debug]: step = {step}, sample[{i}]")
        Logger(f"[Debug]: prompt_length = {prompt_length}, response_length[{len(responses_text[i])}]")
        Logger(f"{'-' * 100}")
        Logger(f"Context begin {'=' * 30}")
        Logger(prompt)
        Logger(f"Context end {'=' * 30} ")
        Logger(f"Response begin {'=' * 30}")
        Logger(responses_text[i])
        Logger(f"Response end {'=' * 30}")
        Logger(f"{'-' * 100}")
        Logger(f"reward: {rewards[i].item():.4f}")



# ---------------------------------------------------------------------------
# 主训练循环
# ---------------------------------------------------------------------------

def ppo_train_epoch(epoch, loader, iters, rollout_engine, ref_model,
                    actor_scheduler, critic_scheduler, reward_model,
                    start_step=0, wandb=None, use_sglang=False):
    """
    单个 epoch 的 PPO 训练。
    每个 step 的生命周期：
        prepare → masks → rollout快照 → GAE → PPO更新 → log → 可能保存 checkpoint
    """
    actor_model.train()
    critic_model.train()
    start_time = time.time()
    grad_accum_step = 0
    for step, raw_batch in enumerate(loader, start_step + 1):
        p = prepare_batch(raw_batch["prompt"], tokenizer, rollout_engine, reward_model, args)  # fix: missing args
        debug_log_samples(step, raw_batch["prompt"], p["responses_text"], p["rewards"], p["prompt_length"], args)  # fix: missing prompt_length, args
        m = build_masks(p["gen_out"], p["prompt_length"], tokenizer, p["prompt_attention_mask"])  # fix: was p["generation"] (wrong key)
        old_values, old_logp, ref_logp = collect_rollout(actor_model, critic_model, ref_model, p["gen_out"], m["full_mask"], m["labels"], m["resp_start"], autocast_ctx,)
        old_values = old_values * m["resp_value_mask"]
        rollout_batch = RolloutBatch(
            gen_out=p["gen_out"],
            full_mask=m["full_mask"],
            labels=m["labels"],
            resp_start=m["resp_start"],
            resp_length=m["resp_lengths"],
            resp_policy_mask=m["resp_policy_mask"],
            resp_value_mask=m["resp_value_mask"],
            rewards=p["rewards"],
            old_logp=old_logp,
            old_value=old_values,
            ref_logp=ref_logp
        )

        adv = compute_gae(old_values, rollout_batch.rewards, rollout_batch.resp_length, rollout_batch.resp_policy_mask, args.gamma, args.lam, args.device)
        stats, grad_accum_step = ppo_update(actor_model=actor_model,
                   critic_model=critic_model,
                   batch=rollout_batch,
                   adv=adv,
                   actor_optimizer=actor_optimizer,
                   critic_optimizer=critic_optimizer,
                   actor_scheduler=actor_scheduler,
                   critic_scheduler=critic_scheduler,
                   args = args,
                   autocast_ctx=autocast_ctx,
                   lm_config=lm_config,
                   grad_accum_step=grad_accum_step
                   )
        if grad_accum_step % args.accumulation_steps != 0:
            _optimizer_step(
                actor_optimizer, critic_optimizer, actor_scheduler, critic_scheduler, actor_model, critic_model, args
            )
        if step % args.log_interval == 0 or step == iters:
            log_step(epoch, step, iters, stats, rollout_batch.rewards, rollout_batch.resp_length, actor_optimizer, critic_optimizer, args, start_time, wandb, )
        if is_main_process() & step % args.save_interval == 0:
            rollout_engine.update_policy(actor_model)
        if step % args.save_interval == 0 or step == iters:
            save_checkpoint(epoch, step, actor_model, critic_model, actor_optimizer, critic_optimizer, actor_scheduler, critic_scheduler, args, lm_config, wandb,)
        del p, m, old_values, old_logp, ref_logp, rollout_batch, adv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default=None, help="模型保存目录（默认 <项目根>/out）")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=5e-7, help="Critic学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', action="store_true", help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/miniMind/dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--gamma", type=float, default=1.0, help="GAE折扣因子")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda参数")
    parser.add_argument("--cliprange_value", type=float, default=0.2, help="Value function裁剪范围")
    parser.add_argument("--ppo_update_iters", type=int, default=2, help="同一批rollout重复更新次数")
    parser.add_argument("--early_stop_kl", type=float, default=0.25, help="PPO early stop 的 KL 阈值")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="PPO每次更新的minibatch大小")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="/root/autodl-tmp/internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    parser.add_argument("--use_compile", action="store_true", help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8997", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="/root/autodl-tmp/miniMind/model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="/root/autodl-tmp/miniMind/trainer/sglang_ckpt_ppo", help="SGLang共享存储路径")
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
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=args.save_dir) if args.from_resume==1 else None
    # ========== 3. 设置混合精度 ==========
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    device_type =  "cuda" if "cuda" in args.device else "cpu"
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=(device_type=="cuda"))
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb:
        import swanlab as wandb
        wandb_id = ckp_data["wandb_id"] if ckp_data is not None else None
        run_name = f"MiniMind PPO epoches : {args.epochs} - batch_size : {args.batch_size} - learning_rate : {args.learning_rate} - use_moe: {args.use_moe}"
        resume = "must" if args.from_resume == 1 else None
        wandb.init(project=args.wandb_project, experiment_name=run_name, id=wandb_id, resume=resume, config=lm_config.to_dict())  # fix: was args.wandb_id (undefined)
    # ========== 5. 初始化模型和数据 ==========
    actor_model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    state_dict = actor_model.state_dict()
    # fix: load_state_dict returns IncompatibleKeys, not the model — must separate init and load
    critic_model = CriticModel(lm_config).to(args.device)
    critic_model.load_state_dict(state_dict, strict=False)
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval().requires_grad_(False)
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=dtype)
    roll_out = create_rollout_engine(engine_type=args.rollout_engine,
                                     policy_model=actor_model,
                                     tokenizer=tokenizer,
                                     device=args.device,
                                     autocast_ctx=autocast_ctx,
                                     sglang_base_url=args.sglang_base_url,
                                     sglang_model_path=args.sglang_model_path,
                                     sglang_shared_path=args.sglang_shared_path
                                     )
    ds = RLAIFDataset(args.data_path, tokenizer, max_length=args.max_seq_len+args.max_gen_len, thinking_ratio=args.thinking_ratio)
    sampler = DistributedSampler(ds) if dist.is_initialized() else None
    loader_for_count = DataLoader(ds, batch_size=args.batch_size, sampler=sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = iters * args.epochs * (args.batch_size // args.mini_batch_size) * args.ppo_update_iters // args.accumulation_steps
    actor_optimizer = torch.optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = torch.optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    actor_scheduler = CosineAnnealingLR(actor_optimizer, total_optimizer_steps, eta_min=args.learning_rate/10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, total_optimizer_steps, eta_min=args.critic_learning_rate/10)
    start_epoch, start_step = 0, 0
    if ckp_data is not None:
        actor_model.load_state_dict(ckp_data["model"])
        critic_model.load_state_dict(ckp_data["critic_model"])
        actor_optimizer.load_state_dict(ckp_data["optimizer"])
        critic_optimizer.load_state_dict(ckp_data["critic_optimizer"])
        actor_scheduler.load_state_dict(ckp_data["scheduler"])
        critic_scheduler.load_state_dict(ckp_data["critic_scheduler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data["step"]
        Logger(f"成功从 checkpoint 恢复: Epoch {start_epoch}, Step {start_step}")

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile:
        actor_model = torch.compile(actor_model)
        critic_model = torch.compile(critic_model)

    if dist.is_initialized():
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
    if is_main_process(): roll_out.update_policy(actor_model)
    # ========== 8. 开始训练 ==========

    for epoch in range(start_epoch,args.epochs):
        sampler and sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler  = SkipBatchSampler(sampler or indices, args.batch_size, skip_batches=skip)
        loader = DataLoader(ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        Logger(f"{epoch + 1}/{args.epochs}  跳过前{skip}个batch, 从{skip}开始")
        ppo_train_epoch(epoch, loader, len(loader), roll_out, ref_model, actor_scheduler, critic_scheduler, reward_model, skip, wandb, True if args.rollout_engine == "sglang" else False)
    if dist.is_initialized(): dist.destroy_process_group()
