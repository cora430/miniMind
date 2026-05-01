import argparse
import os
import json
import random
import re
import torch.nn.functional as F
import torch
import math
from dataclasses import dataclass
from typing import Any
import torch.distributed as dist
from trainer.trainer_utils import init_model, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, LMForRewardModel
from model.model_minimind import MiniMindConfig
from trainer.rollout_engine import create_rollout_engine
from dataset.lm_dataset import AgentRLDataset
from torch.utils.data import DistributedSampler, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel
# =========================================================================
# Section 1. 工具定义与模拟执行
# =========================================================================
 
def _fn(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}

TOOLS = [
    _fn("calculate_math", "计算数学表达式",
        {"expression": {"type": "string"}}, ["expression"]),
    _fn("unit_converter", "单位换算",
        {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}},
        ["value", "from_unit", "to_unit"]),
    _fn("get_current_weather", "获取天气",
        {"location": {"type": "string"}}, ["location"]),
    _fn("get_current_time", "获取时间",
        {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, []),
    _fn("get_exchange_rate", "查询汇率",
        {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}},
        ["from_currency", "to_currency"]),
    _fn("translate_text", "翻译文本",
        {"text": {"type": "string"}, "target_language": {"type": "string"}},
        ["text", "target_language"]),
]
WEATHER_DATA = {
    "北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"),
    "深圳": ("30°C", "晴"), "杭州": ("22°C", "阴"), "成都": ("18°C", "小雨"),
    "武汉": ("25°C", "多云"), "南京": ("20°C", "晴"), "西安": ("16°C", "大风"),
    "重庆": ("26°C", "阴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云"),
    "London": ("5°C", "小雨"), "Paris": ("10°C", "阴"), "Sydney": ("25°C", "晴朗"),
}
TIME_DATA = {
    "Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00",
    "Europe/London": "2025-03-07 06:30:00", "Asia/Tokyo": "2025-03-07 15:30:00",
    "Europe/Paris": "2025-03-07 07:30:00", "Australia/Sydney": "2025-03-07 17:30:00",
}
EXCHANGE_DATA = {
    ("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12,
    ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79,
    ("CNY", "JPY"): 20.83, ("AUD", "CNY"): 4.72,
}
TRANSLATE_DATA = {
    ("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好",
    ("今天天气真好", "english"): "The weather is nice today",
    ("I love programming", "chinese"): "我喜欢编程",
    ("机器学习很有趣", "english"): "Machine learning is interesting",
    ("Happy birthday", "chinese"): "生日快乐",
}
UNIT_DATA = {
    "km_miles": 0.621371, "miles_km": 1.60934,
    "kg_pounds": 2.20462, "pounds_kg": 0.453592,
    "meters_feet": 3.28084, "feet_meters": 0.3048,
    "celsius_fahrenheit": 1.8, "fahrenheit_celsius": 0.5556,
}

# -------- 执行器:每个工具一个独立函数 --------
def _exec_math(args):
    expr = str(args.get("expression", "0"))
    for src, dst in [("^", "**"), ("×", "*"), ("÷", "/"), ("−", "-"), ("(", "("), (")", ")")]:
        expr = expr.replace(src, dst)
    return {"result": str(eval(expr, {"__builtins__": {}, "math": math}))}
 
 
def _exec_unit(args):
    value = float(args.get("value", 0))
    key = f"{args.get('from_unit', '').lower()}_{args.get('to_unit', '').lower()}"
    return {"result": round(value * UNIT_DATA.get(key, 1), 4)}
 
 
def _exec_weather(args):
    city = args.get("location")
    temp, cond = WEATHER_DATA.get(city, ("22°C", "晴"))
    return {"city": city, "temperature": temp, "humidity": "65%", "condition": cond}
 
 
def _exec_time(args):
    tz = args.get("timezone", "Asia/Shanghai")
    return {"datetime": TIME_DATA.get(tz, "2025-03-07 14:30:00"), "timezone": tz}
 
 
def _exec_exchange(args):
    frm, to = args.get("from_currency"), args.get("to_currency")
    return {"from": frm, "to": to, "rate": EXCHANGE_DATA.get((frm, to), 1.0)}
 
 
def _exec_translate(args):
    text = args.get("text", "")
    lang = args.get("target_language")
    return {"translated_text": TRANSLATE_DATA.get((text, lang), text)}

EXECUTORS = {
    "calculate_math": _exec_math,
    "unit_converter": _exec_unit,
    "get_current_weather": _exec_weather,
    "get_current_time": _exec_time,
    "get_exchange_rate": _exec_exchange,
    "translate_text": _exec_translate,
}
 
VALIDATORS = {
    "calculate_math": lambda a: bool(a.get("expression")),
    "unit_converter": lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    "get_current_weather": lambda a: bool(a.get("location")),
    "get_current_time": lambda a: True,
    "get_exchange_rate": lambda a: bool(a.get("from_currency")) and bool(a.get("to_currency")),
    "translate_text": lambda a: bool(a.get("text")) and bool(a.get("target_language")),
}

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
def parse_tool_calls(text):
    """从模型输出里抽取 <tool_call>...</tool_call> 的 JSON 块。"""
    calls = []
    for match in _TOOL_CALL_RE.findall(text):
        try:
            calls.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    return calls

def normalize_tool_args(raw):
    """arguments 字段可能是 dict 也可能是 JSON 字符串,统一成 dict。"""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            return {}
    return {}

def execute_tool(name, args):
    """按名称执行工具;失败返回 None。"""
    executor = EXECUTORS.get(name)
    if executor is None:
        return None
    try:
        return executor(args)
    except:
        return None
# =========================================================================
# Section 2. Reward 计算
# =========================================================================
 
_TOK_RE = re.compile(r"\w+|[^\w\s]")
_NUM_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])")
_PURE_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
REWARD_CLIP = 3.0
def rep_penalty(text, n=3, cap=0.5):
    """n-gram 重复率惩罚,越重复扣分越多(上限 cap)。"""
    toks = _TOK_RE.findall(text)
    grams = [tuple(toks[i : i+n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return 0.0
    dup_ratio = (len(grams) - len(set(grams))) / len(grams)
    return min(cap, dup_ratio * cap * 2)
def _strip_think(turn_text):
    """去掉 </think> 之前的思考部分,只保留答案。"""
    if "</think>" in turn_text:
        return turn_text.split("</think>", 1)[-1].strip()
    return turn_text.strip()
def validate_gt_in_text(text, gt_list):
    """检查 gt_list 中有多少元素出现在 text 里(支持字符串或数值近似匹配)。"""
    text_lower = str(text).strip().lower()
    hit = set()
    nums_in_text = [float(i) for i in _NUM_RE.findall(text.replace(",", ""))]
    for gt in gt_list:
        gt_str = str(gt).strip().lower()
        if not gt_str :
            continue
        if gt_str in text_lower:
            hit.add(gt)
        gt_cleaned = gt_str.replace(",", "")
        if _PURE_NUM_RE.fullmatch(gt_cleaned):
            gt_float = float(gt_cleaned)
            if any([abs(gt_float - x) < 1e-6 for x in nums_in_text]):
                hit.add(gt)
    return hit
def _parse_prompt_messages(prompt):
    """从 chat template 渲染后的字符串里还原 messages(给 RM 用)。"""
    pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
    matches = re.findall(pattern, prompt)
    return [{"role": role, "content": content} for role, content in matches]
def _unbalanced_tag_penalty(turn_answers):
    """<tool_call> 和 </tool_call> 数量不匹配的扣分。"""
    return sum([abs(turn.count("<tool_call>") - turn.count("</tool_call>")) * 0.5 for turn in turn_answers])
def _reward_no_tool_call(response, prompt, reward_model):
    """没有工具调用时的 reward:长度、思考结构、RM 分、重复率。"""
    response_stripped = response.strip()
    reward = 0.0
    reward += 0.5 if 5 <= len(response_stripped) <= 800 else -0.5
    answer = response_stripped
    if "</think>" in response:
        think, answer = response_stripped.split("</think>", 1)
        answer = answer.strip()
        reward += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5
        reward += 0.25 if response.count("</think>") == 1 else -0.25
    messages = _parse_prompt_messages(prompt)
    if reward_model is not None: reward += reward_model.get_score(messages, answer)    
    reward -= rep_penalty(answer,)
    return reward
def _reward_with_tool_call(tool_calls, final_answer, gt, tools, unfinished):
    """有工具调用时的 reward:调用对齐、GT 命中、完成度、重复率。"""
    reward = 0.0
    valid_call = 0
    valid_names = set([t["name"] for t in tools]) if tools else set()
    for call in tool_calls:
        name = call.get("name")
        validator = VALIDATORS.get(name)
        args = normalize_tool_args(call.get("arguments", {}))
        if name in valid_names and validator and validator(args):
            valid_call += 1
    tool_gap = max(len(tool_calls) - valid_call, 0) + abs(len(gt) - valid_call)
    reward += 0.5 if tool_gap == 0 else - 0.5 * tool_gap
    # GT 命中
    if unfinished:
        final_text = ""
    elif "</tool_call>" in final_answer:
        final_text = final_answer.split("</tool_call>", 1)[-1].strip()
    else :
        final_text = final_answer
    if gt:
        hits = validate_gt_in_text(final_text, gt)
        reward += 2.5 * len(hits) / len(gt)
    reward -= rep_penalty(final_text or final_answer)
    return reward
def calculate_rewards(prompts, completions, gt_batch, tools_batch, num_gen,
                      reward_model=None, device="cuda",
                      turn_outputs_batch=None, unfinished_batch=None):
    """对一个 batch 内所有 generation 逐条计算 reward。"""
    rewards = torch.zeros(len(completions), device=device)
    for idx, response in enumerate(completions):
        reward = 0.0
        sample_idx = idx // num_gen
        prompt = prompts[sample_idx]
        gt = gt_batch[sample_idx]
        # tools_batch: 每道题允许使用的工具 schema 列表
        tools = tools_batch[sample_idx]
        turn_outputs = turn_outputs_batch[idx] if turn_outputs_batch else [response]
        turn_answers = [_strip_think(t)  for t in turn_outputs]
        final_answer = turn_answers[-1] if turn_answers else response
        unfinished = unfinished_batch[idx] if unfinished_batch else False
        reward -= _unbalanced_tag_penalty(turn_answers)
        tool_calls = []
        for turn in turn_answers:
            tool_calls.extend(parse_tool_calls(turn))
        if tool_calls:
            reward += _reward_with_tool_call(tool_calls, final_answer, gt, tools, unfinished)
        else:
            reward += _reward_no_tool_call(response, prompt, reward_model)
        rewards[idx] = max(min(reward, REWARD_CLIP), -REWARD_CLIP)
    return rewards

# =========================================================================
# Section 3. 多轮 Rollout
# =========================================================================
 
TOOL_RESULT_MAX_CHARS = 2048
 
 
def _render_chat(tokenizer, messages, tools, open_thinking, add_generation_prompt):
    return tokenizer.apply_chat_template(
        messages, tools = tools, open_thinking = open_thinking, add_generation_prompt = add_generation_prompt, tokenize = False
    )        
def _filter_special_tokens(ids, logps, pad_id, eos_id):
    if len(ids) != len(logps): Logger(f"rollout token/logprob length mismatch: {len(ids)} vs {len(logps)}")
    pairs = [(i, lp)for i, lp in zip(ids, logps) if i != pad_id and i != eos_id]
    return [ i for i, lp in pairs] , [lp for i, lp in pairs]
def _tool_result_to_str(result):
    if not result:
        return '{"error": "tool not found"}'
    return json.dumps(result, ensure_ascii=False)[:TOOL_RESULT_MAX_CHARS]
def _execute_calls_and_append(messages, calls):
    """执行所有 tool_call,把结果作为 tool 消息追加到 messages。"""
    for call in calls:
        name = call.get("name")
        args = normalize_tool_args(call.get("arguments", {}))
        result = execute_tool(name, args)
        messages.append({"role":"tool", "content": _tool_result_to_str(result)})
def rollout_single(rollout_engine, tokenizer, messages, tools,
                   max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    """对单个样本做多轮 rollout,直到模型不再产生 tool_call 或达到 max_turns。"""

    response_ids, response_mask, response_old_logps = [], [], []
    turn_outputs = []
    prompt_ids = None
    final_context = ""
    unfinished = False
    prompt_text = ""   
    open_thinking = random.random() < thinking_ratio
    for turn in range(max_turns):
        context = _render_chat(tokenizer, messages, tools, open_thinking=open_thinking, add_generation_prompt=True)
        inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False, ).to(device)
        if not prompt_ids:
            prompt_ids = inputs["input_ids"][0].tolist()
            prompt_text = context
        rollout_result = rollout_engine.rollout(inputs["input_ids"], inputs["attention_mask"], num_generations=1, max_new_tokens=max_new_tokens)
        new_ids, new_logp = _filter_special_tokens(rollout_result.completions_ids[0].tolist(), rollout_result.per_token_logs[0].tolist(), tokenizer.pad_token_id, tokenizer.eos_token_id)
        new_text = rollout_result.completions[0]
        response_ids.extend(new_ids)
        response_mask.extend([1] * len(new_ids))
        response_old_logps.extend(new_logp)
        final_context = context + new_text
        turn_outputs.append(new_text)
        calls = parse_tool_calls(new_text)
        if not calls:
            break
        unfinished = (turn == max_turns - 1)
        messages.append({"role": "assistant", "content": new_text})
        _execute_calls_and_append(messages, calls)

        observe_context = _render_chat(tokenizer, messages, tools, open_thinking, add_generation_prompt = not unfinished)
        observe_ids = tokenizer(observe_context, return_tensors="pt", add_special_tokens=False,)["input_ids"][0].tolist()
        observe_delta = observe_ids[len(prompt_ids) + len(response_ids):]
        response_ids.extend(observe_delta)
        response_mask.extend([0] * len(observe_delta))
        response_old_logps.extend([0.0] * len(observe_delta))
        final_context = observe_context
    final_output = turn_outputs[-1] if turn_outputs else ""
    return (final_output, final_context, prompt_ids or [],
            response_ids, response_mask, response_old_logps,
            list(turn_outputs), unfinished, prompt_text)
def rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, num_gen,
                  max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    """对 batch 内每个样本做 num_gen 次 rollout,扁平化返回。"""
    fields = [[] for _ in range(9)]
    for messages, tools in zip(messages_batch, tools_batch):
        for _ in range(num_gen):
            out = rollout_single(
                rollout_engine, tokenizer, messages, tools, max_turns, max_new_tokens, thinking_ratio, device
            )
            for acc, val in zip(fields, out):
                acc.append(val)
    return tuple(fields)
def pack_rollout_to_tensors(prompt_ids_batch, response_ids_batch,
                            response_masks_batch, response_old_logps_batch,
                            pad_token_id, max_total_len, device):
    """把可变长度的 rollout 结果 pad 成矩形张量。"""
    packed = []
    for prompt_ids, response_ids, response_mask, response_old_logs in zip(prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch):
        ids = prompt_ids + response_ids
        mask = [0] * len(prompt_ids) + response_mask
        old_logp = [0.0] * (len(prompt_ids) - 1) + response_old_logs
        if len(ids) > max_total_len:
            ids = ids[-max_total_len:]
            mask = mask[-max_total_len:]
            old_logp = old_logp[-(max_total_len-1):]
        prompt_len = next((i for i, v in enumerate(mask) if v == 1), len(mask)) 
        packed.append((ids, mask, old_logp, prompt_len))

    def _pad(seq, fill, length):
        return seq + [fill] * (length - len(seq))
    seq_len = torch.tensor([len(idx) for idx, *_ in packed], device=device, )
    max_length = int(seq_len.max().item())
    input_ids = torch.tensor(
        [_pad(ids, pad_token_id, max_length) for ids, *_ in packed], device=device
    )
    prompt_lens = torch.tensor(
        [pl for *_, pl in packed], device=device
    )
    full_response_mask = torch.tensor(
        [_pad(m, 0, max_length) for _, m, *_ in packed], device=device, dtype=torch.float32
    )
    old_per_token_logs = torch.tensor(
        [_pad(ol, 0.0, max_length -1) for *_, ol, _ in packed], device=device, dtype=torch.float32
    )
    return input_ids, prompt_lens, full_response_mask, old_per_token_logs, seq_len
def compute_policy_logps(model, input_ids):
    """对当前策略取 log-prob。返回 next-token 维度 [B, T-1] 和模型原始输出。"""
    out = model(input_ids=input_ids)
    logits = out.logits[:, :-1, :]
    logp = F.log_softmax(logits, dim=-1) # B, T-1, v
    return torch.gather(logp, dim=2, index=input_ids[:, 1:].unsqueeze(2), ).squeeze(2), out
def truncate_mask_at_eos(full_response_masks, input_ids, eos_token_id):
    """把 completion_mask 在第一个 EOS 之后全部清零。"""
    completion_mask = full_response_masks[:, 1:]
    is_eos = (input_ids[:, 1:] == eos_token_id) * completion_mask.bool()
    eos_idx = torch.full((completion_mask.shape[0]), completion_mask.shape[1] - 1, device=full_response_masks.device)
    has_eos = torch.any(is_eos, dim=-1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    pos = torch.arange(completion_mask.shape[1], device=completion_mask.device).unsqueeze(0) # 1, T
    completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
    token_counts = completion_mask.sum(-1)
    return completion_mask, token_counts
def compute_advantages(rewards, num_generations, eps=1e-4):
    """按 group 做标准化(GRPO 风格)。"""
    grouped = rewards.view(-1, num_generations)
    reward_mean = grouped.mean(1).repeat_interleave(num_generations)
    reward_std = grouped.std(1, unbiased=False).repeat_interleave(num_generations)
    return (rewards - reward_mean) / (reward_std + eps), grouped

def compute_policy_loss(per_token_logps, old_per_token_logps, ref_per_token_logps,
                        advantages, completion_mask, token_counts,
                        loss_type, beta, epsilon, epsilon_high):
    """GRPO 或 CISPO 的 per-token 损失,再做 per-sequence 平均。"""
    x = ref_per_token_logps - per_token_logps
    KL = torch.exp(x) - x - 1
    ratio = torch.exp(per_token_logps - old_per_token_logps)
    adv = advantages.unsqueeze(1)
    if loss_type == "cispo":
        clamped = torch.clamp(ratio, max = epsilon_high).detach()
        per_token_loss = -(clamped * per_token_logps * adv - KL * beta)
    elif loss_type == "grpo":
        clamped = torch.clamp(ratio, min = 1 - epsilon, max = 1 + epsilon)
        per_token_loss = -(torch.min(clamped * adv, ratio * adv) - KL * beta)
    valid_rows = token_counts > 0
    seq_loss = (per_token_loss * completion_mask).sum(dim=1)[valid_rows] / token_counts[valid_rows].clamp(1)
    return seq_loss.mean()
# =========================================================================
# Section 5. 日志 / 调试 / 保存
# =========================================================================
 
def log_debug_batch(args, step, messages_batch, contexts, input_ids,
                    prompt_lens, seq_lens, rewards, gt_batch, tokenizer):
    for i in range(len(messages_batch)):
        Logger(f"[DEBUG] step={step}, gt[{i}]: {repr(gt_batch[i])}")
        Logger('-' * 100)
        for j in range(args.num_generations):
            idx = i * args.num_generations + j
            plen, slen = prompt_lens[idx].item(), seq_lens[idx].item()
            Logger(f"{'=' * 30} [DEBUG] gen[{i}][{j}] CONTEXT_BEGIN {'=' * 30}")
            Logger(contexts[idx])
            Logger(f"{'=' * 31} [DEBUG] gen[{i}][{j}] CONTEXT_END {'=' * 31}")
            Logger(f"[DEBUG] gen[{i}][{j}] prompt_len={plen}, seq_len={slen}")
            tokens = input_ids[idx, plen:slen].tolist()
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            Logger(f"{'=' * 28} [DEBUG] gen[{i}][{j}] COMPLETION_BEGIN [{plen}:{slen}] {'=' * 28}")
            Logger(text)
            Logger(f"{'=' * 29} [DEBUG] gen[{i}][{j}] COMPLETION_END {'=' * 29}")
            Logger(f"[DEBUG] gen[{i}][{j}] reward={rewards[idx].item():.4f}")
            Logger('=' * 100)
def log_step_metrics(epoch, step, iters, args, loss, rewards, token_counts,
                     ref_logps, per_token_logps, completion_mask,
                     grouped_rewards, advantages, optimizer, wandb):
    pl = loss.item() * args.accumulation_steps
    ar = rewards.mean().item()
    al = token_counts.float().mean().item()
    kl = ((ref_logps - per_token_logps) * completion_mask).sum().item() / max(1, token_counts.sum().item())
    gs = grouped_rewards.std(dim=1, unbiased=False).mean().item()
    am = advantages.mean().item()
    ast = advantages.std().item()
    lr = optimizer.param_groups[0]["lr"]
    Logger(
        f"Epoch:[{epoch + 1} / {args.epochs}] step: [{step} / {iters}]"
        f'Reward:{ar:.4f}, KL:{kl:.4f}, GrpStd:{gs:.4f}, AdvStd:{ast:.4f}, '
        f'Loss:{pl:.4f}, AvgLen:{al:.2f}, AdvMean:{am:.4f}, LR:{lr:.8f}'
    ) 
    if is_main_process() and wandb:
        wandb.log({
            "reward": ar, "kl_ref": kl,
            "group_reward_std": gs, "advantages_std": ast,
            "policy_loss": pl, "avg_response_len": al,
            "advantages_mean": am, "learning_rate": lr,
        })
def save_checkpoint(args, lm_config, model, optimizer, scheduler, epoch, step, wandb, tokenizer=None):
    model.eval()
    lm_checkpoint(lm_config, args.save_weight, model, optimizer, epoch, step, wandb,
                  save_dir=args.save_dir, tokenizer=tokenizer, scheduler=scheduler)
    model.train()
def optimizer_step(args, model, optimizer, scheduler, rollout_engine, step):
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    if is_main_process() and step % args.save_interval == 0:
        rollout_engine.update_policy(model)

@dataclass
class TrainCtx:
    """把长期对象打包成一个容器,避免 train_epoch 签名爆炸。"""
    args: Any
    lm_config: Any
    tokenizer: Any
    model: Any
    ref_model: Any
    reward_model: Any
    rollout_engine: Any
    optimizer: Any
    scheduler: Any
    autocast_ctx: Any
    wandb: Any
 
def train_one_step(ctx, batch):
    """单步训练:rollout -> pack -> forward -> loss -> backward。
    返回一个 dict,供主循环拿去记日志。
    """
    args = ctx.args
    messages_batch = batch["messages"]
    tools_batch = batch["tools"]
    gt_batch = batch["gt"]
    with torch.no_grad():
        completions, contexts, prompt_ids_batch, responses_ids_batch, respose_mask_batch, response_old_logp_batch, turn_outputs_batch, unfinished_batch, prompts = rollout_batch(ctx.rollout_engine, ctx.tokenizer, messages_batch, tools_batch, args.num_generations, max_turns=3, max_new_tokens=args.max_gen_len, thinking_ratio=args.thinking_ratio, device=args.device)
    input_ids, prompt_lens, full_response_mask, old_per_token_logs, seq_len = pack_rollout_to_tensors(prompt_ids_batch, responses_ids_batch, respose_mask_batch, response_old_logp_batch, ctx.tokenizer.pad_token_id, args.max_total_len, device=args.device)
    with torch.no_grad():
        ref_per_token_logp, _ = compute_policy_logps(ctx.ref_model, input_ids)
    with ctx.autocast_ctx:
        per_token_logp, out = compute_policy_logps(ctx.model, input_ids)
        aux_loss = (out.aux_loss if ctx.lm_config.use_moe
                    else torch.tensor(0.0, device=args.device))
    
    completion_mask, token_counts = truncate_mask_at_eos(full_response_mask, input_ids, ctx.tokenizer.eos_token_id)
    
    rewards = calculate_rewards(prompts, completions, gt_batch, tools_batch, args.num_generations, ctx.reward_model, args.device, turn_outputs_batch, unfinished_batch)
    advantages, grouped_rewards = compute_advantages(rewards, args.num_generations)
    policy_loss = compute_policy_loss(per_token_logp, old_per_token_logs, ref_per_token_logp, advantages, completion_mask, token_counts, args.loss_type, args.beta, epsilon=args.epsilon, epsilon_high=args.epsilon_high,)
    loss = (policy_loss + aux_loss) / args.accumulation_steps
    loss.backward()
    return {
        'loss': loss,
        'rewards': rewards,
        'grouped_rewards': grouped_rewards,
        'advantages': advantages,
        'token_counts': token_counts,
        'completion_mask': completion_mask,
        'per_token_logps': per_token_logp,
        'ref_per_token_logps': ref_per_token_logp,
        # 下面仅给 debug 日志用
        'contexts': contexts,
        'input_ids': input_ids,
        'prompt_lens': prompt_lens,
        'seq_lens': seq_len,
        'messages_batch': messages_batch,
        'gt_batch': gt_batch,
    }
def rl_train_epoch(ctx, epoch, loader, iters, start_step=0):
    """一个 epoch 的训练循环,只负责编排,细节都在 train_one_step / 日志函数里。"""
    last_step = start_step
    args = ctx.args
    for step, batch in enumerate(loader, start_step + 1):
        last_step = step
        out = train_one_step(ctx, batch)
        if step % args.accumulation_steps == 0:
            optimizer_step(args, ctx.model, ctx.optimizer, ctx.scheduler, ctx.rollout_engine, step)
        if step % args.save_interval == 0 or step == iters:
            save_checkpoint(args, ctx.lm_config, ctx.model, ctx.optimizer, ctx.scheduler, epoch, step, ctx.wandb, tokenizer=ctx.tokenizer)
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            log_debug_batch(args, step, out["messages_batch"], out["contexts"], input_ids=out["input_ids"], prompt_lens=out["prompt_lens"], seq_lens=out["seq_lens"], rewards=out["rewards"], gt_batch=out["gt_batch"], tokenizer=ctx.tokenizer)
        if step % args.log_interval == 0 or step == iters:
            log_step_metrics(epoch, step, iters, args, out["loss"], out["rewards"], out["token_counts"], out["ref_per_token_logps"], out["per_token_logps"], out["completion_mask"], out["grouped_rewards"], out["advantages"], ctx.optimizer, ctx.wandb)
    if last_step % args.accumulation_steps != 0 and last_step > start_step:
        optimizer_step(args, ctx.model, ctx.optimizer, ctx.scheduler, ctx.rollout_engine, step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Agent RL")
    parser.add_argument("--save_dir", type=str, default=None, help="模型保存目录（默认 <项目根>/out）")
    parser.add_argument('--save_weight', default='agent', type=str, help="保存权重名称")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型 bfloat16/float16")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="模型隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="模型层数")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="最大序列长度")
    parser.add_argument("--max_gen_len", type=int, default=768, help="单次最大生成长度")
    parser.add_argument("--max_total_len", type=int, default=2500, help="训练侧最终总长度上界")
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/miniMind/dataset/agent_rl.jsonl", help="训练数据路径")
    parser.add_argument("--num_generations", type=int, default=4, help="每个prompt生成数量")
    parser.add_argument("--beta", type=float, default=0.1, help="KL散度惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="加载预训练权重名称")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否从checkpoint恢复")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Agent-RL", help="wandb项目名称")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile")
    parser.add_argument("--debug_mode", action="store_true", help="调试模式")
    parser.add_argument("--debug_interval", type=int, default=20, help="调试日志间隔")
    parser.add_argument("--thinking_ratio", type=float, default=0.1, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--reward_model_path", type=str, default="/root/autodl-tmp/internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="/root/autodl-tmp/miniMind/model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="/root/autodl-tmp/miniMind/trainer/sglang_ckpt_agent", help="SGLang共享存储路径")
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
        run_name = f"MiniMind AGENT epochs:{args.epochs} bs:{args.batch_size} G:{args.num_generations} lr:{args.learning_rate} moe:{args.use_moe}"
        resume = "must" if args.from_resume == 1 else None
        wandb.init(project=args.wandb_project, experiment_name=run_name, id=wandb_id, resume=resume, config=lm_config.to_dict())
    # ========== 5. 初始化模型（GRPO 无 Critic）==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval().requires_grad_(False)
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=dtype)
    Logger(f'Loaded reward model from {args.reward_model_path}')
    roll_out = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path
    )
    ds = AgentRLDataset(args.data_path, tokenizer, max_length=args.max_total_len)
    sampler = DistributedSampler(ds) if dist.is_initialized() else None
    def collate_fn(batch): return {"messages": [b["messages"] for b in batch], "gt": [b["gt"] for b in batch], "tools": [b["tools"] for b in batch]}
    loader_for_count = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,collate_fn=collate_fn)
    iters = len(loader_for_count)
    total_optimizer_steps = (
        iters * args.epochs
        // args.accumulation_steps
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, total_optimizer_steps, eta_min=args.learning_rate / 10)
    start_epoch, start_step = 0, 0
    if ckp_data is not None:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scheduler.load_state_dict(ckp_data["scheduler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data["step"]
        Logger(f"成功从 checkpoint 恢复: Epoch {start_epoch}, Step {start_step}")
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile:
        model = torch.compile(model)

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    if is_main_process(): roll_out.update_policy(model)
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        sampler and sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(sampler or indices, args.batch_size, skip_batches=skip)
        loader = DataLoader(ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        Logger(f"{epoch + 1}/{args.epochs}  跳过前{skip}个batch, 从{skip}开始")
        ctx = TrainCtx(args=args, 
                       lm_config=lm_config, 
                       tokenizer=tokenizer, 
                       model=model, 
                       ref_model=ref_model,
                       reward_model=reward_model,
                       rollout_engine=roll_out,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       autocast_ctx=autocast_ctx,
                       wandb=wandb
                       )
        rl_train_epoch(ctx, epoch, loader, iters, skip)
        
    if dist.is_initialized(): dist.destroy_process_group()


    






    





    


















    