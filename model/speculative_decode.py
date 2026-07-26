"""基于链式 MTP 的自投机解码（self-speculative decoding）。

用模型自带的链式 MTP 模块（见 model_minimind.py 的 MTPModule）廉价地"起草"接下来
几个 token，再用主模型一次前向"验证"，从而用一次较贵的主模型前向换回多个 token，
以此摊薄 Engram 等模块给单次前向带来的额外开销。

限制：
- 只实现了贪心验证（等价于 do_sample=False），没有做保持采样分布的 rejection sampling，
  如果需要随机采样，请直接使用标准的 model.generate()。
- 仅支持 batch_size=1（教学/单会话演示场景），不支持批量并发请求共用一次调用。
"""
from typing import Optional, Iterator, Tuple

import torch


def _truncate_kv_cache(past_key_values, drop_count: int):
    if drop_count <= 0:
        return past_key_values
    truncated = []
    for kv in past_key_values:
        if kv is None:
            truncated.append(None)
            continue
        k, v = kv
        truncated.append((k[:, :k.shape[1] - drop_count, :, :], v[:, :v.shape[1] - drop_count, :, :]))
    return truncated


@torch.no_grad()
def mtp_speculative_generate_stream(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> Iterator[int]:
    if not (model.config.use_mtp and model.mtp_modules is not None):
        raise ValueError("mtp_speculative_generate_stream 需要 config.use_mtp=True 且模型带 mtp_modules")
    assert input_ids.shape[0] == 1, "自投机解码当前实现仅支持 batch_size=1"

    model.eval()
    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id
    if model.model.engram is not None:
        model.model.engram.reset_state()

    prompt_len = input_ids.shape[1]
    generated = input_ids

    # ---- 首次前向：喂入完整 prompt，建立 KV cache ----
    out = model(input_ids=generated, use_cache=True, past_key_values=None)
    past_key_values = out.past_key_values
    last_hidden = out.hidden_states[:, -1:, :]
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    while generated.shape[1] - prompt_len < max_new_tokens:
        remaining = max_new_tokens - (generated.shape[1] - prompt_len)
        cur_pos = generated.shape[1]

        # ---- 起草：链式 MTP 依次预测接下来最多 mtp_depth 个 token ----
        # 注意：每个深度用于内部 attention 的 RoPE 位置都固定在锚点位置 cur_pos，
        # 与训练时 compute_mtp_loss 里"每个深度用同一批绝对位置、只有预测目标偏移"的约定保持一致
        cis = (model.model.freqs_cos[cur_pos:cur_pos + 1], model.model.freqs_sin[cur_pos:cur_pos + 1])
        draft_tokens = [next_token]
        prev_hidden, prev_token = last_hidden, next_token
        for module in model.mtp_modules:
            if len(draft_tokens) >= remaining:
                break
            token_embed = model.model.embd_tokens(prev_token)
            h_d = module(prev_hidden, token_embed, cis)
            logits_d = model.lm_head(h_d)
            draft_token = logits_d[:, -1, :].argmax(dim=-1, keepdim=True)
            draft_tokens.append(draft_token)
            prev_hidden, prev_token = h_d, draft_token
            if eos_token_id is not None and draft_token.item() == eos_token_id:
                break
        candidate = torch.cat(draft_tokens, dim=1)  # 1, 1+k

        # ---- 验证：主模型对 candidate 做一次批量前向 ----
        verify_out = model(input_ids=candidate, use_cache=True, past_key_values=past_key_values)
        verify_logits = verify_out.logits
        verify_pred = verify_logits.argmax(dim=-1)  # 1, 1+k；位置 i 预测的是 candidate[i+1]

        # candidate[0] 就是主模型自己上一轮的贪心预测，天然通过验证；
        # 从 candidate[1] 起，需要 verify_pred[i-1] == candidate[i] 才算验证通过
        accept_len = 1
        for i in range(1, candidate.shape[1]):
            if verify_pred[0, i - 1].item() == candidate[0, i].item():
                accept_len += 1
            else:
                break
        accept_len = min(accept_len, remaining)

        accepted = candidate[:, :accept_len]
        drop_count = candidate.shape[1] - accept_len
        past_key_values = _truncate_kv_cache(verify_out.past_key_values, drop_count)
        last_hidden = verify_out.hidden_states[:, accept_len - 1:accept_len, :]

        if model.model.engram is not None:
            # 草稿里被拒绝的 token 不会真正进入序列，Engram 的滑窗历史需要用"最终确认"的
            # token 重新对齐，不能依赖 verify 前向里基于完整 candidate（含被拒绝部分）算出的历史
            pad_len = model.model.engram.ngram_order - 1
            if pad_len > 0:
                confirmed = torch.cat([generated, accepted], dim=1)
                model.model.engram._decode_history = confirmed[:, -pad_len:].clone()

        stop = False
        for i in range(accept_len):
            tok = accepted[0, i].item()
            yield tok
            if eos_token_id is not None and tok == eos_token_id:
                stop = True
                break
        generated = torch.cat([generated, accepted], dim=1)
        if stop:
            return

        next_token = verify_out.logits[:, accept_len - 1, :].argmax(dim=-1, keepdim=True)


@torch.no_grad()
def mtp_speculative_generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, dict]:
    tokens = list(mtp_speculative_generate_stream(model, input_ids, max_new_tokens, eos_token_id))
    new_ids = torch.tensor([tokens], dtype=input_ids.dtype, device=input_ids.device) if tokens else input_ids[:, :0]
    generated = torch.cat([input_ids, new_ids], dim=1)
    return generated, {"new_tokens": len(tokens)}
