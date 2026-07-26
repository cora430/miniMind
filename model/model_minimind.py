from textwrap import indent

import torch
import torch.nn as nn
from transformers import GenerationMixin, PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN
from typing import Optional, Tuple, Dict, List, Union
import math
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(
            self,
            vocab_size: int = 6400,
            max_position_embeddings: int = 32768,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            dropout: float = 0.0,
            hidden_act: str = "silu",
            hidden_size: int = 512,
            intermediate_size: int = None,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            rms_norm_eps: float = 1e-5,
            rope_theta: int = 1000000,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            # moe
            use_moe:bool = False,
            num_experts_per_tokens: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = "softmax",
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            # gated attention (Qwen3-Next 风格注意力输出门控)
            use_gated_attention: bool = False,
            # engram（n-gram 哈希记忆增强）
            use_engram: bool = False,
            engram_ngram_order: int = 2,
            engram_num_buckets: int = 16384,
            engram_embed_dim: int = 64,
            # 链式 MTP（DeepSeek-V3 风格多 token 预测）
            use_mtp: bool = False,
            mtp_depth: int = 1,
            mtp_loss_weight: float = 0.3,
            **kwargs
    ):
        
        super().__init__(**kwargs)
        self.auto_map = {
            "AutoConfig": "model.model_minimind.MiniMindConfig",
            "AutoModelForCausalLM": "model.model_minimind.MiniMindForCausalLM"
        }
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.rope_scaling = {
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn",
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        self.use_moe = use_moe
        self.num_experts_per_tokens = num_experts_per_tokens
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.use_gated_attention = use_gated_attention
        self.use_engram = use_engram
        self.engram_ngram_order = engram_ngram_order
        self.engram_num_buckets = engram_num_buckets
        self.engram_embed_dim = engram_embed_dim
        self.use_mtp = use_mtp
        self.mtp_depth = mtp_depth
        self.mtp_loss_weight = mtp_loss_weight

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x: torch.Tensor):
        rsqrt = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps) # ... dim
        return (rsqrt * self.weight).type_as(x)

def precompute_freqs_cis(dim: int, end: int = 16*2048, rope_base: float = 1e6, rope_scaling: Optional[dict] = None):
    freqs, attention_factor = 1 / (rope_base ** (torch.arange(0, dim, 2).float() / dim)), 1.0
    # base_theta -> dim // 2,
    # freqs 是每个token转多少弧度
    if rope_scaling is not None:
        beta_fast, beta_slow, factor, original_max_position_embeddings, attention_factor = (
            rope_scaling.get("beta_fast"),
            rope_scaling.get("beta_slow"),
            rope_scaling.get("factor"),
            rope_scaling.get("original_max_position_embeddings"),
            rope_scaling.get("attention_factor"),
        )
        if end > original_max_position_embeddings:
            inv_dim = lambda b: (dim*math.log(original_max_position_embeddings / (b*2*math.pi))) / (2*math.log(rope_base))
            low, high  = max(math.floor(inv_dim(beta_slow)), 0), min(math.ceil(inv_dim(beta_fast)), dim//2-1)
            ramp = torch.clamp((torch.arange(dim//2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            # ramp 决定 多少是慢速（freqs/factor）,多少是原速
            # 因为i小的就是低维的他的freqs大，大意味着整个圆都转过了，没有盲区
            # i大的就是高维的他的freqs小，而小就意味着训练时候可能只转了很小一部分，还有很大的盲区没有转过，当推理的时候上下文一长，角度一大他就容易到盲区
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float() # end, dim//2
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attention_factor # end, dim
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attention_factor # end, dim
    return freqs_cos, freqs_sin 

def apply_rotary_embeddings(q, k, cos, sin, unsqueeze_dim = 1):
    # q,k -> bsz, seqlen, num_heads, head_dim
    # cos, sin -> seq_len, head_dim  
    def rotate_half(x):
        return torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)
    # seq_len, 1, head_dim 右对齐自动变成 1, seq_len, 1, head_dim
    q_embd = q * (cos.unsqueeze(unsqueeze_dim)) + rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    k_embd = k * (cos.unsqueeze(unsqueeze_dim)) + rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    return q_embd, k_embd # bsz, seq_len, num_heads, head_dim

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bsz, seq_len, num_key_value_heads, head_dim = x.shape
    # bsz, seq_len, num_key_value_heads*n_rep, head_dim
    return x[:, :, :, None, :].expand(bsz, seq_len, num_key_value_heads, n_rep, head_dim).reshape(bsz, seq_len, num_key_value_heads*n_rep, head_dim)

class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = args.num_key_value_heads if args.num_key_value_heads is not None else self.n_local_heads
        assert self.n_local_heads % self.n_local_kv_heads == 0
        self.dropout = args.dropout
        self.head_dim = args.hidden_size // self.n_local_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.q_proj = nn.Linear(args.hidden_size, self.head_dim * self.n_local_heads,bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.head_dim * self.n_local_kv_heads, bias = False)
        self.v_proj = nn.Linear(args.hidden_size, self.head_dim * self.n_local_kv_heads, bias = False)
        self.o_proj = nn.Linear(self.head_dim * self.n_local_heads, args.hidden_size, bias = False)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        self.use_gated_attention = args.use_gated_attention
        if self.use_gated_attention:
            self.gate_proj = nn.Linear(args.hidden_size, self.head_dim * self.n_local_heads, bias=True)
            # 门控初始化为近似恒等（sigmoid(3)≈0.95），避免续训旧 checkpoint 时行为被剧烈打乱
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, 3.0)

        self.flash = args.flash_attn and hasattr(torch.nn.functional, "scaled_dot_product_attention")
    def forward(self, 
                x: torch.Tensor, 
                use_cache: bool = False, 
                past_key_value: Optional[tuple[torch.Tensor, torch.tensor]] = None,
                cis: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                attention_mask: Optional[torch.tensor] = None, 
    ):
        # x -> bsz, seq_len, hidden_size
        bsz, seq_len, hidden_size = x.shape
        qx, kx, vx = self.q_proj(x), self.k_proj(x), self.v_proj(x) 
        qx = qx.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        kx = kx.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        vx = vx.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        qx, kx = apply_rotary_embeddings(qx, kx, cis[0], cis[1])
        # kv cache
        if past_key_value is not None:
            kx = torch.cat([past_key_value[0], kx], dim = 1)
            vx = torch.cat([past_key_value[1], vx], dim = 1)
        past_kv = (kx, vx) if use_cache else None
        qx = qx.transpose(1, 2).contiguous()# bsz, num_heads, seq_len , head_dim
        kx = repeat_kv(kx, self.n_rep).transpose(1, 2).contiguous() # bsz, num_heads, seq_len + ?, head_dim
        vx = repeat_kv(vx, self.n_rep).transpose(1, 2).contiguous() # bsz, num_heads, seq_len + ?, head_dim
        if self.flash and seq_len > 1 and past_key_value is None and (attention_mask is None or torch.all(attention_mask == 1)):
            # training / 无 cache 的整段 prefill：is_causal=True 在 query/key 等长时才是对的因果对齐，
            # 一旦带着 KV cache 再喂入多个新 token（如投机解码的批量验证），
            # SDPA 的 is_causal 不会自动做 bottom-right 对齐，必须退回下面手写的因果 mask 分支
            output = F.scaled_dot_product_attention(qx, kx, vx, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = qx @ kx.transpose(-1, -2) / math.sqrt(self.head_dim) # bsz, num_heads, seq_len, seq_len + ?
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)
            if attention_mask is not None:
                # attn_mask -> bsz, seq_len + ?
                extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # 有了attention_mask算分数的时候就会几乎为0
                scores = scores + (1.0 - extended_mask) * (-1e9)
            scores = F.softmax(scores.float(), dim=-1).type_as(x)
            scores = self.attn_dropout(scores)
            output = scores @ vx # output -> bsz, num_heads, seq_len, head_dim
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # output -> bsz, seq_len, hidden_dim
        # past_kv -> k: bsz, seq_len + ?, self.n_local_kv_heads, self.head_dim
        #            v: bsz, seq_len + ?, self.n_local_kv_heads, self.head_dim
        if self.use_gated_attention:
            # 门控信号来自本层输入 x（而非 attention 输出本身），避免自环
            output = output * torch.sigmoid(self.gate_proj(x))
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
        
class FeedForward(nn.Module):
    def __init__(self, config:MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size  = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 63) // 64)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x: torch.Tensor, ):
        # bsz, seq_len, hidden_size
        return self.dropout(self.down_proj(self.up_proj(x) * self.act_fn(self.gate_proj(x))))
class MoEGate(nn.Module):
    def __init__(self, config:MiniMindConfig):
        super().__init__()
        self.topk =config.num_experts_per_tokens
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.norm_topk_prob = config.norm_topk_prob
        self.n_routed_experts = config.n_routed_experts
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.n_routed_experts, self.gating_dim),)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    def forward(self, hidden_states: torch.Tensor,):
        # x -> bsz, seq_len, hidden_dim
        bsz, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(bsz * seq_len, hidden_dim)
        logits = F.linear(hidden_states, weight=self.weight, bias=None) # bsz*seq_len n_routed_experts
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f"insupportable scoring function for MoE gating:{self.scoring_func}")
        topk_weight, topk_idx = torch.topk(scores, k=self.topk, dim=-1, sorted=False)
        # print(scores.mean(0))
        # print(torch.bincount(topk_idx.view(-1)))
        # topk_weight, topk_idx -> bsz*seq_len topk
        if self.norm_topk_prob and self.topk > 1:
            topk_weight = topk_weight / (topk_weight.sum(-1, keepdim=True) + 1e-20)
        if self.training and self.alpha > 0.0:
            topk_idx_for_aux = topk_idx.view(bsz, -1) # bsz seq_len*topk
            if self.seq_aux:
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                # src 和 index 的维度是一样的，要把 src[0][0] 里的 1 加到 self[0][index[0][0]]，那这样 self 里的就是每个句子每个专家的启用数量
                ce.scatter_add_(dim=1, index=topk_idx_for_aux, src=torch.ones(bsz, seq_len*self.topk, device=hidden_states.device)).div_(seq_len*self.topk/self.n_routed_experts)
                aux_loss = (ce * scores.reshape(bsz, seq_len, self.n_routed_experts).mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux.view(-1), num_classes=self.n_routed_experts,)
                ce = mask_ce.float().mean(dim=0) # n_routed_experts,
                Pi = scores.mean(dim=0) # n_routed_experts,
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_weight, topk_idx, aux_loss
class MoEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [FeedForward(config=config) for _ in range(config.n_routed_experts)]
        )
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config=config) for _ in range(config.n_shared_experts)]
            )
        self.gate = MoEGate(config)
    def forward(self, x: torch.Tensor):
        # x -> bsz, seq_len, hidden_dim
        identity = x
        orig_shape = x.shape
        bsz, seq_len, hidden_dim = x.shape
        x = x.reshape(-1, hidden_dim) # bsz*seq_len, hidden_dim
        topk_weight, topk_idx, aux_loss = self.gate(identity)
        # topk_weight, topk_idx -> bsz*seq_len topk
        topk = self.config.num_experts_per_tokens
        flat_topk_idx = topk_idx.view(-1)# bsz*seq_len*topk
        if self.training:
            x = x.repeat_interleave(topk, dim=0)# bsz*seq_len*topk, hidden_dim
            y = torch.empty_like(x, device=x.device, dtype=x.dtype)
            
            for i, expert in enumerate(self.experts):  
                mask = flat_topk_idx == i # bsz*seq_len*topk
                if mask.sum() > 0:
                    input_x = x[mask]
                    output_x = expert(input_x) # 该专家要处理的tokens数量，hidden_dim
                    y[mask] = output_x.to(x.dtype)
                else:
                    y[mask] = expert(x.new_zeros(0, hidden_dim)).to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            y = (y.reshape(bsz*seq_len, topk, hidden_dim) * topk_weight.unsqueeze(-1)).sum(1)
        else:
            
            y = self.moe_infer(identity.view(-1, hidden_dim), flat_topk_idx, topk_weight.view(-1, 1))
        y = y.view(orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y
        
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        优化的推理路径，减少内存使用和计算量
        
        参数:
        - x: 展平的输入，(num_tokens, hidden_size)
        - flat_expert_indices: 每个token选择的专家索引，(num_tokens*num_experts_per_tok,)
        - flat_expert_weights: 对应的权重，(num_tokens*num_experts_per_tok, 1)
        
        返回:
        - expert_cache: 加权后的专家输出，(num_tokens, hidden_size)
        """
        # 1. 对专家索引进行排序，将相同专家的 token 聚合在一起
        idxs = torch.argsort(flat_expert_indices) 
        
        # 2. 核心修复：强制 bincount 长度为专家总数，防止 IndexError
        # 这里bincount只能 input 为 1D，统计在所句子里每个专家的使用次数，minlength 保证输出长度为为专家总数
        # ce = torch.zeros(len(self.experts))
        # ce.scatter_add_(dim=0, index=flat_expert_indices, src=torch.ones_like(flat_expert_indices)).float()
        counts = torch.bincount(input=flat_expert_indices, minlength=len(self.experts))
        tokens_per_expert = counts.cpu().numpy().cumsum(0)
        
        # 3. 映射回原始 token 索引
        num_tokens = x.shape[0]
        topk = self.config.num_experts_per_tokens
        # 建立映射：每个推理位置对应原始输入 x 的哪个 token
        orig_token_map = torch.arange(num_tokens, device=x.device).repeat_interleave(topk)
        token_ids = orig_token_map[idxs] # 这里token_ids 是按照相同专家排序的token id

        experts_cache = torch.zeros_like(x, device=x.device)
        
        for i, expert in enumerate(self.experts):
            end = tokens_per_expert[i]
            start = tokens_per_expert[i-1] if i != 0 else 0
            
            # 如果该专家没有被选中，直接跳过，避免 start == end 导致切片为空
            if start == end:
                continue
                
            # 找到当前专家需要处理的原始 token 索引
            tokens_input_ids = token_ids[start:end]
            x_input = x[tokens_input_ids]
            
            # 专家前向计算
            expert_output = expert(x_input)
            
            # 乘以对应的 Gate 权重
            # idxs[start:end] 获取了这些 token 在展平后的权重数组中的原始位置
            expert_output.mul_(flat_expert_weights[idxs[start:end]])
            
            # 使用 scatter_add 将结果累加回缓存中对应的 token 位置
            experts_cache.scatter_add_(
                dim=0,
                index=tokens_input_ids.view(-1, 1).expand(-1, x.shape[-1]),
                src=expert_output
            )
            
        return experts_cache
  
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.attn = Attention(config)
        self.mlp = MoEFeedForward(config) if config.use_moe else FeedForward(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layer_id = layer_id
    def forward(self, 
                hidden_states,
                use_cache: bool = False, 
                past_key_value: Optional[tuple[torch.Tensor, torch.tensor]] = None,
                cis: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                attention_mask: Optional[torch.tensor] = None, ):
        residual = hidden_states
        attn_out, past_kv = self.attn(self.input_layernorm(hidden_states), use_cache, past_key_value, cis, attention_mask)
        hidden_states = attn_out + residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        # 记录当前层的 aux_loss
        #current_aux_loss = self.mlp.aux_loss if hasattr(self.mlp, 'aux_loss') else 0
        return hidden_states, past_kv


class EngramMemory(nn.Module):
    """n-gram 哈希记忆增强：用最近 ngram_order 个 token id 的固定滚动哈希查一张
    低秩 embedding 表，门控融合进 hidden state，给模型一条不经过 attention/FFN
    的"记忆"通路。哈希函数不可训练（Horner 法滚动哈希 mod 桶数），只有查表结果
    和融合门控是可训练参数。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.ngram_order = max(2, config.engram_ngram_order)
        self.num_buckets = config.engram_num_buckets
        self.hash_base = 1000003
        self.bucket_embed = nn.Embedding(self.num_buckets, config.engram_embed_dim)
        self.up_proj = nn.Linear(config.engram_embed_dim, config.hidden_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        # 门控初始化为接近关闭（sigmoid(-3)≈0.047），让模型从"几乎不用记忆"起步，
        # 训练中按需学着打开，避免一开始就把随机初始化的记忆表噪声注入主干
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -3.0)
        # 增量 decode（每步只喂 1 个新 token）时用来记住最近 ngram_order-1 个 token 的滑窗。
        # 注意：这是模块级可变状态，只对"单序列顺序生成"正确；每次全新 prefill
        # （cache 为空）会被 MiniMindModel.forward 调用 reset_state() 清空，
        # 但同一模型实例被用于并发/交织的多会话生成时会互相污染，需要调用方自行隔离实例。
        self._decode_history: Optional[torch.Tensor] = None

    def reset_state(self):
        self._decode_history = None

    def _hash_to_bucket(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        # ngram_ids -> ..., ngram_order（最旧的 token 在前），逐步取模避免溢出
        hashed = torch.zeros(ngram_ids.shape[:-1], dtype=torch.long, device=ngram_ids.device)
        for i in range(ngram_ids.shape[-1]):
            hashed = (hashed * self.hash_base + ngram_ids[..., i].long()) % self.num_buckets
        return hashed

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, use_cache: bool = False):
        bsz, seq_len = input_ids.shape
        pad_len = self.ngram_order - 1
        if use_cache and self._decode_history is not None:
            history = self._decode_history
        elif pad_len > 0:
            # 序列最开头没有真实历史，用当前块的第一个 token 自我填充（causal-safe 的近似）
            history = input_ids[:, :1].expand(bsz, pad_len)
        else:
            history = input_ids[:, :0]
        full_ids = torch.cat([history, input_ids], dim=1)  # bsz, pad_len+seq_len
        windows = full_ids.unfold(dimension=1, size=self.ngram_order, step=1)  # bsz, seq_len, ngram_order
        bucket_ids = self._hash_to_bucket(windows)
        embed = self.up_proj(self.bucket_embed(bucket_ids))
        gate = torch.sigmoid(self.gate_proj(hidden_states))
        if use_cache:
            self._decode_history = full_ids[:, -pad_len:].detach() if pad_len > 0 else full_ids[:, :0]
        return hidden_states + gate * embed


class MTPModule(nn.Module):
    """链式 MTP 的单个预测深度（DeepSeek-V3 风格）：把上一深度的 hidden state 和
    未来第 k 个 token 的真实 embedding 拼接、降维，过一个浅层 transformer block，
    输出喂给下一深度，实现"链式"多 token 预测。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.embed_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.block = MiniMindBlock(layer_id=-1, config=config)

    def forward(self, prev_hidden, next_token_embeds, cis, attention_mask=None):
        fused = self.proj(torch.cat([self.hidden_norm(prev_hidden), self.embed_norm(next_token_embeds)], dim=-1))
        hidden_states, _ = self.block(fused, use_cache=False, past_key_value=None, cis=cis, attention_mask=attention_mask)
        return hidden_states


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.embd_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList([MiniMindBlock(i, config) for i in range(config.num_hidden_layers)])
        self.engram = EngramMemory(config) if config.use_engram else None

        self.head_dim = config.hidden_size // config.num_attention_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.head_dim, config.max_position_embeddings, config.rope_theta, config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                past_key_values: Optional[List[tuple[torch.Tensor, torch.Tensor]]] = None,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs):
        batch_size, seq_len = input_ids.shape
        if hasattr(past_key_values, "layers"): past_key_values = None
        past_key_values = past_key_values or [None] * self.config.num_hidden_layers
        
        # 计算 RoPE 的起始位置
        start = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        cis = (
            self.freqs_cos[start : start + seq_len],
            self.freqs_sin[start : start + seq_len]
        )

        hidden_states = self.dropout(self.embd_tokens(input_ids))

        if self.engram is not None:
            if start == 0:
                # 新的 prefill（cache 为空）意味着一段新的生成/前向序列开始，清空滑窗历史
                self.engram.reset_state()
            hidden_states = self.engram(hidden_states, input_ids, use_cache=use_cache)

        presents = []
        # 初始化 aux_loss，确保其具有梯度追踪能力
        total_aux_loss = hidden_states.new_zeros(1).squeeze()
        
        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            # 假设你的 MiniMindBlock forward 已经修改为返回 (hidden_states, past_kv, l_aux)
            # 或者依然通过 layer.mlp.aux_loss 获取
            hidden_states, past_kv = layer(
                hidden_states, use_cache, past_key_value, cis, attention_mask
            )
            presents.append(past_kv)
            
            # 每一层如果使用了 MoE，累加其负载均衡损失
            if self.config.use_moe and hasattr(layer.mlp, 'aux_loss'):
                total_aux_loss = total_aux_loss + layer.mlp.aux_loss
                
        hidden_states = self.norm(hidden_states)
        
        return hidden_states, presents, total_aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        # past_key_values 用的是自定义的 list[tuple(k, v)] 格式，不是 transformers.Cache，
        # 禁止 generate() 自动注入 DynamicCache，否则首步会被 model_minimind.py 里的
        # `hasattr(past_key_values, "layers")` 兜底逻辑强制清空 cache。
        return False

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重共享
        self.model.embd_tokens.weight = self.lm_head.weight
        self.mtp_modules = (
            nn.ModuleList([MTPModule(self.config) for _ in range(self.config.mtp_depth)])
            if self.config.use_mtp else None
        )

    def compute_mtp_loss(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, labels: torch.Tensor):
        # 链式多 token 预测的训练侧辅助 loss：深度 d 的模块用位置 i 的 h^{d-1} 和
        # token_{i+d} 的真实 embedding 预测 token_{i+d+1}，链式指下一深度接着用上一深度的输出
        seq_len = input_ids.shape[1]
        cis = (self.model.freqs_cos[:seq_len], self.model.freqs_sin[:seq_len])
        prev_hidden = hidden_states
        total_loss = hidden_states.new_zeros(1).squeeze()
        computed = 0
        for depth, module in enumerate(self.mtp_modules, start=1):
            length = seq_len - depth - 1
            if length <= 0:
                break
            cur_prev_hidden = prev_hidden[:, :length, :]
            next_token_embeds = self.model.embd_tokens(input_ids[:, depth:depth + length])
            step_cis = (cis[0][:length], cis[1][:length])
            h_d = module(cur_prev_hidden, next_token_embeds, step_cis)
            logits_d = self.lm_head(h_d)
            target = labels[:, depth + 1:depth + 1 + length]
            loss_d = F.cross_entropy(
                logits_d.reshape(-1, self.config.vocab_size),
                target.reshape(-1),
                ignore_index=-100
            )
            total_loss = total_loss + loss_d
            prev_hidden = h_d
            computed += 1
        return total_loss / computed if computed > 0 else total_loss

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                past_key_values: Optional[List[tuple[torch.Tensor, torch.Tensor]]] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                logits_to_keep: Optional[Union[int, torch.Tensor]] = 0,
                **kwargs):
        
        # 1. 前向传播获取隐藏状态和 MoE 辅助损失
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids, use_cache, past_key_values, attention_mask, **kwargs
        )
        
        # 2. 计算 Logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        loss = None
        mtp_loss = hidden_states.new_zeros(1).squeeze()
        if labels is not None:
            # 3. 计算交叉熵损失 (Token Prediction Loss)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            logits_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

            # 4. 【核心修复】将 aux_loss 合并到主 loss 中，否则 aux_loss 不会产生梯度
            # 这样 Gate 权重才会根据负载均衡情况进行更新
            loss = logits_loss + aux_loss

            # 5. 链式 MTP 的辅助 loss（仅训练时计算），按权重合并进主 loss
            if self.config.use_mtp and self.mtp_modules is not None:
                mtp_loss = self.compute_mtp_loss(hidden_states, input_ids, labels)
                loss = loss + self.config.mtp_loss_weight * mtp_loss
        else:
            # 推理阶段不需要 loss
            loss = None

        # 5. 封装输出，确保包含合并后的 loss
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states
        )
        
        # 记录原始的 aux_loss / mtp_loss 方便日志打点
        output.aux_loss = aux_loss
        output.mtp_loss = mtp_loss

        return output

        
        


        
        




    




        


    

