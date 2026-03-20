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
            **kwargs
    ):
        
        super().__init__(**kwargs)
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
    #  base_theta -> dim // 2,
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
            low, high  = math.max(math.floor(inv_dim(beta_slow)), 0), math.min(math.ceil(inv_dim(beta_fast)), dim//2-1)
            ramp = torch.clamp((torch.arange(dim//2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
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
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            # training
            output = F.scaled_dot_product_attention(qx, kx, vx, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = qx @ kx.transpose(-1, -2) / math.sqrt(self.head_dim) # bsz, num_heads, seq_len, seq_len + ?
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)
            if attention_mask is not None:
                # attn_mask -> bsz, seq_len + ?
                extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                scores = scores + (1.0 - extended_mask) * (-1e9)
            scores = F.softmax(scores.float(), dim=-1).type_as(x)
            scores = self.attn_dropout(scores)
            output = scores @ vx # output -> bsz, num_heads, seq_len, head_dim
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # output -> bsz, seq_len, hidden_dim
        # past_kv -> k: bsz, seq_len + ?, self.n_local_kv_heads, self.head_dim
        #            v: bsz, seq_len + ?, self.n_local_kv_heads, self.head_dim
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
        print(scores.mean(0))
        print(torch.bincount(topk_idx.view(-1)))
        # topk_weight, topk_idx -> bsz*seq_len topk
        if self.norm_topk_prob and self.topk > 1:
            topk_weight = topk_weight / (topk_weight.sum(-1, keepdim=True) + 1e-20)
        if self.training and self.alpha > 0.0:
            topk_idx_for_aux = topk_idx.view(bsz, -1) # bsz seq_len*topk
            if self.seq_aux:
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux, torch.ones(bsz, seq_len*self.topk, device=hidden_states.device)).div_(seq_len*self.topk/self.n_routed_experts)
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
        counts = torch.bincount(flat_expert_indices, minlength=len(self.experts))
        tokens_per_expert = counts.cpu().numpy().cumsum(0)
        
        # 3. 映射回原始 token 索引
        num_tokens = x.shape[0]
        topk = self.config.num_experts_per_tokens
        # 建立映射：每个推理位置对应原始输入 x 的哪个 token
        orig_token_map = torch.arange(num_tokens, device=x.device).repeat_interleave(topk)
        token_ids = orig_token_map[idxs]

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
    


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.embd_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList([MiniMindBlock(i, config) for i in range(config.num_hidden_layers)])
        
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

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重共享
        self.model.embd_tokens.weight = self.lm_head.weight

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
        
        # 记录原始的 aux_loss 方便日志打点
        output.aux_loss = aux_loss
        
        return output

        
        


        
        




    




        


    

