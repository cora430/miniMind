from networkx import multi_source_dijkstra
from ngrok import forward
import torch
from torch import nn
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        # 矩阵A,B初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

        self.rank = rank
    def forward(self, x):
        return self.B(self.A(x))
def apply_lora(model, rank=8):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank).to(model.device)
            setattr(module, "lora", lora)
            origin_forward = module.forward
            def forward_with_lora(x, layer1=origin_forward, layer2=lora):
                return layer1(x) + layer2(x)
            module.forward = forward_with_lora
def load_lora(model, path):
    # 1. 加载权重到模型所在设备
    state_dict = torch.load(path, map_location=next(model.parameters()).device)
    
    # 2. 移除 DDP 包装产生的 "module." 前缀
    state_dict = {(k[7:] if k.startswith("module.") else k) : v for k, v in state_dict.items()}
    
    # 3. 遍历模型模块进行精准匹配
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            # 这里的改进点：使用 strip('.') 彻底清除 key 前后多余的点
            # 并且增加对 key 的校验，防止误匹配
            target_prefix = f"{name}.lora"
            
            lora_state = {}
            for k, v in state_dict.items():
                if k.startswith(target_prefix):
                    # 替换前缀，并去除可能残留在开头的 "."
                    new_key = k.replace(target_prefix, "").lstrip('.')
                    lora_state[new_key] = v
            
            # 4. 加载到子模块中
            if lora_state:
                module.lora.load_state_dict(lora_state, strict=True)
                
    print(f"✅ LoRA 权重已成功从 {path} 加载")
def save_lora(model, path):
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        clean_name = name[7:] if name.startswith("module.") else name
        if hasattr(module, "lora"):
            lora_state = {f"{clean_name}.lora.{k}": v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


