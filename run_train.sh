#!/bin/bash
# ==============================================================
# MiniMind-3 完整训练脚本（hidden_size=768, ~64M 参数）
# 运行方式: bash run_train.sh
# 建议后台挂起: nohup bash run_train.sh > train.log 2>&1 &
#
# 可选开关（在此处修改或通过环境变量覆盖）:
#   USE_MOE=1      启用 Mixture-of-Experts
#   USE_COMPILE=1  启用 torch.compile 加速
#   USE_TMUX=1     在 tmux session（minimind-train）中运行
# ==============================================================
set -euo pipefail

: "${USE_MOE:=0}"
: "${USE_COMPILE:=0}"
: "${USE_TMUX:=0}"

# ----------------------------------------------------------
# tmux: 若未在 tmux 内且 USE_TMUX=1，则自动重入 tmux session
# ----------------------------------------------------------
if [[ "$USE_TMUX" == "1" && -z "${TMUX:-}" ]]; then
    SESSION="minimind-train"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Session '$SESSION' 已存在，请先 tmux kill-session -t $SESSION 或手动 attach"
        exit 1
    fi
    echo "以 tmux session '$SESSION' 启动训练..."
    exec tmux new-session -s "$SESSION" \
        "USE_MOE=$USE_MOE USE_COMPILE=$USE_COMPILE bash \"${BASH_SOURCE[0]}\"; \
         echo '训练结束，按任意键退出'; read -n1"
fi

# ----------------------------------------------------------
# 根据开关构建附加参数
# ----------------------------------------------------------
EXTRA_ARGS=""
[[ "$USE_MOE"      == "1" ]] && EXTRA_ARGS="$EXTRA_ARGS --use_moe"
[[ "$USE_COMPILE"  == "1" ]] && EXTRA_ARGS="$EXTRA_ARGS --use_compile"

echo "USE_MOE     : $USE_MOE"
echo "USE_COMPILE : $USE_COMPILE"
echo "USE_TMUX    : $USE_TMUX"
echo "EXTRA_ARGS  : ${EXTRA_ARGS:-(none)}"
echo ""

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REWARD_MODEL_DIR="$PROJECT_DIR/../internlm2-1_8b-reward"
DATASET_DIR="$PROJECT_DIR/dataset"
OUT_DIR="$PROJECT_DIR/out"

mkdir -p "$OUT_DIR" "$DATASET_DIR"

echo "PROJECT_DIR  : $PROJECT_DIR"
echo "REWARD_MODEL : $REWARD_MODEL_DIR"
echo "DATASET_DIR  : $DATASET_DIR"
echo "OUT_DIR      : $OUT_DIR"
echo ""

# ==============================================================
# Step 1: 安装 Python 依赖
# ==============================================================
echo "========== [1/5] 安装 Python 依赖 =========="
# 查当前 CUDA 版本，自动选 PyTorch wheel
CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/' || echo "12.1")
echo "检测到 CUDA 版本: $CUDA_VER"

CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
if [[ "$CUDA_MAJOR" -ge 12 && "$CUDA_MINOR" -ge 8 ]]; then
    TORCH_URL="https://download.pytorch.org/whl/cu128"
elif [[ "$CUDA_MAJOR" -ge 12 && "$CUDA_MINOR" -ge 4 ]]; then
    TORCH_URL="https://download.pytorch.org/whl/cu124"
elif [[ "$CUDA_MAJOR" -ge 12 ]]; then
    TORCH_URL="https://download.pytorch.org/whl/cu121"
else
    TORCH_URL="https://download.pytorch.org/whl/cu118"
fi

# 镜像已预装 PyTorch，跳过重复安装可节省时间；若需强制安装取消注释
# pip install torch==2.7.0 torchvision torchaudio --index-url "$TORCH_URL" -q
echo "PyTorch wheel URL（如需重装）: $TORCH_URL"
pip install modelscope -q
pip install -r "$PROJECT_DIR/requirements.txt" -q
echo "依赖安装完成"
echo ""

# ==============================================================
# Step 2: 下载奖励模型（与 miniMind 同级目录）
# ==============================================================
echo "========== [2/5] 下载奖励模型 =========="
if [ ! -d "$REWARD_MODEL_DIR" ]; then
    git clone https://www.modelscope.cn/Shanghai_AI_Laboratory/internlm2-1_8b-reward.git \
        "$REWARD_MODEL_DIR"
    echo "奖励模型下载完成: $REWARD_MODEL_DIR"
else
    echo "奖励模型已存在，跳过: $REWARD_MODEL_DIR"
fi
echo ""

# ==============================================================
# Step 3: 下载数据集（仅下载训练需要的 4 个文件）
# ==============================================================
echo "========== [3/5] 下载数据集 =========="
python3 - <<PYEOF
from modelscope.hub.snapshot_download import snapshot_download
import os

local_dir = os.environ.get("DATASET_DIR", "./dataset")
snapshot_download(
    "gongjy/minimind_dataset",
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=[
        "pretrain_t2t_mini.jsonl",  # Pretrain 用 (1.24GB)
        "sft_t2t_mini.jsonl",       # Full SFT 用 (1.74GB)
        "dpo.jsonl",                # DPO 用
        "rlaif.jsonl",              # GRPO 用
    ]
)
print("数据集下载完成")
PYEOF
export DATASET_DIR
echo ""

# ==============================================================
# 进入项目根目录，后续所有训练相对路径均基于此
# ==============================================================
cd "$PROJECT_DIR"

# ==============================================================
# Step 4: Pretrain（预训练）
# 数据: pretrain_t2t_mini ~0.25B tokens × 5 epochs
# 预计耗时: ~3h
# 输出: ./out/pretrain_768.pth
# ==============================================================
echo "========== [4/5] Pretrain =========="
python -m trainer.train_pretrain \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --epochs 5 \
    --batch_size 64 \
    --learning_rate 5e-4 \
    --accumulation_steps 4 \
    --max_seq_len 512 \
    --data_path ./dataset/pretrain_t2t_mini.jsonl \
    --save_dir ./out \
    --save_weight pretrain \
    --save_interval 2000 \
    --log_interval 200 \
    --dtype bfloat16
echo "Pretrain 完成"
echo ""

# ==============================================================
# Step 5: Full SFT（全参监督微调）
# 数据: sft_t2t_mini ~0.26B tokens × 5 epochs（含 Tool Call 数据）
# 预计耗时: ~4h
# 输出: ./out/full_sft_768.pth
# ==============================================================
echo "========== Full SFT =========="
python -m trainer.train_full_sft \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --epochs 5 \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --accumulation_steps 2 \
    --max_seq_len 768 \
    --data_path ./dataset/sft_t2t_mini.jsonl \
    --from_weight pretrain \
    --save_dir ./out \
    --save_weight full_sft \
    --save_interval 1000 \
    --log_interval 100 \
    --dtype bfloat16
echo "Full SFT 完成"
echo ""

# ==============================================================
# Step 6: DPO（偏好对齐）
# 预计耗时: ~2h
# 输出: ./out/dpo_768.pth
# ==============================================================
echo "========== DPO =========="
python -m trainer.train_dpo \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --epochs 2 \
    --batch_size 8 \
    --learning_rate 5e-8 \
    --max_seq_len 1024 \
    --data_path ./dataset/dpo.jsonl \
    --from_weight full_sft \
    --save_dir ./out \
    --save_weight dpo \
    --save_interval 200 \
    --log_interval 50 \
    --beta 0.15 \
    --dtype bfloat16
echo "DPO 完成"
echo ""

# ==============================================================
# Step 7: GRPO（强化学习）
# 预计耗时: ~25h（剩余时间全给 RL，收益最大）
# 输出: ./out/grpo_actor_768.pth
# 从 full_sft 出发（非 dpo），RL 对齐效果更稳定
# ==============================================================
echo "========== GRPO =========="
python -m trainer.train_grpo \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --epochs 4 \
    --batch_size 4 \
    --num_generations 8 \
    --learning_rate 3e-7 \
    --max_seq_len 512 \
    --max_gen_len 512 \
    --grpo_update_iters 2 \
    --kl_coef 0.1 \
    --clip_epsilon 0.2 \
    --data_path ./dataset/rlaif.jsonl \
    --from_weight full_sft \
    --save_dir ./out \
    --save_weight grpo_actor \
    --reward_model_path "$REWARD_MODEL_DIR" \
    --rollout_engine torch \
    --save_interval 20 \
    --log_interval 5 \
    --dtype bfloat16
echo "GRPO 完成"
echo ""

echo "========== 全部训练完成 =========="
echo "最终权重位于 $OUT_DIR"
