#!/bin/bash
# ==============================================================
# MiniMind-3 Pretrain + Full SFT 训练脚本（hidden_size=768, ~64M 参数）
# 运行方式: bash run_train.sh
# 建议后台挂起: nohup bash run_train.sh > train.log 2>&1 &
#
# 可选开关（在此处修改或通过环境变量覆盖）:
#   USE_MOE=1              启用 Mixture-of-Experts（显存开销大，6GB 卡不建议开）
#   USE_GATED_ATTENTION=1  启用门控注意力（Qwen3-Next风格，+4.7M参数，默认开）
#   USE_ENGRAM=1           启用Engram记忆增强（+1.7M参数，默认开）
#   USE_MTP=1              启用链式MTP辅助训练（多一层的前反向开销，默认关）
#   USE_COMPILE=1          启用 torch.compile 加速
#   USE_TMUX=1             在 tmux session（minimind-train）中运行
#   PRETRAIN_BATCH_SIZE    Pretrain 单步 batch size（默认 16，按 6GB 显存卡调低）
#   PRETRAIN_ACCUM_STEPS   Pretrain 梯度累积步数（默认 16，等效 batch=256）
#   FULLSFT_BATCH_SIZE     Full SFT 单步 batch size（默认 8）
#   FULLSFT_ACCUM_STEPS    Full SFT 梯度累积步数（默认 8，等效 batch=64）
# 若显存更大，可调大 *_BATCH_SIZE、调小 *_ACCUM_STEPS 以提速。
#
# 断点续训: Pretrain / Full SFT 都已带 --from_resume 1，中途被杀（OOM、
# SSH 断线、手动 Ctrl-C）后直接重跑 bash run_train.sh 即可从最近一次
# save_interval 的 checkpoint 继续，不会从头开始；没有 checkpoint 时会
# 自动退化为从零训练，不会报错。SSH 连接本身建议配合 USE_TMUX=1（或自
# 己开 tmux/screen）使用，避免断连直接杀掉前台进程。
# ==============================================================
set -euo pipefail

: "${USE_MOE:=0}"
: "${USE_GATED_ATTENTION:=1}"
: "${USE_ENGRAM:=1}"
: "${USE_MTP:=0}"
: "${USE_COMPILE:=0}"
: "${USE_TMUX:=0}"
: "${PRETRAIN_BATCH_SIZE:=16}"
: "${PRETRAIN_ACCUM_STEPS:=16}"
: "${FULLSFT_BATCH_SIZE:=8}"
: "${FULLSFT_ACCUM_STEPS:=8}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 优先用项目自带的 uv venv（本机开发环境）；云端全新机器（如 AutoDL）通常没有
# .venv，退回用系统/镜像自带的 python3（镜像一般已预装 PyTorch，直接能用）。
if [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
    PY="$PROJECT_DIR/.venv/bin/python"
else
    PY="$(command -v python3 || command -v python)"
fi

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
        "USE_MOE=$USE_MOE USE_GATED_ATTENTION=$USE_GATED_ATTENTION USE_ENGRAM=$USE_ENGRAM USE_MTP=$USE_MTP USE_COMPILE=$USE_COMPILE bash \"${BASH_SOURCE[0]}\"; \
         echo '训练结束，按任意键退出'; read -n1"
fi

# ----------------------------------------------------------
# 根据开关构建附加参数
# 注意：train_pretrain.py 的 --use_moe/--use_gated_attention/--use_engram/
# --use_mtp/--use_compile 是 type=int, choices=[0,1]（必须显式传值），而
# train_full_sft.py 里同名参数是 action="store_true"（只看是否出现，不能
# 传值）。两边 argparse 定义不一致，因此这里分别构造两份 EXTRA_ARGS，不能
# 共用一份。
# ----------------------------------------------------------
PRETRAIN_EXTRA_ARGS="--use_moe $USE_MOE --use_gated_attention $USE_GATED_ATTENTION --use_engram $USE_ENGRAM --use_mtp $USE_MTP --use_compile $USE_COMPILE"

FULLSFT_EXTRA_ARGS=""
[[ "$USE_MOE"             == "1" ]] && FULLSFT_EXTRA_ARGS="$FULLSFT_EXTRA_ARGS --use_moe"
[[ "$USE_GATED_ATTENTION" == "1" ]] && FULLSFT_EXTRA_ARGS="$FULLSFT_EXTRA_ARGS --use_gated_attention"
[[ "$USE_ENGRAM"          == "1" ]] && FULLSFT_EXTRA_ARGS="$FULLSFT_EXTRA_ARGS --use_engram"
[[ "$USE_MTP"             == "1" ]] && FULLSFT_EXTRA_ARGS="$FULLSFT_EXTRA_ARGS --use_mtp"
[[ "$USE_COMPILE"         == "1" ]] && FULLSFT_EXTRA_ARGS="$FULLSFT_EXTRA_ARGS --use_compile"

echo "USE_MOE             : $USE_MOE"
echo "USE_GATED_ATTENTION : $USE_GATED_ATTENTION"
echo "USE_ENGRAM          : $USE_ENGRAM"
echo "USE_MTP             : $USE_MTP"
echo "USE_COMPILE         : $USE_COMPILE"
echo "USE_TMUX            : $USE_TMUX"
echo "PRETRAIN_EXTRA_ARGS : $PRETRAIN_EXTRA_ARGS"
echo "FULLSFT_EXTRA_ARGS  : ${FULLSFT_EXTRA_ARGS:-(none)}"
echo ""

DATASET_DIR="$PROJECT_DIR/dataset"
OUT_DIR="$PROJECT_DIR/out"

mkdir -p "$OUT_DIR" "$DATASET_DIR"

echo "PROJECT_DIR  : $PROJECT_DIR"
echo "DATASET_DIR  : $DATASET_DIR"
echo "OUT_DIR      : $OUT_DIR"
echo ""

# ==============================================================
# Step 1: 安装 Python 依赖
# ==============================================================
echo "========== [1/3] 检查/安装 Python 依赖 =========="
echo "使用python: $PY"
if command -v uv >/dev/null 2>&1; then
    PIP_INSTALL=(uv pip install --python "$PY" -q)
else
    PIP_INSTALL=("$PY" -m pip install -q)
fi
if "$PY" -c "import torch, transformers, modelscope" >/dev/null 2>&1; then
    echo "核心依赖（torch/transformers/modelscope）已就绪，跳过 requirements.txt 安装"
else
    # requirements.txt 锁的部分老版本包（如 ujson==5.1.0）在较新 Python（如 3.12）
    # 上可能没有预编译 wheel，需要联网拉构建依赖源码编译，容易被网络问题卡住。
    # 云端建议选 Python 3.10/3.11 的镜像来规避这个问题。
    "${PIP_INSTALL[@]}" modelscope
    "${PIP_INSTALL[@]}" -r "$PROJECT_DIR/requirements.txt"
fi
echo "依赖检查完成"
echo ""

# ==============================================================
# Step 2: 下载数据集（仅下载 Pretrain / Full SFT 需要的 2 个文件）
# ==============================================================
echo "========== [2/3] 下载数据集 =========="
DATASET_DIR="$DATASET_DIR" "$PY" - <<PYEOF
from modelscope.hub.snapshot_download import snapshot_download
import os

local_dir = os.environ["DATASET_DIR"]
snapshot_download(
    "gongjy/minimind_dataset",
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=[
        "pretrain_t2t_mini.jsonl",  # Pretrain 用 (1.24GB)
        "sft_t2t_mini.jsonl",       # Full SFT 用 (1.74GB)
    ]
)
print("数据集下载完成")
PYEOF
echo ""

# ==============================================================
# 进入项目根目录，后续所有训练相对路径均基于此
# ==============================================================
cd "$PROJECT_DIR"

# 减少显存碎片，避免 OOM 时明明有空闲却分配失败
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==============================================================
# Step 3: Pretrain（预训练）
# 数据: pretrain_t2t_mini ~0.25B tokens × 5 epochs
# 输出: ./out/pretrain_768.pth
# ==============================================================
echo "========== [3/3] Pretrain =========="
"$PY" -m trainer.train_pretrain \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --epochs 5 \
    --batch_size "$PRETRAIN_BATCH_SIZE" \
    --learning_rate 5e-4 \
    --accumulation_steps "$PRETRAIN_ACCUM_STEPS" \
    --max_seq_len 512 \
    --data_path ./dataset/pretrain_t2t_mini.jsonl \
    --save_dir ./out \
    --save_weight pretrain \
    --save_interval 2000 \
    --log_interval 200 \
    --dtype bfloat16 \
    --from_resume 1 \
    $PRETRAIN_EXTRA_ARGS
echo "Pretrain 完成"
echo ""

# ==============================================================
# Step 4: Full SFT（全参监督微调）
# 数据: sft_t2t_mini ~0.26B tokens × 5 epochs（含 Tool Call 数据）
# 输出: ./out/full_sft_768.pth
# ==============================================================
echo "========== Full SFT =========="
"$PY" -m trainer.train_full_sft \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --epochs 5 \
    --batch_size "$FULLSFT_BATCH_SIZE" \
    --learning_rate 1e-5 \
    --accumulation_steps "$FULLSFT_ACCUM_STEPS" \
    --max_seq_len 768 \
    --data_path ./dataset/sft_t2t_mini.jsonl \
    --from_weight pretrain \
    --save_dir ./out \
    --save_weight full_sft \
    --save_interval 1000 \
    --log_interval 100 \
    --dtype bfloat16 \
    --from_resume 1 \
    $FULLSFT_EXTRA_ARGS
echo "Full SFT 完成"
echo ""

echo "========== 全部训练完成 =========="
echo "最终权重位于 $OUT_DIR"
