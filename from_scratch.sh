#!/usr/bin/env bash
# 环境搭建 + 数据下载（gpugeek RTX A5000, CUDA 12.1, conda3）
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODELS_DIR="/gz-data/models"
export HF_DATASETS_CACHE="${PROJ_ROOT}/data/hf_datasets"
export HF_HOME="/gz-data/hf_cache"
export HF_ENDPOINT="https://hf-mirror.com"

mkdir -p "$MODELS_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"

# ---------- 1. conda 环境 ----------
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx research; then
    conda create -n research python=3.10 -y
fi
conda activate research
pip install -q --upgrade pip

# ---------- 2. 安装依赖 ----------
pip install -q --index-url https://download.pytorch.org/whl/cu121 torch
pip install -q "transformers>=4.40" "datasets>=2.18" numpy scipy scikit-learn matplotlib tqdm modelscope

# ---------- 3. 下载模型 ----------
python -c "
from modelscope import snapshot_download
import os
snapshot_download('LLM-Research/Meta-Llama-3.1-8B-Instruct', cache_dir=os.environ['MODELS_DIR'])
"

# ---------- 4. 下载数据集 ----------
python -c "
from datasets import load_dataset
import os
for s in ['gsm8k', 'math', 'olympiadbench', 'omnimath']:
    ds = load_dataset('Qwen/ProcessBench', s, cache_dir=os.environ['HF_DATASETS_CACHE'])
    print(f'{s}: {len(ds[\"test\"])} examples')
"

echo "Done. 模型: $MODELS_DIR, 数据: $HF_DATASETS_CACHE"
