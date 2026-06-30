#!/usr/bin/env bash
# =============================================================================
# 轨迹几何验证实验 - 远程服务器执行脚本
# =============================================================================
#
# 验证假设：
# H1: 错误推理的轨迹smoothness低于正确推理
# H2: Shallow Lock-in模式在error中更频繁
# H3: Deep Decay模式在error中更频繁
# H4: 基于轨迹的检测器优于基于单步几何的检测器
#
# 数据位置（在box上）：
#   /gz-data/research/demo/data/features/full_*.npz
#
# 结果输出：
#   ~/trajectory_results/
#
# =============================================================================

set -euo pipefail

# =============================================================================
# 配置
# =============================================================================

DATA_BASE="/gz-data/research/demo/data/features"
OUTPUT_BASE="$HOME/trajectory_results"
SCRIPTS_DIR="$PWD"

# 要测试的数据集（按难度递增）
DATASETS=("full_gsm8k.npz" "full_math.npz" "full_omnimath.npz")

# 要分析的层（L14是主分析层，添加多层用于跨层验证）
LAYERS=(14 10 18 22)

# =============================================================================
# 函数
# =============================================================================

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
    --datasets LIST    数据集列表（空格分隔，默认: gsm8k math omnimath）
    --layers LIST      要分析的层（默认: 14 10 18 22）
    --output DIR       输出目录（默认: ~/trajectory_results）
    --quick            快速模式（只运行L14 on omnimath）
    --help             显示此帮助

示例:
    # 完整验证（所有数据集，所有层）
    $0

    # 快速测试（omnimath L14）
    $0 --quick

    # 自定义数据集和层
    $0 --datasets "math omnimath" --layers "14 18"
EOF
}

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

run_validation() {
    local npz_file=$1
    local output_dir=$2
    shift 2
    local layers=("$@")

    local dataset_name=$(basename "$npz_file" .npz)
    local dataset_output="$output_dir/$dataset_name"

    log "=================================================="
    log "Processing dataset: $dataset_name"
    log "Input: $npz_file"
    log "Output: $dataset_output"
    log "Layers: ${layers[*]}"
    log "=================================================="

    # 创建输出目录
    mkdir -p "$dataset_output"

    # 运行轨迹几何验证
    python3 "$SCRIPTS_DIR/trajectory_geometry_validation.py" \
        "$npz_file" \
        --output_dir "$dataset_output" \
        --layers "${layers[@]}" \
        2>&1 | tee "$dataset_output/validation.log"

    log "✓ Completed: $dataset_name"
}

# =============================================================================
# 参数解析
# =============================================================================

QUICK_MODE=false
CUSTOM_DATASETS=()
CUSTOM_LAYERS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets)
            shift
            IFS=' ' read -ra CUSTOM_DATASETS <<< "$1"
            ;;
        --layers)
            shift
            IFS=' ' read -ra CUSTOM_LAYERS <<< "$1"
            ;;
        --output)
            OUTPUT_BASE="$2"
            shift
            ;;
        --quick)
            QUICK_MODE=true
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

# 应用参数
if $QUICK_MODE; then
    DATASETS=("full_omnimath.npz")
    LAYERS=(14)
fi

if [[ ${#CUSTOM_DATASETS[@]} -gt 0 ]]; then
    DATASETS=()
    for ds in "${CUSTOM_DATASETS[@]}"; do
        DATASETS+=("full_${ds}.npz")
    done
fi

if [[ ${#CUSTOM_LAYERS[@]} -gt 0 ]]; then
    LAYERS=("${CUSTOM_LAYERS[@]}")
fi

# =============================================================================
# 前置检查
# =============================================================================

log "Starting trajectory geometry validation..."
log "Data base: $DATA_BASE"
log "Output base: $OUTPUT_BASE"

# 检查数据目录
if [[ ! -d "$DATA_BASE" ]]; then
    log "ERROR: Data directory not found: $DATA_BASE"
    log "This script must be run on the box (gz-data)."
    exit 1
fi

# 检查脚本文件
if [[ ! -f "$SCRIPTS_DIR/trajectory_geometry_validation.py" ]]; then
    log "ERROR: Script not found: trajectory_geometry_validation.py"
    log "Current directory: $SCRIPTS_DIR"
    exit 1
fi

# 检查依赖
log "Checking dependencies..."
python3 -c "import numpy, scipy, tqdm" 2>/dev/null || {
    log "ERROR: Missing required packages. Run: pip install numpy scipy tqdm"
    exit 1
}

# =============================================================================
# 执行验证
# =============================================================================

mkdir -p "$OUTPUT_BASE"

for npz_file in "${DATASETS[@]}"; do
    full_path="$DATA_BASE/$npz_file"

    if [[ ! -f "$full_path" ]]; then
        log "WARNING: File not found: $full_path"
        log "Skipping..."
        continue
    fi

    run_validation "$full_path" "$OUTPUT_BASE" "${LAYERS[@]}"
done

# =============================================================================
# 结果汇总
# =============================================================================

log "=================================================="
log "All validations completed!"
log "Results directory: $OUTPUT_BASE"
log "=================================================="

# 生成汇总表
python3 << 'EOF'
import json
import os
from pathlib import Path

output_base = os.path.expanduser(os.getenv("OUTPUT_BASE", "~/trajectory_results"))
results_path = Path(output_base)

summary = []
for result_file in results_path.glob("*/trajectory_validation_results.json"):
    if not result_file.exists():
        continue

    with open(result_file) as f:
        data = json.load(f)

    dataset = data['metadata']['subset']
    for key, val in data['results'].items():
        summary.append({
            'dataset': dataset,
            'test': key,
            'metric': val['metric'],
            'p_value': val['p_value'],
            'cohens_d': val['cohens_d'],
            'interpretation': val['interpretation'],
        })

print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"{'Dataset':<12} {'Test':<12} {'Metric':<20} {'Cohen\\'s d':<10} {'p-value':<10}")
print("-"*80)

for item in sorted(summary, key=lambda x: (x['dataset'], x['test'])):
    sig = "*" if item['p_value'] and item['p_value'] < 0.05 else ""
    print(f"{item['dataset']:<12} {item['test']:<12} {item['metric']:<20} "
          f"{item['cohens_d']:<10.3f} {item['p_value']:<10.4f} {sig}")

print("="*80)
print("* p < 0.05")
EOF

log "Done!"
