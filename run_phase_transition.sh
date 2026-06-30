#!/usr/bin/env bash
# =============================================================================
# 轨迹几何相变检测 - 执行脚本
# =============================================================================
#
# 论文: "Trajectory of Thought: Geometric Phase Transitions in Multi-Step Reasoning"
#
# 核心假设：错误推理是几何轨迹的相变，不是单点异常
#
# 三层架构：
#   Layer 1: Step-wise Geometry (每步的几何特征)
#   Layer 2: Trajectory Geometry (步骤间几何关系)
#   Layer 3: Phase Transition Detection (相变检测)
#
# 验证假设：
#   H1: 错误推理的轨迹smoothness低于正确推理
#   H2: Shallow Lock-in模式在error中更频繁
#   H3: Deep Decay模式在error中更频繁
#   H4: 基于轨迹的检测器优于基于单步几何的检测器
#
# =============================================================================

set -euo pipefail

# 配置
DATA_BASE="/gz-data/research/demo/data/features"
OUTPUT_BASE="$HOME/trajectory_phase_results"
SCRIPTS_DIR="$PWD"

# 要测试的数据集（按难度递增）
DATASETS=("full_gsm8k.npz" "full_math.npz" "full_omnimath.npz")

# 要分析的层（L14是主分析层）
LAYERS=(14)

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
    --quick            快速模式（只运行L14 on omnimath）
    --datasets LIST    数据集列表（默认: gsm8k math omnimath）
    --output DIR       输出目录（默认: ~/trajectory_phase_results）

示例:
    # 完整验证
    $0

    # 快速测试（难任务）
    $0 --quick

    # 只测试omnimath
    $0 --datasets omnimath
EOF
}

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

# 参数解析
QUICK_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            ;;
        --datasets)
            shift
            DATASETS=()
            IFS=' ' read -ra ds <<< "$1"
            for d in "${ds[@]}"; do
                DATASETS+=("full_${d}.npz")
            done
            ;;
        --output)
            OUTPUT_BASE="$2"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

if $QUICK_MODE; then
    DATASETS=("full_omnimath.npz")
fi

# 检查
log "Starting trajectory phase transition analysis..."
log "Data: $DATA_BASE"
log "Output: $OUTPUT_BASE"

if [[ ! -d "$DATA_BASE" ]]; then
    log "ERROR: Data directory not found. Run on box (gz-data)."
    exit 1
fi

if [[ ! -f "$SCRIPTS_DIR/trajectory_phase_transition.py" ]]; then
    log "ERROR: Script not found"
    exit 1
fi

mkdir -p "$OUTPUT_BASE"

# 执行
for npz_file in "${DATASETS[@]}"; do
    full_path="$DATA_BASE/$npz_file"

    if [[ ! -f "$full_path" ]]; then
        log "WARNING: File not found: $npz_file"
        continue
    fi

    dataset_name=$(basename "$npz_file" .npz)
    output_dir="$OUTPUT_BASE/$dataset_name"

    log "=================================================="
    log "Processing: $dataset_name"
    log "=================================================="

    mkdir -p "$output_dir"

    python3 "$SCRIPTS_DIR/trajectory_phase_transition.py" \
        "$full_path" \
        --output_dir "$output_dir" \
        --layers "${LAYERS[@]}" \
        2>&1 | tee "$output_dir/validation.log"

    log "✓ Completed: $dataset_name"
done

log "=================================================="
log "All analyses completed!"
log "Results: $OUTPUT_BASE"
log "=================================================="
