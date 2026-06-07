#!/usr/bin/env bash
# run_v2_pipeline.sh - 端到端 v2 数据生成 + 统计管道, 串行运行防 OOM
#
# 用途:
#   把上次 OOM 那条踩坑全部修掉:
#     - 串行 (绝不并发两个 generate 进程)
#     - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 减碎片
#     - 每步独立 log + 时间戳, 单步失败不污染下一步状态
#     - 中间存 done-marker, 重启时跳过已完成步骤 (resume 友好)
#
# 默认任务序:
#     1. 抽 v2 custom_zeroshot 数据  (~10-12 h)
#     2. 抽 v2 lm_eval_5shot 数据    (~10-12 h)
#     3. 跑 26 在 v2 custom 上       (~3-5 min)
#     4. 跑 26 在 v2 5shot 上        (~3-5 min)
#     5. (可选) 把 JSON 拷到 results/ 并 push GitHub
#
# 用法:
#   chmod +x run_v2_pipeline.sh
#   ./run_v2_pipeline.sh                          # 用默认参数跑全套
#   GPU=0 N_PROBLEMS=300 K=12 ./run_v2_pipeline.sh # 重定参数
#   SKIP_SAMPLE=1 ./run_v2_pipeline.sh            # 跳过抽数据, 只跑 26
#   SKIP_PUSH=1 ./run_v2_pipeline.sh              # 跑完不 push
#   STYLES="custom_zeroshot" ./run_v2_pipeline.sh # 只跑一种 prompt
#
# 重启友好:
#   每个完成的步骤会写一个 .done marker; 重新跑会跳过. 想强制重跑某步:
#     rm logs/.done_v2_custom_extract
#     ./run_v2_pipeline.sh
#
# 退出码:
#   0 = 全部成功; 非 0 = 某步失败, 看 logs/*.log 找原因.

set -e                    # 任何命令失败立即退出 (不要 set -u, $@ 展开会炸)
set -o pipefail           # tee 后面的命令真实退出码

# ============================================================================
# 配置 (用环境变量覆盖)
# ============================================================================

PROJECT_DIR="${PROJECT_DIR:-/gz-data/research/demo}"
N_PROBLEMS="${N_PROBLEMS:-300}"
K="${K:-12}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
SEED="${SEED:-42}"
CLOUD_LAYERS="${CLOUD_LAYERS:-16}"
STYLES="${STYLES:-custom_zeroshot lm_eval_5shot}"        # 空格分隔的 prompt list
BOOTSTRAP_N="${BOOTSTRAP_N:-1000}"

# 控制开关
SKIP_PULL="${SKIP_PULL:-0}"          # 1 = 跳过 git pull
SKIP_SAMPLE="${SKIP_SAMPLE:-0}"      # 1 = 跳过抽数据 (10), 只跑 26
SKIP_STATS="${SKIP_STATS:-0}"        # 1 = 跳过 26
SKIP_PUSH="${SKIP_PUSH:-0}"          # 1 = 不 push GitHub
GPU="${GPU:-}"                       # 如 "0" 或 "0,1"; 留空 = 用所有可见 GPU

# ============================================================================
# 环境
# ============================================================================

cd "$PROJECT_DIR"

mkdir -p logs data results

# OOM 修复: 减显存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 绑 GPU (可选)
if [ -n "$GPU" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# 时间戳 helper
ts() { date '+%Y-%m-%d %H:%M:%S'; }

log_step() {
    echo ""
    echo "============================================================"
    echo "[$(ts)] $*"
    echo "============================================================"
}

run_or_skip() {
    # $1 = done marker name (基本名, 自动加 logs/.done_ 前缀)
    # $2+ = 命令
    local marker="logs/.done_$1"
    shift
    if [ -f "$marker" ]; then
        echo "[$(ts)] SKIP (已完成: $marker)  ==>  $*"
        return 0
    fi
    echo "[$(ts)] RUN   ==>  $*"
    if eval "$@"; then
        touch "$marker"
        echo "[$(ts)] DONE  ==>  marker $marker 已写"
    else
        echo "[$(ts)] FAIL  ==>  $*" >&2
        echo "[$(ts)] 查看对应 logs/*.log; 不写 done marker, 修好后重跑会接着这一步开始" >&2
        return 1
    fi
}

# ============================================================================
# 启动信息
# ============================================================================

log_step "v2 pipeline 启动"
echo "PROJECT_DIR    = $PROJECT_DIR"
echo "STYLES         = $STYLES"
echo "N_PROBLEMS     = $N_PROBLEMS"
echo "K              = $K"
echo "MAX_NEW_TOKENS = $MAX_NEW_TOKENS"
echo "TEMPERATURE    = $TEMPERATURE / TOP_P=$TOP_P / SEED=$SEED"
echo "CLOUD_LAYERS   = $CLOUD_LAYERS"
echo "BOOTSTRAP_N    = $BOOTSTRAP_N"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-'(unset, 用全部)'}"
echo "PYTORCH_CUDA_ALLOC_CONF = $PYTORCH_CUDA_ALLOC_CONF"
echo "SKIP_PULL/SAMPLE/STATS/PUSH = $SKIP_PULL/$SKIP_SAMPLE/$SKIP_STATS/$SKIP_PUSH"

# 看一眼 GPU
if command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    echo "GPU 状态 (启动时):"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
fi

# ============================================================================
# 步骤 0: git pull (拉最新代码)
# ============================================================================

if [ "$SKIP_PULL" != "1" ]; then
    log_step "步骤 0: git pull origin main"
    run_or_skip "git_pull_$(date +%Y%m%d)" "git pull origin main"
else
    echo "[$(ts)] SKIP_PULL=1, 跳过 git pull"
fi

# ============================================================================
# 步骤 1+2: 抽数据 (每个 prompt style 串行一次)
# ============================================================================

if [ "$SKIP_SAMPLE" != "1" ]; then
    for STYLE in $STYLES; do
        case "$STYLE" in
            custom_zeroshot) TAG="custom"   ;;
            lm_eval_5shot)   TAG="5shot"    ;;
            lm_eval_8shot)   TAG="8shot"    ;;
            *)               TAG="$STYLE"   ;;
        esac
        OUT="data/gsm8k_v2_${TAG}.npz"
        LOG="logs/10_v2_${TAG}.log"

        log_step "抽 v2 数据: prompt_style=$STYLE  ->  $OUT"

        # 进程绑独占 GPU 用 CUDA_VISIBLE_DEVICES 已经在顶部设过; 这里串行所以
        # 同一 GPU 不会出现两个 generate 进程并发的 OOM
        run_or_skip "v2_${TAG}_extract" \
            "python 10_sample_and_extract.py \
                --dataset_format gsm8k --dataset openai/gsm8k --subset main --split test \
                --n_problems $N_PROBLEMS --k_samples $K \
                --max_new_tokens $MAX_NEW_TOKENS \
                --temperature $TEMPERATURE --top_p $TOP_P --seed $SEED \
                --prompt_style $STYLE --step_split line \
                --store_vectors --store_clouds --cloud_layers $CLOUD_LAYERS \
                --output $OUT \
                2>&1 | tee $LOG"

        # 抽完打 audit 摘要 (从 log 里捞最后 30 行)
        echo ""
        echo "----- audit 块 ($STYLE) -----"
        if [ -f "$LOG" ]; then
            tail -40 "$LOG" | grep -E "(generation / labeling audit|drop reasons|Kept|Contrastive|Format|Lenient|Strict|Gap|Saved -> )" || \
                tail -20 "$LOG"
        fi
        echo "----- /audit -----"
    done
else
    echo "[$(ts)] SKIP_SAMPLE=1, 跳过抽数据"
fi

# ============================================================================
# 步骤 3+4: 跑 26 全套统计 (每份 npz 一次)
# ============================================================================

if [ "$SKIP_STATS" != "1" ]; then
    for STYLE in $STYLES; do
        case "$STYLE" in
            custom_zeroshot) TAG="custom"   ;;
            lm_eval_5shot)   TAG="5shot"    ;;
            lm_eval_8shot)   TAG="8shot"    ;;
            *)               TAG="$STYLE"   ;;
        esac
        IN="data/gsm8k_v2_${TAG}.npz"
        OUT_JSON="data/v2_${TAG}.comprehensive_stats.json"
        LOG="logs/26_v2_${TAG}.log"

        if [ ! -f "$IN" ]; then
            echo "[$(ts)] WARN: $IN 不存在, 跳过 26 (上一步抽数据失败了?)"
            continue
        fi

        log_step "跑 26 (comprehensive_stats) on $IN"

        run_or_skip "v2_${TAG}_stats" \
            "python 26_comprehensive_stats.py \
                --input $IN \
                --output $OUT_JSON \
                --bootstrap_n $BOOTSTRAP_N \
                2>&1 | tee $LOG"

        echo "  -> $OUT_JSON ($(du -h $OUT_JSON | cut -f1))"
    done
else
    echo "[$(ts)] SKIP_STATS=1, 跳过 26"
fi

# ============================================================================
# 步骤 5: 拷 JSON 到 results/ 并 push (data/ 在 .gitignore 里, results/ 不在)
# ============================================================================

if [ "$SKIP_PUSH" != "1" ]; then
    log_step "步骤 5: 拷 JSON 到 results/ 并 push"

    # 只拷 *.json, 不带 *.npz (大文件别上 git)
    if compgen -G "data/v2_*.comprehensive_stats.json" > /dev/null; then
        cp -v data/v2_*.comprehensive_stats.json results/
    fi
    if [ -f "data/v1_comprehensive_stats.json" ]; then
        cp -v data/v1_comprehensive_stats.json results/ 2>/dev/null || true
    fi

    if [ -n "$(git status --porcelain results/)" ]; then
        git add results/
        git commit -m "results: v2 comprehensive stats ($(date +%Y-%m-%d), styles=$STYLES, n=$N_PROBLEMS, K=$K)"
        git push origin main && \
            echo "[$(ts)] push 成功"
    else
        echo "[$(ts)] results/ 没新文件, 跳过 push"
    fi
else
    echo "[$(ts)] SKIP_PUSH=1, 跳过 push"
fi

# ============================================================================
# 收尾
# ============================================================================

log_step "v2 pipeline 完成"

echo "产物清单:"
for STYLE in $STYLES; do
    case "$STYLE" in
        custom_zeroshot) TAG="custom"   ;;
        lm_eval_5shot)   TAG="5shot"    ;;
        lm_eval_8shot)   TAG="8shot"    ;;
        *)               TAG="$STYLE"   ;;
    esac
    NPZ="data/gsm8k_v2_${TAG}.npz"
    JSON="data/v2_${TAG}.comprehensive_stats.json"
    [ -f "$NPZ" ]  && echo "  $NPZ   $(du -h $NPZ | cut -f1)"
    [ -f "$JSON" ] && echo "  $JSON  $(du -h $JSON | cut -f1)"
done

echo ""
echo "重启提示: 想强制重跑某步, 删对应 done marker:"
ls -1 logs/.done_* 2>/dev/null || echo "  (尚无 done marker)"

echo ""
echo "[$(ts)] OK"
