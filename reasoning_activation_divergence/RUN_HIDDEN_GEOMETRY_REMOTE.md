# 远端前台运行 hidden-state geometry

## 1. 环境与路径

```bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate research

PROJECT_ROOT=/share/home/tm902089733300000/a903202310/lys/research/demo/reasoning_activation_divergence
DATA_ROOT=/share/home/tm902089733300000/a903202310/lys/research/demo/data/exact/processbench_observer_llama31_full
OUTPUT_ROOT="$PROJECT_ROOT/outputs/hidden_state_geometry"
PYTHON_BIN="$CONDA_PREFIX/bin/python"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" -m pip install -e '.[test]'
"$PYTHON_BIN" -c 'import sys, numpy, scipy, sklearn, tqdm; print(sys.executable); print(sklearn.__version__); print(sklearn.__file__)'
mkdir -p "$OUTPUT_ROOT"
```

真实输入不只是 `trace.npz`。每个数据集至少需要：

```text
$DATA_ROOT/<dataset>/selected/trace.raw_residual_stream.npz
$DATA_ROOT/<dataset>/selected/trace.npz
```

其中 `trace.raw_residual_stream.npz` 的 `response_token_state_files` 指向真正的
`[response_token, stored_layer, hidden]` `.npy` 分片；`trace.npz` 提供对齐的标签、step
边界及 entropy/NLL 摘要。若 residual manifest 或它引用的任一 `.npy` 不存在，preflight
会直接失败。`run_hidden_geometry_remote.sh` 启动时也会打印四域的解析路径；可用
`DATA_ROOT=/另一目录` 显式覆盖默认 full 路径。

`sys.executable` 和 `sklearn.__file__` 必须位于同一个 `research` conda 环境。安装包名是
`scikit-learn`，导入名才是 `sklearn`。所有命令都使用同一个 `PYTHON_BIN`，没有缺包
退化实现。

下列完整命令也已封装为普通前台脚本，可分别执行：

```bash
PYTHON_BIN="$CONDA_PREFIX/bin/python" bash run_hidden_geometry_remote.sh preflight
PYTHON_BIN="$CONDA_PREFIX/bin/python" bash run_hidden_geometry_remote.sh smoke
PYTHON_BIN="$CONDA_PREFIX/bin/python" bash run_hidden_geometry_remote.sh full
```

## 2. 全量预检

```bash
"$PYTHON_BIN" -m functional_divergence.hidden_state_geometry.cli preflight \
  --data-root "$DATA_ROOT" \
  --domains gsm8k,math,olympiadbench,omnimath \
  --response-generator llama3.1-8b \
  --observer-model llama3.1-8b \
  --acquisition-mode observer_teacher_forcing_replay \
  --output-features token_entropy,token_nll \
  --max-records-per-domain 0 \
  --seed 17 | tee "$OUTPUT_ROOT/preflight_console.json"
```

预检会报告每域 manifest/selected records、正确/错误数量、problem groups、stored layers、
首个 shard shape、strict-prefix rows 和 step-0 left truncation。它会 mmap 打开全部
eligible shards，逐个验证文件存在性、shape、count，以及四域 layer/hidden schema。
若 manifest 只有数值 `problem_ids`，同域聚类仍可正常运行，但跨域题目哈希审计会标为
`unavailable`；只有真实 `problem_sha256:*` 存在时才报告 `complete` 或检测跨域重题。

## 3. 真实数据 smoke

这仍从 full manifest 的 Llama-3.1-8B cohort 抽样，不是本地指标、代理数据或另一套
pilot artifact：

```bash
"$PYTHON_BIN" -m functional_divergence.hidden_state_geometry.cli run \
  --data-root "$DATA_ROOT" \
  --domains gsm8k,math,olympiadbench,omnimath \
  --response-generator llama3.1-8b --observer-model llama3.1-8b \
  --acquisition-mode observer_teacher_forcing_replay \
  --output-features token_entropy,token_nll \
  --tasks whole_chain,strict_prefix --method raw_functional_probe \
  --max-records-per-domain 32 \
  --pca-dim 8 --positions-per-chain 16 \
  --time-basis 3 --layer-basis 3 \
  --l2 1.0 --restarts 1 --max-iter 150 \
  --null-repeats 2 \
  --bootstrap 200 --seed 17 \
  --output-dir "$OUTPUT_ROOT/smoke_32_seed17"
```

## 4. 四域全量正式实验

`0` 表示使用过滤后的全部 eligible records：

```bash
"$PYTHON_BIN" -m functional_divergence.hidden_state_geometry.cli run \
  --data-root "$DATA_ROOT" \
  --domains gsm8k,math,olympiadbench,omnimath \
  --response-generator llama3.1-8b --observer-model llama3.1-8b \
  --acquisition-mode observer_teacher_forcing_replay \
  --output-features token_entropy,token_nll \
  --tasks whole_chain,strict_prefix --method raw_functional_probe \
  --max-records-per-domain 0 \
  --pca-dim 16 --positions-per-chain 32 \
  --time-basis 3 --layer-basis 3 \
  --l2 1.0 --restarts 3 --max-iter 500 \
  --null-repeats 3 \
  --bootstrap 2000 --seed 17 \
  --output-dir "$OUTPUT_ROOT/full_seed17"
```

这是普通前台 CPU/sklearn 实验，没有 `nohup`、`screen`、`tmux` 或 `&`。终端会显示
task、LODO domain、PCA chain、投影、编码、各 probe arm 和 bootstrap 进度。它只读取
已有 hidden/output artifacts，不会加载 Llama 权重或重新前向传播。

建议每次正式运行使用新的 `--output-dir`（例如在目录名附加 seed 或时间戳），避免把
不同配置混入同一目录。

## 5. 结果位置与判断顺序

输出目录包含 `preflight.json`、`results.json`、`oof_predictions.csv`、
`fold_audit.csv` 和 `model_factors.npz`。
`artifact_manifest.json` 记录同一次 run 的 ID 和文件哈希；正式分析前应保留它一起归档。

先分别看 whole-chain 与 strict-prefix 的：

```text
tasks.<task>.summary.increments.hidden_given_output_summary_nll
```

- point 与 95% CI 都大于 0：hidden 在 entropy/NLL summaries 之外有跨域增量；
- whole-chain 强而 strict-prefix 弱：信号主要来自错误后果；
- 两者都弱：当前低秩跨域可迁移表示没有挖到稳定增量，不等于 hidden 完全无信息；
- strict-prefix 稳定为正：支持提前关联，但仍不支持功能性或因果结论。
