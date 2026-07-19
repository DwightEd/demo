# 在已有全量提取数据上运行

已有 `selected` 数据是逐 token、稀疏深度状态。本实验测量相邻已观测深度之间的方向变化，
例如 `h[10]-h[8]`，结论范围仅为 sparse depth-interval pilot。

## 1. 检查数据

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/demo

DATA_ROOT=data/exact/processbench_observer_llama31_pilot
find "$DATA_ROOT" -path '*/selected/trace.npz' -print
du -sh "$DATA_ROOT"/*/selected
```

应看到 `gsm8k`、`math`、`olympiadbench`、`omnimath` 四个 manifest。

## 2. 单 manifest 预检

```bash
python -m token_residual_dispersion.cli \
  --input "$DATA_ROOT/gsm8k/selected/trace.npz" \
  --legacy-sparse-pilot \
  --max-traces 1 \
  --preflight
```

## 3. 使用现成的 20 条链数据做 smoke test

```bash
MAX_TRACES=20 bash token_residual_dispersion/run_existing_selected_pilot.sh \
  "$DATA_ROOT" \
  outputs/token_residual_dispersion_sparse_pilot20
```

检查：

```bash
find outputs/token_residual_dispersion_sparse_pilot20 -name audit_summary.json -print
python - <<'PY'
import json
from pathlib import Path

for path in Path("outputs/token_residual_dispersion_sparse_pilot20").glob("*/audit_summary.json"):
    report = json.loads(path.read_text())
    print(path.parent.name, len(report["traces"]), report["delta_kinds"])
PY
```

每个 `delta_kinds` 必须是：

```text
sparse_multi_block_depth_interval_delta_pilot
```

## 4. 全量运行

使用新的输出目录，避免与 smoke test 混合：

```bash
DATA_ROOT=data/exact/processbench_observer_llama31_full

bash token_residual_dispersion/run_existing_selected_pilot.sh \
  "$DATA_ROOT" \
  outputs/token_residual_dispersion_sparse_full
```

结果按链写入 `<output>/<subset>/traces/*.npz`，汇总写入
`<output>/<subset>/audit_summary.json`。程序每处理 100 条链报告一次进度。
