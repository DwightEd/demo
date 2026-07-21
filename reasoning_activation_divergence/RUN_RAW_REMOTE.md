# Raw residual-stream remote runs

## What “full” and “pilot” mean

There are two data families on the remote box.

### Canonical ProcessBench full

```text
data/features/full_gsm8k.npz
data/hidden/gsm8k/gsm8k-<row>.npy
```

- 395 ProcessBench observer chains;
- first-error labels and step ranges are in `full_gsm8k.npz`;
- each hidden shard is response-token state `[R,4,4096]`, layers
  `[10,14,18,22]`, fp16 on disk;
- depths are sparse, so depth operators mean intervals 10→14→18→22, not
  individual transformer blocks;
- “canonical pilot” below means the first 20 matched pairs from this full
  artifact. It is a compute smoke test, not a separate dataset.

### Exact observer pilot/full

```text
data/exact/processbench_observer_llama31_pilot/<subset>/selected/trace.npz
data/exact/processbench_observer_llama31_full/<subset>/selected/trace.npz
```

- `pilot` is the small extraction used to validate exact prompt/token alignment,
  shard loading and analysis before scale-up;
- `full` is the scaled extraction for GSM8K, MATH, OlympiadBench and Omni-MATH;
- each `trace.npz` points to per-chain response-token state `.npy` files;
- the loader requires
  `response_token_state_snapshot_kind=raw_residual_stream` and fails closed for
  unverified/final-normalized snapshots;
- selected depths may still be sparse. Always read the preflight
  `depth_semantics` and `layers` fields before interpreting results.

These are benchmark-observer residual streams. They are not the unknown original
generator's internal states. Stored states support empirical local transport and
rotation analyses, but not an autograd Jacobian without loading and replaying the
model.

## Update and test

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/demo
git pull --ff-only origin main

python -m pip install -e 'reasoning_activation_divergence[test]'
export PYTHONPATH="$PWD/reasoning_activation_divergence/src:$PWD"
python -m pytest reasoning_activation_divergence/tests -q
```

The editable install is required. It installs scikit-learn as a first-class
dependency used by both the raw layer-time operators and the task-probe Fisher
analysis; there is no dependency-free fallback implementation.

If the checkout is instead `/gz-data/research/demo`, use that directory; the
script derives `REPO_ROOT` from its own location.

## Recommended sequence

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/demo

# 1. Confirm canonical manifest, raw shard and layer shape.
bash reasoning_activation_divergence/run_raw_remote.sh canonical-preflight

# 2. Twenty matched pairs; verifies runtime and output schema.
bash reasoning_activation_divergence/run_raw_remote.sh canonical-pilot

# 3. Full 395-chain GSM8K observer analysis.
bash reasoning_activation_divergence/run_raw_remote.sh canonical-full
```

For the newer exact extraction:

```bash
bash reasoning_activation_divergence/run_raw_remote.sh exact-pilot
bash reasoning_activation_divergence/run_raw_remote.sh exact-full
```

Override the interpreter or checkout root when needed:

```bash
PYTHON_BIN=/path/to/conda/env/bin/python \
REPO_ROOT=/gz-data/research/demo \
bash /gz-data/research/demo/reasoning_activation_divergence/run_raw_remote.sh canonical-full
```

## Direct command

```bash
export PYTHONPATH="$PWD/reasoning_activation_divergence/src:$PWD"
python -m functional_divergence.raw_residual_experiment \
  --input data/features/full_gsm8k.npz \
  --hidden-dir data/hidden/gsm8k \
  --output-dir outputs/raw_layer_time/canonical_gsm8k_full \
  --offsets=-2,-1,0,1 \
  --layers 10,14,18,22 \
  --rank 16 --folds 5 --bootstrap 2000 --seed 17
```

Outputs are `results.json`, `pair_scores.csv`, and `metric_comparison.png`.
Check `representation_scope`, `snapshot_kind`, `depth_semantics`, actual
`projection_rank`, and `max_component_overlap` before reading AUROC values.
