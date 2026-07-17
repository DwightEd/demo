# Causal Belief Routing

This project tests a three-stage claim about pretrained decoder-only
Transformers:

1. residual states preserve an analytically known constraint belief that is
   absent from the current task-output distribution;
2. evidence-token attention/OV paths write in the direction of the exact
   belief update;
3. replacing those source-specific paths changes future answers in the donor
   direction.

It does not infer a manifold from generic hidden-state distances. The geometric
coordinates are finite-field Fourier characters of the exact posterior.

## Code map

| file | responsibility |
|---|---|
| `world.py` | Generate exact predictive aliases over \(\mathbb F_p^n\) |
| `finite_field.py`, `geometry.py` | Exact affine supports, Fourier coordinates, query inversion |
| `extraction.py`, `schema.py` | Compact all-layer boundary states and output sketches |
| `charts.py`, `audit.py`, `metrics.py` | Pair-grouped cross-fitting and representation gate |
| `routing_extraction.py`, `routing.py`, `routing_schema.py` | Evidence-source attention and per-head \(W_O\) writes |
| `routing_audit.py` | Cross-fitted head selection and routing gate |
| `patching.py`, `patch_schema.py`, `patch_audit.py` | Donor/recipient source-path interventions and causal gate |

See [GEOMETRY_PRIMER.md](GEOMETRY_PRIMER.md), [METHOD.md](METHOD.md),
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md), and [RELATED_WORK.md](RELATED_WORK.md)
for the mathematical object, hypothesis, run order, and novelty boundary.

## Remote pilot

Run from:

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/demo
```

The exact observer path is:

```text
/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct
```

### 1. Build 200 exact alias pairs

```bash
python build_predictive_aliases.py \
  --output data/causal_belief_routing/alias_pilot_200.jsonl \
  --num_pairs 200 \
  --modulus 3 \
  --num_variables 4 \
  --common_rank 2 \
  --template_families 3 \
  --seed 17
```

### 2. Extract boundary states and a 256-dimensional output sketch

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python extract_causal_belief_states.py \
  --input data/causal_belief_routing/alias_pilot_200.jsonl \
  --model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct \
  --output data/causal_belief_routing/alias_pilot_200_trace.npz \
  --layers 0,4,8,12,16,20,24,28,32 \
  --batch_size 16 \
  --max_batch_tokens 4096 \
  --logit_sketch_dim 256 \
  --device cuda \
  --dtype bfloat16
```

### 3. Run the representation gate

```bash
/opt/conda/bin/python audit_causal_belief_routing.py \
  --input data/causal_belief_routing/alias_pilot_200_trace.npz \
  --output_dir outputs/causal_belief_routing/alias_pilot_200/representation \
  --folds 5 \
  --projection_dim 64 \
  --ridge_alpha 10 \
  --bootstrap 2000 \
  --compute_device cuda
```

Stop if `ready for routing analysis` is false. A failed representation gate
means attention extraction cannot support the intended mechanism claim.

### 4. Extract source-specific OV writes

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python extract_causal_belief_routing.py \
  --trace data/causal_belief_routing/alias_pilot_200_trace.npz \
  --charts outputs/causal_belief_routing/alias_pilot_200/representation/layer_charts.npz \
  --model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct \
  --output data/causal_belief_routing/alias_pilot_200_routing.npz \
  --batch_size 4 \
  --max_batch_tokens 2048 \
  --device cuda \
  --dtype bfloat16

/opt/conda/bin/python audit_causal_belief_routing_mechanism.py \
  --input data/causal_belief_routing/alias_pilot_200_routing.npz \
  --output_dir outputs/causal_belief_routing/alias_pilot_200/routing \
  --folds 5 \
  --top_heads 16 \
  --bootstrap 2000
```

Stop if `ready for causal patching` is false.

### 5. Run a 50-pair causal patch pilot

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python extract_causal_belief_patches.py \
  --trace data/causal_belief_routing/alias_pilot_200_trace.npz \
  --routing_summary outputs/causal_belief_routing/alias_pilot_200/routing/summary.json \
  --model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct \
  --output data/causal_belief_routing/alias_pilot_50_patches.npz \
  --max_pairs 50 \
  --device cuda \
  --dtype bfloat16

/opt/conda/bin/python audit_causal_belief_patches.py \
  --input data/causal_belief_routing/alias_pilot_50_patches.npz \
  --output_dir outputs/causal_belief_routing/alias_pilot_50/patches \
  --bootstrap 2000 \
  --min_coverage 0.80
```

## What is actually controlled

- Alias pair is the split and bootstrap unit.
- Current exact answer distributions are identical across branches.
- Branch support size, posterior entropy, and information gain are equal.
- The output baseline contains residue logits and a fixed 256-dimensional
  full-vocabulary logit sketch.
- Belief labels are shuffled only inside training folds for the null model.
- Routing uses the opposite branch update and a same-length non-evidence source
  window as controls.
- Patching uses the same selected heads with a same-length source null and
  same-layer, same-count random-head null.
- Layer 32 is retained for representation decoding but excluded from head-write
  projection because its stored state is post-final-normalization rather than a
  raw block output.

The local unit suite is `tests/test_causal_belief_routing.py`.
