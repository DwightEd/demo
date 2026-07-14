# OC-GPI Experiment Plan

## Implementation architecture

| module | responsibility |
|---|---|
| `ocgpi/schema.py` | versioned packed output-trace contract and validation |
| `ocgpi/logit_trace.py` | shift-invariant token output features and step aggregation |
| `ocgpi/extraction.py` | causal teacher-forcing replay and chunked GPU vocabulary projection |
| `ocgpi/geometry_features.py` | label-free depth/time/coupling geometry and legacy adapters |
| `ocgpi/dataset.py` | response, online-prefix, and future-output task construction |
| `ocgpi/models.py` | nested problem-group cross-fitting and output-conditioned residual chart |
| `ocgpi/metrics.py` | usable information, partial $R^2$, calibration, and cluster bootstrap |
| `ocgpi/gates.py` | pure result-to-claim gate, independent of GPU dependencies |
| `ocgpi/report.py` | deterministic Markdown, JSON, CSV, and OOF outputs |
| `ocgpi/audit.py` | orchestration, saturation ladder, group ablations, and integrity checks |

The root scripts `extract_ocgpi_traces.py` and `audit_ocgpi.py` are direct-path
Linux entry points. They contain no experiment logic.

## Stage 0: frozen decisions

- Main task: any ProcessBench process error at causal prefix checkpoints;
  final-answer correctness is a secondary label-policy run.
- Deployable detector task: one shared model over every observed absolute
  prefix; relative-length checkpoints are diagnostic only.
- Mechanism task: one-step future change of compact output state.
- Group split: `problem_id`, never random rows.
- Primary geometry: all intermediate depth/time/coupling groups.
- Negative control: `final_control` only.
- Null: length-matched geometry permutation.
- Primary datasets: ProcessBench GSM8K, MATH, and OmniMath.
- Main observer: the same Llama-3.1-8B-Instruct checkpoint used for the saved
  hidden states. A different observer is an explicit transfer experiment.

## Stage 1: compact logits replay

Run once per subset from the canonical local ProcessBench source or an
exact-token trace artifact. The canonical `full_*.npz` files do not contain
problem text or exact input IDs and therefore cannot be used alone for valid
logits replay. The dedicated source loader reproduces the existing at-least
three-step filter, kept-row indexing, and single-newline response rendering;
the subsequent response-hash join refuses any remaining mismatch.

```bash
python extract_ocgpi_traces.py \
  --input data/hf_datasets/ProcessBench \
  --input_format processbench_source \
  --subset gsm8k \
  --geometry_reference data/features/full_gsm8k.npz \
  --model /gz-data/models/Meta-Llama-3.1-8B-Instruct \
  --output outputs/ocgpi/gsm8k_output_trace.npz \
  --max_seq_len 4096 \
  --top_k 64 \
  --sketch_dim 64 \
  --token_chunk_size 32 \
  --dtype bfloat16 \
  --device cuda
```

Repeat for `math.jsonl` and `omnimath.jsonl`. Check:

- chain and problem coverage are at least 0.95;
- source/reference IDs, steps, labels, and response text pass before model loading;
- no chain is silently truncated;
- model and tokenizer identity match exact traces when present;
- `full_logits_persisted=false` in metadata.

## Stage 2: exploratory reuse of current geometry

The current sparse `stepvec` can test pipeline behavior without re-extracting
hidden states:

```bash
python audit_ocgpi.py \
  --trace outputs/ocgpi/gsm8k_output_trace.npz \
  --geometry data/features/full_gsm8k.npz \
  --output_dir outputs/ocgpi/gsm8k_sparse_audit \
  --compute_device cuda \
  --geometry_batch_size 32 \
  --outer_folds 5 \
  --inner_folds 4 \
  --bootstrap 2000
```

This run is exploratory because `stepvec` contains eight sparse layers and
legacy step pooling. It can reject the feature family, estimate runtime, and
identify broken joins. It cannot establish a whole-layer mechanism.

## Stage 3: confirmatory whole-layer run

If Stage 2 produces positive future-output partial $R^2$, extract contiguous
whole-layer arithmetic-mean states with the existing mechanism extractor's
`--geometry_only` path, then rerun OC-GPI with that artifact.

The confirmatory suite is frozen before seeing MATH/OmniMath results. Report:

- four response prefix checkpoints;
- future-output forecast;
- output-to-geometry $R^2$;
- depth, temporal, coupling, ICR, legacy, and final-layer ablations;
- matched-null confidence intervals;
- per-dataset and macro-average effects.

## Stage 4: mechanism localization

Only groups with positive forecast increment in at least two datasets proceed.
For each surviving group:

1. inspect `conditional_geometry_importance.csv`;
2. freeze a small set of layers/features using GSM8K only;
3. verify the direction and effect on MATH/OmniMath;
4. align events to the first future logit collapse, not merely first-error
   position;
5. compare easy and hard problem strata without changing the detector.

## Stage 5: causal test

The final mechanism claim requires an intervention. Patch or steer the selected
residual geometric mode at the earliest detected prefix and test whether:

\[
\Delta Z_{t\rightarrow t+k}
\quad\text{and}\quad
P(Y=1)
\]

move in the direction predicted by the observational adapter. Activation
patching must use held-out problems and include equal-norm random-direction and
final-layer controls.

## Stop rules

Stop this branch if either condition holds across all three datasets:

- future-output partial $R^2$ confidence interval includes zero and does not
  beat the matched null;
- geometry improves response AUROC but not conditional NLL or future-output
  prediction.

Those outcomes mean the geometry is a correlated detector feature or capacity
proxy, not the proposed anticipatory mechanism.
