# Prompt-Controlled Residual Flow

Data provenance, label semantics, teacher-forcing alignment, and the distinction
between observer traces and generation-matched self traces are defined in
[DATA_CONTRACT_REASONING_TRACES.md](DATA_CONTRACT_REASONING_TRACES.md). Run
`audit_reasoning_data.py` before selecting an artifact for a new experiment.

The repository-wide experiment history, including negative results, superseded
claims, evidence grades, and the current frozen conclusion, is maintained in
[`../EXPERIMENT_LEDGER.md`](../EXPERIMENT_LEDGER.md).

This subproject tests whether reasoning failures are visible as a shift in
residual-stream writes away from the problem prompt and toward the generated
prefix.  It deliberately avoids using step position or step length to define
"healthy" reasoning; those variables are reported only as controls.

## Core Question

For a generated token or step, does the residual update remain aligned with the
information subspace induced by the original prompt?

Let the prompt token cloud at layer `l` be `H_Q^l`, and let `Delta r_t^l` be
the residual update used to predict a generated token.  The prompt-controlled
write fraction is

```text
|| P_{S_Q^l} Delta r_t^l ||^2 / (|| Delta r_t^l ||^2 + eps)
```

where `S_Q^l` is obtained from an SVD basis of the prompt token cloud.  A matched
prefix basis and a random basis are computed as controls.

## Why Prompt SVD Is Only a Hypothesis

Prompt SVD is not assumed to be correct.  It is one chart among several:

- `prompt_svd`: per-problem prompt-induced subspace.
- `prefix_svd`: generated-prefix subspace, used to detect prefix lock-in.
- `random_svd`: random basis with the same rank, a negative control.
- `representation_geometry`: cross-fitted point-cloud geometry over pooled
  step hidden states, including boundary projection, local intrinsic
  dimensionality, spectral entropy, kNN label topology, and cross-layer
  neighborhood instability.
- `spectral_chain_dynamics`: whole-chain spectral-manifold validation.  It
  learns a graph-Laplacian / diffusion-map chart from training chains and
  scores held-out response trajectories by healthy-tube distance, spectral
  leakage, tangent off-manifold motion, and error-basin committor.

The current claim-driven method is
[METHOD_REASONING_FLOW_SIGNATURES.md](METHOD_REASONING_FLOW_SIGNATURES.md).
It tests whether same-problem correct hidden-state trajectories form a
concentrated class of ordered flows using first- and second-order path
log-signatures.  The required controls are net displacement, shuffled
increments, response length, step count, total variation, and same-problem
paired evaluation.

[METHOD_LAYER_TIME_GEOMETRY.md](METHOD_LAYER_TIME_GEOMETRY.md) is retained as
a compatibility/negative baseline.  Its layer-time plaquette must not be
described as validated reasoning curvature.  The older
[METHOD_SPECTRAL_CHAIN_DYNAMICS.md](METHOD_SPECTRAL_CHAIN_DYNAMICS.md) remains
as another trajectory baseline.

[METHOD_DIRECTIONAL_CONSENSUS.md](METHOD_DIRECTIONAL_CONSENSUS.md) is the
targeted follow-up to the observed length confound in raw token spread. It uses
the exact off-diagonal mean cosine (a linear-time spherical U-statistic), then
tests same-problem response separability under cross-fitted length controls,
fixed token windows, token-length matching, problem bootstrap, and within-
problem label permutation. It consumes the existing `sv_clouds`; no new model
forward pass is required.

[METHOD_FEASIBLE_TANGENT_GATE.md](METHOD_FEASIBLE_TANGENT_GATE.md) is the
retained low-rank negative baseline. Its rank-four existence gate failed, so
normal escape from a single linear tangent is not the current mainline claim.

[METHOD_CONDITIONAL_FLOW_FIELD.md](METHOD_CONDITIONAL_FLOW_FIELD.md) is the
current geometry-first validation. It replaces one fitted tangent with the
same-problem, causal-phase-conditioned empirical distribution of normalized
hidden-state updates on the unit sphere. A proper energy score measures target
compatibility while matched donor counts, time-shuffled fields, wrong-problem
fields, cross-fitted length controls, and two ordered decision gates prevent a
response-length or finite-reference artifact from becoming the claim. It does
not use logits or train a classifier. The existing `sv_vec_step_exp`
multisample artifact is sufficient; run `audit_conditional_flow_field.py`.

The field-existence gate passed while error excursion failed, so geometry-only
density is no longer the mainline detector. The preregistered follow-up is
[METHOD_CAUSAL_PULLBACK_FLOW.md](METHOD_CAUSAL_PULLBACK_FLOW.md). It derives a
field-normal witness from same-problem correct responses, intervenes only after
the corresponding transition is causally observable, and estimates the full
source-step to future-output categorical-Fisher pullback by central KL
curvature. Time-shuffled and random tangent interventions are mandatory. The
audit then asks whether this causal operator adds usable information beyond
entropy, chosen-token NLL, margins, and length, instead of treating another
static geometry scalar as the result.

[METHOD_PREDICTIVE_STATE_GEOMETRY.md](METHOD_PREDICTIVE_STATE_GEOMETRY.md)
implements the next claim-driven pilot after directional debiasing failed to
improve AUROC over raw spread. It learns a correct-only reduced-rank chart from
future-window predictability, not VAE reconstruction, and requires ordered
innovation to beat shuffled futures, same-problem mismatched futures, static
latent density, lexical bigram NLL, length controls, and fixed-window
directional consensus. Existing legacy multisample files have `sv_clouds` but
not exact token IDs: they run an explicitly exploratory state-only tier without
re-extraction. Exact-trace artifacts enable token-ID residualization and the
full confirmatory gate.

[METHOD_CROSS_DATASET_REPLICATION.md](METHOD_CROSS_DATASET_REPLICATION.md)
freezes the existing spread, jump, transition-surprise, and CUSUM signals and
tests them without per-dataset sign selection on ProcessBench GSM8K, MATH, and
OmniMath. It reports raw and length/position-residualized AUROC, cluster
bootstrap intervals, and fixed-direction within-chain localization. The three
canonical `full_*.npz` files are sufficient; no re-extraction is needed.
The first three-subset result and its claim boundary are recorded in
[RESULT_CROSS_DATASET_TRANSITION_2026-07-14.md](RESULT_CROSS_DATASET_TRANSITION_2026-07-14.md).

Use `--store_step_vectors` during extraction to save the shared per-step
residual-flow vector store for PCA/VAE/spectral chart comparisons.  This is
important: prompt SVD, VAE, and other charts should be compared on the same
underlying residual-flow object, not on incompatible feature sets.

Use `--store_step_state_vectors` when the target analysis is whole-layer
representation geometry.  These vectors are pooled hidden states for each
reasoning step and selected layer:

```text
z_{chain,step}^{layer} = mean_pool hidden_state[layer, tokens_in_step]
```

This is the point cloud required for LID, spectral entropy, neighborhood
overlap, and manifold-fracture style audits.  Residual update vectors and state
vectors answer different questions and should not be mixed silently.

For the canonical ProcessBench full files documented in
`md/guides/DATA.md`, the same object already exists as `stepvec`:

```text
data/features/full_gsm8k.npz
data/features/full_math.npz
data/features/full_omnimath.npz

stepvec: object array, each chain shaped (T, 8, 4096)
```

`cli/audit_spectral_chain.py` can consume these `full_*.npz` files directly.
It unfolds `stepvec` into the same vector-bank protocol used by extracted
mechanism files.  Do not use `data/full_gsm8k.npz` or
`data/processbench_gsm8k_stepvec.npz`; those are non-canonical paths for this
project layout.

The method is useful only if `prompt_svd` beats length, position, and random
controls, especially under within-problem response comparisons.

## Folder Layout

```text
prompt_control_flow/
  README.md
  __init__.py
  config.py
  data.py
  geometry.py
  representation_geometry.py
  spectral_chain_dynamics.py
  layer_time_geometry.py
  layer_time_evaluate.py
  flow_signatures.py
  flow_signature_data.py
  flow_signature_audit.py
  directional_consensus.py
  predictive_state_data.py
  predictive_state_model.py
  predictive_state_audit.py
  metrics.py
  extraction.py
  evaluate.py
  reports.py
  schema.py
  visualize.py
  cli/
    __init__.py
    extract_prompt_flow.py
    audit_prompt_flow.py
    inspect_source_npz.py
    visualize_prompt_flow.py
    audit_geometry.py
    audit_spectral_chain.py
    audit_layer_time_geometry.py
    evaluate_layer_time_geometry.py
```

The primary flow-signature entry point is intentionally a direct script so it
works from the Linux repo root without package-mode invocation:

```bash
python audit_reasoning_flow_signatures.py \
  --input data/gsm8k_v2_custom.npz \
  --output outputs/reasoning_flow_signatures/gsm8k_custom_scores.npz \
  --output_dir outputs/reasoning_flow_signatures/gsm8k_custom_audit \
  --vector_key sv_vec_step_exp \
  --label_policy answer_format_ok \
  --compute_device cuda
```

Same-problem local geometry is audited separately because these files usually
have response labels but no first-error step annotation:

```bash
python audit_multisample_geometry.py \
  --input data/gsm8k_v2_custom.npz \
  --output outputs/multisample_geometry/gsm8k_custom_scores.npz \
  --output_dir outputs/multisample_geometry/gsm8k_custom_audit \
  --vector_key sv_vec_step_exp \
  --layers all \
  --label_policy answer_format_ok \
  --compute_device cuda
```

The fixed comparison is dynamic phase-profile support versus static geometry
support under same-problem paired AUROC. Per-layer means/maxima are explicitly
exploratory, and length-residualized scores are reported alongside every
headline geometry score.

Method and claim gates: `METHOD_MULTISAMPLE_GEOMETRY.md`.

The debiased directional-consensus follow-up uses the token clouds in the same
files and has a direct Linux entry point:

```bash
python audit_directional_consensus.py \
  --input data/gsm8k_v2_custom.npz \
  --output outputs/directional_consensus/gsm8k_custom_scores.npz \
  --output_dir outputs/directional_consensus/gsm8k_custom_audit \
  --vector_key sv_vec_step_exp \
  --cloud_layers all \
  --label_policy answer_format_ok \
  --compute_device cuda \
  --bootstrap 2000 \
  --permutations 2000
```

This audit is response-level. Its estimator, controls, stopping rule, and
replication command are frozen in `METHOD_DIRECTIONAL_CONSENSUS.md`.

The predictive-state pilot also uses a direct Linux entry point:

```bash
python audit_predictive_state.py \
  --input data/gsm8k_v2_custom.npz \
  --output outputs/predictive_state/gsm8k_custom_scores.npz \
  --output_dir outputs/predictive_state/gsm8k_custom_audit \
  --vector_key sv_vec_step_exp \
  --cloud_layers all \
  --label_policy answer_format_ok \
  --projection_dim 96 \
  --window_tokens 16 \
  --window_stride 16 \
  --horizons 1,2 \
  --latent_dim 16 \
  --compute_device cuda \
  --bootstrap 2000 \
  --permutations 2000
```

Run `--preflight` first. It reports `legacy_cloud_order` for the current
`gsm8k_v2_*` files and `exact_trace` when cloud states can be matched exactly
to stored `input_ids`. Legacy runs are state-only and cannot pass the full
gate. Method, null models, and frozen stop/go gates are documented in
`METHOD_PREDICTIVE_STATE_GEOMETRY.md`.

## File Responsibilities and Main Interfaces

### `config.py`

- `ExtractionConfig`: model-forward and metric settings.
- `MetricNames`: canonical metric name constants.

### `data.py`

- `load_chain_records(path, max_chains=0) -> list[ChainRecord]`
- `is_processbench_full(path, npz=None) -> bool`
- `is_multisample(path, npz=None) -> bool`

This module handles ProcessBench `full_*.npz` and same-problem
`*_multisample_sv.npz` records without mixing their evaluation semantics.

### `geometry.py`

- `orthonormal_basis(x, k, center=True) -> BasisResult`
- `random_basis(dim, k, rng) -> ndarray`
- `projection_energy_fraction(x, basis) -> ndarray`
- `principal_angle_distance(a, b) -> float`

These are pure NumPy utilities.  They are tested with synthetic data.

### `metrics.py`

- `compute_step_prompt_flow_metrics(...) -> dict[str, ndarray]`
- `compute_step_residual_vectors(...) -> ndarray`
- `compute_step_state_vectors(...) -> ndarray`
- `summarize_step_metrics(metric_series) -> dict[str, float]`

This is the mathematical core: build prompt/prefix/random subspaces and score
residual updates by projection fractions.  The residual vector function builds
the common step-level representation that later learned latent charts consume.

`compute_step_state_vectors` builds the whole-layer hidden-state point cloud
used by the representation-geometry audit.  It is closer to papers that study
hidden-state manifolds, intrinsic dimensionality, and neighborhood topology.

### `extraction.py`

- `build_prompt_response(problem, steps) -> tuple[str, str]`
- `extract_chain_prompt_flow(model, tokenizer, record, cfg) -> ChainExtraction`
- `pack_extractions(extractions) -> dict[str, ndarray]`
- `save_extractions(extractions, path) -> None`

This module runs teacher forcing, extracts hidden states/logits, and computes
per-step prompt-control metrics.

### `evaluate.py`

- `load_metric_npz(path) -> dict`
- `evaluate_first_error(metrics) -> dict`
- `evaluate_response(metrics) -> dict`
- `rank_first_errors(metrics, score_name) -> dict`
- `save_json(obj, path) -> None`

This module evaluates step localization and response-level scores.  It reports
position and step-length controls next to geometry metrics.  If `is_correct`
exists, response-level labels use `is_correct == 0`; otherwise they use
`gold_error_step >= 0`.

### `schema.py`

- `inspect_npz_schema(path) -> dict`

This checks whether an old `.npz` contains enough information for prompt-control
analysis.  Prompt text alone is not enough for prompt SVD; we need prompt-token
hidden states, full hidden shards, or a fresh teacher-forcing pass.

### `reports.py`

- `render_markdown(summary) -> str`
- `write_step_csv(metrics, path) -> None`

### `visualize.py`

- `write_separability_csv(metrics, out_path) -> list[dict]`
- `write_trajectory_csv(metrics, out_path, ...) -> list[dict]`
- `write_first_error_aligned_csv(metrics, out_path, ...) -> list[dict]`
- `make_plots(metrics, output_dir, ...) -> None`

This module answers the key empirical question: are the prompt-control signals
actually separable between correct and incorrect traces, and how do they evolve
over steps?

### `cli/inspect_source_npz.py`

Inspect whether an existing dataset can avoid re-extraction:

```bash
python -m prompt_control_flow.cli.inspect_source_npz data/features/full_gsm8k.npz
```

Interpretation:

- `can_reconstruct_prompt_text=true`: the question/prompt text can be recovered.
- `can_compute_prompt_svd_without_reextract=true`: prompt-token hidden states or
  full hidden shards are available.
- `needs_teacher_forcing_reextract=true`: old step vectors are insufficient for
  prompt SVD; rerun extraction.

### `cli/extract_prompt_flow.py`

Run teacher-forcing extraction:

```bash
python prompt_control_flow/cli/extract_prompt_flow.py \
  --input data/features/full_gsm8k.npz \
  --model /path/to/model \
  --output outputs/prompt_control_flow/full_gsm8k_metrics.npz \
  --layers 8,10,12,14,16,18,20,22 \
  --subspace_k 16 \
  --store_step_vectors
```

### `cli/extract_mechanisms.py`

Run the refactored residual-flow mechanism extraction framework.  This is the
preferred entry point for new experiments because it can combine hidden-only
prompt-flow, logit uncertainty, and optional ICR-style attention/residual
mismatch without saving full attention tensors to disk:

The implementation follows five explicit boundaries:

1. `replay_protocols.py` freezes observer prompts and problem grouping.
2. `data.py` loads labels/text without inferring unavailable process labels.
3. `teacher_forcing.py` owns one exact token axis and causal prediction shift.
4. `extractors.py` computes optional mechanism views from a shared cache.
5. `storage.py` validates precision and atomically commits state artifacts.

Model forward passes are currently one chain at a time. Within each chain,
selected-layer geometry and chunked output summaries run on the accelerator;
the code does not claim ragged multi-chain forward batching. This keeps the
first exact-trace implementation auditable while leaving length-bucketed model
batching as a throughput-only extension.

```bash
python extract_mechanisms.py \
  --input data/hf_datasets/ProcessBench \
  --input_format processbench_source \
  --subset gsm8k \
  --model /gz-data/models/Meta-Llama-3.1-8B-Instruct \
  --output outputs/mechanisms/gsm8k_llama31.npz \
  --layers 8,10,12,14,16,18,20,22 \
  --replay_protocol processbench_observer_chat_v1 \
  --enable_prompt_flow \
  --enable_uncertainty \
  --enable_icr \
  --store_step_state_vectors \
  --store_response_token_states \
  --state_storage_dtype float16
```

The ICR branch is opt-in via `--enable_icr` because it requires
`output_attentions=True` and forces eager attention. It is deliberately limited
to sequences no longer than `--full_attention_token_threshold` (default 1200).
There is no model-agnostic exact layerwise-attention implementation yet, so a
longer ICR sample fails before allocating the quadratic tensor instead of
silently using an approximate path. The output includes the
exact replay axis, separate process/final labels, causal pre-step and
retrospective step-state views, compact scores, and `profile_summary.json`.
Large response-token states are sharded instead of embedded in the NPZ.
The current ICR score is explicitly a routing/state-alignment proxy: it compares
attention-selected source states with the total block update and is not exact
per-head OV attribution. Artifacts record this limitation in `icr_semantics`.

### OC-GPI: output-conditioned geometry

The current geometry/logits mainline is documented in
[`METHOD_OUTPUT_CONDITIONED_GEOMETRY.md`](METHOD_OUTPUT_CONDITIONED_GEOMETRY.md)
and its frozen run order in
[`PLAN_OUTPUT_CONDITIONED_GEOMETRY.md`](PLAN_OUTPUT_CONDITIONED_GEOMETRY.md).
It asks whether internal geometry predicts future output drift and response
failure after conditioning on a rich causal logits history.

Extract a compact output trace without saving full logits:

```bash
python extract_ocgpi_traces.py \
  --input data/hf_datasets/ProcessBench \
  --input_format processbench_source \
  --subset gsm8k \
  --geometry_reference data/features/full_gsm8k.npz \
  --model /gz-data/models/Meta-Llama-3.1-8B-Instruct \
  --output outputs/ocgpi/gsm8k_output_trace.npz \
  --top_k 64 \
  --sketch_dim 64 \
  --dtype bfloat16 \
  --device cuda
```

Audit against the existing geometry artifact:

```bash
python audit_ocgpi.py \
  --trace outputs/ocgpi/gsm8k_output_trace.npz \
  --geometry data/features/full_gsm8k.npz \
  --output_dir outputs/ocgpi/gsm8k_sparse_audit \
  --compute_device cuda \
  --geometry_batch_size 32 \
  --bootstrap 2000
```

The sparse `full_*.npz` run is exploratory. A confirmatory mechanism result
requires contiguous whole-layer states, a positive future-output partial
\(R^2\), and improvement over the length-matched null.
Layer-time state tensors with identical shapes are bucketed and evaluated in
GPU batches; this changes throughput only, not the feature definitions.

### `cli/audit_geometry.py`

Append cross-fitted point-cloud geometry scores to an extracted mechanism npz:

```bash
python -m prompt_control_flow.cli.audit_geometry \
  --input outputs/mechanisms/gsm8k_llama31.npz \
  --output outputs/mechanisms/gsm8k_llama31_geom.npz \
  --output_dir outputs/mechanisms/gsm8k_llama31_geom_audit \
  --vector_key step_state_vectors \
  --folds 5 \
  --knn_k 20
```

The geometry reference is fitted on training chains only and scored on held-out
chains.  This avoids the invalid shortcut of computing one global positive vs
negative mean from all samples before evaluating separability.

The first geometry metrics are:

- `geom_boundary_proj`: projection onto the train-fold first-error vs healthy
  boundary direction.
- `geom_lid`: local intrinsic dimensionality against the healthy step-state
  cloud.
- `geom_healthy_residual`: fraction of energy outside the healthy PCA bundle.
- `geom_knn_error_frac` / `geom_knn_label_entropy`: local neighborhood
  composition and fragmentation.
- `geom_local_spec_entropy`: spectral entropy of the local healthy
  neighborhood.
- `geom_layer_nbr_instability`: cross-layer neighborhood rearrangement.
- `geom_compartment_score`: bounded diagnostic combination of the above, used
  only as a summary score.

### `cli/audit_spectral_chain.py`

Validate whole-chain spectral-manifold dynamics from either canonical
ProcessBench full data or mechanism extraction outputs.

Canonical full data, no teacher-forcing re-extraction:

```bash
python -m prompt_control_flow.cli.audit_spectral_chain \
  --input data/features/full_gsm8k.npz \
  --output outputs/spectral_chain/full_gsm8k_sd.npz \
  --output_dir outputs/spectral_chain/full_gsm8k_sd_audit \
  --vector_key step_state_vectors \
  --folds 5 \
  --modes 12 \
  --low_modes 4
```

Mechanism extraction output:

```bash
python -m prompt_control_flow.cli.audit_spectral_chain \
  --input outputs/mechanisms/gsm8k_llama31.npz \
  --output outputs/mechanisms/gsm8k_llama31_sd.npz \
  --vector_key step_state_vectors
```

The first `sd_*` diagnostics are:

- `sd_tube_dist`: distance from the held-out trajectory point to the healthy
  trajectory tube at matched normalized phase.
- `sd_spectral_leak`: fraction of diffusion-map energy outside the low modes.
- `sd_tangent_off`: fraction of the next-step update outside the local healthy
  tangent space.
- `sd_committor`: kNN estimate of entering an error basin in spectral
  coordinates.
- `sd_step_speed`: phase-local speed in spectral coordinates.
- `sd_curve_efficiency`: response-level chord/path efficiency.
- `sd_path_length_per_phase`: response-level normalized path length.
- `sd_tube_auc`, `sd_committor_auc`, `sd_leak_auc`: phase-normalized curve
  integrals, not raw length-dependent sums.

Important evaluation guardrail: `full_*.npz` supports ProcessBench
first-error localization, within-chain rank, and cross-problem response
diagnosis.  It does **not** support same-problem paired AUROC.  For that,
use `*_multisample_sv.npz` and a separate multisample response audit.

### `cli/audit_layer_time_geometry.py`

Build the current whole-layer field and immediately run its claim-driven
validation:

```bash
python prompt_control_flow/cli/audit_layer_time_geometry.py \
  --input data/gsm8k_ltg_smoke.npz \
  --output outputs/layer_time/gsm8k_ltg_v2.npz \
  --output_dir outputs/layer_time/gsm8k_ltg_v2_audit \
  --fiber_rank_mode fixed \
  --tangent_rank 6 \
  --max_transport_residual 0.35 \
  --phase_mode linear \
  --compute_backend auto \
  --compute_device cuda \
  --validation_bootstrap 2000
```

Primary V2 observables are `fiber_rank`,
`plaquette_wilson_curvature`, `plaquette_transport_residual`, and
`plaquette_reliable_wilson`.  The legacy Frobenius
`plaquette_holonomy` remains for comparison.

Run structural nulls into separate artifacts:

```bash
python prompt_control_flow/cli/audit_layer_time_geometry.py \
  --input data/gsm8k_ltg_smoke.npz \
  --output outputs/layer_time/gsm8k_ltg_phase_null.npz \
  --null_mode phase_shuffle

python prompt_control_flow/cli/audit_layer_time_geometry.py \
  --input data/gsm8k_ltg_smoke.npz \
  --output outputs/layer_time/gsm8k_ltg_id_null.npz \
  --null_mode reference_id_shuffle
```

For an existing field, rerun statistics without recomputing hidden-state
geometry:

```bash
python prompt_control_flow/cli/evaluate_layer_time_geometry.py \
  --input outputs/layer_time/gsm8k_ltg_v2.npz \
  --output_dir outputs/layer_time/gsm8k_ltg_v2_validation \
  --bootstrap 2000
```

### `cli/audit_mechanisms.py` and `cli/visualize_mechanisms.py`

Evaluate and visualize the refactored mechanism outputs:

```bash
python -m prompt_control_flow.cli.audit_mechanisms \
  --input outputs/mechanisms/gsm8k_llama31.npz \
  --output_dir outputs/mechanisms/gsm8k_llama31_audit

python -m prompt_control_flow.cli.visualize_mechanisms \
  --input outputs/mechanisms/gsm8k_llama31.npz \
  --output_dir outputs/mechanisms/gsm8k_llama31_viz
```

### `cli/audit_prompt_flow.py`

Evaluate extracted metrics:

```bash
python -m prompt_control_flow.cli.audit_prompt_flow \
  --input outputs/prompt_control_flow/full_gsm8k_metrics.npz \
  --output_dir outputs/prompt_control_flow/full_gsm8k_audit
```

### `cli/visualize_prompt_flow.py`

Generate distribution and trajectory views:

```bash
python -m prompt_control_flow.cli.visualize_prompt_flow \
  --input outputs/prompt_control_flow/full_gsm8k_metrics.npz \
  --output_dir outputs/prompt_control_flow/full_gsm8k_viz
```

Outputs include:

- `separability_summary.csv`: correct/error means, differences, AUROC.
- `trajectory_curves.csv`: correct vs error mean curves over normalized step.
- `first_error_aligned_curves.csv`: metric curves aligned to the first error.
- `trajectory_<metric>.png` and `first_error_aligned_<metric>.png`.

## First-Pass Claim Gate

Do not claim a prompt-control mechanism unless all of the following hold:

1. Prompt-control metrics beat `rel_pos` and `step_len` controls.
2. Prompt SVD beats random matched subspace.
3. Gold first-error steps rank highly within chains.
4. On multisample data, wrong responses score higher than correct responses for
   the same problem.
5. Case cards show interpretable prompt-to-prefix control shifts, not only
   late-position spikes.

## First-Error Geometry Event Audit

Before combining geometry with logits, use the direct root script to test
whether hidden-state motion itself changes around ProcessBench first errors:

```bash
python audit_first_error_geometry.py \
  --input data/features/full_gsm8k.npz \
  --hidden_dir data/hidden/gsm8k \
  --output_dir outputs/first_error_geometry/full_gsm8k \
  --modes step,token \
  --step_layers all \
  --token_layers 10,14,18,22 \
  --step_offsets=-2,-1,0,1,2 \
  --token_radius 32 \
  --device cuda \
  --bootstrap 2000 \
  --permutations 5000
```

This uses existing `stepvec` and per-token hidden shards; no extraction is
needed. It computes update norm, relative update norm, turning angle, Menger
curvature, and scale-free curvature. The primary result is an event-aligned,
matched-control, cross-fitted nuisance-residual curve, not a raw first-error
AUROC. See `METHOD_FIRST_ERROR_GEOMETRY.md` for definitions and claim gates.

## Question-Conditioned Feasible-Tangent Escape

The direct audit below tests a stronger alternative to generic curvature or
dispersion: whether a normalized reasoning update persistently leaves a
group-held-out, question-and-phase-conditioned low-rank transition space.

```bash
python audit_conditional_tangent_escape.py \
  --input data/features/full_gsm8k.npz \
  --output_dir outputs/conditional_tangent/full_gsm8k \
  --layers 8,10,12,14,16,18,20,22 \
  --device cuda \
  --bootstrap 2000
```

The canonical `full_*.npz` files already contain `stepvec`, `qvec`,
`step_token_ranges`, labels, and the legacy token-cloud summaries needed for
the conditional-space, first-error, persistence, and length-confound gates. No
re-extraction is needed for those tests. They do not contain an exact
downstream logit cotangent, so output-sensitive transverse coupling remains an
explicitly untested gate unless a `[step,layer,hidden]` cotangent array is
merged. When present, the audit separates transverse magnitude from normalized
normal/cotangent alignment and tests the latter beyond instantaneous escape.
See `METHOD_CONDITIONAL_TANGENT_ESCAPE.md` for equations, structural nulls,
schema, novelty boundary, and kill criteria.

## Relation to Prior Work

This project should not be sold as "prompt SVD is new".  Nearby work already
uses prompt-guided hidden states, residual-stream dynamics, context subspaces,
and latent geometry.  The intended empirical gap is narrower: measure whether
generated residual writes remain controlled by the original problem constraints
or drift toward self-generated prefix/template dynamics, then verify this with
matched-rank random controls, position/length controls, and trajectory plots.
