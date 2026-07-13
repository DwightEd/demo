# Prompt-Controlled Residual Flow

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

The canonical method is now
[METHOD_LAYER_TIME_GEOMETRY.md](METHOD_LAYER_TIME_GEOMETRY.md).  It preserves
the full layer axis, separates LID/rank fronts from a fixed-rank connection,
and validates reliability-gated Wilson-loop curvature.  The older
[METHOD_SPECTRAL_CHAIN_DYNAMICS.md](METHOD_SPECTRAL_CHAIN_DYNAMICS.md) remains
as a one-dimensional trajectory baseline, not the current main method.

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
python -m prompt_control_flow.cli.extract_prompt_flow \
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

```bash
python prompt_control_flow/cli/extract_mechanisms.py \
  --input data/processbench/gsm8k.jsonl \
  --input_format processbench_jsonl \
  --model /path/to/Llama-3.1-8B-Instruct \
  --output outputs/mechanisms/gsm8k_llama31.npz \
  --layers 8,10,12,14,16,18,20,22 \
  --enable_prompt_flow \
  --enable_uncertainty \
  --enable_icr \
  --store_step_state_vectors
```

The ICR branch is opt-in via `--enable_icr` because it requires
`output_attentions=True` and forces eager attention.  The output includes
`step_token_ranges`, `generator`, `dataset`, compact step scores, chain scores,
and a `profile_summary.json` next to the `.npz`.

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

## Relation to Prior Work

This project should not be sold as "prompt SVD is new".  Nearby work already
uses prompt-guided hidden states, residual-stream dynamics, context subspaces,
and latent geometry.  The intended empirical gap is narrower: measure whether
generated residual writes remain controlled by the original problem constraints
or drift toward self-generated prefix/template dynamics, then verify this with
matched-rank random controls, position/length controls, and trajectory plots.
