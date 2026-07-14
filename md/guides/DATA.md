# SMCD / NTS data map (box: `/gz-data/research/demo/data/`)

## 2026-07-13 conditional feasible-tangent escape audit

Use the canonical cross-problem artifacts:

```text
/gz-data/research/demo/data/features/full_gsm8k.npz
/gz-data/research/demo/data/features/full_math.npz
/gz-data/research/demo/data/features/full_omnimath.npz
```

The audit reads `stepvec`, `sv_layers`, `qvec`, `gold_error_step`,
`problem_ids`, and `step_token_ranges`. Stored `stepcloud/resultant` and
`respcloud` are optional legacy length-audit inputs. These fields are already
present, so the conditional tangent, matched first-error, persistence, and old
direction/spectrum length audits do not require re-extraction.

The strong output-sensitivity gate is different: current canonical artifacts
do not store an exact downstream cotangent. That gate needs a record-aligned
`step_output_cotangent` object array with one `[step,layer,hidden]` tensor per
chain plus `step_output_cotangent_layers` and an exact kind label. Missing this
array must be reported as untested, not replaced with entropy or NLL.

Direct entry point and method note:

```text
audit_conditional_tangent_escape.py
prompt_control_flow/METHOD_CONDITIONAL_TANGENT_ESCAPE.md
```

## 2026-07-13 first-error geometry event audit

Use the existing canonical artifact and hidden shards; do not re-extract:

```text
/gz-data/research/demo/data/features/full_gsm8k.npz
/gz-data/research/demo/data/hidden/gsm8k/*.npy
```

Repo-relative paths on the GPU box:

```text
data/features/full_gsm8k.npz
data/hidden/gsm8k
```

The step audit reads `stepvec`, `sv_layers`, `gold_error_step`, and
`step_token_ranges`. The token audit additionally reads `hidden_files`,
`hidden_layers`, and response-token hidden shards. The ranges are legacy
inclusive absolute token indices; hidden shards start at the first response
token, and the loader performs this offset conversion explicitly.

Direct script and method note:

```text
audit_first_error_geometry.py
prompt_control_flow/METHOD_FIRST_ERROR_GEOMETRY.md
```

## 2026-07-13 ordered reasoning-flow signature inputs

Primary same-problem files:

```text
/gz-data/research/demo/data/gsm8k_v2_5shot.npz
/gz-data/research/demo/data/gsm8k_v2_custom.npz
```

The required raw trajectory key is `sv_vec_step_exp`, shaped per sample as
`(T, L, D)`. It exists only in multisample artifacts extracted with
`--store_vectors`. Run `audit_reasoning_flow_signatures.py --preflight` before
the audit; the loader refuses to guess a mismatched layer mapping.

For response-level local displacement/turning/curvature diagnostics, use
`audit_multisample_geometry.py`. This is deliberately separate from
`audit_first_error_geometry.py`: the multisample files provide final-answer
labels but normally do not provide `gold_error_step`.

```bash
python audit_multisample_geometry.py \
  --input data/gsm8k_v2_custom.npz \
  --preflight
```

The same artifacts also contain the full-dimensional token clouds needed by
the debiased directional-consensus audit:

```text
sv_clouds       object array; each response is (N, L, D)
cloud_sizes     semantic-step token counts; sum equals N
cloud_layers    actual model layer ids for the L slices
```

They also contain the exact trace fields required by the predictive-state
geometry pilot:

```text
input_ids                  exact teacher-forced model input IDs
time_axis_token_ranges     inclusive absolute ranges concatenated into sv_clouds
```

`audit_predictive_state.py --preflight` reconstructs the token ID for every
cloud row and rejects any range/cloud-size mismatch. Therefore the primary
token-ID nuisance control and reduced-rank predictive-state experiment do not
need re-extraction. The frozen method is documented in
`prompt_control_flow/METHOD_PREDICTIVE_STATE_GEOMETRY.md`.

No re-extraction is needed for this test. Verify alignment before the audit:

```bash
python audit_directional_consensus.py \
  --input data/gsm8k_v2_custom.npz \
  --vector_key sv_vec_step_exp \
  --cloud_layers all \
  --label_policy answer_format_ok \
  --preflight
```

Method and frozen evaluation protocol:
`prompt_control_flow/METHOD_DIRECTIONAL_CONSENSUS.md`.

The canonical files below use `stepvec` and support the cross-problem global
baseline, but each problem normally has one response, so they cannot provide
the same-problem support diagnostic:

```text
/gz-data/research/demo/data/features/full_gsm8k.npz
/gz-data/research/demo/data/features/full_math.npz
/gz-data/research/demo/data/features/full_omnimath.npz
```

Method and exact direct commands:
`prompt_control_flow/METHOD_REASONING_FLOW_SIGNATURES.md`.

## 2026-07-09 current ProcessBench feature inputs

Use these files for learned-latent / hidden-state reasoning experiments:

```text
/gz-data/research/demo/data/features/full_gsm8k.npz
/gz-data/research/demo/data/features/full_math.npz
/gz-data/research/demo/data/features/full_omnimath.npz
```

From repo root on the GPU box, the relative paths are:

```text
data/features/full_gsm8k.npz
data/features/full_math.npz
data/features/full_omnimath.npz
```

Do **not** use `data/full_gsm8k.npz` or
`data/processbench_gsm8k_stepvec.npz`; those are not canonical files in this
project layout.

For `latent_separatrix_audit.py`, the required fields are:

```text
stepvec              object array, each chain shaped (T, L, d)
gold_error_step      -1 for correct response, otherwise first wrong step index
problem_ids          grouping id for leakage-safe CV
steps_text           optional, used only for nuisance operation bins
step_token_ranges    optional, used to estimate step length
```

Confirmed 2026-06-29 via `inspect_data.py`. `data/` is gitignored (big files live only on the box).

## ⚠️ LABEL CONVENTION (red line, audited 2026-07-03)
`is_correct_strict` / `is_correct`: **1 = correct, 0 = error** (writer: `extract_features._pb_record`).
Ground-truth anchor: `gold_error_step < 0 ⟺ correct` (ProcessBench: -1 = all steps fine).
The 7/1 trajectory pipeline (`data_loading*.py`, `validate_phase_instability.py`, `analyze_results.py`,
`diagnose_results.py`, `validate_local.py` mock) had this INVERTED (assumed 0=correct) — fixed 2026-07-03;
**any `chain_*.pkl` caches under `data/hidden/cache/` built before that date carry inverted `is_correct`
and mis-sliced step windows (absolute-index + open-interval bug) and MUST be deleted and rebuilt.**
The `nts/` package always used `gold_error_step` and was never affected.

## ✅ CANONICAL — use these

### Cross-problem · ProcessBench · full per-token hidden + κ + qvec
`data/features/full_gsm8k.npz` (543M), `full_math.npz` (2G), `full_omnimath.npz` (2.8G) — **confirmed (full_gsm8k: 395 chains)**:
- `gold_error_step`, `is_correct(_strict)`, `problem_ids`, `steps_text` (T≈4), `step_token_ranges` (T,2)
- `stepvec` **(T, 8, 4096)** — pooled step vectors at **8** sv-layers (`sv_layers`, values via re-run of the fixed inspect)
- `qvec` **(8, 4096)** — question/prompt baseline (orientation signal feasible)
- `stepcloud` **(T, 33, 9)** + `cloud_feature_names` = **9 incl `resultant`** → **κ IS available** (all 33 layers)
- `tokgeom` (R,33,9), `respcloud` (R, 4, 256) JL-projected per-token cloud
- `hidden_stored=True` → per-token full hidden in shards ↓

`data/hidden/<subset>/<id>.npy` — **confirmed**: per-chain full per-token hidden **(R, 4, 4096) fp16**, layers **[10, 14, 18, 22]** (so **layer 14 IS present**), keyed `<subset>-<n>.npy` (e.g. `gsm8k-0.npy`). 395 shards for gsm8k. Read via `hidden_io.load_chain`.

### Within-problem · self-sampled · token clouds
`data/gsm8k_v2_5shot.npz` — **confirmed**: 2646 chains / **291 problems**; `sv_clouds` **(≈70, 1, 4096)** full-dim token cloud at **1 layer (16)**; `sv_vec_{last,mean,linear,step_exp}` (3,33,467); `whiten_baseline` present; layers_used = all 33. (`gsm8k_v2_custom.npz` 3600 chains, same.)

### Prior results (baselines — do NOT recompute)
`../progress/results_summary.md` + `data/{probe,ens,mc,within,...}_*.npz` → probe within **0.71**, SPE **0.68**, scalar **0.55**.

## ❌ DROP — useless / stale, do not use
- `data/features/processbench_gsm8k_features.npz` — **old/early variant** (pooled stepvec only, layers [8,16,24,31], 3-feature cloud, no `resultant`/`sv_layers`/`hidden_*`). This is what the loader was wrongly pointed at.
- `data/features/pb_*_cloud.npz` / `pb_*_8L.npz` / `pb_*_coh.npz` — early ProcessBench feature variants, superseded by `full_*.npz`.
- `config/data/*.yaml` currently point at `processbench_*_features.npz` → **must re-point to `full_*.npz`**.

## ⚠️ Spectral scripts print-only (no npz)
`lam1_within.py`, `traj_stability.py`, `seq_gram.py`, `step_gram.py`, `mfoc_action.py` print tables to terminal, don't save. Add `--save` to persist (TODO).

## What this means for the NTS gates
1. **Re-point loader** → `full_*.npz` (labels + κ from `stepcloud["resultant"]`) + `data/hidden/<subset>/<id>.npy` (per-token, **layer 14**). Drop `processbench_*_features.npz`.
2. **κ available** (resultant, 33 layers) → cbw region testable, no re-extraction.
3. **Per-token full hidden available** → enables **step-free cloud NTS** (see below) and within-step κ.
4. **Audit result (read 8 Gram/spectrum scripts):** none compute *off-correct-subspace energy step-free* — they're all trajectory-INTRINSIC (own eff_rank/lam1/HS/ME) or off-SEMANTIC (01_extract_spectral_field projects onto a fixed HARP/unembedding subspace, not a correct-reasoning one). So that signal is the genuine gap NTS should fill.
