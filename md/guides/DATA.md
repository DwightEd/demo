# SMCD / NTS data map (box: `/gz-data/research/demo/data/`)

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
