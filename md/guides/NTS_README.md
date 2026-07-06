# NTS Evidence Gates (`nts/`)

> **UPDATE 2026-06-29 (supersedes the gsm8k-feature notes below).** The loader now reads the
> canonical cross-problem data — `data/features/full_*.npz` (labels + `resultant`/κ + `qvec`) +
> `data/hidden/<subset>/<id>.npy` (full per-token hidden, layers `[10,14,18,22]`, **analysis layer 14**),
> via `load_full_table`. Two NTS variants are kept: **(1) step-free cloud** — `nts_cloud` signal =
> off-correct-subspace energy of the token cloud (`geom/subspace.py`), evaluated chain-level
> within-problem by **`gate_cloud`** (vs `is_correct`, benchmarked against recorded probe 0.71 /
> SPE 0.68); **(2) step-displacement** — the original `nts` signal + `gate0..3`. See `DATA.md`.

Post-hoc geometric falsification gates over **already-extracted** ProcessBench hidden states (no model/GPU). Mirrors the `hallucination-detection` architecture: `core` (Registry + `StepTable`/`ChainData` + `GeomCfg`), `data` (npz→table loader), `geom` (PCA-whiten reducer, kNN bank, local-PCA tangent/normal decomposition, TwoNN), `signals` (`BaseSignal` + `@SIGNALS.register`: `nts`, `rema`, `kappa`, `mahalanobis`), `eval` (AUROC/bucket/residualize/bootstrap), `gates` (`BaseGate.run→GateResult`). One Hydra entry: `scripts/run_gate.py`.

**Hypothesis under test (NTS):** a reasoning error = step displacement Δh **escaping off the manifold of correct reasoning** (large *normal* component to the local tangent space), while hard-but-correct reasoning stays *tangent*. Gates 0–3 try to **kill** this cheaply; **Gate 2** is decisive (NTS-resid must beat the isotropic REMA baseline after triple residualization).

## Run
```bash
python check_stepvec.py data/features/processbench_gsm8k_features.npz   # inspect schema
python scripts/run_gate.py gate=gate0 data=gsm8k                        # honest Mahalanobis floor
python scripts/run_gate.py gate=gate1 data=gsm8k                        # ★ estimability + null control
python scripts/run_gate.py gate=gate2 data=gsm8k                        # ★★ NTS vs REMA (命门)
python scripts/run_gate.py gate=gate3 data=gsm8k                        # curvature debiasing
# override layer/params: ... geom.layer=8 geom.k=64
```
Each gate prints a `KILL?` line and writes `outputs/nts_gates/<gate>_<subset>_L<layer>.json`. Unit tests: `python -m pytest tests -q` (pure numpy+sklearn, no model needed; 11 tests).

## Feature-file contract (what `data/loader.py` expects)
`data/features/processbench_<subset>_features.npz`, loaded `np.load(..., allow_pickle=True)`, per-chain object arrays.
- **Required:** `stepvec` (T,n_sv,d raw pooled step vectors), `gold_error_step` (−1=correct, else first-error step), `problem_ids`, `step_token_ranges` (T,2), `steps_text`, `layers_used`.
- **Optional:** `stepcloud`+`cloud_feature_names` containing `"resultant"` → enables κ (else κ=NaN, cbw region skipped); `sv_layers` → exact stepvec layer mapping (else assumed == `layers_used`); `qvec` → enables a future orientation signal.

## Inspected file — `processbench_gsm8k_features.npz` (recorded 2026-06-29)

| Field | Value / shape | Note |
|---|---|---|
| chains | **395** (correct **190** / error **205**) | one chain per problem (395 problems) |
| `stepvec` | `(T, 4, 4096)`, T≈4 | 4 stored layers; gsm8k chains are short |
| stored layers | `layers_used = [8, 16, 24, 31]` | `sv_layers` ABSENT → loader assumes stepvec layers == `layers_used` (n_sv=4 matches) |
| `stepcloud` | `(T, 4, 3)` = `cloud_D, cloud_V, cloud_C` | **no `resultant`** → κ not available here |
| `stepgeom` | `(T, 4, 9)` | `norm, pr, ae, ed_half, e50, e90, ae_robust, anom_k5, anom_k10` — alternate signals |
| `qvec` | **present** | question/prompt baseline vector → orientation signal feasible |
| `chain_intrinsic` | present (+`intrinsic_names`) | precomputed per-chain intrinsic dim |
| uncertainty | `tok_U_D`, `tok_U_C`, `tok_U_E` present | entropy/EDIS channels |
| model | Llama-3.1-8B-Instruct | `source_tag`/`pb_subset` mark provenance |

## Implications for the gates (read before running)
1. **Analysis layer:** prior κ work referenced **layer 14**, which is **not stored** here. Available = `{8, 16, 24, 31}`. Default `config/geom/default.yaml` set to **16** (nearest to 14). Confirm the best layer from Gate 1's per-layer TwoNN ID curve; override with `geom.layer=8|24|31`.
2. **κ unavailable** (no `resultant`): Gate 0/1 run fully; Gate 2's **ALL-steps NTS-vs-REMA test runs** (still the core decisive comparison); Gate 2/3's **coherent-but-wrong sub-region is auto-skipped**. To test the cbw region, re-extract with the `resultant` cloud feature (`--cloud_eff_rank` or equivalent), or designate an available concentration proxy.
3. **`qvec` is present** — contrary to earlier project notes that it was un-extracted. The orientation / goal-alignment signal (`cos(step direction, qvec)`) is now buildable; not yet implemented as a gate signal.

> This documents the **data contract and one inspected file**, not experimental findings. Gate verdicts (the `KILL?` outputs) are recorded separately once run on the server.
