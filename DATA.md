# SMCD / NTS data map (box: `/gz-data/research/demo/data/`)

Recorded 2026-06-29 from the project owner's description. **Confirm each file with `python inspect_data.py <path>` and update the "confirmed" notes.** `data/` is gitignored (big files live only on the box).

## 1. Self-sampled · within-problem · with token clouds
> for within-problem analysis; `lam1_within.py` reads these.

| file | size | content |
|---|---|---|
| `data/gsm8k_v2_5shot.npz` | 4.7G | `sv_clouds` (layer **16**) + `problem_ids` + `is_correct(_strict)`, **2646** chains |
| `data/gsm8k_v2_custom.npz` | 8.9G | same, **3600** chains |
| `data/gsm8k_multisample_sv.npz` | 25.5G | step vectors only, **no cloud** (`11_within_problem` uses it) |
| `data/features/sampled_sv.npz` | 323M | self-sampled step vectors, no cloud |

## 2. ProcessBench · cross-problem · full hidden states
> `seq_gram.py` / `step_gram.py` / `mfoc_action.py` read these.

| file | content |
|---|---|
| `data/features/full_gsm8k.npz` (543M), `full_math.npz` (2G), `full_omnimath.npz` (2.8G) | `respcloud` + `hidden_stored` → shards + gold labels, **cross-problem** |
| `data/hidden/<subset>/<id>.npy` | per-chain **full per-token hidden**, layers **10/14/18/22**; `full_*.npz` reads full-dim through these |
| `data/features/pb_*_cloud.npz` / `pb_*_8L.npz` / `pb_*_coh.npz` … | early ProcessBench feature variants |

## 3. Prior within-problem results (small npz)
> probe **0.71** / SPE **0.68** / scalar **0.55** batch. Indexed in `demo/results_summary.md`.

`data/probe_*.npz`, `ens_*.npz`, `mc_*.npz` (SPE), `within_*.npz`, `temporal_*.npz`, `decomp_*.npz`, `frac_*.npz`, `sparse_*.npz`, `norm_*.npz`, `pairwise.npz`, `amplifier.npz`, `decouple.npz`, `probe_interpret.npz` …

## ⚠️ Recent spectral scripts print only — no npz
`lam1_within.py`, `traj_stability.py`, `seq_gram.py`, `step_gram.py`, `mfoc_action.py` print a table to the terminal and **do not save**. To persist, add a `--save` flag (TODO).

## Quick routing
- spectral scalar / lam1 **within-problem** → `data/gsm8k_v2_5shot.npz` (+`_custom`)
- **cross-problem** full geometry → `data/features/full_*.npz` + `data/hidden/`
- existing within-problem conclusions → `results_summary.md` + `data/{probe,ens,mc}_*.npz`

---

## Implications for the `nts/` evidence gates (action items)

1. **The loader is pointed at the WRONG file.** `nts/data/loader.py` + `config/data/*.yaml` currently read `data/features/processbench_gsm8k_features.npz` — an **old/early variant** (pooled `stepvec`, layers `[8,16,24,31]`, no `resultant`, no `sv_layers`/`hidden_*` keys). It is **not** in the canonical list above. → **Re-point to the cross-problem canonical: `data/features/full_*.npz` + `data/hidden/<subset>/`.**
2. **Layer 14 IS available.** The real per-token dump (`data/hidden/`) stores layers `10/14/18/22`. The prior-work layer 14 is present → revert the gates' default layer to **14** (the `→16` change was based on the wrong file).
3. **κ (`resultant`) and within-step token clouds are available** in the cross-problem `full_*.npz` (`respcloud`) and the within-problem cloud files (`sv_clouds`). So κ can be computed/used **without re-extraction** once the loader reads the right files.
4. **Do not recompute existing baselines.** Mahalanobis / intrinsic-dim / κ / probe / SPE are already in `results_summary.md` + the small npz. The gates should **benchmark NTS against those recorded numbers (probe 0.71 / SPE 0.68 / κ)**, not re-derive them. The only genuinely new signal is the tangent/normal escape decomposition.
5. **Loader rewrite pending confirmation** — run `inspect_data.py` on `full_gsm8k.npz` and a `data/hidden/gsm8k/` shard so the exact keys/shapes are known before adapting the loader to read `hidden_stored` shards (via `hidden_io.load_chain`) and pool per-token → per-step.
