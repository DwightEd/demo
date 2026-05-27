# Demo: (Step × Layer) Spectral Field Low-Rank Hypothesis Test

## What this validates

A single empirical claim:

> For each reasoning chain, build the (T × L) effective-rank matrix
>
>   $M_{j,\ell} = D_j^{(\ell)} = \exp\!\big(-\sum_i p_i \log p_i\big),\ \ p_i = (\sigma_i^{(j,\ell)})^2 / \sum_k (\sigma_k^{(j,\ell)})^2$
>
> where $\sigma_i^{(j,\ell)}$ are the singular values of the token cloud
> $\mathbf{H}_j^{(\ell)} \in \mathbb{R}^{n_j \times d}$ at step $j$, layer $\ell$.
>
> **For correct chains, $M$ is approximately low-rank** — a few principal
> components explain most of its variation. **For error chains, $M$ exhibits
> a sparse departure from that low-rank structure**, localised at a single
> $(j^*, \ell^*)$ cell.

The demo computes three observable signals from a single SVD per chain:

| Signal | Definition | Granularity | Tested label |
|---|---|---|---|
| Chain low-rankness | $\sigma_1^2 / \sum_k \sigma_k^2$ | per chain | "has any error step" |
| Step residual norm | $\|(M - L_k)_{j,:}\|_2$ | per step | "is the first-error step" |
| Layer-profile corr. | $\mathrm{corr}\!\big(M_{j,:},\, \overline{M}_{<j,:}\big)$ | per step (j ≥ 2) | "is the first-error step" |

Three AUROC numbers fall out. They decide whether the hypothesis holds and
at what granularity.

## Pipeline

```
01_extract_spectral_field.py  →  per-trajectory (T, L) matrices  (data/spectral_field.npz)
                                 channels: M_D (effective rank),
                                           M_V (spectral energy, auxiliary),
                                           M_C (top concentration, auxiliary)
                ↓
02_lowrank_analysis.py        →  three signals + AUROCs           (data/analysis.npz)
                ↓
03_plot_results.py            →  4 figures into output/
```

The pipeline is **training-free, label-free at analysis time, and runs in
seconds per trajectory once hidden states are extracted**. Each trajectory's
SVD is the only computation; no probe, no GRU, no NF.

## Files

```
demo/
├── README.md                       this file
├── requirements.txt                deps
├── .gitignore                      ignores data/, output/
├── 01_extract_spectral_field.py    HF model + ProcessBench → (T, L) matrices
├── 02_lowrank_analysis.py          per-chain SVD → 3 signals + AUROCs
├── 03_plot_results.py              violin / heatmap / decomposition figures
├── smoke_test.py                   synthetic-data validation (no GPU needed)
└── utils/
    ├── __init__.py
    ├── spectral.py                 effective rank, SVD, lowrank decompose
    └── step_boundaries.py          ProcessBench-step → token-range alignment
```

## Quick start

```bash
# 0. Sanity-check the math on synthetic data (no GPU, ~5 s)
python smoke_test.py

# 1. Extract spectral field from real model + ProcessBench (GPU recommended)
python 01_extract_spectral_field.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset Qwen/ProcessBench \
    --subset gsm8k \
    --n_correct 50 \
    --n_error 50 \
    --layers all \
    --output data/spectral_field.npz

# 2. Low-rank analysis (CPU, seconds)
python 02_lowrank_analysis.py \
    --input data/spectral_field.npz \
    --channel D \
    --rank_k 1 \
    --output data/analysis.npz

# 3. Figures
python 03_plot_results.py \
    --spectral data/spectral_field.npz \
    --analysis data/analysis.npz \
    --outdir output/
```

## What to look at

`02_lowrank_analysis.py` prints four AUROCs. The decision matrix is:

| Outcome | Interpretation | Action |
|---|---|---|
| `AUROC(chain_lowrank_k=1) ≥ 0.75` AND `AUROC(step_residual) ≥ 0.65` | Both hypotheses validated | Proceed with this as the main method |
| `AUROC(chain) ≥ 0.75` only | Chain-level signal strong, step-level weak | Reframe as trajectory-level detection; use SVD-residual for localization as a secondary feature |
| `AUROC(step) ≥ 0.65` only | Step-level localization works, chain-level weak | Possible when error chains still look "self low-rank"; switch to constrained-RPCA against a global subspace learned from correct chains |
| Both `< 0.6` | Hypothesis fails on this channel | Try `--channel V` or `--channel C`; or run on a subset of layers; or revisit the hypothesis |

`output/fig3_spectral_field_heatmaps.png` is the visual sanity check —
correct chains should show a stable layer-gradient pattern across steps;
error chains should show a color-block disruption near `τ`.

`output/fig4_lowrank_decomposition.png` shows $M$, $L_1$ and the residual $R$
side by side — the residual heatmap is where the (j*, l*) localization lives.

## Design notes

- All tokens in a step are kept (not just the last token); the spectral
  signal is in the cloud, not in any single hidden state.
- Layer 0 = embedding output; layers 1..L = transformer block outputs.
  By default all layers are sampled; pass e.g. `--layers 0,8,16,24,30,31`
  for a quick check.
- Three channels (D, V, C) are extracted in one pass; analysis defaults to
  D (effective rank) but you can re-run `02_lowrank_analysis.py` with
  `--channel V` or `--channel C` without re-extracting.
- Step labels follow ProcessBench convention: `label = -1` means all
  correct; `label ≥ 0` is the (0-indexed) first-error step. Trajectories
  whose first-error step is truncated by `--max_seq_len` are dropped.
- `data/` and `output/` are gitignored. Re-extraction overwrites; analysis
  and figures are derived and may be re-generated freely.
