# Cross-Dataset Replication of Existing Reasoning Signals

## Scope

This audit does not introduce a new detector. It asks whether the already
implemented divergence and transition signals retain the same error direction
on all three canonical ProcessBench subsets:

```text
data/features/full_gsm8k.npz
data/features/full_math.npz
data/features/full_omnimath.npz
```

No extraction is required. The audit reuses `stepcloud`, `stepvec`, `qvec`,
token uncertainty traces, `gold_error_step`, and `step_token_ranges` through
`chain_dynamics_audit.py`.

## Frozen Signals

The following signals are fixed before examining the new cross-dataset table:

- `spread`: within-step directional dispersion;
- `d_spread`: one-step increase in dispersion;
- `step_direction_jump`: change in the pooled hidden-state direction;
- `transition_surprise__spread`: deviation from a correct-chain transition
  model using spread only;
- `transition_cusum__spread`: persistent accumulation of that deviation;
- `transition_surprise__spread_anchor_unc`: transition deviation using spread,
  question-anchor loss, and uncertainty;
- `transition_cusum__spread_anchor_unc`: its persistent version.

The expected direction is fixed as **higher means more likely to be the first
error**. The audit never flips a signal independently on GSM8K, MATH, or
OmniMath.

## Labels and Controls

For chain (i) with first-error step (g_i), candidate steps are restricted to
the correct prefix and first error:

\[
y_{i,t}=\mathbb{1}[t=g_i],
\qquad
t\le g_i,
\]

while every step in a fully correct chain is a negative. Post-error steps are
excluded.

Each signal is reported raw and after cross-fitted nuisance removal using only
non-error training steps:

\[
s^{\perp}_{i,t}
=s_{i,t}-\widehat{\mathbb E}
\left[s_{i,t}\mid \log(1+N_{i,t}),\operatorname{relpos}_{i,t}\right].
\]

The held-out chain is never used to fit its nuisance baseline. Confidence
intervals resample complete problem/chain clusters rather than flattened
steps.

## Replication Gates

Two gates are deliberately separated:

1. **Directional consistency:** the fixed-direction AUROC is above (0.5) on
   every requested dataset.
2. **CI replication:** the lower endpoint of the cluster-bootstrap 95% CI is
   above (0.5) on every requested dataset.

The second is the confirmatory result. A good macro average with one failed
dataset does not count as universal replication.

Within-chain localization is also reported with the same fixed direction:

\[
\operatorname{Top1}
=\Pr\left(s_{i,g_i}=\max_{t\le g_i}s_{i,t}\right),
\]

alongside the per-chain uniform-random expectation. This prevents a
cross-problem difficulty effect from being mistaken for first-error
localization.

## Command

Run the frozen layer-14 replication on the GPU server:

```bash
cd /gz-data/research/demo
python mainline_validation_suite.py \
  --datasets gsm8k,math,omnimath \
  --layers 14 \
  --data_dir data \
  --folds 5 \
  --n_boot 2000 \
  --top 50 \
  --output_dir outputs/cross_dataset_replication_l14
```

The main outputs are:

```text
outputs/cross_dataset_replication_l14/mainline_validation_summary.md
outputs/cross_dataset_replication_l14/mainline_validation_summary.json
outputs/cross_dataset_replication_l14/cross_dataset_replication.csv
outputs/cross_dataset_replication_l14/mechanism_component_ablation.csv
outputs/cross_dataset_replication_l14/transition_additive_value.csv
```

Layer 14 is primary because it was used in the earlier GSM8K audit. A later
layer sweep is a robustness analysis, not a replacement for this frozen run.

## Additive-Value Audit

`anchor_uncertainty` is a shorthand for a supervised OOF logistic model, not a
single anchor cosine. Its input is

\[
x_t=
\left[
\log(1+N_t),\operatorname{relpos}_t,
\operatorname{spread}_t,
1-\cos(h_t,q),
\operatorname{uncertainty}_t
\right].
\]

Here, `qvec` is the exponentially pooled representation of all prompt tokens
and `stepvec` is the same pooling over one reasoning step. The component audit
removes spread, anchor loss, or uncertainty one at a time while preserving the
same candidate rows, GroupKFold splits, fold-local imputation, scaling, and
classifier capacity.

The additive audit then compares the full baseline with a model that adds
exactly one frozen temporal signal:

\[
\Delta\operatorname{AUROC}
=
\operatorname{AUROC}(x_t,s_t)
-
\operatorname{AUROC}(x_t).
\]

For each signal, both models are refit on the identical finite-support rows.
Undefined first-step transition values are excluded rather than imputed, and
the resulting coverage is reported.

Both AUROC and AUPRC deltas use paired cluster bootstrap intervals. The signal
has cross-dataset incremental value only if every requested dataset has a
lower confidence endpoint above zero. This distinguishes three claims that
must not be conflated:

1. a score is a useful standalone detector;
2. a score contains information beyond the static baseline;
3. a score measures Transformer residual-stream updates.

`d_spread` and `step_direction_jump` test temporal changes between pooled
reasoning-step states at one layer. They are not layerwise Transformer
residual updates. For the third claim, the audit uses adjacent stored
`stepvec` layers. At primary layer 14 this is the cumulative 12-to-14 update

\[
\Delta^{12\rightarrow14}_t=h^{14}_t-h^{12}_t.
\]

It also removes the prompt's own depth drift:

\[
\Delta^{\mathrm{pc}}_t=
(h^{14}_t-q^{14})-(h^{12}_t-q^{12}).
\]

Their scale-normalized norms are added one at a time to the full baseline.
This is a pooled two-block-band approximation available in the existing NPZ,
not an exact tokenwise write from one Transformer block. Exact per-block
attribution would require contiguous hidden states or component hooks.
