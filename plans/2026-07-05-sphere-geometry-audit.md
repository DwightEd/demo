# Sphere Geometry Audit: Directional Hidden-State Hypothesis

## Mechanism Hypothesis

This branch does not claim that the residual stream is literally a unit
hypersphere.  The operational claim is narrower:

```text
Transformer blocks read normalized states through LayerNorm/RMSNorm, so the
direction distribution of hidden states may be a more mechanistic first-order
object than raw Euclidean magnitude.
```

Paper-facing version:

```text
Reasoning steps are directional distributions on a normalized representation
sphere. Errors occur not simply when the distribution becomes diffuse, but when
anchor-conditioned directional transport becomes ungrounded, structurally split,
or confidently attached to a wrong direction.
```

## Code Added

```text
sphere_geometry_audit.py
```

The audit turns the hypersphere analogy into falsifiable gates:

| gate | question | metrics |
|---|---|---|
| direction dominance | Does unit direction beat norm-only? | `unit_sphere` vs `norm_only` |
| vMF proxy | Is existing `resultant/kappa` a mean-resultant signal? | `unit_resultant`, `unit_spread`, `vmf_kappa_hat` |
| beyond one vMF | Is there split / multimodal structure beyond one mean direction? | `shape_top1_frac`, `shape_eff_rank`, `two_vmf_gain_bal`, `bipolarity` |
| mechanism hygiene | Does the result survive existing strong signals? | `U_D_mean`, `anchor_loss_unitmean`, `logN`, `pos` controls |

Important implementation choice:

```text
`norm_only` contains only token norm mean/std/CV plus logN/pos.
`raw_mean_norm` is not norm-only because it already mixes in directional
concentration: diffuse directions lower the raw mean norm even if token norms
are unchanged.
```

## Local Validation

Local machine has no real `data/features/full_*.npz` or hidden shards, so only a
synthetic selftest was run.  This is implementation validation, not research
evidence.

Commands:

```bash
python -m py_compile sphere_geometry_audit.py
python sphere_geometry_audit.py --selftest --folds 3 --n_boot 50 --output_dir outputs/sphere_geometry_selftest
```

Selftest summary:

```text
rows 617 | err 90

OOF groups:
  controls                   AUROC 0.690
  norm_only                  AUROC 0.689
  raw_magnitude_geometry     AUROC 0.824
  unit_sphere                AUROC 0.824
  unit_plus_entropy          AUROC 0.970
  anchor_uncertainty_like    AUROC 1.000
  shape_over_spread          AUROC 0.846
  shape_anchor_entropy       AUROC 1.000

OOF increments:
  raw_magnitude_geometry over norm_only      +0.135 SIG
  unit_sphere over norm_only                 +0.135 SIG
  unit_plus_entropy over unit_sphere         +0.146 SIG
  anchor over unit_plus_entropy              +0.030 SIG
  shape_over_spread over unit_sphere         +0.022 SIG
  shape_anchor_entropy over anchor baseline  +0.000 ns
```

## Result Analysis

1. The selftest confirms the implementation can separate pure token norm from
   unit-direction geometry.  `norm_only` is weak while `unit_sphere` is strong.
2. `unit_resultant`, `unit_spread`, and `vmf_kappa_hat` move together, so the
   current `kappa/resultant` should be described as a vMF mean-resultant proxy,
   not as a full vMF fit.
3. Shape / mixture features add over unit spread in the synthetic split-state
   setting, but do not add over the full anchor+entropy baseline.  This matches
   the project constraint: a pretty geometric feature is not enough unless it
   beats current effective signals.
4. The synthetic anchor signal is intentionally strong and should not be
   interpreted as evidence for real data.

## Real-Data Decision Rules

Run the audit on GSM8K, MATH, and OmniMath.  Interpret outcomes as follows:

```text
A. unit_sphere > norm_only and raw_plus_unit does not beat unit_sphere
   -> normalized directional geometry is justified; norm is not the main
      mechanism.

B. unit_sphere ~= precomputed_resultant/spread
   -> existing spread/resultant is confirmed as directional concentration,
      not an implementation accident.

C. shape/mix beats anchor_uncertainty_like in high-spread or low-entropy subsets
   -> upgrade from single-kappa to anchor-conditioned mixture-vMF / spherical
      shape mechanisms.

D. shape/mix does not beat anchor_uncertainty_like
   -> hypersphere is useful explanatory language, but not a new method mainline.
```

## Follow-Up Research Direction

If C holds, the next mechanism story should not be "errors are more diffuse".
It should be typed:

```text
diffuse drift              -> low resultant / high spread
wrong-anchor concentration -> high concentration but low q/prompt alignment
split-state reasoning      -> high two-vMF gain / bipolarity
```

Each type can map to a different intervention:

```text
diffuse drift              -> local constraint replay
wrong-anchor concentration -> anchor re-injection
split-state reasoning      -> bridge proof or branch selection
```

If C fails, stop optimizing spherical shape scalars and redirect effort toward
attention lookback, real prompt-anchor hidden banks, verifier traces, or
intervention actuators.

## Optimization Suggestions

1. First run layer 14 on GSM8K/MATH/OmniMath.
2. Only expand to layers 10/14/18/22 if layer 14 shows a real increment.
3. Report residualized versions over `U_D_mean`, `logN`, and `pos` before making
   mechanism claims.
4. For paper utility, prioritize high-spread and low-entropy/confident-wrong
   subsets over marginal global AUROC gains.

## Remote GPU Commands

```bash
cd /gz-data/research/demo
git pull

for d in gsm8k math omnimath; do
  python sphere_geometry_audit.py \
    --dataset $d \
    --data_dir /gz-data/research/demo/data \
    --hidden_dir /gz-data/research/demo/data/hidden \
    --layer 14 \
    --folds 5 \
    --n_boot 200 \
    --output_dir outputs/sphere_geometry_l14
done
```

Layer sweep, only if L14 is promising:

```bash
for d in gsm8k math omnimath; do
  for l in 10 14 18 22; do
    python sphere_geometry_audit.py \
      --dataset $d \
      --data_dir /gz-data/research/demo/data \
      --hidden_dir /gz-data/research/demo/data/hidden \
      --layer $l \
      --folds 5 \
      --n_boot 200 \
      --output_dir outputs/sphere_geometry_layers
  done
done
```
