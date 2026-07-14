# Cross-Dataset Transition Result (2026-07-14)

## Run

Frozen layer-14 validation on ProcessBench GSM8K, MATH, and OmniMath using
`full_gsm8k.npz`, `full_math.npz`, and `full_omnimath.npz`.

## Stable Baseline

The supervised OOF `anchor_uncertainty` model reached step-level first-error
AUROC `0.811`, `0.781`, and `0.809`. The broader sequence model reached
`0.790`, `0.779`, and `0.816`, for increments `-0.021`, `-0.002`, and `+0.007`.
None supported promoting the broad sequence feature stack.

This baseline combines spread, anchor loss, uncertainty, step length, and
relative position. It must not be described as question-vector similarity
alone.

## Frozen Scalar Replication

| Signal | Raw macro/min AUROC | Length-position residual macro/min | Raw gate | Residual gate |
|---|---:|---:|---:|---:|
| `d_spread` | 0.662 / 0.638 | 0.540 / 0.504 | CI | direction only |
| `spread` | 0.726 / 0.702 | 0.600 / 0.521 | CI | direction only |
| `step_direction_jump` | 0.517 / 0.500 | 0.560 / 0.526 | direction only | direction only |
| spread CUSUM | 0.518 / 0.491 | 0.555 / 0.528 | fail | CI |
| joint CUSUM | 0.593 / 0.579 | 0.637 / 0.628 | CI | CI |
| spread surprise | 0.521 / 0.461 | 0.579 / 0.560 | fail | CI |
| joint surprise | 0.609 / 0.574 | 0.647 / 0.621 | CI | CI |

`joint` denotes a healthy-transition model over spread, anchor loss, and
uncertainty. `CI` means that every dataset-level cluster-bootstrap interval
has a lower endpoint above chance in the predeclared direction.

## Supported Interpretation

The evidence does not support an unconditional claim that errors are simply
more dispersed. Raw spread and its change lose most of their discrimination
after removing step length and position.

The strongest replicated scalar is instead a conditional innovation. A ridge
transition model is fit on correct chains:

\[
z_t=A z_{t-1}+B c_t+b+\varepsilon_t,
\qquad
z_t=[\operatorname{spread}_t,\operatorname{anchorloss}_t,
\operatorname{uncertainty}_t],
\]

with controls

\[
c_t=[\log(1+N_t),\operatorname{relpos}_t].
\]

Its surprise is the covariance-normalized prediction residual

\[
r_t=\varepsilon_t^\top\Sigma_{\varepsilon}^{-1}\varepsilon_t.
\]

The current result supports this limited claim:

> First-error steps reproducibly violate a healthy joint transition law more
> than they exhibit unconditional geometric dispersion.

## Not Yet Supported

- The transition signal has not yet shown AUROC or AUPRC increment over the
  full `anchor_uncertainty` baseline.
- The result does not prove that the complete hidden-state trajectory lies on
  a low-dimensional manifold.
- `step_direction_jump` is a change between pooled step representations, not
  a Transformer layerwise residual-stream update.
- Online alarms recover only about `0.30-0.37` of error chains near
  `0.10-0.12` false-alarm rate and are not deployment ready.
- The best high-spread and localization rows are selected from many features
  and remain exploratory.

## Decisive Next Gate

The rerun must compare the full baseline against adding exactly one fixed
transition score on identical OOF folds. A mechanism is promoted only if both
AUROC and AUPRC increments replicate with positive paired confidence intervals
on all three datasets. It also tests the pooled 12-to-14 residual-state update
and a prompt-depth-drift-subtracted version. These depth-band scores are marked
as approximations rather than exact per-block writes. The implementation
writes:

```text
mechanism_component_ablation.csv
transition_additive_value.csv
```
