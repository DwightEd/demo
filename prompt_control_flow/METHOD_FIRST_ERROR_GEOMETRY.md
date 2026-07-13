# First-Error-Aligned Hidden-State Geometry

## Purpose

This audit tests a narrow empirical hypothesis before building another detector:

> If a first reasoning error corresponds to a local loss of dynamical coherence,
> hidden-state motion should exhibit a reproducible, layer-dependent kinematic
> event near the first-error boundary after position and length effects are
> removed.

The analysis is deliberately descriptive and falsifiable. It does not assume
that geometric quantities predict correctness, and it does not combine them
with logits until their independent signal has been established.

## State and Geometry

Let (z_t^{(\ell)}\in\mathbb{R}^d) be the representation at step or token
(t), layer (\ell). Define the incoming update

\[
\Delta z_t^{(\ell)}=z_t^{(\ell)}-z_{t-1}^{(\ell)}.
\]

The audit computes five fields:

\[
v_t^{(\ell)}=\lVert\Delta z_t^{(\ell)}\rVert_2,
\qquad
\widetilde v_t^{(\ell)}=
\frac{\lVert\Delta z_t^{(\ell)}\rVert_2}
{\tfrac12(\lVert z_t^{(\ell)}\rVert_2+\lVert z_{t-1}^{(\ell)}\rVert_2)}.
\]

\[
\theta_t^{(\ell)}=
\arccos\!\left(
\frac{\langle\Delta z_t^{(\ell)},\Delta z_{t+1}^{(\ell)}\rangle}
{\lVert\Delta z_t^{(\ell)}\rVert_2
 \lVert\Delta z_{t+1}^{(\ell)}\rVert_2}
\right).
\]

\[
\kappa_t^{(\ell)}=
\frac{2\sin\theta_t^{(\ell)}}
{\lVert z_{t+1}^{(\ell)}-z_{t-1}^{(\ell)}\rVert_2},
\qquad
\overline\kappa_t^{(\ell)}=
\kappa_t^{(\ell)}
\frac{\lVert\Delta z_t^{(\ell)}\rVert_2+
      \lVert\Delta z_{t+1}^{(\ell)}\rVert_2}{2}.
\]

Here (\kappa_t) is Menger curvature. It is not the earlier directional
concentration statistic that happened to use the same symbol.

## Two Resolutions

### Step axis

`stepvec` has shape `(T, L, D)`. Offset `0` denotes the gold first-error step.
The value (\lVert\Delta z_t\rVert) at offset `0` is the update entering that
step. This directly reproduces the step-level construction used by
Reasoning-Flow, but tests it against ProcessBench labels.

### Token axis

The existing hidden shards have shape `(R, L, D)`. The event is the first token
of the gold first-error step, obtained from the inclusive
`step_token_ranges`. This version avoids step pooling and exposes whether a
step-level result was only a token-count or semantic-boundary artifact.

Turning angle and curvature at offset `0` need (z_{t+1}), so they are
diagnostic signals available one state later, not strict pre-error alarms.

## Matched Counterfactual Event

Every erroneous response is matched to a pseudo-event in a correct response.
The Hungarian assignment sees only nuisance variables:

- number of steps;
- relative event position;
- event-step token length;
- same-problem identity when such a correct response exists.

No hidden-state geometry or label-derived score enters the match cost. If
errors outnumber correct chains, nearest-control reuse is retained but marked
in `matched_pairs.csv`.

## Cross-Fitted Nuisance Removal

For every layer and geometric field, a ridge nuisance model is fitted on
correct training chains only and evaluated out of fold. The step model uses

\[
(r_t,r_t^2,\log(1+n_t),\log(1+n_{t-1}),
\log(1+n_{t+1}),\log(1+T)),
\]

where (r_t) is relative step position and (n_t) is step length. The token
model uses relative token position, containing-step length, within-step
fraction, and response-token count. The batched geometry and nuisance solves
run through PyTorch on the selected device. Same-problem records and every
matched error/control pair are assigned to the same held-out fold, so both
sides of a paired contrast use the same nuisance model and the matching
control cannot leak into the error event's baseline.

The primary statistic is the paired error-minus-control residual at each
layer and event offset. Reports include paired standardized effect,
bootstrap confidence interval, matched-event AUROC, sign-flip permutation
test at offset `0`, Benjamini-Hochberg correction, and pair coverage.

First-error AUROC/rank is secondary because it can still benefit from the
causal ordering of steps even after nuisance adjustment.

## Existing Data

No teacher-forcing re-extraction is required.

```text
data/features/full_gsm8k.npz
  stepvec: per-chain (T, 8, 4096)
  sv_layers: [8, 10, 12, 14, 16, 18, 20, 22]
  gold_error_step: -1 for correct, otherwise first wrong step
  step_token_ranges: inclusive absolute token ranges

data/hidden/gsm8k/*.npy
  response-token hidden states: (R, 4, 4096)
  hidden_layers: [10, 14, 18, 22]
```

## Direct Commands

Step event audit:

```bash
python audit_first_error_geometry.py \
  --input data/features/full_gsm8k.npz \
  --output_dir outputs/first_error_geometry/full_gsm8k \
  --modes step \
  --step_layers all \
  --step_offsets=-2,-1,0,1,2 \
  --device cuda \
  --batch_size 32 \
  --bootstrap 2000 \
  --permutations 5000
```

Token event audit using the already saved shards:

```bash
python audit_first_error_geometry.py \
  --input data/features/full_gsm8k.npz \
  --hidden_dir data/hidden/gsm8k \
  --output_dir outputs/first_error_geometry/full_gsm8k \
  --modes token \
  --token_layers 10,14,18,22 \
  --token_radius 32 \
  --device cuda \
  --batch_size 16 \
  --bootstrap 2000 \
  --permutations 5000
```

Use `--modes step,token` to run both. Use `--preflight` to verify paths,
layers, labels, matches, and skipped records without computing geometry.

## Outputs

Each `step/` or `token/` directory contains:

- `event_curves.csv`: layer-by-offset raw and nuisance-residualized statistics;
- `first_error_discrimination.csv`: secondary within-trajectory localization;
- `matched_pairs.csv`: auditable event matching and reuse flags;
- `geometry_audit.npz`: raw fields, residual fields, events, layers, and matches;
- `event_effect_heatmaps.png`: layer-by-offset paired effect maps;
- `event_curve_*.png`: error versus matched-control event curves per layer;
- `summary.md` and `summary.json`: coverage-aware headline results.

The hypothesis receives support only if a signal survives nuisance removal,
has high pair coverage, appears at a coherent depth band, and is reproduced on
the token axis. Otherwise the correct conclusion is that this simple local
geometry does not add a reliable correctness signal.
