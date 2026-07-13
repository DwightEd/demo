# Same-Problem Multisample Geometry Audit

## Question and Claim Boundary

This audit tests one response-level hypothesis:

> For the same problem, incorrect reasoning trajectories have a different
> local hidden-state geometry from correct trajectories, and the temporal
> shape of that geometry contains information beyond static magnitude and
> response length.

The input files contain final-answer labels but normally do not contain
`gold_error_step`. Therefore this method cannot establish where the first error
occurs. Its target is same-problem response ranking.

## Input

The canonical inputs are:

```text
data/gsm8k_v2_5shot.npz
data/gsm8k_v2_custom.npz
```

The raw trajectory key is `sv_vec_step_exp`. For sample (i), it stores

\[
h_{i,t,\ell}\in\mathbb{R}^{D},
\qquad t=0,\ldots,T_i-1.
\]

`problem_ids` groups independently sampled responses to the same question.
The default `answer_format_ok` policy removes format failures before defining
error labels from `is_correct`.

## Local Geometry

For each transition, the audit computes

\[
\Delta h_{t,\ell}=h_{t,\ell}-h_{t-1,\ell},
\qquad
v_{t,\ell}=\lVert\Delta h_{t,\ell}\rVert_2,
\]

and the scale-normalized displacement

\[
\widetilde v_{t,\ell}=
\frac{\lVert\Delta h_{t,\ell}\rVert_2}
{\tfrac12(\lVert h_{t,\ell}\rVert_2+\lVert h_{t-1,\ell}\rVert_2)}.
\]

For an interior state, it also computes the turning angle

\[
\theta_{t,\ell}=\arccos
\frac{\langle\Delta h_{t,\ell},\Delta h_{t+1,\ell}\rangle}
{\lVert\Delta h_{t,\ell}\rVert_2\lVert\Delta h_{t+1,\ell}\rVert_2},
\]

the three-point Menger curvature

\[
\kappa_{t,\ell}=
\frac{2\sin\theta_{t,\ell}}
{\lVert h_{t+1,\ell}-h_{t-1,\ell}\rVert_2},
\]

and its scale-free version obtained by multiplying by the mean adjacent edge
length.

These are descriptive Euclidean quantities. They are not treated as proof
that the hidden states lie on a Riemannian manifold.

## Dynamic Versus Static Test

Each ragged trajectory is aligned to normalized reasoning phase

\[
\tau_t=\frac{t}{T_i-1}\in[0,1]
\]

and linearly interpolated onto a fixed phase grid. This produces a profile

\[
g_i(\tau,\ell,m),
\]

where (m) indexes the five local geometry channels.

The fixed dynamic representation is the complete phase profile over all
selected layers and channels. The static control contains only per-chain mean,
maximum, standard deviation, and late-phase mean. Per-layer scalar searches
are saved separately and labeled exploratory.

## Correct-Support Energy

Two support regimes are evaluated.

1. `global`: correct trajectories from training problems define a robust
   phase-conditioned center and scale; target problems are held out.
2. `support`: labeled correct samples of the target problem define a
   problem-conditioned center. Correct candidates are leave-one-out.

For the same-problem diagnostic, the phase energy is

\[
E_i(\tau)=\frac{1}{F}\sum_{f=1}^{F}
\left(
\frac{g_{i,f}(\tau)-\mu_{p,-i,f}(\tau)}{s_f(\tau)}
\right)^2.
\]

The robust scale (s_f(\tau)) is fitted only from correct training-problem
trajectories. `support` is an oracle diagnostic because it uses correctness
labels from other samples of the target problem; it is not a single-sample
online detector.

## Evaluation

The headline metric is same-problem paired AUROC:

\[
\Pr\left(s(x^-_p)>s(x^+_p)\mid p\right),
\]

micro-averaged over all correct/incorrect response pairs and accompanied by a
problem bootstrap confidence interval and within-problem label-permutation
test.

Every headline geometry score is also residualized, using problem-grouped
cross-fitting, against

\[
\log(1+T_i),\qquad \log(1+\text{response characters}_i).
\]

The dynamic-minus-static AUROC difference is computed only on samples finite
for both scores. This prevents unequal coverage from masquerading as a dynamic
increment.

## Decision Gates

Continue the geometry branch only if all of the following hold:

1. dynamic support has useful same-problem AUROC with adequate coverage;
2. the bootstrap interval for dynamic-minus-static AUROC is strictly positive;
3. the increment remains after length residualization;
4. the global score is tested separately from the oracle support score;
5. any selected layer/metric replicates on the other prompt regime.

If only static support works, the result is geometric level separation rather
than trajectory-shape separation. If only oracle support works, the geometry
is problem-conditioned but not yet deployable.

## Commands

Preflight:

```bash
python audit_multisample_geometry.py \
  --input data/gsm8k_v2_custom.npz \
  --preflight
```

Primary run:

```bash
python audit_multisample_geometry.py \
  --input data/gsm8k_v2_custom.npz \
  --output outputs/multisample_geometry/gsm8k_custom_scores.npz \
  --output_dir outputs/multisample_geometry/gsm8k_custom_audit \
  --vector_key sv_vec_step_exp \
  --layers all \
  --label_policy answer_format_ok \
  --phase_points 16 \
  --compute_device cuda \
  --bootstrap 2000 \
  --permutations 2000
```

Run the same frozen configuration on `gsm8k_v2_5shot.npz` as replication.
