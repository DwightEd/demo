# Spectral Chain Dynamics

This document is the implementation-aligned method draft for the current
`prompt_control_flow` project. It replaces the earlier speculative
HoloReason / Reasoning Spectral Field drafts. Those older drafts proposed
holonomy, curvature fields, and broader spectral surfaces; the current code
implements a narrower and testable method: cross-fitted spectral-manifold
dynamics over complete reasoning chains.

## One-Paragraph Story

Reasoning failure should not be diagnosed by asking whether a single step has a
large hidden-state norm, a high spread value, or a late position. A reasoning
trace is a trajectory through the model's hidden representation space. Correct
traces should remain compatible with a healthy low-dimensional trajectory tube:
their motion is phase-aligned with previous healthy chains, concentrated in
low-frequency manifold coordinates, and locally tangent to the healthy flow.
Wrong traces may still be smooth or confident, but they should either leave the
healthy tube, leak energy into higher spectral modes, move off the local
healthy tangent bundle, or enter a neighborhood whose training trajectories
mostly belong to error basins. We therefore learn a graph-Laplacian spectral
chart on training chains and score held-out chains as whole trajectories, not
as isolated toy scalars.

## Implemented Object

For each reasoning chain $i$, step $t$, and selected layer set, the current
implementation uses a pooled step hidden-state vector:

$$
x_{i,t}
= \operatorname{vec}\left(
  h_{i,t}^{\ell_1}, h_{i,t}^{\ell_2}, \ldots, h_{i,t}^{\ell_L}
\right)
\in \mathbb{R}^{LD}.
$$

In canonical ProcessBench `full_*.npz` files this object is already stored as:

```text
stepvec[i].shape == (T_i, L, D)
```

and is flattened into the vector bank consumed by
`spectral_chain_dynamics.py`. For mechanism extraction outputs, the same role
is played by `step_state_vectors`. The method can also run on
`step_vectors`, but then the curve describes residual-write motion rather than
state motion.

The current implementation is step-level and layer-pooled. It is not yet the
token-level hidden-shard version. Token-level geometry should be added as a
separate extension rather than silently mixed into this method.

## Core Hypothesis

The implemented hypothesis is:

$$
\text{valid reasoning}
\approx
\text{a phase-normalized low-frequency trajectory in a healthy manifold tube}.
$$

A failure is not simply "larger spread." It is a dynamical compatibility defect:

$$
\text{failure}
\Rightarrow
\text{tube departure}
\;\lor\;
\text{spectral leakage}
\;\lor\;
\text{off-tangent motion}
\;\lor\;
\text{entry into an error basin}.
$$

This is weaker than claiming a full causal mechanism, but stronger than a
single static geometry score. It is explicitly designed to answer the user's
main empirical concern: response-level diagnosis should preserve dynamic
information without diluting the first-error signal into a length proxy.

## Cross-Fitted Training Protocol

All geometry is learned out-of-fold by chain. Test-chain labels are never used
to construct the chart, healthy tube, tangent reference, or committor
neighborhood.

For each fold:

1. Split complete chains into train and held-out rows.
2. Flatten train step vectors into a point cloud
   $\mathcal{X}_{\mathrm{train}} = \{x_{i,t}\}$.
3. Standardize train vectors with train-only mean and scale.
4. Optionally subsample landmarks, controlled by `max_landmarks`.
5. Build an adaptive-kernel neighborhood graph.
6. Compute a graph-Laplacian / diffusion-map chart on train landmarks.
7. Extend held-out points into the spectral chart by Nyström-style kernel
   extension.
8. Score held-out chains against train-only healthy and error references.

The fold boundary is important. Without it, a global manifold can leak the
test chain into its own reference geometry and inflate the apparent
separability.

## Spectral Chart

Let standardized train landmarks be $\tilde{x}_a$. The implementation uses a
local adaptive kernel whose scale is derived from k-nearest-neighbor distances:

$$
K_{ab}
=
\exp\left(
  -\frac{\lVert \tilde{x}_a - \tilde{x}_b\rVert_2^2}
        {\sigma_a \sigma_b + \epsilon}
\right).
$$

After normalization, eigenvectors of the resulting diffusion operator define
coordinates:

$$
\Phi(x)
=
\left(
  \lambda_1^\tau \psi_1(x),
  \lambda_2^\tau \psi_2(x),
  \ldots,
  \lambda_m^\tau \psi_m(x)
\right).
$$

Here `m == n_modes` and `tau == diffusion_time`. Low modes are controlled by
`low_modes`.

This chart is not used as a visualization trick. It defines the coordinate
system in which the trajectory tube, spectral leakage, local tangent motion,
and committor are computed.

## Healthy Reference Points

Each train point receives a weak process label:

$$
y_{i,t} =
\begin{cases}
0, & \text{chain is correct, or } t < g_i, \\
1, & \text{chain is wrong and } t \ge g_i,
\end{cases}
$$

where $g_i$ is the annotated first error step. Healthy reference points are
the training points with $y_{i,t}=0$. If too few healthy points are available,
the implementation falls back to all train points to avoid a degenerate fold.

This means `sd_committor` is weakly supervised by training labels. It should
not be described as an unsupervised discovery score. The chart and metric
evaluation remain out-of-fold.

## Phase Alignment

A chain with $T_i$ steps is mapped to normalized phase:

$$
\rho_{i,t} =
\begin{cases}
0, & T_i \le 1, \\
\frac{t}{T_i - 1}, & T_i > 1.
\end{cases}
$$

Phase is used only to compare a held-out step with train healthy points at
similar trajectory progress. It is not a claim that late steps are intrinsically
unhealthy. Evaluation still reports position and length controls separately.

## Step-Level Diagnostics

The implementation appends five `sd_*` step metrics.

### Healthy-Tube Distance

For held-out point $x_{i,t}$, let $z_{i,t}=\Phi(x_{i,t})$. Let
$\mathcal{H}(\rho_{i,t})$ be healthy train points with nearby phase. The score
is a local distance to the phase-matched healthy tube:

$$
\operatorname{tube}(i,t)
=
\operatorname{mean}_{k}
\left[
  \lVert z_{i,t} - z_h \rVert_2
  \cdot w(\rho_{i,t}, \rho_h)
\right],
$$

where $w$ is a phase penalty and the mean is taken over the nearest healthy
neighbors. In code this is `sd_tube_dist`.

### Spectral Leakage

Let $z_{i,t}^{(1:q)}$ be the low-mode part and $z_{i,t}^{(q+1:m)}$ be the
remaining high-mode part. The spectral leakage score is:

$$
\operatorname{leak}(i,t)
=
\frac{\lVert z_{i,t}^{(q+1:m)} \rVert_2^2}
       {\lVert z_{i,t}^{(1:m)} \rVert_2^2 + \epsilon}.
$$

In code this is `sd_spectral_leak`. Its purpose is to test whether the chain's
state requires high-frequency manifold coordinates rather than staying in the
low-frequency healthy chart.

### Tangent Off-Manifold Motion

For consecutive steps, define spectral displacement:

$$
\Delta z_{i,t}=z_{i,t+1}-z_{i,t}.
$$

The local healthy tangent space is estimated by PCA over nearby healthy
spectral coordinates. If $P_{\mathcal{T}}$ is the local tangent projection,
the off-tangent fraction is:

$$
\operatorname{offtan}(i,t)
=
\frac{\lVert (I-P_{\mathcal{T}})\Delta z_{i,t}\rVert_2^2}
       {\lVert \Delta z_{i,t}\rVert_2^2 + \epsilon}.
$$

In code this is `sd_tangent_off`.

### Error-Basin Committor

The committor is a k-nearest-neighbor estimate of how much a held-out point's
spectral neighborhood resembles train error-basin points:

$$
\operatorname{committor}(i,t)
=
\frac{
  \sum_{j \in \mathcal{N}_k(z_{i,t})} \omega_j y_j
}{
  \sum_{j \in \mathcal{N}_k(z_{i,t})} \omega_j + \epsilon
}.
$$

In code this is `sd_committor`. Because $y_j$ comes from train first-error
annotations, this is best described as a process-supervised basin score.

### Step Speed

The step speed is:

$$
\operatorname{speed}(i,t)=\lVert z_{i,t+1}-z_{i,t}\rVert_2.
$$

In code this is `sd_step_speed`. It is not the main claim; it is included to
separate mere movement magnitude from tube, leakage, tangent, and basin effects.

## Response-Level Diagnostics

The method also appends response-level summaries. These are designed to avoid
raw length-dependent sums.

### Curve Efficiency

$$
\operatorname{efficiency}(i)
=
\frac{
  \lVert z_{i,T_i-1}-z_{i,0}\rVert_2
}{
  \sum_{t=0}^{T_i-2}\lVert z_{i,t+1}-z_{i,t}\rVert_2 + \epsilon
}.
$$

In code this is `sd_curve_efficiency`.

### Path Length Per Phase

$$
\operatorname{pathlen}(i)
=
\frac{
  \sum_{t=0}^{T_i-2}\lVert z_{i,t+1}-z_{i,t}\rVert_2
}{
  \max(\rho_{i,T_i-1}-\rho_{i,0}, \epsilon)
}.
$$

In code this is `sd_path_length_per_phase`.

### Phase-Normalized Curve Integrals

For selected step curves $s_{i,t}$, the implementation adds phase-normalized
integrals such as:

$$
\operatorname{auc}_s(i)
=
\int_0^1 s_i(\rho)\,d\rho.
$$

The implemented names are `sd_tube_auc`, `sd_committor_auc`, and
`sd_leak_auc`.

The general summarizer also produces `mean_*`, `max_*`, `top20_mean_*`, and
`survival_*` chain scores for the new step metrics.

## Evaluation Tasks

The current audit uses the shared evaluator in `evaluate.py` and reports:

- step-level first-error AUROC;
- within-chain first-error rank;
- response-level AUROC and AUPRC;
- controls such as `rel_pos` and `step_len`;
- ablation group `spectral_chain_dynamics`, which collects `sd_*` metrics.

For canonical `full_*.npz`, the valid claims are:

- ProcessBench first-error localization;
- within-chain first-error ranking;
- cross-problem response diagnosis.

Canonical `full_*.npz` does not support same-problem paired response AUROC.
Same-problem tests require a multisample dataset such as `*_multisample_sv.npz`
and a separate paired response audit.

## Main Command

Run directly on canonical ProcessBench full data:

```bash
python -m prompt_control_flow.cli.audit_spectral_chain \
  --input data/features/full_gsm8k.npz \
  --output outputs/spectral_chain/full_gsm8k_sd.npz \
  --output_dir outputs/spectral_chain/full_gsm8k_sd_audit \
  --vector_key step_state_vectors \
  --folds 5 \
  --modes 12 \
  --low_modes 4
```

Run on mechanism extraction output:

```bash
python -m prompt_control_flow.cli.audit_spectral_chain \
  --input outputs/mechanisms/gsm8k_llama31.npz \
  --output outputs/mechanisms/gsm8k_llama31_sd.npz \
  --vector_key step_state_vectors
```

## Implementation Map

- `spectral_chain_dynamics.py`
  - `SpectralChainConfig`
  - `append_spectral_chain_dynamics`
  - `canonicalize_spectral_input`
  - diffusion-map fitting and held-out scoring
- `cli/audit_spectral_chain.py`
  - command-line entry point
  - writes enriched `.npz`, `summary.json`, `summary.md`, and `step_scores.csv`
- `evaluate.py`
  - reports step, response, rank, and ablation summaries
- `visualize.py`
  - includes default visualization support for `sd_*` curves
- `README.md`
  - engineering usage and data-path guardrails

## What This Method Is Not

This method is not:

- a VAE method;
- a raw prompt-SVD detector;
- a holonomy implementation;
- a token-level hidden-shard manifold analysis;
- an unsupervised proof that correct and wrong reasoning occupy separated
  manifolds;
- a causal intervention result.

It is a cross-fitted spectral-manifold trajectory audit. Its value depends on
whether `sd_*` metrics beat length, position, random, and simpler geometry
controls on held-out chains.

## Next Necessary Extensions

1. Token-level version using `data/hidden/<subset>/<id>.npy`.
2. Same-problem paired response audit on multisample datasets.
3. Random-subspace and shuffled-phase negative controls for `sd_*` metrics.
4. Visual case cards showing tube departure, leakage, tangent defect, and
   committor curves for individual chains.
5. Optional causal follow-up: patch or steer states at high `sd_tangent_off`
   or high `sd_committor` locations.

## Current Claim Boundary

The current implementation supports the following conservative claim:

> A reasoning chain can be represented as a phase-normalized curve in a
> train-fold spectral chart of hidden states. Out-of-fold deviations from the
> healthy trajectory tube, high-mode spectral leakage, off-tangent motion, and
> train-error-basin committor provide testable diagnostics for first-error
> localization and response-level failure detection.

It does not yet support the stronger claim that the model internally computes
a unique correct low-dimensional manifold or that these signals are causal.
