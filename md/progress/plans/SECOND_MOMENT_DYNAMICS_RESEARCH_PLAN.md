# Second-Moment Dynamics Research Plan

## Result Analysis

The recent EM/HMM audit should not be treated as a successful latent-state
method.  It technically used EM, but the modeled object was too weak: a
diagonal Gaussian HMM over old per-step scalars mostly reclustered
`cloud_spread`, entropy, PR/AE, and step jumps.  On the same-problem
`gsm8k_v2_5shot.npz` setting, the HMM risk scores were near chance, while the
best baseline remained `cloud_spread.max`:

| Family | Best observed pattern |
|---|---|
| Static spread | `base.cloud_spread.max` within-problem AUROC 0.634 |
| Entropy | useful but weaker in same-problem contrast; often measures difficulty/confidence rather than rupture |
| PR/AE moments | mostly near chance in this run |
| HMM posterior risk | did not beat the base signals |

So the conclusion is not "EM is useless".  The conclusion is that EM over
hand-crafted scalar emissions is the wrong level of abstraction for our current
hypothesis.  If the hypothesis is about constrained hidden geometry, the latent
object should be a matrix-valued state: a token-cloud scatter/Gram shape, a
local covariance flow, or a low-rank directional support, not merely a list of
scalar summaries.

The earlier second-moment results also give a sharper boundary:

- On GSM8K L14, second-moment shape over `[kappa + logN]` was not robust
  globally; over `[kappa-bin + logN]` it had a small positive increment
  (`+0.019`, significant), and in the low-kappa bottom 30% it had a stronger
  increment (`+0.107`, significant).
- Older scatter-spectrum results suggested hard-task or low-concentration
  regimes can expose useful covariance structure, while easy regimes are
  already dominated by first-moment spread/resultant.

This means second moments should not be sold as a universal replacement for
spread.  They are more plausibly a **conditional detector** for cases where
first-moment coherence is ambiguous: high spread, low kappa, same-problem
contrastive samples, or hard-but-format-clean problems.

## Core Reframing

The main mathematical object should be the step token-cloud Gram matrix:

```text
H_t = raw hidden token matrix for step t
G_t = H_t H_t^T / n_t
G_t^c = (H_t - mean(H_t)) (H_t - mean(H_t))^T / n_t
```

`G_t` is small because its size is the number of tokens in the step, not the
hidden width.  Its non-zero eigenvalues match those of the hidden-dimension
scatter matrix `H_t^T H_t / n_t`.  This gives us a cheap way to study the
second-order geometry without explicitly building a 4096 x 4096 covariance
matrix.

2026-07-06 correction: row-normalized unit-token Gram matrices should not be
the main object for this branch.  They are useful only as a directional
ablation, because they discard token-norm/radial information before the
spectral analysis.  The Geometry-of-Reason-style implementation should first
compute spectra directly from the raw and centered token matrix, then report
unit-row spectra only as a control.

The story becomes:

```text
Correct reasoning:
  token directions remain coordinated;
  spectral mass stays concentrated in a controlled low-rank shape;
  changes in G_t are locally bounded and prompt-conditioned.

Incorrect reasoning:
  spectral tail inflates;
  effective rank rises or becomes unstable;
  G_t undergoes a local covariance rupture that is larger than expected from
  the chain prefix and problem context.
```

This is closer to the user's original hypothesis: wrong reasoning is not just
"larger scalar spread"; it is a relaxation of a hidden constraint, visible as a
change in the geometry of the token-cloud second moment.

## Research Survey

### 1. Spectral Tail Inflation

The low-rank transition-tube experiments already suggest that diagnostic
same-problem correct support can separate errors with tail/off-tube quantities,
for example `support_oracle.off_late` and `support_oracle.tail_k75`.  The next
version should remove the explicit tube and ask whether the current step's
Gram spectrum itself shows tail inflation:

```text
tail_auc_t      = mean_k (1 - cumulative_energy_t[k])
tail_k75_t      = smallest k reaching 75% energy
residual_k_t    = 1 - cumulative_energy_t[k]
eff_rank_t      = exp(entropy(eigenvalue_distribution_t))
gap12_t         = lambda_1 - lambda_2
```

This is not copying alpha from spectral-geometry papers.  It is a different
operationalization: rather than fitting a global power-law exponent, we measure
how much hidden-token energy escapes the leading coordinated directions inside
each step.

Relevant inspiration:

- The Spectral Geometry of Thought: phase transitions, instruction reversal,
  token-level dynamics, and spectral cascade.
- Effective Reasoning Chains Reduce Intrinsic Dimensionality: successful
  reasoning can be framed as lower intrinsic-dimensional trajectory structure.
- Geometry of Reason: spectral signatures of valid mathematical reasoning.

For our data, the falsifiable claim is narrower:

```text
In same-problem contrastive samples, error chains should have higher late or
peak second-moment spectral tail than correct chains, especially after
conditioning on spread, entropy, and step length.
```

### 2. Covariance Change-Point Rather Than CUSUM Drift

The user is right that simple cumulative scores are fragile: they often grow
late in every chain.  The better formulation is online covariance change-point
detection:

```text
rupture_t = distance(shape(G_t), expected_shape_from_recent_prefix)
```

The expected shape should be local and non-cumulative.  Candidate distances:

- spectral L2/JS distance between normalized eigenvalue distributions;
- log-Euclidean distance on a regularized covariance/Gram shape;
- Bures-Wasserstein distance between covariance matrices when dimensions align;
- robust prefix z-score using only previous steps from the same chain.

This directly targets "previous transitions stayed in a constrained range, but
the wrong step changes abruptly."  It is a better match than the earlier
`transition_cusum`, which mainly rewarded late positions.

Relevant algorithmic family:

- Online change-point detection for high-dimensional covariance matrices.
- Sequential subspace-change detection and subspace CUSUM.
- Random-matrix/spiked-covariance tests for tail inflation and rank changes.

### 3. Subspace/Grassmann Drift

If we can keep top hidden-space PCs for each step, we can compare subspaces
rather than only spectra:

```text
S_t = top-r left singular vectors of U_t
theta_t = principal_angles(S_t, S_{t-1})
drift_t = ||sin(theta_t)||_2
```

This gives a mechanism-facing interpretation:

```text
correct step transition = tangent rotation inside a constrained support;
wrong step transition   = subspace rotation into directions not used by the
recent prefix or by same-problem correct samples.
```

This should be tested carefully because same-problem samples have variable step
lengths and variable token counts.  If eigenvectors are unstable, use spectral
tail and Gram-distance first, then add subspace drift only where token clouds
are sufficiently long.

### 4. Directional Distributions Instead of Euclidean Gaussians

LayerNorm and row normalization make a directional model more natural than a
Euclidean Gaussian over raw hidden vectors.  The useful distributions to borrow
from are:

- von Mises-Fisher for rank-one concentration;
- Bingham for antipodally symmetric directional scatter;
- Angular Central Gaussian for covariance-like directional shape.

This is where EM could re-enter in a meaningful way:

```text
z_t = latent geometric regime
p(U_t | z_t) = directional matrix distribution with state-specific scatter
p(z_t | z_{t-1}, prompt, position) = online regime transition
```

That would be a true latent geometry model.  The failed scalar HMM is only a
weak placeholder for this idea.

### 5. Matrix-Valued HMM / Wishart-Style State Space

A more direct matrix model would treat each step as an observed SPD/PSD object:

```text
G_t | z_t ~ regularized Wishart / matrix-variate distribution
z_t follows an HMM or semi-Markov process
```

This is fancy enough mathematically, but it may be too heavy for the current
data.  It should only be attempted after a nonparametric Gram-spectrum audit
shows a reliable same-problem signal.  Otherwise it will become another
expensive wrapper around weak features.

## What To Verify Next

The immediate next code should be `second_moment_dynamics_audit.py`.  It should
not train a large model first.  It should answer four falsifiable questions.

### Gate A: Is There Same-Problem Spectral Tail Separation?

Compute per-step Gram-spectrum features from saved `sv_clouds`:

```text
sm_eff_rank
sm_tail_auc
sm_k50 / sm_k75 / sm_k90
sm_resid_k1 / sm_resid_k2 / sm_resid_k4 / sm_resid_k8 / sm_resid_k16
sm_gap12
```

Evaluate chain summaries under `answer_format_ok`:

```text
level_late, level_max, local_jump_max, prefix_z_max, contrast_max, volatility
```

Primary metric: same-problem paired AUROC.  Cross-problem AUROC is only
secondary because it can be inflated by difficulty.

### Gate B: Does Second Moment Add Over Spread/Entropy?

The test must include the known strong baselines:

```text
spread/resultant
log step length
out_entropy / committal when available
format policy
```

Accept only if second-moment groups add stable OOF or paired-ranking increment
over spread plus entropy.  A raw AUROC that merely matches spread is not enough.

### Gate C: Is It Local, Not Just Late?

For every rupture-like score, report:

```text
argpos_error
argpos_correct
correct-chain false alarm position
early-warning recall at fixed correct-chain FPR
```

If both correct and error chains fire near the final step, the score is an
endpoint artifact.

### Gate D: Does It Work Where Spread Is Ambiguous?

The most important subset is not the whole dataset.  Test:

```text
high-spread subset
low-kappa / low-resultant subset
low-entropy confident subset
same-problem format-ok contrastive subset
hard problems with both correct and wrong samples
```

This is where second moments could become paper-worthy: not as a better
replacement for spread, but as the extra geometry that explains failures when
spread alone is underdetermined.

## Expected Implementation

Use the saved token-cloud data:

```text
sv_clouds
sv_cloud_sizes
cloud_layers
```

For each step:

1. take the token hidden matrix for that step;
2. compute direct raw-token spectra from `H_t`;
3. compute centered-token spectra from `H_t - mean(H_t)`;
4. separately compute exp-weighted unit-row kappa as the first-moment baseline;
5. eigendecompose `G_t`;
6. compute spectral-tail and effective-rank features;
7. compute local change relative to the previous step and previous prefix.

Implementation note: `second_moment_dynamics_audit.py` follows this split.
Feature prefixes `tok_raw_*` and `tok_cen_*` are the primary direct token-matrix
features; `unit_raw_*` and `unit_cen_*` are ablations only.

The first implementation should be transparent and nonparametric.  If it works,
then a second phase can fit a directional mixture/HMM or matrix-valued latent
state model.

## Result Analysis To Record After Running

When the audit is run, record:

- whether second-moment spectra beat `cloud_spread` under same-problem AUROC;
- whether they add over spread/entropy/length;
- whether rupture positions are early or merely terminal;
- whether they help specifically in high-spread or confident-error subsets;
- whether centered and uncentered Gram spectra behave differently.

## Follow-Up Research Direction

If Gate A and B pass, the paper story becomes:

```text
Reasoning failures are not just uncertain or diffuse.  They relax the
second-order directional constraint of the hidden token cloud.  A plug-in
single-forward monitor can detect this as spectral-tail inflation or covariance
rupture before or at the failed step.
```

If only same-problem support/oracle settings work, then the next direction is
prompt-conditioned pseudo-support:

- retrieve nearest training problems by prompt hidden vector;
- build a local reference distribution of healthy Gram spectra;
- score deviations with conformal p-values instead of raw residuals.

If all second-moment gates fail, retire the second-moment branch as a mainline
claim and return to richer single-forward signals: boundary logits, attention
lookback/prompt connectivity, and token-level reconvergence.

## Remote Gram Dynamics Result

Remote run on `gsm8k_v2_custom.npz` does **not** support a second-moment Gram
dynamics increment over the static baseline.

Setting:

- samples: 1658; errors: 462; problems: 147;
- source: `sv_clouds`, layer 16;
- primary metric: OOF same-problem paired AUROC;
- baseline: exp-weighted cloud spread/resultant + step length + available
  uncertainty/static controls.

Headline:

- baseline: 0.685;
- best Gram group: `token_matrix_level`, 0.660;
- increment: -0.025, bootstrap CI [-0.058, +0.005];
- decision: no robust OOF same-problem increment over the static baseline.

Group increments:

| Group | AUROC | Increment |
|---|---:|---:|
| `baseline+token_matrix_level` | 0.660 | -0.025 |
| `baseline+token_raw_matrix` | 0.658 | -0.027 |
| `baseline+token_centered_matrix` | 0.654 | -0.031 |
| `baseline+token_matrix_dynamics` | 0.644 | -0.041 |
| `baseline+token_matrix_all` | 0.635 | -0.051 |
| `baseline+token_spectral_tail` | 0.634 | -0.051 |
| `baseline+unit_direction_ablation` | 0.633 | -0.052 |
| `baseline` | 0.685 | 0.000 |

Top scalar observations:

- best scalar: `tok_norm_mean_late`, 0.667;
- best Gram scalar: `tok_raw_log_energy_mean`, best-direction 0.662 but raw
  orientation 0.338;
- best residual over baseline: `unit_cen_log_energy_volatility`, 0.614.

Subsets:

| Subset | n | problems | baseline | group | residual |
|---|---:|---:|---:|---:|---:|
| all | 1658 | 147 | 0.685 | 0.660 | 0.614 |
| ambiguous high spread q50 | 829 | 106 | 0.679 | 0.631 | 0.641 |
| ambiguous high spread q75 | 415 | 68 | 0.728 | 0.636 | 0.654 |
| confident low entropy q50 | 829 | 97 | 0.690 | 0.661 | 0.653 |
| hard high-error-rate problems | 886 | 81 | 0.661 | 0.659 | 0.617 |

Interpretation:

- Direct token-matrix Gram features are not unlocking information beyond the
  static spread/resultant baseline.  The strongest Gram scalar is mostly a
  raw-energy/norm quantity and even has reversed raw orientation, which suggests
  it is not a stable risk mechanism.
- The dynamic Gram features are worse than level features.  This agrees with
  the previous event-study result: the error signal is synchronous/late rather
  than a clean precursor or covariance rupture.
- Centered, raw, unit-direction, and spectral-tail variants all fall below the
  baseline.  This rules out the simple explanation that the first
  implementation merely used the wrong centering or normalization.
- Subset tests do not rescue the branch.  Even in high-spread, confident
  low-entropy, and hard-problem subsets, Gram groups do not overtake the static
  baseline.
- OOF `baseline+Gram` can be lower than `baseline` because the added Gram
  block is high-dimensional, correlated with spread/length/norm, and noisy
  under same-problem splits.  Regularized OOF models are not monotone: adding a
  weak correlated block can shift weights away from the robust static scalar and
  reduce held-out paired ranking.

Decision:

- Retire direct step-token Gram/spectrum dynamics as a mainline method claim.
- Keep `second_moment_dynamics_audit.py` as a negative-result audit and as a
  guard against rebranding second moments as a new signal.
- Future work should move to genuinely orthogonal channels: operation/premise
  choice consistency, causal evidence flow, fork-token progress value, or
  knowledge-boundary/output-commitment mismatch.

## Optimization Suggestions

1. Do not use cumulative drift as the main detector.
2. Do not report oracle same-problem tubes as deployable detectors.
3. Do not claim alpha-style spectral geometry unless we actually estimate a
   new quantity with a different operational role.
4. Treat entropy as a necessary control, not a novel contribution.
5. Treat second-moment methods as conditional, designed to explain high-spread
   or low-kappa ambiguity.
6. Only introduce EM after the nonparametric Gram-spectrum audit shows a real
   same-problem signal.

## Source Pointers

- The Spectral Geometry of Thought:
  https://arxiv.org/abs/2604.15350
- Effective Reasoning Chains Reduce Intrinsic Dimensionality:
  https://arxiv.org/abs/2602.09276
- Geometry of Reason:
  https://arxiv.org/abs/2601.00791
- Reasoning emerges from constrained inference manifolds:
  https://arxiv.org/abs/2605.08142
- Limited Reasoning Space:
  https://arxiv.org/abs/2602.19281
- Reasoning Fails Where Step Flow Breaks:
  https://arxiv.org/abs/2604.06695
- EDIS: Diagnosing LLM Reasoning via Entropy Dynamics:
  https://arxiv.org/abs/2602.01288
- Online change-point detection for high-dimensional covariance structure:
  https://jmlr.org/papers/v24/20-1101.html
- Bures-Wasserstein covariance geometry:
  https://epubs.siam.org/doi/10.1137/22M149168X
- Spectral regularization/random-matrix background:
  https://optml.mit.edu/papers/sra_dirchap.pdf
