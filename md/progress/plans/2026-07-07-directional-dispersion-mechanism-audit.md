# Directional Dispersion Mechanism Audit Plan

Date: 2026-07-07

## Motivation

The validated observation is:

```text
incorrect reasoning steps have lower kappa / resultant
```

where kappa is the weighted mean resultant length of unit token directions
inside a step:

```text
u_i = h_i / ||h_i||
mu = sum_i w_i u_i
kappa = ||mu|| / sum_i w_i
```

This proves a loss of local directional consensus.  It does **not** by itself
prove the mechanism.  In particular, low kappa may come from several distinct
geometries:

1. **high-rank dispersion**: token directions spread into many unrelated
   directions;
2. **bipolar / axial cancellation**: tokens lie mostly on one axis but split
   into opposite signs, so the mean cancels;
3. **multi-cluster semantic mixing**: the step contains several coherent
   sub-computations or semantic fragments;
4. **legitimate long-step complexity**: correct but difficult steps naturally
   use several directions;
5. **length/statistical artifact**: longer steps have lower kappa for reasons
   unrelated to correctness.

The existing Gram/spectral experiments mostly asked:

```text
Do second-moment / spectral features improve classification over strong
baselines?
```

This new audit asks a different mechanism question:

```text
When kappa is low, what geometric morphology produced it?
```

## What Was Already Tested

### Already Done

- `cloud_D`, PR/AE, effective-rank-like features were tested as scalar
  predictors.  They often track length.
- `second_moment_dynamics_audit.py` tested direct token-matrix Gram/spectrum
  features under same-problem controls on `gsm8k_v2_custom`.
- Result on `gsm8k_v2_custom`: baseline around 0.685, best Gram group around
  0.660, spectral-tail group around 0.634, negative increment.
- Historical scatter-spectrum notes found that on harder `omnimath`,
  `eff_rank` may add a small clean increment over `[kappa + length]`, while
  GSM8K was negative.

### Not Yet Done

The following mechanism decomposition has not been cleanly executed:

```text
condition on kappa and length, then ask whether low-kappa errors are
high-rank dispersed, bipolar, multi-clustered, or merely long/complex.
```

The old experiments mostly tested whether spectra improve a detector.  They did
not produce a morphology taxonomy of the low-kappa event.

## Mathematical Decomposition

Use unit token directions `u_i` and normalized weights `w_i`, `sum_i w_i = 1`.

Mean direction:

```text
mu = sum_i w_i u_i
kappa = ||mu||
```

Uncentered directional second moment:

```text
A = sum_i w_i u_i u_i^T
trace(A) = 1
```

Centered residual scatter:

```text
C = A - mu mu^T
trace(C) = 1 - kappa^2
```

This gives a strict identity:

```text
total directional energy = mean-direction energy + residual scatter
1 = kappa^2 + trace(C)
```

So the total residual energy is not new information.  The new information, if
any, is in the **shape** of `C`.

Normalize residual eigenvalues:

```text
lambda(C) / trace(C)
```

and analyze whether residual scatter is concentrated on one axis, spread over
many axes, or split into clusters.

## Mechanism Features

### 1. Consensus Strength

Baseline, already validated:

```text
kappa
spread = 1 - kappa
residual_energy = 1 - kappa^2
```

`residual_energy` is mathematically tied to kappa and should not be sold as a
new signal.

### 2. Residual Scatter Shape

Computed from centered residual scatter `C`:

```text
res_lam1
res_lam2
res_gap12
res_eff_rank
res_entropy
res_stable_rank
res_topk_mass
```

Interpretation:

- high `res_eff_rank`: directions spread across many axes;
- high `res_lam1` with low `res_eff_rank`: low-dimensional axial split;
- high `res_gap12`: one dominant residual axis;
- high top-k mass with k > 1: structured multi-axis spread.

### 3. Bipolar / Axial Cancellation

Let `v1` be the top eigenvector of `C`.

Project tokens:

```text
p_i = u_i dot v1
```

Features:

```text
axis_balance = min(sum_{p_i>0} w_i, sum_{p_i<0} w_i)
axis_separation = mean(p_i | p_i>0) - mean(p_i | p_i<0)
bipolarity = res_lam1 * axis_balance
sign_flip_rate over token order
```

Interpretation:

- low kappa + high residual lam1 + high axis balance means two-sided
  cancellation, not high-dimensional chaos.

### 4. Multi-Cluster / Semantic Mixing

Use the token-token Gram matrix:

```text
G_ij = u_i dot u_j
```

Features:

```text
pair_cos_q10/q25/q50/q75/q90
frac_negative_cos
frac_low_cos
spectral_cluster_gap
best_spherical_silhouette_k2/k3/k4
cluster_balance_k2/k3/k4
within_minus_between_cos
```

If external clustering dependencies are undesirable, start with:

- eigenvectors of `G`;
- simple 2-means on the first two spectral coordinates;
- pairwise cosine quantiles;
- sign split along residual `v1`.

### 5. Order / Segment Structure

Low kappa may happen because a step contains multiple sub-actions in order.

Features:

```text
early_mu, late_mu
early_late_cos
within_early_kappa
within_late_kappa
boundary_jump = 1 - cos(early_mu, late_mu)
running_kappa_min/mean
```

Interpretation:

- if early and late halves each have high kappa but point to different
  directions, the step may be a multi-substep semantic mixture rather than
  noisy failure.

## Primary Hypotheses

### H1a: Low-Kappa Error As High-Rank Dispersion

Incorrect low-kappa steps should have higher residual effective rank than
correct low-kappa steps of the same length.

Evidence required:

```text
res_eff_rank(error | kappa bin, length bin)
  > res_eff_rank(correct | same kappa bin, length bin)
```

### H1b: Low-Kappa Error As Bipolar Cancellation

Some incorrect steps should show low kappa but high axial concentration:

```text
low kappa
high res_lam1
high axis_balance
```

This would mean the step did not spread isotropically; it split between two
opposing directions.

### H1c: Low-Kappa Error As Multi-Cluster Mixing

Some incorrect steps should contain multiple internally coherent token groups:

```text
low kappa
high clusterability
high within_minus_between_cos
```

This may correspond to mixing incompatible semantic sub-computations.

### H1d: Correct Long-Step Complexity

Correct difficult steps may also have low kappa, but should differ in one of:

```text
higher prompt/anchor support
higher internal segment order
lower uncertainty
more stable early->late transition
```

This connects the dispersion audit to the anchor-source branch.

## Evaluation Design

### Dataset Tiers

1. ProcessBench full step-level data:
   - `full_gsm8k.npz`
   - `full_math.npz`
   - `full_omnimath.npz`
   - `full_olympiad.npz`
2. Same-problem generated samples:
   - `gsm8k_v2_custom.npz`
   - `gsm8k_v2_5shot.npz`
3. Later:
   - counterfactual sibling traces.

### Labels

ProcessBench:

- positive: first wrong step;
- controls:
  - correct-chain steps;
  - pre-error correct steps;
  - matched position/length correct steps.

Same-problem data:

- chain-level final incorrectness;
- optional step text/ranges only for descriptive alignment.

### Primary Controls

Every morphology claim must report:

- length buckets;
- kappa buckets;
- position buckets;
- density / formula-token controls if available;
- GroupKFold or cluster bootstrap by chain/problem;
- same-problem paired AUROC where applicable.

The key test is:

```text
Does morphology distinguish error vs correct after matching kappa and length?
```

If not, the mechanism is only a restatement of kappa.

## Analysis Blocks

### Block A: All-Step Distribution

Report distributions of:

```text
kappa
res_eff_rank
res_lam1
bipolarity
clusterability
early_late_cos
```

for correct and error steps.

This is descriptive, not enough for claims.

### Block B: Fixed-Length And Fixed-Kappa Matching

Within each `(length_bin, kappa_bin)`:

- compare error vs correct morphology;
- aggregate with weighted AUROC / effect size;
- bootstrap by chain.

This is the main mechanism test.

### Block C: Low-Kappa Morphology Taxonomy

Restrict to low-kappa steps, e.g. bottom 30%.

Classify each low-kappa step into:

```text
high_rank_dispersion
bipolar_split
multi_cluster
ordered_substep_shift
unclassified
```

Report:

- fraction of error vs correct in each morphology;
- odds ratio / enrichment;
- examples from step text;
- whether each morphology has different entropy / final correctness.

### Block D: Coherent-Wrong Complement

Restrict to:

```text
wrong chain or wrong step
high kappa
low entropy
```

This is not expected to be explained by dispersion.  Use it as a boundary:

```text
directional dispersion explains fragmented errors, not coherent wrong errors.
```

This subset should be handed to anchor-source attribution.

### Block E: Cross-Dataset Stress

Expected outcomes:

- GSM8K: kappa may saturate; residual shape may add little.
- OmniMath / harder tasks: residual shape may matter more.
- If this pattern repeats, it becomes a meaningful difficulty-dependent
  mechanism claim.

## Pass / Kill Criteria

### Strong Pass

At least one morphology variable has:

```text
positive increment over [kappa + length + position + entropy]
CI excludes 0
replicates on at least one hard dataset
interpretable enrichment in low-kappa taxonomy
```

### Mechanism Pass Without Detector Increment

No classification gain, but low-kappa errors are morphologically different from
low-kappa correct steps:

```text
e.g. errors are high-rank dispersed, correct steps are ordered substep shifts
```

This can support a mechanism section, but not a detector claim.

### Kill

If all morphology differences vanish after conditioning on kappa and length:

```text
low kappa is the whole available direction-geometry signal
```

Then retire "multi-direction spectrum" as a paper claim.  Keep only the
mathematical explanation:

```text
kappa is the rank-one mean component of the directional moment.
```

## Proposed Script

```text
directional_dispersion_mechanism_audit.py
```

Inputs:

- `.npz` with hidden shards or `sv_clouds`/`stepcloud`;
- labels: `gold_error_step`, `is_correct`;
- optional step text for examples.

Outputs:

```text
*.json
*.md
*.taxonomy.csv
*.matched_bins.csv
*.examples.jsonl
```

Core functions:

```text
unit_step_matrix(H)
weighted_kappa(U, w)
residual_scatter_spectrum(U, w)
bipolar_features(U, w)
pairwise_cosine_features(U, w)
simple_spectral_cluster_features(U, w)
early_late_features(U, w)
low_kappa_taxonomy(row)
matched_kappa_length_audit(rows)
```

Implementation priority:

1. residual scatter spectrum + bipolarity;
2. pairwise cosine quantiles;
3. early/late substep shift;
4. simple spectral clustering;
5. text examples.

## Paper-Level Interpretation

Safe claim if the experiment passes:

```text
The kappa drop is not a single failure morphology.  It decomposes into
structured subtypes: high-rank dispersion, bipolar cancellation, and semantic
multi-cluster mixing.  These subtypes differ in their relation to correctness
and task difficulty.
```

Safe claim if it fails:

```text
Within source-free direction geometry, the discriminative information is
saturated by the mean resultant length.  This explains why more complex
Gram/spectral variants failed and motivates moving to source-aware anchors.
```

Either outcome is useful.  The experiment is worth running because it tells us
whether the current kappa observation can be mechanized internally, or whether
we must move entirely to anchor-source / semantic-flow variables.
