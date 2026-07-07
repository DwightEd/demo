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

## 2026-07-07 Implementation V1

Implemented `directional_dispersion_mechanism_audit.py` with a step-level
mechanism protocol rather than another detector sweep.

### What the code verifies

1. **Kappa identity check**

   For unit token directions:

   ```text
   C = sum_i w_i (u_i - mu)(u_i - mu)^T
   trace(C) + ||mu||^2 = 1
   ```

   The script reports median and q90 numerical error.  This prevents us from
   accidentally presenting residual energy as a new signal.

2. **Residual-shape mechanism tests**

   The four explicit hypotheses are:

   | hypothesis | feature |
   | --- | --- |
   | H1a high-rank dispersion | `res_eff_rank` |
   | H1b bipolar cancellation | `bipolarity` |
   | H1c multi-cluster mixing | `signed_clusterability` |
   | H1d ordered substep shift | `ordered_shift` |

   Each is evaluated by error-vs-correct pair comparisons inside matched
   `(length bin, kappa bin, position bin)` strata, with chain-cluster bootstrap
   confidence intervals.

3. **Low-kappa taxonomy**

   Low-kappa rows are assigned multi-label flags and a primary visual class:

   ```text
   high_rank_dispersion
   bipolar_split
   multi_cluster
   ordered_substep_shift
   unclassified_low_kappa
   ```

   Thresholds are calibrated from low-kappa correct controls when available.
   Taxonomy is descriptive; the main evidence is the conditioned morphology
   test above.

4. **Examples for qualitative audit**

   The script writes top examples by morphology class to `*.examples.jsonl`.
   If `steps_text` exists, each example includes the step text snippet.

### Local validation

```text
python -m py_compile directional_dispersion_mechanism_audit.py
python directional_dispersion_mechanism_audit.py --selftest
python -m pytest tests/test_directional_dispersion_mechanism_audit.py -q
python -m pytest tests/test_token_stream_geometry_audit.py tests/test_second_moment_dynamics_audit.py -q
```

Selftest constructs matched low-kappa steps where errors have zero-mean
multi-axis residual scatter and correct controls have low-rank axial/ordered
scatter.  It verifies that `res_eff_rank` is recovered after matching on
length/kappa/position.

### Remote ProcessBench command

Use the full hidden ProcessBench files first because they contain true
`gold_error_step` labels.

```bash
cd /gz-data/research/demo
git pull

python directional_dispersion_mechanism_audit.py \
  --input data/features/full_gsm8k.npz \
  --policy gold_error_step \
  --label_mode first_error \
  --layer 14 \
  --nearest_layer \
  --kappa_beta 1.0 \
  --min_tokens 4 \
  --length_bins 4 \
  --kappa_bins 4 \
  --pos_bins 3 \
  --bootstrap 500 \
  --output_dir outputs/directional_dispersion_full_gsm8k
```

Harder datasets should use the same command with:

```bash
--input data/features/full_math.npz
--input data/features/full_omnimath.npz
```

If the full hidden path stored inside the npz is stale, pass:

```bash
--hidden_dir data/hidden/gsm8k
```

or the corresponding hidden shard directory for the dataset.

### Same-problem descriptive command

Same-problem sampled GSM8K does not have first-error step labels, so this is
only descriptive and must not be used for first-error mechanism claims:

```bash
python directional_dispersion_mechanism_audit.py \
  --input data/gsm8k_v2_custom.npz \
  --policy answer_format_ok \
  --label_mode chain_final \
  --layer 16 \
  --kappa_beta 1.0 \
  --length_bins 4 \
  --kappa_bins 4 \
  --pos_bins 3 \
  --bootstrap 500 \
  --output_dir outputs/directional_dispersion_gsm8k_v2_custom
```

### How to read the result

Pass as a mechanism claim if:

```text
H1a/H1b/H1c/H1d has positive conditioned delta with CI excluding 0,
and taxonomy enrichment agrees with the direction of the conditioned test.
```

Kill the spectral-mechanism story if:

```text
All morphology differences vanish after matching length/kappa/position.
```

In that case the correct conclusion is not "try another spectrum"; it is:

```text
direction-only geometry is saturated by kappa, so coherent wrong requires
source-aware anchor analysis.
```

## 2026-07-07 Full GSM8K Result

Command run on the remote full-hidden ProcessBench GSM8K file:

```text
directional_dispersion_mechanism_audit.py
  --input data/features/full_gsm8k.npz
  --policy gold_error_step
  --label_mode first_error
  --layer 14
```

### Headline

```text
rows 1560 | err 205 | chains 395 | source full_hidden L14 | label first_error
low-kappa err-rate 0.280 vs non-low 0.068
identity median 1.11e-16
```

This confirms two things:

1. The original directional-consensus observation still holds at true
   first-error-step level: low-kappa steps are substantially enriched for
   first errors.
2. The residual-scatter decomposition is numerically exact, so the following
   mechanism claims are about residual **shape**, not a hidden restatement of
   residual energy.

### Mechanism Finding

The first clean positive mechanism signal is **high-rank residual dispersion**:

```text
H1a res_eff_rank: conditioned AUROC 0.602
delta error-correct +3.669
CI [+1.983, +5.528]
```

The conditioning already matches length, kappa, and step position bins.  Thus,
among steps with comparable concentration and length, wrong first-error steps
still spread their residual directional energy across more dimensions.

Taxonomy agrees:

```text
high_rank_dispersion n=142 | error_rate=0.373 | outside=0.107 | OR=4.96
```

This is the important new result.  It supports the refined statement:

```text
The kappa drop is not only loss of the rank-one consensus component.  At true
first-error steps, the remaining residual scatter is more high-dimensional.
```

### What Did Not Survive

`bipolarity` and `signed_clusterability` do not explain GSM8K first errors:

```text
H1b bipolarity            cond 0.433 | delta -0.002 | CI [-0.005, +0.000]
H1c signed_clusterability cond 0.452 | delta -0.003 | CI [-0.006, +0.000]
```

This argues against a simple picture where wrong steps split into two opposing
directions or a clean small number of semantic clusters.

`ordered_shift` is not independently convincing after conditioning:

```text
H1d ordered_shift cond 0.502 | delta +0.000 | CI [-0.004, +0.005]
```

The taxonomy enrichment for `ordered_substep_shift` is therefore likely a
low-kappa correlate, not an independent mechanism.

### Shape Interpretation

Top conditioned features also point to diffuse angular flattening:

```text
pair_cos_q90 / q75 lower in errors
res_top8_mass lower in errors
res_eff_rank higher in errors
```

So the best current wording is:

```text
First-error steps are not primarily bipolar or cleanly multi-clustered.  They
look like a diffuse high-dimensional angular dispersion of token directions.
```

This is more precise than "the model loses consistency" and stronger than a
static detector claim.

### Caveats

`kappa`, `spread`, `residual_energy`, and `res_trace` still appear among the top
conditioned features because quantile bins leave within-bin continuous
variation.  The next version should add one stricter control:

```text
res_eff_rank residualized against continuous [logN, kappa, pos]
or nearest-neighbor matching on continuous kappa/length/position.
```

The H1a result is still meaningful because `res_eff_rank` is residual shape, not
the trace identity, but a continuous-control re-run is needed before paper
wording.

### Literature Connection: Activations And Residual Stream

This result connects to the mechanistic-interpretability residual-stream line:

- Elhage et al., *A Mathematical Framework for Transformer Circuits*,
  Transformer Circuits, 2021:
  residual stream as the shared communication channel that components read from
  and write to.
- Anthropic, *Privileged Bases in the Transformer Residual Stream*,
  Transformer Circuits, 2023:
  individual residual coordinates can become special in practice despite the
  nominal arbitrary-basis view.
- Sun et al., *Massive Activations in Large Language Models*, arXiv/COLM 2024:
  a few very large activation coordinates can act as bias/attention-sink
  mechanisms.
- Cunningham et al., *Sparse Autoencoders Find Highly Interpretable Features in
  Language Models*, ICLR 2024, and Templeton et al., *Scaling
  Monosemanticity*, Transformer Circuits/arXiv, 2024:
  residual-stream activations can be decomposed into sparse, interpretable
  feature directions.
- Lawson et al., *Residual Stream Analysis with Multi-Layer SAEs*, ICLR 2025:
  directly studies residual-stream features across layers and representation
  drift.
- Anthropic, *Circuit Tracing* / *On the Biology of a Large Language Model*,
  Transformer Circuits, 2025:
  traces computational graphs through residual/MLP/attention pathways for
  behaviors including multi-hop reasoning and hallucination.

Our result is not yet a circuit analysis.  It is a population-level residual
geometry finding.  The next bridge is to ask whether the high-rank residual
dispersion corresponds to:

```text
many SAE features becoming weakly active,
loss of a small valid-reasoning feature set,
or transport away from prompt/prefix anchor features.
```

### Next Experiments

1. **Continuous-control re-run**

   Add residualization or nearest-neighbor matching for
   `[logN, kappa, pos]`, then re-test `res_eff_rank`.

2. **Layer sweep**

   Run L10/L14/L18/L22 on `full_gsm8k`; mechanism should peak in the same
   middle-layer region as kappa if it is the same physiological event.

3. **Hard dataset replication**

   Run the exact script on `full_math.npz` and `full_omnimath.npz`.  If H1a
   strengthens with task difficulty, it becomes a credible mechanism story.

4. **Cascade test**

   Re-run with:

   ```text
   --label_mode error_and_after
   ```

   If high-rank dispersion grows after the first wrong step, we can discuss a
   cascade.  If it is only at the first wrong step, it is a local rupture.

5. **Source-aware follow-up**

   High-rank dispersion explains fragmented failures.  It does not solve
   coherent-but-wrong.  The coherent-wrong subset should go to AnchorFlow /
   source attribution rather than more direction-only spectra.

## 2026-07-07 Follow-Up Replication Result

The next remote run, with a larger step-row count than `full_gsm8k`, replicated
the same qualitative pattern:

```text
H1a res_eff_rank cond 0.576 | delta +3.169 | CI [+1.907, +4.496]
H1b bipolarity   cond 0.437 | delta -0.002 | CI [-0.004, -0.000]
H1c cluster      cond 0.464 | delta -0.002 | CI [-0.005, +0.001]
H1d ordered      cond 0.510 | delta +0.002 | CI [-0.002, +0.007]
```

Taxonomy again points in the same direction:

```text
high_rank_dispersion n=362 | error_rate=0.304 | outside=0.119 | OR=3.23
not_low_kappa        n=3038 | error_rate=0.091 | outside=0.238 | OR=0.32
```

Interpretation:

1. **H1a replicates**: wrong reasoning steps have higher residual effective
   rank even after coarse matching on length/kappa/position.
2. **Effect is real but modest**: conditioned AUROC `0.576` is weaker than the
   GSM8K pilot `0.602`; this supports a mechanism story, not a standalone
   detector.
3. **Bipolar and clean clustering remain negative**: first-error dispersion is
   not mainly a two-pole cancellation or a small number of stable semantic
   clusters.
4. **Ordered substep shift remains non-independent**: taxonomy enrichment exists
   but the conditioned hypothesis test is nearly null.

The repeated top-feature pattern is also informative:

```text
pair_cos_q90/q75 lower in errors
res_top8_mass/top4_mass lower in errors
res_participation higher in errors
res_eff_rank higher in errors
```

This converges on a narrower phrase:

```text
first-error steps exhibit diffuse high-dimensional residual angular
dispersion, not discrete cluster splitting.
```

### Immediate Methodological Risk

`kappa`, `spread`, `res_trace`, and `residual_energy` remain among the top
conditioned features.  This means the current binning is not a strict enough
control: within each kappa bin, residual continuous kappa variation still leaks
into the matched comparison.

Before using H1a in the paper, implement and run:

```text
continuous residualization:
  feature ~ spline/logistic controls of [logN, kappa, pos, n_steps]

nearest-neighbor matching:
  match each error step to correct steps with close [logN, kappa, pos]

within-chain/prefix controls when available:
  compare first-error step to pre-error correct steps in the same chain class
```

Pass criterion for the mechanism claim:

```text
res_eff_rank remains positive with CI > 0 under continuous controls,
and the effect replicates across at least GSM8K plus one harder dataset.
```

## 2026-07-07 Joint Kappa-Rank Trajectory Audit Code

Implemented `kappa_rank_joint_trajectory_audit.py` to answer two follow-up
questions:

```text
1. Do consensus loss and high-rank residual dispersion jointly define a stronger
   failure state than either signal alone?
2. Along a reasoning trajectory, do spread and residual rank rise at the same
   point, before the first error, or only after the first error?
```

### Signals

The script computes:

```text
spread = 1 - kappa
res_eff_rank = effective rank of centered residual directional scatter
spread_resid_lenpos = spread residualized over [logN, pos, n_steps]
rank_resid_lenkappapos = res_eff_rank residualized over [logN, kappa, pos, n_steps]
joint_raw_zsum = z(spread) + z(res_eff_rank)
joint_strict_zsum = z(spread_resid_lenpos) + z(rank_resid_lenkappapos)
joint_strict_min = min(z_spread_resid, z_rank_resid)
```

The important version is `joint_strict_zsum`: it keeps the consensus-loss
channel while asking whether residual rank adds beyond continuous kappa,
length, and position controls.

### Joint Failure State

The script defines four quadrants using control-step quantile thresholds:

```text
low_spread_low_rank
consensus_loss_only
rank_dispersion_only
dual_high_spread_high_rank
```

The most interpretable state is:

```text
dual_high_spread_high_rank
```

This means both:

```text
the step has weak directional consensus
and
its residual scatter is unusually high-rank after controlling for kappa/length
```

### Trajectory Outputs

The script keeps all steps and uses `gold_error_step` to label:

```text
correct_chain
pre_error
first_error
post_error
```

It writes:

```text
*.rows.csv         per-step signals and phases
*.profiles.csv     profiles by phase, normalized position, and relative step
*.transitions.csv  pre->first, first->post, and correct-chain adjacent deltas
*.json / *.md      headline summary
```

Key trajectory questions:

```text
pre->first jump > correct-chain adjacent jump?
first->post stays high or falls back?
rank_resid rises with spread or lags it?
dual_high quadrant appears mainly at first-error or also post-error?
```

### Local Validation

```text
python -m py_compile kappa_rank_joint_trajectory_audit.py
python kappa_rank_joint_trajectory_audit.py --selftest
python -m pytest tests/test_kappa_rank_joint_trajectory_audit.py -q
python -m pytest tests/test_directional_dispersion_mechanism_audit.py -q
```

### Remote Command

```bash
cd /gz-data/research/demo
git pull

python kappa_rank_joint_trajectory_audit.py \
  --input data/features/full_gsm8k.npz \
  --policy gold_error_step \
  --layer 14 \
  --nearest_layer \
  --kappa_beta 1.0 \
  --min_tokens 4 \
  --quadrant_q 0.75 \
  --bootstrap 500 \
  --output_dir outputs/kappa_rank_joint_full_gsm8k
```

For harder datasets:

```bash
python kappa_rank_joint_trajectory_audit.py \
  --input data/features/full_math.npz \
  --policy gold_error_step \
  --layer 14 \
  --nearest_layer \
  --bootstrap 500 \
  --output_dir outputs/kappa_rank_joint_full_math

python kappa_rank_joint_trajectory_audit.py \
  --input data/features/full_omnimath.npz \
  --policy gold_error_step \
  --layer 14 \
  --nearest_layer \
  --bootstrap 500 \
  --output_dir outputs/kappa_rank_joint_full_omnimath
```

If the hidden shard path in the npz is stale:

```bash
--hidden_dir data/hidden/gsm8k
```

### How To Interpret

Strong joint evidence:

```text
dual_high_spread_high_rank has high odds ratio and useful recall at moderate FPR;
joint_strict_zsum improves over spread and rank_resid with bootstrap CI > 0;
pre->first jumps are larger than correct-chain adjacent jumps for both spread
and rank_resid.
```

Mechanism-only evidence:

```text
joint score does not improve AUROC, but trajectory profiles show spread and
rank_resid co-activate at first-error and the dual-high quadrant is enriched.
```

Negative result:

```text
rank_resid does not rise at first-error after continuous controls, or dual-high
is no more enriched than consensus_loss_only.
```

If negative, the honest conclusion is:

```text
residual rank was a coarse-bin artifact; kappa/spread remains the saturated
direction-only signal, and coherent wrong must be handled by source attribution.
```

## 2026-07-07 Full GSM8K Joint Kappa-Rank Result

Remote run:

```text
kappa_rank_joint_trajectory_audit.py
  --input data/features/full_gsm8k.npz
  --policy gold_error_step
  --layer 14
```

Headline:

```text
rows 2072 | first-error 205 | pre 385 | post 512 | correct-step 970
best score spread AUROC 0.772
```

### Detection Result

Global first-error ranking is still dominated by spread:

```text
spread                 AUROC 0.772
joint_raw_zsum         AUROC 0.754
joint_strict_zsum      AUROC 0.714
res_eff_rank           AUROC 0.714
rank_resid_lenkappapos AUROC 0.605
```

Bootstrap increments confirm no all-error AUROC gain over spread:

```text
joint_strict_vs_spread -0.058 [-0.098, -0.018]
joint_raw_vs_spread    -0.018 [-0.031, -0.004]
```

Conclusion:

```text
Residual rank is not a stronger global detector than kappa/spread.
```

This should not be sold as detector improvement.

### Useful Increment: High-Precision Subtype

The dual-high quadrant is strongly enriched:

```text
dual_high_spread_high_rank n=128
error_rate 0.453
OR 7.23
recall 0.283
FPR 0.052
```

This is the useful operational signal:

```text
when consensus loss and residual-rank dispersion co-occur, the step is a
high-risk fragmented-error subtype.
```

It catches only ~28% of first errors, but at ~5% control FPR and 45% precision.
So its role is **not** "better AUROC than spread"; its role is:

```text
high-confidence alarm subtype / intervention selector.
```

The other quadrants are weaker:

```text
consensus_loss_only  error_rate 0.175 | OR 1.56
rank_dispersion_only error_rate 0.124 | OR 0.93
low-low              error_rate 0.065 | OR 0.28
```

So rank alone is not enough.  Rank matters mainly when paired with consensus
loss.

### Trajectory Result

The two signals jump together at the first-error step:

```text
spread         pre->first +0.023 | correct adjacent -0.013 | first->post -0.027
res_eff_rank   pre->first +7.674 | correct adjacent -3.686 | first->post -9.814
z_spread_resid pre->first +0.467 | correct adjacent +0.045 | first->post -0.081
z_rank_resid   pre->first +0.796 | correct adjacent +0.015 | first->post -0.557
joint_strict   pre->first +1.263 | correct adjacent +0.061 | first->post -0.638
```

Interpretation:

```text
The event is local and synchronous: consensus weakens and residual rank rises
at the first wrong step, then both partially fall back after that step.
```

This does **not** support a long precursor or accumulating cascade story on
GSM8K.  It supports:

```text
first-error = local rupture / fragmented computation event
```

### Why The Earlier Gram Audit Looked Negative

The earlier second-moment / Gram audit answered a different question:

```text
Can chain-level aggregated Gram summaries improve final-answer same-problem
classification over spread/entropy baselines?
```

This joint audit answers:

```text
At the gold first-error step, does residual-rank dispersion co-activate with
consensus loss and define a high-risk subtype?
```

Those are not equivalent.  The Gram audit diluted the effect because:

1. it aggregated over the chain, while the rank signal is sharply localized at
   first error;
2. it optimized global AUROC, while the useful result here is a low-FPR
   dual-high subtype;
3. the strongest global scalar, spread, already captures most rank-one
   consensus loss;
4. raw Gram/effective-rank features are length-sensitive unless residualized;
5. the OOF logistic fusion can learn linear combinations, but it is not designed
   to discover a sparse "both high" quadrant unless explicitly represented.

Therefore the earlier null result does not contradict this finding.  It says:

```text
Gram spectra are not a better broad final-answer detector.
```

The new result says:

```text
residual rank is a mechanistic subtype marker when aligned to the first-error
step and combined with spread.
```

### Paper Wording

Safe wording:

```text
Directional concentration remains the strongest source-free detector.  However,
conditioning on the first-error event reveals a second axis: when low
concentration is accompanied by high residual effective rank, the step enters a
high-precision fragmented-error regime.
```

Unsafe wording:

```text
Residual rank improves detection over kappa.
```

The data does not support the unsafe version.

## 2026-07-07 Note On Geometric Hallucination Metrics

Reference checked from the local `papers/推理` folder:

```text
What do Geometric Hallucination Detection Metrics Actually Measure?
arXiv:2602.09158, ICML 2025 Reliable and Responsible Foundation Models Workshop
```

The paper studies three metrics:

1. **Hidden Score (HS)**:

   ```text
   G_l = H_l H_l^T
   HS(H_l) = (1/m) log det(G_l) = (1/m) sum_i log lambda_i
   ```

   This is a token-sequence log-volume / logdet statistic.

2. **Matrix Entropy (ME)**:

   ```text
   q_i = lambda_i / trace(G_l)
   ME(H_l) = - sum_i q_i log q_i
   ```

   This is spectral entropy / effective-rank family.

3. **Attention Score (AS)**:

   ```text
   AS(A_l) = (1/(m n_heads)) sum_head sum_token log diag(A_l^head)
   ```

   This measures self-attention diagonal strength, not just attention entropy
   or prompt attention mass.

### Relation To Our Current Code

Important correction: in the paper, HS and ME are computed on the Gram matrix
of **all tokens in the whole response / reasoning chain**:

```text
H_l = [h_1^l, ..., h_m^l]^T for the whole generated response
G_l = H_l H_l^T
```

This is not the same object as a within-step token Gram.  Our existing code has
covered several related but distinct versions:

```text
seq_gram.py:
  whole-response raw H H^T HS / ME / lam1, close to the paper protocol

lam1_within.py:
  whole-response kappa / alpha / lam1 / effrank / HS with within-problem
  difficulty control for sampled rollouts

second_moment_dynamics_audit.py:
  tok_raw_logdet_mean / tok_cen_logdet_mean  ~= HS
  tok_raw_entropy / tok_cen_entropy
  tok_raw_eff_rank / tok_cen_eff_rank        ~= ME / effective rank
  but computed on step-level token slices / aggregated step summaries

vector_detect.py:
  explicitly tests [HS, eff-rank D, lam1, logE, twoNN_d] at step level
```

Therefore the precise statement is:

```text
HS/ME as a spectral family has been probed, and step-level Gram variants did
not beat kappa/spread.  However, the exact whole-chain paper protocol has not
yet been folded into the current controlled audit/reporting pipeline.
```

This matters because whole-chain Gram statistics are response-level detectors:
they can judge final correctness or best-of-chain selection, but they cannot
directly localize a first wrong step unless converted into prefix or
step-difference statistics.

AS has **not** been exactly replicated.  Existing `attn_audit.py` uses:

```text
q_frac, sink_frac, attn_entropy
```

These are anchoring/sink summaries, not the paper's diagonal attention logdet.
An exact AS replication requires saved per-token self-attention diagonal values
or re-extraction from the model.

### Why Their AUROC Can Be Much Higher

The paper's strongest AUROCs come from a different protocol:

1. synthetic prompt-response QA templates, not natural generated CoT;
2. teacher-forced prompt+response evaluation, not online detection while the
   model is generating;
3. best layer selected across all layers;
4. large hallucination types such as irrelevance, incoherence, and
   incompleteness, not only subtle first wrong reasoning steps;
5. domain-normalized perturbation baselines around the answer token.

The most important methodological point is their perturbation normalization:

```text
score_norm = (score(response) - mean score(perturbed responses))
             / std score(perturbed responses)
```

This removes domain/template effects and makes all-domain factual
incorrectness much easier.  It is closer to a local counterfactual comparison
than a plain static hidden-state scalar.

### Actionable Takeaway

Do not re-run step-level HS/ME as if they were new signals.  They are already
in the tested Gram/spectral family.

The useful follow-up is:

```text
1. exact whole-chain HS/ME replication under the current leak-free controls:
   length, same-problem grouping, bootstrap CI, and comparison to kappa/spread;
2. exact AS diagonal-logdet replication if attention diagonals can be saved;
3. perturbation-normalized reasoning audit:
   compare a generated step to local counterfactual/sibling steps for the same
   problem and same position.
```

If perturbation normalization works, the novelty is not "another geometric
metric"; it is:

```text
same-problem local counterfactual normalization turns weak geometry into a
calibrated reasoning-step monitor.
```

## 2026-07-07 Strict Whole-Chain Gram Replication Code

Implemented:

```text
whole_chain_gram_metrics_audit.py
```

Purpose:

```text
strictly reproduce the paper's HS/ME object level:
whole response / whole reasoning-chain token Gram H_l H_l^T
```

This is different from `step_gram.py`, `vector_detect.py`, and most of
`second_moment_dynamics_audit.py`, which operate on step-local token clouds or
step-level summaries.

### Exact Metrics

For each chain and layer:

```text
G_l = H_l H_l^T
HS = (1/m) log det(G_l)
ME = -sum_i q_i log q_i, q_i = lambda_i / trace(G_l)
```

Strict guardrails:

```text
HS is reported only when G_l is full rank.
No pseudo-logdet is substituted.
AS is reported only if exact self-attention diagonals are stored.
attn_entropy/q_frac/sink_frac are not used as AS substitutes.
```

### Two Evaluation Levels

1. **Chain-level replication**

   Tests whether whole-chain HS/ME/AS detect final wrong reasoning chains,
   with:

   ```text
   cross AUROC
   same-problem within-pair AUROC
   OOF group scores
   increment over [length + chain_spread]
   bootstrap CI
   ```

2. **Prefix localization adaptation**

   For ProcessBench-style first-error labels, the script computes:

   ```text
   prefix_hs / prefix_me / prefix_as
   delta_hs / delta_me / delta_as
   ```

   at reasoning-step endpoints.  This is not the original paper protocol; it
   is the deployable adaptation that asks whether whole-chain Gram metrics can
   localize the first wrong step.

### Local Validation

```bash
python -m py_compile whole_chain_gram_metrics_audit.py
python whole_chain_gram_metrics_audit.py --selftest
python -m pytest tests/test_whole_chain_gram_metrics_audit.py -q
```

The tests verify:

```text
HS/ME equal the analytic values on scaled identity matrices;
HS becomes NaN when the Gram is rank-deficient;
prefix traces are true whole-chain prefix Grams;
synthetic whole-chain and prefix signals are recovered.
```

### Remote Commands

Full hidden ProcessBench GSM8K:

```bash
cd /gz-data/research/demo
git pull

python whole_chain_gram_metrics_audit.py \
  --input data/features/full_gsm8k.npz \
  --policy gold_error_step \
  --layer 14 \
  --nearest_layer \
  --bootstrap 500 \
  --output_dir outputs/whole_chain_gram_full_gsm8k
```

Same-problem sampled GSM8K:

```bash
python whole_chain_gram_metrics_audit.py \
  --input data/gsm8k_v2_custom.npz \
  --policy answer_format_ok \
  --layer 16 \
  --nearest_layer \
  --bootstrap 500 \
  --output_dir outputs/whole_chain_gram_gsm8k_v2_custom
```

If the full-hidden shard directory is stale:

```bash
--hidden_dir data/hidden/gsm8k
```

### How To Read

Important rows:

```text
paper_exact_hs_me          strict paper HS+ME
paper_exact_hs_me_as       strict paper HS+ME+AS, only if AS exists
baseline_length_spread     length + whole-chain directional spread
baseline_plus_exact_paper  whether strict paper metrics add over baseline
```

Pass condition for a detector claim:

```text
baseline_plus_exact_paper improves over baseline_length_spread
with same-problem bootstrap CI excluding 0.
```

Pass condition for localization:

```text
delta_hs / delta_me / delta_as has high within-error-chain localization
against pre-error steps.
```

If chain-level HS/ME is strong but prefix deltas are weak, then the paper metric
is a final-response detector, not a real-time first-error monitor.

## 2026-07-07 Structural Signal Validity Plan Inspired By Know More, Know Clearer

Reference checked from local `papers/推理`:

```text
Know More, Know Clearer: A Meta-Cognitive Framework for Knowledge
Augmentation in Large Language Models
ICML 2026 / PMLR 306, arXiv:2602.12996
```

### What The Paper Actually Does

The useful methodological lesson is not their training recipe itself, but their
signal-validity argument.  They first ask whether an internal confidence signal
contains structured information aligned with performance.

Their signal:

```text
U(y | q) = -(1/T) sum_t log p_theta(x_t | q, x_<t)
```

For each query, they sample multiple reasoning paths:

```text
K = 16 responses per query
U_bar(q) = mean_k U(y^(k) | q)
Acc(q) = mean_k I(correct(y^(k)))
```

Then they aggregate across many instances:

```text
50,000 sampled instances
M = 100 uncertainty intervals
bin centroid:
  x_m = mean_{i in bin m} U_bar_i
  y_m = mean_{i in bin m} Acc_i
```

They fit:

```text
E[Acc | U] ~= a * exp(-b U) + c
```

and show this relationship across Qwen, Llama, and Mistral model families.
This is the basis for their statement:

```text
internal confidence signals carry structured information aligned with
performance, not random fluctuations.
```

They then add two additional validity layers:

1. **Calibration validity**

   Convert uncertainty to confidence:

   ```text
   c = exp(-NLL)
   ```

   and report ECE with equal-mass confidence bins:

   ```text
   ECE = sum_m |B_m|/N * |acc(B_m) - conf(B_m)|
   ```

2. **Behavioral self-knowledge validity**

   Evaluate answer/refuse decisions through:

   ```text
   AR  = TP / (TP + FP)
   KEI = TP / (TP + FN)
   NPV = TN / (TN + FN)
   CBS = harmonic_mean(AR, KEI)
   CAE = (TP + TN) / all
   ```

The key lesson for our paper:

```text
Do not only report AUROC.  Demonstrate that the internal signal induces a
stable, monotonic, calibrated, and behaviorally meaningful performance law.
```

### Our Corresponding Core Hypothesis

For reasoning traces, the analogous hypothesis should be:

```text
Internal geometric disorder is a structured meta-cognitive signal:
as directional consensus decreases and high-rank residual dispersion increases,
the empirical probability of reasoning failure rises monotonically after
controlling for length, task difficulty, and problem identity.
```

This is stronger and more paper-worthy than:

```text
low kappa has AUROC above chance.
```

### Signals To Audit

Primary source-free hidden-state signals:

```text
spread = 1 - kappa
res_eff_rank = effective rank of residual directional scatter
dual_high = high spread AND high residual rank
```

Auxiliary channels:

```text
entropy / NLL / token uncertainty
whole-chain HS / ME / AS if available
length, step position, n_steps, problem id
```

Localization signals:

```text
step spread
step residual rank
prefix_delta_HS / prefix_delta_ME
pre->first jumps
```

### Experiment 1: Structural Decay Law For Geometry

Goal:

```text
replace "signal has AUROC" with "signal defines a stable risk law".
```

For same-problem multi-sampling data:

```text
for each generated chain i:
  compute chain-level signals:
    mean_spread, max_spread, late_spread
    mean_res_eff_rank, dual_high_fraction
    entropy baselines
  y_i = final incorrectness
```

For ProcessBench first-error data:

```text
for each step j:
  compute step-level spread, res_eff_rank, dual_high
  y_j = first-error indicator
```

Analysis:

```text
1. equal-mass bins by signal value, M in {10, 20, 50}
2. within each bin:
     err_rate = mean(y)
     mean_length, mean_entropy, mean_position
3. fit monotonic / exponential / logistic laws:
     P(error | s) ~= sigmoid(a + b s)
     P(correct | s) ~= a * exp(-b s) + c
4. report:
     Spearman rho between bin signal and bin error rate
     Kendall tau
     monotonic violation count
     bootstrap CI by problem/chain
     R^2 or deviance reduction over length-only baseline
```

Required controls:

```text
length-stratified bins
same-problem paired bins
permutation within problem and length bucket
label shuffle null distribution
residualized signal after [logN, position, n_steps, entropy]
```

Pass condition:

```text
error rate rises monotonically across signal bins;
bootstrap CI excludes zero for trend;
trend remains after length/problem controls;
permuted labels/signals destroy the trend.
```

Kill condition:

```text
trend vanishes under same-problem or length-stratified controls.
```

### Experiment 2: Calibration Validity / ECE For Geometry

The paper reports ECE to show that confidence aligns with accuracy.  Our analog
should calibrate geometric risk.

Map raw signal to probability using only train folds:

```text
risk_hat = isotonic_regression(spread or joint signal -> error probability)
confidence_hat = 1 - risk_hat
```

Evaluate out of fold:

```text
ECE over equal-mass risk/confidence bins
Brier score
negative log-likelihood
reliability diagram
risk-coverage / selective answering curve
AURC / coverage at fixed error rate
```

Baselines:

```text
length-only
entropy-only
spread-only
spread + entropy
spread + residual-rank interaction
whole-chain HS/ME if chain-level task
```

Important interpretation:

```text
If geometry improves AUROC but not ECE, it is a ranking signal, not calibrated
meta-cognition.

If geometry improves ECE especially in low-entropy/confident subsets, that is a
strong "Know Clearer" style claim.
```

### Experiment 3: Multi-Sample Latent State Estimation

The paper uses K=16 samples per query to estimate latent knowledge state rather
than trusting one generation.  We should mirror this exactly on same-problem
multi-sampling data.

For each problem p with rollouts r:

```text
Acc_bar(p) = mean_r I(correct_{p,r})
S_bar(p) = mean_r signal_{p,r}
S_var(p) = var_r signal_{p,r}
S_q25/q50/q75(p)
```

Questions:

```text
Does mean geometric disorder predict problem-level failure rate?
Does within-problem signal variance identify "confused" problems?
Do correct and wrong rollouts separate within the same problem?
```

Region assignment without LLM prompting:

```text
Mastered:
  high Acc_bar, low S_bar, low S_var

Missing:
  low Acc_bar, high S_bar, low-to-medium S_var

Confused:
  intermediate Acc_bar or high S_var, mixed correctness among rollouts
```

This gives us an empirical analog of:

```text
Mastered / Confused / Missing
```

without relying on prompt-based self-evaluation.

Validation:

```text
confused region should have high within-problem disagreement;
missing region should have high error rate and high geometry risk;
mastered region should have low error and calibrated low risk.
```

### Experiment 4: First-Error Structural Law

For ProcessBench:

```text
align steps by relative event time:
  pre-error
  first-error
  post-error
  correct-chain controls
```

Report:

```text
P(first-error | spread decile)
P(first-error | residual-rank decile)
P(first-error | dual_high quadrant)
pre->first jump vs correct-chain adjacent jump
post-error relaxation or cascade
```

This connects to our current finding:

```text
dual_high_spread_high_rank:
  error_rate 0.453
  OR 7.23
  recall 0.283
  FPR 0.052
```

The refined claim should be:

```text
the strongest structural event is not a long precursor but a local rupture:
consensus drops and residual rank rises synchronously at the first wrong step.
```

### Experiment 5: Coherent-Wrong Boundary Test

The paper explicitly cares about confidence-accuracy mismatch.  Our equivalent
hard case is:

```text
coherent-but-wrong:
  low entropy
  normal/high kappa
  incorrect final answer or first wrong step
```

Required report:

```text
fraction of errors that are geometry-visible:
  high spread or dual_high

fraction of errors that are geometry-blind:
  coherent wrong

does whole-chain HS/ME or entropy catch geometry-blind errors?
does anchor/source attribution catch them?
```

This prevents overclaiming.  It also gives the paper a clean boundary:

```text
geometry detects fragmented failures; source attribution is needed for
coherent-but-wrong failures.
```

### Scripts Needed / Existing Coverage

Existing:

```text
kappa_rank_joint_trajectory_audit.py
  first-error spread/rank joint state and trajectory

whole_chain_gram_metrics_audit.py
  strict whole-chain HS/ME/AS replication and prefix adaptation

token_stream_geometry_audit.py
  segmentation-free token stream geometry
```

New script to implement:

```text
signal_structural_validity_audit.py
```

Core outputs:

```text
*.json
*.md
*.bins.csv          binned structural law
*.calibration.csv   ECE / reliability bins
*.regions.csv       Mastered / Confused / Missing analog
*.nulls.csv         permutation nulls
```

Required command targets:

```bash
python signal_structural_validity_audit.py \
  --input data/gsm8k_v2_custom.npz \
  --policy answer_format_ok \
  --layer 16 \
  --mode chain_final \
  --signals spread,res_eff_rank,dual_high,entropy,whole_hs,whole_me \
  --bins 20 \
  --permutations 1000 \
  --bootstrap 1000 \
  --output_dir outputs/signal_structural_validity_gsm8k_v2_custom

python signal_structural_validity_audit.py \
  --input data/features/full_gsm8k.npz \
  --policy gold_error_step \
  --layer 14 \
  --mode first_error \
  --signals spread,res_eff_rank,dual_high,entropy,prefix_hs,prefix_me \
  --bins 20 \
  --permutations 1000 \
  --bootstrap 1000 \
  --output_dir outputs/signal_structural_validity_full_gsm8k
```

### Paper-Level Claim If Experiments Pass

Strong claim:

```text
Reasoning failures obey a structural risk law in internal geometry: as
directional consensus deteriorates and residual scatter becomes high-rank,
empirical error probability rises monotonically under length-, problem-, and
entropy-controlled analyses.  This transforms kappa from a weak scalar detector
into a calibrated meta-cognitive signal of reasoning state.
```

Moderate claim:

```text
Directional geometry is a reliable high-risk subtype marker, especially for
fragmented first-error steps, but it does not cover coherent wrong reasoning.
```

Negative but useful claim:

```text
After rigorous structural-validity tests, geometry remains a weak ranking
signal rather than a calibrated meta-cognitive law.  The research should then
shift to anchor-source attribution.
```

### Why This Is Necessary

Top-venue reviewers will not accept:

```text
AUROC is above 0.5, therefore the hidden signal is meaningful.
```

The required evidence chain is:

```text
1. signal-performance relationship is monotonic and stable;
2. trend survives length/difficulty/problem controls;
3. calibration metrics improve over entropy/length baselines;
4. permutation/null tests destroy the trend;
5. failure cases are explicitly characterized.
```

This is the exact role played by the Structural Decay Law and ECE analyses in
`Know More, Know Clearer`, and it should become the validity backbone of our
paper.
