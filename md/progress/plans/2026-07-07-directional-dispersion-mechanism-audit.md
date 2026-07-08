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

## 2026-07-08 Geometry / Topology Reference Audit

### Why Run More Geometry If AUROC Does Not Increase?

The purpose cannot be "try another geometry scalar and hope for AUROC".  The
results so far already show that many token-direction geometry variants are
mostly proxies for the same dispersion axis:

```text
resultant = 1 - spread
lambda_1 / spectral entropy / effective rank = spectral views of concentration
unit effective rank = angular dispersion proxy
raw effective rank = angular dispersion + norm/length/difficulty proxy
```

Therefore a geometry experiment is necessary only if it serves one of three
roles:

1. **Detector role**: it gives a leak-free same-problem / length-controlled
   increment over spread, entropy, and difficulty baselines.
2. **Mechanism role**: it explains what kind of failure a low-kappa step is
   (for example high-rank residual dispersion vs bipolar cancellation), even if
   it does not improve global AUROC.
3. **Boundary role**: it cleanly rules out a tempting family of explanations, so
   the paper can justify moving from "shape of a token cloud" to source-aware
   anchor attribution.

If an experiment satisfies none of these roles, it should not be repeated.  The
stop rule is: once a leak-free controlled audit shows no increment, the signal is
kept only as a mechanism/control variable, not as a proposed detector.

### What Gram Matrix Are We Talking About?

Different papers and scripts use different matrices.  They should not be
collapsed into one "Gram" bucket.

```text
Step-local token Gram:
  G_j = H_j H_j^T for tokens inside one reasoning step.
  This is the matrix behind step-level spread / effective-rank mechanism tests.

Sliding-window token Gram:
  G_t(W) = H_{t-W:t} H_{t-W:t}^T along the generated stream.
  This asks whether dispersion evolves continuously without step segmentation.

Whole-chain token Gram:
  G = H_all H_all^T over all generated tokens in the answer.
  This is closest to geometric hallucination metrics such as HS / ME.

Step-trajectory Gram:
  S_{ij} = v_i^T v_j over step embeddings or step summary vectors.
  This is about path geometry across reasoning steps, not token scatter inside a
  step.

Attention graph Laplacian:
  L = graph_laplacian(A) from token attention weights.
  This is not a hidden-state Gram.  It treats hidden states as graph signals over
  an attention-induced token graph.

TDA distance matrix:
  D_{ij} = distance(e_i, e_j) over step or sentence embeddings.
  This is a point-cloud topology object, usually analyzed via persistent
  homology rather than Gram eigenvalues.
```

### `Geometry of Reason` Implementation

`Geometry of Reason: Spectral Signatures of Valid Mathematical Reasoning` does
not simply compute the effective rank of hidden-token Gram matrices.  Its core
object is an **attention graph**:

1. Tokens are graph nodes.
2. Attention weights define weighted edges, aggregated across heads into one
   graph per layer.
3. The graph is symmetrized / normalized into a Laplacian.
4. Layer hidden states are treated as graph signals living on this token graph.
5. The hidden signal is projected onto Laplacian eigenmodes.
6. The paper extracts graph-spectral diagnostics:
   - high-frequency energy ratio (HFER),
   - graph smoothness / Dirichlet energy,
   - spectral entropy,
   - Fiedler value / algebraic connectivity.

The classifier is deliberately simple: select a layer / metric / threshold on
validation data, then classify by threshold.  The paper reports strong
valid/invalid proof separation, but this is not directly equivalent to our
setting because their signal combines two sources:

```text
attention topology  +  hidden-state graph signal
```

This is genuinely different from our step-local kappa family.  If we can store
attention maps, this is worth testing as an attention-graph spectral audit.  If
we only use hidden states, then re-implementing HFER as another hidden Gram
spectral scalar would not reproduce the paper's method.

### `The Shape of Reasoning` Implementation

`The Shape of Reasoning: Topological Analysis of Reasoning Traces in Large
Language Models` is also not a token-hidden Gram method.  It operates on
reasoning traces as paths / point clouds:

1. Split model and expert solutions into reasoning steps.
2. Embed each step in a semantic embedding space.
3. Use Smith-Waterman local alignment to compare model traces with expert
   traces.
4. Construct Vietoris-Rips filtrations from pairwise step distances.
5. Compute persistent homology features, especially H0 and H1:
   - H0: fragmentation / connected components / clustering.
   - H1: loops, detours, leaving and returning to reasoning regions.
6. Regress alignment quality using graph features, TDA features, and combined
   features.

Its target is not first-error AUROC.  Its target is how well the geometry /
topology of a reasoning path explains alignment with expert reasoning.  For our
project, this becomes useful only if we have either:

```text
expert/reference traces, or
same-problem multi-sampling where high-confidence correct traces act as an
empirical reference manifold.
```

Without a reference path, TDA can still describe trajectory shape, but it cannot
by itself prove that a loop or component is an error rather than a valid
alternative route.

### Concrete Implication For This Project

The next geometry work should be split into three bins:

```text
Stop as detector:
  More kappa / effective-rank / hidden Gram variants inside the same step token
  cloud, unless they show controlled increment.

Keep as mechanism:
  Residual effective rank and dual-high spread+rank quadrants.  They identify a
  high-risk fragmented subtype but do not beat spread globally.

Test as genuinely new information:
  Attention-graph spectral metrics from Geometry of Reason, if attention maps
  are available.
  Reference-relative TDA / path topology from Shape of Reasoning, if expert
  traces or same-problem correct-trace references are available.
  Anchor-source attribution, because it asks where the coherent direction points
  rather than only how concentrated it is.
```

The paper narrative should therefore be:

```text
Directional consensus loss is the validated base phenomenon.
Hidden-token geometry around this phenomenon is mostly saturated by spread.
Residual rank reveals a fragmented-error mechanism but not a stronger detector.
To address coherent-but-wrong failures, the method must move from geometry of
dispersion to geometry of source attribution: what premise, prior step, or
self-generated claim the current step is anchored to.
```

## 2026-07-08 Hidden Graph Signal / Chain Risk / Anchor Attribution Plan

### Hidden Graph Signal Means Graph Fourier Energy Of Hidden States

In the attention-graph papers, a hidden state matrix is not treated as an
ordinary token cloud.  It is treated as a **graph signal**:

```text
tokens = graph nodes
attention weights = graph edges
hidden state coordinates = scalar/vector signals living on nodes
```

For one layer `l`:

```text
A_l^h in R^{N x N}       attention matrix for head h
A_l = aggregate_h(A_l^h) aggregated token graph
W_l = symmetrize(A_l)    undirected weighted graph for spectral analysis
D_l = diag(W_l 1)
L_l = I - D_l^{-1/2} W_l D_l^{-1/2} normalized graph Laplacian
H_l in R^{N x d}         hidden states for the same tokens
L_l = U Lambda U^T
C_l = U^T H_l            graph Fourier coefficients of hidden states
e_k = ||C_l[k, :]||_2^2  hidden energy in graph frequency k
```

Then the usable features are:

```text
HFER          = sum_{lambda_k > cutoff} e_k / sum_k e_k
smoothness    = tr(H_l^T L_l H_l) / ||H_l||_F^2
spectral_ent  = -sum_k q_k log q_k, q_k = e_k / sum_j e_j
graph_effrank = exp(spectral_ent)
fiedler       = second-smallest eigenvalue of L_l
```

This is different from hidden-token Gram effective rank:

```text
hidden Gram eff_rank: spectrum of H H^T
graph signal spectrum: energy of H after projection onto attention-Laplacian modes
```

The second one asks whether hidden representations vary smoothly over the
model's own attention graph.  It can catch a failure mode that a token cloud
cannot: the hidden states may still look concentrated, but their energy may move
to high-frequency attention modes, meaning neighboring / mutually attending
tokens no longer carry compatible information.

### Whole-Chain Effective Rank Is Already Computable

`whole_chain_gram_metrics_audit.py` already computes response-level Gram
statistics:

```text
H_l = all generated tokens in one response at layer l
G_l = H_l H_l^T
q_i = lambda_i / trace(G_l)
paper_me = -sum_i q_i log q_i
paper_eff_rank = exp(paper_me)
paper_lam1 = max_i q_i
paper_hs = mean_i log lambda_i, only if full rank
```

It also computes prefix versions at step endpoints:

```text
prefix_eff_rank
prefix_me
prefix_lam1
delta_eff_rank
delta_me
delta_lam1
```

The important distinction:

```text
step-local eff_rank     = local mechanism / first-error subtype
whole-chain eff_rank    = final response-level geometry
prefix delta eff_rank   = online adaptation for first-error localization
```

Whole-chain effective rank may have high final-answer AUROC, but it is very
likely entangled with response length, task difficulty, and verbosity.  It must
therefore be reported with length/problem controls, not as a standalone
"reasoning detector".

### Keeping Step Detection Power While Predicting Whole Response Correctness

The clean response-level object is not an average of step scores.  Averaging
dilutes local failures and recreates the length confound.  Use a survival /
hazard view:

```text
h_j = P(step j is the first wrong step | no previous wrong step, prefix up to j)
P(response wrong) = 1 - product_j (1 - h_j)
```

Recommended chain-level detector:

```text
step hazard features:
  spread_j
  residual_rank_j
  dual_high_j
  entropy_j
  anchor/source features_j
  length_j, position_j, problem controls

chain aggregators:
  noisy_or = 1 - prod_j(1 - h_j)
  max_hazard
  top2_mean_hazard
  early_prefix_hazard

chain global features:
  whole_spread
  whole_eff_rank / ME / HS
  whole attention-graph HFER / smoothness if attention is available
```

Evaluation must include:

```text
chain-level AUROC / AUPRC for final answer correctness
same-problem grouped splits
length- and entropy-controlled baselines
calibration / ECE
rescue cases where step hazard catches errors missed by whole-chain features
```

This preserves the local detection power because a single high-risk step can
raise `noisy_or`, while whole-chain geometry acts as a slower response-level
context variable.

### Anchor / Source Attribution Implementation

The current AnchorFlow code is a first-pass scaffold.  It parses anchors from
prompt text, but if prompt-span hidden vectors are unavailable it builds anchor
vectors by partitioning `qvec`.  That mode is explicitly named
`q_partition_fallback`; it is useful as a software scaffold but not enough for a
semantic "失锚" claim.

The real implementation should define source spans:

```text
prompt anchors:
  goal span
  number spans
  entity/context spans
  constraint spans

generated anchors:
  previous step spans
  current recent-token span
  first-error / tainted-step spans when labels are available for analysis
  answer span
```

Then build two source-attribution channels.

#### Channel A: Direct Attention Source Attribution

If full attention maps are saved:

```text
for current step tokens T_j and source span S_k:
  mass_{j,k}^{layer,head} = mean_{t in T_j} sum_{i in S_k} A_{layer,head}[t, i]
```

Features:

```text
prompt_core_mass      mass to goal/number/constraint anchors
previous_step_mass    mass to earlier generated reasoning
self_recent_mass      mass to current local text
tainted_mass          mass to wrong previous step, analysis-only when labels exist
anchor_entropy        entropy over source spans
source_jump           ||mass_j - mass_{j-1}||
detach                1 - max_k mass_{j,k}
coherent_wrong_flag   high kappa + low prompt_core_mass + high self/tainted mass
```

Attention is necessary if the claim is "the model is reading from / routing
through the wrong source", because attention is closer to an internal routing
object than hidden-vector similarity.

#### Channel B: Hidden Source Transport

If attention is not stored, use hidden states but with real source-span vectors,
not qvec partitions:

```text
a_k = pooled hidden vector of source span S_k
v_j = pooled hidden vector of current step j
P_{j,k} = softmax(cos(v_j, a_k) / tau)
```

Features mirror attention:

```text
hidden_core_mass
hidden_self_mass
hidden_tainted_mass
hidden_anchor_entropy
hidden_detach
hidden_transport_jump
```

This can be useful and cheaper than attention, but it supports a weaker claim:

```text
current representation is geometrically closer to source X
```

not the stronger causal/routing claim:

```text
the model attends to source X while computing the current tokens
```

### Minimal Strong Experiment

The next serious experiment should compare four models of evidence:

```text
M0: length + position + entropy
M1: M0 + spread / kappa
M2: M1 + whole-chain / prefix Gram features
M3: M1 + attention source attribution
M4: M1 + hidden source transport
M5: M1 + attention source + hidden source
```

Required controls:

```text
random anchors
shuffled anchor kinds
wrong-problem anchors
q_partition_fallback explicitly separated from real prompt-span anchors
same-problem grouped split
bootstrap CI for increments over M1
coherent-wrong subset: high kappa but wrong final answer / wrong step
```

The decisive outcome is not only global AUROC.  The key paper-level question is:

```text
Can source attribution rescue coherent-but-wrong cases where spread/kappa is
high and ordinary dispersion geometry says the step looks healthy?
```

If yes, the method becomes a two-axis theory:

```text
fragmented failure = loss of directional consensus
coherent wrong     = coherent anchoring to the wrong source
```

## 2026-07-08 ReDeEP Paper-Strategy Notes

Reference: `REDEEP: Detecting Hallucination in Retrieval-Augmented Generation
via Mechanistic Interpretability`, ICLR 2025.

### What ReDeEP Actually Uses

ReDeEP is powerful partly because it does **not** introduce many metrics.  It
uses two mechanism-grounded scores:

```text
ECS: External Context Score
PKS: Parametric Knowledge Score
```

ECS measures whether external context attended by a head is semantically retained
in the generated token:

```text
1. For generated token t, take attention from t to retrieved-context tokens.
2. Select top-k% attended context tokens for a layer/head.
3. Mean-pool final-layer hidden states of those attended context tokens.
4. Compute cosine similarity to the final-layer hidden state of t.
5. Average over response tokens.
```

Chunk ECS replaces token hidden cosine with chunk-level attention pairing plus
embedding cosine between response chunk and highest-attended context chunk.

PKS measures how much a FFN layer changes the token's vocabulary distribution:

```text
x_mid^l = residual stream after attention, before FFN
x^l     = residual stream after FFN
q(x)    = softmax(LogitLens(x))
PKS_l   = JSD(q(x_mid^l), q(x^l))
```

The detector is a simple signed combination:

```text
hallucination_score = sum_l alpha * PKS_l - sum_{l,h} beta * ECS_{l,h}
```

Attention heads and FFN layers are selected by validation-set correlation /
grid search.  The intervention AARF then amplifies Copying Head outputs and
reduces Knowledge FFN outputs when the token-level score exceeds a threshold.

### How The Paper Is Organized

The paper's strength is not metric complexity.  It is the evidence chain:

```text
1. Problem reframing:
   RAG hallucination is separated from knowledge conflict.  The retrieved
   context can be correct while the response still conflicts with it.

2. Causal framing:
   Existing methods are grouped by what they confound:
   - parametric knowledge confounded by external context;
   - external context confounded by parametric knowledge;
   - mixed methods that do not decouple the two.

3. Mechanistic empirical study before the method:
   RQ1: Are ECS/PKS statistically related to hallucination?
   RQ2: Do Copying Heads / Knowledge FFNs matter under intervention?
   RQ3: What changes when parametric knowledge already knows the answer?

4. Method:
   A very simple regression / threshold detector becomes credible because the
   mechanism study already made the variables meaningful.

5. Necessary experiments:
   - multiple datasets: RAGTruth and Dolly(AC);
   - multiple backbones: LLaMA2-7B/13B and LLaMA3-8B;
   - broad baselines across parametric-only, external-only, and mixed methods;
   - ablation: Only PKS, Only ECS, Full ReDeEP;
   - intervention: noise Copying Heads, amplify FFNs, matched controls;
   - mitigation: AARF pairwise truthfulness comparison;
   - efficiency analysis;
   - sensitivity to Top-K heads / FFNs and weights;
   - case study.
```

### What We Should Borrow

We should not copy their metrics.  The transferable pattern is:

```text
few telemetry variables + strong mechanism story + hard controls
```

For our reasoning project, the analogous structure should be:

```text
Axis 1: directional consensus / fragmentation
  spread, kappa, residual effective rank

Axis 2: anchor-source attribution
  prompt/source mass, self-source mass, tainted-prefix mass,
  transport jump, anchor entropy

Optional Axis 3: graph-signal roughness
  attention-graph HFER / smoothness if attention maps are stored
```

Then the paper-level RQs become:

```text
RQ1: Do first-error steps show a stable directional-consensus loss after
     length, position, entropy, and problem controls?

RQ2: Does residual rank explain what kind of low-kappa event this is, or is it
     merely a proxy for spread?

RQ3: Do coherent-but-wrong cases preserve directional consensus but shift source
     anchoring toward self-generated or tainted prefixes?

RQ4: Can a step-hazard / noisy-or model turn local telemetry into final response
     correctness without losing localization?

RQ5: Do interventions or re-anchoring prompts reduce downstream errors when
     triggered by the telemetry state?
```

The novelty should not be `q_frac`, `sink_frac`, or `attn_entropy`.  Those are
standard-ish attention lookback / sink / focus summaries.  The novelty must be
the **typed reasoning-state model**:

```text
fragmented wrong = direction consensus collapses;
coherent wrong   = direction remains stable but source attribution is wrong;
response risk    = accumulated first-error hazard, not averaged scalar scores.
```

### Current Local Attention Metrics Are Not The Novel Contribution

Our existing attention features are:

```text
q_frac       = attention mass from current step tokens to prompt/question span
sink_frac    = attention mass to position 0 / BOS sink
attn_entropy = entropy of token attention distributions
```

They are extracted in `extract_features.py --attn_sink` and audited in
`attn_audit.py` as an increment over length, uncertainty, and geometry.  They
are useful as baseline source-flow summaries, but they are close to existing
ideas:

```text
q_frac       resembles attention lookback / context-use ratio;
sink_frac    resembles attention-sink / massive-activation diagnostics;
entropy      is a generic attention focus/diffuseness measure.
```

Therefore they should be presented as controls or first-pass proxies.  A genuine
contribution needs typed anchors and source attribution:

```text
goal / number / constraint / entity / previous-step / self-recent /
tainted-prefix source masses,
plus source jump and anchor entropy,
tested specifically on coherent-but-wrong failures.
```

## 2026-07-08 Core Hypothesis And Organic Metric Synthesis

The working hypothesis should be stated more narrowly than "correct reasoning
is concentrated":

```text
During stepwise reasoning, a step is reliable when the model maintains a
coherent local computation and keeps that computation anchored to valid
sources: the question constraints and the verified prefix.  Errors arise when
either the local computation fragments, or the computation remains coherent but
is anchored to the wrong source.
```

This gives two distinct failure modes:

```text
fragmented wrong:
  the step has low directional consensus / high residual rank;
  this is the phenomenon already observed by kappa and spread.

coherent-but-wrong:
  the step can keep high directional consensus and low entropy;
  the hidden state is internally stable, but attention/source flow points to a
  self-generated, tainted, stale, or irrelevant anchor.
```

The role of attention is therefore not "another feature."  It is the missing
source-attribution axis.  Hidden geometry answers:

```text
Are the tokens in this semantic step forming one computation or many competing
directions?
```

Attention/source flow answers:

```text
Which earlier evidence is this computation using?
```

Entropy dynamics answers:

```text
Is the output distribution unstable while this computation is generated?
```

### Metric Blocks

For step `j`, define the following blocks with problem / length / position
controls and leak-free same-problem splits.

```text
H_j: hidden consensus block
  spread_j = 1 - ||sum_t u_t|| / n_j
  rank_j   = effective_rank of residual token covariance
  frag_j   = relu(z(spread_j)) * relu(z(rank_resid_j))

A_j: typed source-flow block
  prompt_mass_j      = attention/saliency mass to question and constants
  verified_mass_j    = mass to previous correct / non-tainted steps
  self_recent_mass_j = mass to current step and immediately previous text
  tainted_mass_j     = mass to steps after the first known wrong step
  anchor_entropy_j   = entropy over typed source bins
  source_jump_j      = distance between source distributions at j-1 and j

G_j: hidden-source alignment block
  hidden anchor vector for source bin b:
    a_b^l = mean hidden vector of tokens in source bin b
  step vector:
    s_j^l = mean / resultant direction of tokens in step j
  align_j(b) = cosine(s_j^l, a_b^l)
```

The key interaction is not linear fusion but typed state assignment:

```text
fragmentation risk:
  high frag_j

lost-anchor risk:
  low prompt/verified flow + high self_recent/tainted flow

coherent-wrong risk:
  low spread_j + low entropy_j + high source_mismatch_j

uncertain-wrong risk:
  high EDIS-style entropy instability + high frag_j
```

This can be implemented as a step hazard model:

```text
p(error at step j | prefix) =
  sigmoid(b + controls
          + beta_f * frag_j
          + beta_a * lost_anchor_j
          + beta_c * coherent_wrong_j
          + beta_u * uncertainty_instability_j
          + beta_fa * frag_j * lost_anchor_j)

p(chain wrong) = 1 - product_j (1 - p(error at step j)).
```

This preserves localization while producing a whole-response risk.  It also
gives a clean ablation story:

```text
hidden only        -> detects fragmented wrong;
attention only     -> detects source loss but may miss internal fragmentation;
entropy only       -> detects unstable uncertainty;
typed interaction  -> targets coherent-but-wrong and propagated-error cases.
```

### Relation To StepFlow And EDIS

StepFlow is an attention-gradient flow paper.  It pools token saliency into
question / thinking-step / summary blocks and identifies two recurring failures:

```text
shallow lock-in:
  shallow layers over-focus on current or adjacent steps.

deep decay:
  deep layers lose saliency on the thinking segment, and the summary becomes
  dominated by itself and the last few steps.
```

Its intervention repairs those failure modes:

```text
Odds-Equal Bridge:
  shallow-layer attention mass is rebalanced toward bridge context.

Step Momentum Injection:
  a small residual vector from the previous step is injected at selected deep
  layers to maintain step-to-step continuity.
```

EDIS is an output-entropy dynamics paper.  It computes token entropy over the
generated trajectory and detects:

```text
burst spikes:
  sustained entropy growth over a local window.

peak-valley spikes:
  rebounds from a previous low-entropy valley.

EDIS = instability_count * (1 + entropy_variance).
```

EDIS is strong for unstable failures, but it is not designed for confident
wrong steps.  StepFlow is strong for information-flow failures, but it does not
use our hidden consensus phenomenon.  Our story should therefore be:

```text
Reasoning failure is a coupled hidden-flow event.
  hidden geometry tells whether the local computation is coherent;
  source flow tells whether the coherent computation is anchored correctly;
  entropy dynamics tells whether the decoder is uncertain.
```

### Step-Labeled Datasets To Use

Priority datasets:

```text
ProcessBench:
  human first-error labels for stepwise mathematical reasoning; best match for
  our first-error localization setting.

PRM800K:
  large human step-level correctness labels on MATH solutions; good for scale,
  but labels are step rewards rather than necessarily first-error trajectories.

MR-GSM8K:
  includes first-error step and error reason annotations for meta-reasoning
  variants; useful for easier GSM-style controlled validation.

AgentProcessBench:
  human step-level annotations for tool-using agent trajectories; useful only
  after the method works on math, because the task semantics are broader.
```

Secondary / synthetic supervision:

```text
Math-Shepherd:
  automatically constructed process supervision for GSM8K/MATH; useful for
  pretraining or stress tests, weaker as final evidence because labels are not
  human first-error labels.

FG-PRM:
  synthetic fine-grained hallucination categories at the reasoning-step level;
  useful for testing whether source-attribution separates error types.
```

### Next Code Tasks

```text
1. Build typed step-source bins from available token/step spans:
   prompt, current step, previous step, earlier verified prefix, tainted prefix,
   summary/answer.

2. Implement source-flow summaries from stored attention if available:
   source mass, source entropy, source jump, shallow/deep layer split.

3. Implement hidden-source alignment without attention:
   source anchor vectors from hidden states, step-to-source cosine and nearest
   source type.  This tests whether residual geometry alone can recover
   anchor/source attribution.

4. Fit the step hazard model with grouped same-problem splits:
   controls -> hidden -> source-flow -> entropy -> typed interactions.

5. Evaluate three endpoints:
   first-error step AUROC / AUPRC,
   coherent-wrong subset lift,
   whole-chain wrong risk via noisy-or aggregation.
```

### Implemented: Constraint Anchor Flow Audit

`constraint_anchor_flow_audit.py` implements the first falsifiable version of
the anchor-flow idea.

Core object:

```text
p_hidden_j(a), a in {question, earlier_prefix, recent_prev, other}
```

This is not a single hidden-vector cosine.  For each source type it builds a
small hidden subspace from the corresponding token rows, computes the current
step's projection-density into that subspace, and converts the source scores
into an anchor posterior.

The script also builds a numeric text posterior:

```text
p_text_j(a)
```

from the step's surface numbers and their support in the question / earlier
prefix / immediate previous step.  The key tests are:

```text
text_hidden_kl:
  the step claims to use one set of constraints, but hidden energy is anchored
  elsewhere.

anchor_transition_js:
  the hidden anchor posterior changes abruptly from the previous step.

risk_coherent_hijack:
  kappa is high, but hidden source mass moves toward recent/self/other anchors.
```

Run:

```bash
python constraint_anchor_flow_audit.py \
  --input data/features/full_gsm8k.npz \
  --layer 14 \
  --nearest_layer \
  --hidden_dir data/hidden/gsm8k \
  --anchor_rank 4 \
  --posterior_temp 0.08 \
  --control_pool pre_and_correct \
  --folds 5 \
  --bootstrap 1000 \
  --output_dir outputs/constraint_anchor_flow_full_gsm8k
```

For `sv_clouds` files:

```bash
python constraint_anchor_flow_audit.py \
  --input data/gsm8k_v2_custom.npz \
  --layer 16 \
  --nearest_layer \
  --anchor_rank 4 \
  --posterior_temp 0.08 \
  --control_pool pre_and_correct \
  --folds 5 \
  --bootstrap 1000 \
  --output_dir outputs/constraint_anchor_flow_custom
```

Decision rule:

```text
Support the anchor-flow hypothesis only if:
  OOF:baseline+anchor or an anchor score improves same-problem paired AUROC over
  the strongest spread/length/entropy baseline, and
  the coherent low-spread slice still shows useful lift.

Reject or revise if:
  gains disappear after same-problem pairing, or the best anchor score is just
  position/length in disguise.
```
