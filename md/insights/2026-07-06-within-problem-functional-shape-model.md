# Within-Problem Functional Shape Model

## Why the Full-Data Dynamic Audit Is Not a Research Increment

The `full_gsm8k.npz` chain-dynamics run should be treated as a negative/control result, not as a new main experiment.

Observed result:

- `anchor_uncertainty` remains the strongest OOF group: `0.811`.
- `explicit_uncertainty` is slightly worse: `0.808`.
- `dynamic_online`, `transition_ablation`, `trajectory_pattern`, and `uncertainty_dynamics` are all below `anchor_uncertainty`, with negative bootstrap increments.
- `pos` reaches within-chain localization top1 `1.000`, proving that first-error localization on this full ProcessBench split is heavily contaminated by late-error position.
- CUSUM-style scores also localize well, but this is not credible because cumulative scores naturally peak late when most errors occur late.

Conclusion:

> Cross-problem full-data dynamic detection mostly recovers difficulty, length, and late-position structure. It is useful as a sanity check, but it does not answer the research question.

The main modeling target must move back to same-problem multi-sampling, where problem difficulty is controlled by design.

## Core Hypothesis

For the same problem `p`, correct and incorrect sampled solutions share the same problem difficulty and prompt semantics. The meaningful signal is therefore not an absolute level like "high spread", but a **problem-conditioned trajectory-shape deviation**:

```text
Correct chains stay inside a problem-specific healthy trajectory tube.
Incorrect chains exhibit localized or persistent departures from that tube,
visible in hidden-cloud spread, uncertainty shape, and anchor/direction dynamics.
```

Mathematically, for sample `i` of problem `p`, channel `c`, and normalized progress `u`:

```text
X_{p,i,c}(u) = mu_{p,c}(u) + A_{p,i,c}(u) + noise
```

- `mu_p(u)` is the problem-specific healthy trajectory.
- `A_{p,i}(u)` is the failure-deviation process.
- The goal is to test whether `A` differs between correct and incorrect samples without using raw problem identity, raw position, or endpoint accumulation as shortcuts.

## Method 1: Problem-Conditioned Functional Residual Curves

Build a robust correct-chain baseline for each problem:

```text
mu_{p,c}(u) = median_{i correct} X_{p,i,c}(u)
sigma_{p,c}(u) = MAD_{i correct} X_{p,i,c}(u)
Z_{p,i,c}(u) = (X_{p,i,c}(u) - mu_{p,c}(u)) / sigma_{p,c}(u)
```

Use leave-one-out baselines for correct chains so a chain never defines its own normality.

Then compare incorrect and correct samples using:

- same-problem paired AUROC;
- sign-flip or label-permutation tests within each problem;
- cluster permutation over `(channel, u)` to identify significant non-endpoint regions.

This is the cleanest statistical test of the hypothesis.

## Method 2: Conditional Functional Logistic / Ranking Model

Avoid problem difficulty by training only on within-problem pair differences:

```text
D_{p,e,k}(u) = Z_{p,error,e}(u) - Z_{p,correct,k}(u)
label = 1 means the first member is the error chain
```

Fit a regularized functional model:

```text
score(D) = sum_c integral beta_c(u) D_c(u) du
```

Implementation choices:

- represent `beta_c(u)` with B-splines or low-order DCT coefficients;
- use elastic-net logistic regression on pairwise differences;
- smoothness penalty or total-variation penalty to avoid endpoint spikes;
- GroupKFold by problem;
- report selected time regions and coefficient signs.

This gives an interpretable mathematical object: which trajectory phases and channels separate error from correct samples under same-problem control.

## Method 3: Local Shapelet / Motif Discovery

Global curve averages may miss short failure motifs. Add train-only shapelet discovery:

1. Generate candidate windows from training problems only.
2. Score each window by within-problem effect size:

```text
effect(window) = mean_error(window residual energy) - mean_correct(window residual energy)
```

3. Keep non-overlapping windows after permutation/FDR control.
4. Use distances or pooled residuals over those windows as features.

Anti-artifact rule:

- windows in the final 20% must be reported separately;
- a method only counts as useful if non-endpoint windows remain significant or if endpoint-only behavior is explicitly framed as late-answer monitoring, not reasoning-process detection.

## Method 4: Monotone Hidden Semi-Markov Failure State

Use a small latent-state model instead of raw CUSUM:

```text
state_t in {healthy, uncertain-drift, detached, collapsed}
```

Constraints:

- mostly monotone transition from healthy toward failure states;
- duration penalty prevents one-step noise from becoming a failure state;
- emissions are problem-conditioned residual vectors `Z_{p,i}(u_t)`;
- fit by EM on training problems, evaluate by held-out same-problem pair AUROC and state localization.

This is the mature version of the earlier "EM-like" idea. It is not just fitting a scalar latent score; it models failure regimes and dwell times.

## Method 5: Transport to Correct Barycenter

For each problem, compute a correct-chain barycenter trajectory:

```text
B_p(u) = barycenter({X_{p,i}(u): i correct})
```

Measure incorrect-chain deviation by local transport cost:

```text
cost_{p,i}(u) = || Z_{p,i}(u) - B_p(u) ||_{Sigma_p(u)^{-1}}
```

Use soft-DTW or local window alignment only as an ablation, because excessive warping can erase real failure timing.

## Required Anti-Degradation Checks

Every proposed method must report:

- within-problem paired AUROC, not only cross-problem AUROC;
- increment over static same-problem baselines such as mean/max `cloud_spread`;
- endpoint-only fraction of selected windows or alarms;
- performance with the last 20% of steps censored;
- length and number-of-step controls;
- train-only feature/shapelet discovery under GroupKFold by problem;
- permutation p-values by shuffling labels within each problem.

## Immediate Code Target

Build `within_problem_shape_model_audit.py` for `gsm8k_v2_5shot.npz` and `gsm8k_v2_custom.npz`.

First version:

1. Load available same-problem channels: `cloud_spread`, `out_entropy`, `pr_mid`, `ae_mid`, and later `out_committal` if re-extracted.
2. Interpolate each chain to a fixed progress grid.
3. Construct problem-conditioned residual curves against correct-chain baselines.
4. Run:
   - functional cluster permutation;
   - pairwise functional logistic ranking;
   - local shapelet discovery.
5. Compare all methods against static same-problem baselines and endpoint controls.

Pass condition:

```text
The method improves same-problem paired AUROC over static spread,
has a significant within-problem permutation result,
and does not reduce to endpoint/length artifacts.
```

