# Localized Methods for Correct-vs-Error Reasoning Trajectory Differences

Date: 2026-07-06

This note answers a narrower question than the monitor plan:

> What methods can study trajectory changes and distinguish statistically different correct/error reasoning samples on our local data, without collapsing back to simple EM or raw AUROC feature hunting?

## 1. Ground Rules From Our Data

Any method must obey the project constraints already learned the hard way:

1. Use `answer_format_ok` as the main same-problem policy where applicable.
2. Report same-problem paired AUROC, not only cross-problem AUROC.
3. Residualize or stratify by `log step length`, normalized position, and format behavior.
4. Treat raw volume/effective-rank metrics as suspicious until length-controlled.
5. Report where the method fires: early, around first error, or only near the endpoint.
6. Use GroupKFold/cluster bootstrap by problem, not random token/step splits.

## 2. Best First Method: Paired Functional Trajectory Test

### Idea

Represent each chain as a time-normalized multichannel function:

```text
X_i(u) = [
  spread(u),
  entropy(u),
  committal_uncertainty(u),
  transition_residual(u),
  anchor_detachment(u)
],  u in [0, 1]
```

Then compare correct vs incorrect samples within the same problem using a paired or grouped functional two-sample test.

### Why it fits

This directly answers:

```text
At which phase of the trajectory do wrong samples differ from correct samples?
```

It gives more than AUROC: it identifies statistically significant time regions.

### Local adaptation

1. Split each response into step sequence or fixed token windows.
2. Interpolate each signal to a common grid, e.g. 20 or 50 points.
3. Within each problem, subtract the problem-level correct mean trajectory if correct samples exist.
4. Fit a null by label permutation within problem.
5. Use cluster-based permutation testing over adjacent time bins to avoid multiple-comparison fishing.

### Outputs

- significant time clusters;
- effect direction;
- effect size curve with confidence band;
- whether the effect survives length residualization.

### Pass condition

The method is useful if it finds a stable mid-trajectory or first-error-adjacent cluster, not just an endpoint cluster.

## 3. Most Useful Classifier: Path Signature / Logsignature

### Idea

A path signature is a compact representation of trajectory shape.  It captures ordered interactions such as:

```text
entropy rises before spread rises
spread rises while anchor mass drops
uncertainty rebound follows a confident valley
```

These are exactly the "shape over magnitude" effects that simple averages miss.

### Local adaptation

Use channels already available or cheap to extract:

```text
[spread, U_D, out_entropy, out_committal, step_jump, direction_jump]
```

Then:

1. length-normalize each channel;
2. add time as an explicit channel;
3. compute logsignature up to order 2 or 3;
4. classify with L1/L2 logistic regression;
5. validate with GroupKFold by problem;
6. test increment over `cloud_spread + U_D + length`.

### Why better than EM

EM/HMM tends to learn broad latent states.  Logsignature features encode the ordering of changes without requiring fragile latent-state assumptions.

### Pass condition

Stable OOF increment over static spread/uncertainty, plus interpretable top terms such as `entropy -> spread` or `anchor_detachment -> spread`.

## 4. Best Online Detector: Conformal Kernel Change-Point

### Idea

Instead of scalar CUSUM over one score, use a multivariate kernel or energy-distance residual:

```text
r_t = distance( current prefix state, healthy prefix/tube distribution )
```

Then calibrate the alarm on correct chains using conformal thresholds.

### Local adaptation

Healthy reference choices, from safest to riskiest:

1. prefix-only self baseline within the same chain;
2. problem-type / dataset-level correct-chain baseline;
3. same-problem correct support, only as oracle diagnostic;
4. real prompt-anchor transition tube once anchors are available.

Candidate distances:

- energy distance;
- MMD with RBF kernel;
- robust Mahalanobis in residualized feature space;
- sliced Wasserstein over token-cloud summaries.

### Outputs

- FPR-controlled alarm;
- recall at FPR 5% and 20%;
- delay from first-error step;
- early-warning fraction.

### Pass condition

Improves over current conformal CUSUM without just firing at the last step.

## 5. Best Statistical Model: Mixed-Effects Functional Logistic Model

### Idea

Use interpretable temporal basis coefficients with problem-level random effects:

```text
logit P(error) =
  beta0
  + problem_random_intercept
  + sum_k beta_k * basis_k(trajectory)
  + controls(length, position, format)
```

Basis choices:

- cubic splines over normalized time;
- functional PCA components;
- wavelets for local spikes;
- hand-built EDIS/EDRM basis features.

### Why it fits

It gives a statistical test:

```text
Do correct and wrong trajectories differ after controlling for problem identity and length?
```

### Local adaptation

Use same-problem multisample data first, because it naturally supplies problem random effects.

### Pass condition

Significant trajectory coefficients survive:

- problem grouping;
- length controls;
- bootstrap confidence intervals;
- held-out problems.

## 6. Best EM-Like Upgrade: Monotone Hidden Semi-Markov Model

If we still want an EM-family method, use a constrained model rather than a generic HMM.

### States

```text
Healthy constrained flow
Uncertain exploration
Committed wrong basin
Persistent uncertainty
Detached from prompt constraints
Answer finalization
```

### Constraints

- left-to-right or weakly monotone transitions;
- minimum duration per state;
- no arbitrary rapid state flipping;
- emissions are residualized features, not raw features;
- correct chains should rarely enter committed/detached states.

### Why this is better than the previous simple EM

The previous scalar latent EM was too unconstrained and could only rediscover static spread.  A hidden semi-Markov model can test a real hypothesis:

```text
wrong chains spend more time in committed/detached states,
and enter them earlier than correct chains,
after length/problem controls.
```

### Pass condition

The inferred state-entry time predicts final correctness and localizes first-error better than static `cloud_spread`.

## 7. Shapelet Discovery for Local Ruptures

### Idea

Find short discriminative subsequences:

```text
[entropy rebound, spread increase, direction jump]
```

or:

```text
[anchor target mass drop, transport entropy increase]
```

### Local adaptation

1. Build residualized multichannel trajectories.
2. Search short windows of length 2-5 steps or 64-256 tokens.
3. Compare each candidate window to all chains with DTW or Euclidean distance.
4. Select shapelets only inside cross-validation.
5. Validate with same-problem paired AUROC and permutation p-values.

### Caution

Shapelet search can overfit badly.  It must be nested inside GroupKFold and compared against random-label shapelets.

## 8. Recommended Order

Do not start with a large neural model.  The clean order is:

1. Paired functional trajectory test.
2. Logsignature classifier.
3. Conformal kernel change-point.
4. Mixed-effects functional logistic model.
5. Monotone HSMM if the first four show phase/state evidence.
6. Shapelet discovery for local interpretability.

## 9. Minimal Implementation Plan

Create one audit script first:

```text
trajectory_difference_audit.py
```

Inputs:

- same-problem `.npz` outputs;
- ProcessBench full features;
- selected signal groups.

Outputs:

- `functional_test.json`;
- `signature_oof.json`;
- `change_point_alarm.json`;
- `mixed_effects_summary.md`;
- plots of effect curves and alarm locations.

Primary metrics:

- same-problem paired AUROC;
- GroupKFold OOF AUROC/AUPR;
- cluster permutation p-values;
- FPR/recall/delay;
- increment over `cloud_spread + U_D + length`.

Implemented initial version:

```bash
python trajectory_difference_audit.py --selftest
```

The script currently runs:

- paired functional trajectory cluster-permutation tests;
- path-signature classifiers with GroupKFold by problem;
- static baseline + length controls for the anti-degradation comparison;
- correct-only conformal alarm calibration on training problems;
- endpoint-alarm reporting to catch late-only detectors.
