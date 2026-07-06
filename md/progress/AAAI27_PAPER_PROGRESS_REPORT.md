# AAAI27 Paper Progress Report

## Working Title

**Constrained Reasoning Geometry: Single-Forward Detection of Reasoning Failures from Hidden-State Manifold Health**

## Core Story

Large language model reasoning can be viewed as a trajectory through a constrained hidden-state geometry.  Correct reasoning does not merely stay "confident"; it maintains a compact, coordinated, prompt-conditioned representation of the current step.  Failure appears as a relaxation of that constraint: token clouds become more diffuse, active dimensions increase, uncertainty rises in some cases, and structured readers over token-step-layer relations can recover failure information.

The paper should therefore be framed around **manifold health during reasoning**:

```text
Correct reasoning:
  step token cloud stays concentrated/coordinated
  effective active dimensions remain controlled
  boundary uncertainty is limited or recoverable
  hidden trajectory remains compatible with a constrained transition tube

Incorrect reasoning:
  token cloud becomes more diffuse
  resultant/coherence drops
  active dimensions or entropy-like measures rise
  hidden trajectory requires additional directions or leaves the healthy tube
```

This story connects hidden geometry, spectral/intrinsic-dimensionality views, and a plug-in monitoring module.  It avoids relying on generated chain-of-thought text as the main signal.

## Main Hypotheses

**H1. Geometric Dispersion Hypothesis**

Reasoning failures are associated with a more diffuse hidden token cloud.  In practice this is measured by `spread = 1 - resultant`, where `resultant` is the norm of the mean normalized token direction within a step.

**H2. Constrained Dimensionality Hypothesis**

Effective reasoning is constrained to fewer coordinated directions.  Incorrect reasoning activates a broader or less organized set of dimensions, visible through participation ratio, activation entropy, and low-rank transition residuals.

**H3. Confidence-Geometry Complementarity**

Logit entropy captures uncertainty-aware failures, while hidden geometry also captures confident failures.  A useful monitor should combine geometry and confidence, rather than reduce to entropy alone.

**H4. Structured Hidden Relations Hypothesis**

Token-level hidden states contain relational signals that are not fully captured by hand-crafted scalars.  Hypergraph message passing over token-step-layer structure can read part of this information and provide a structured branch for model-based detection.

## Current Evidence

### 1. ProcessBench Step-Level Geometry

On `full_gsm8k.npz`, layer 14, the strongest single-step geometry signal is hidden dispersion:

| Signal | AUROC | Non-error Mean | Error Mean |
|---|---:|---:|---:|
| `spread` | 0.772 | 0.366 | 0.407 |
| `resultant` | 0.772 | 0.634 | 0.593 |
| `transition_surprise` | 0.720 | 6.788 | 22.306 |
| `d_spread` | 0.702 | -0.010 | 0.023 |

Interpretation: first-error steps have lower directional coherence and higher geometric spread.  The spread/resultant pair gives the cleanest current geometric readout of reasoning health.

### 2. Geometry + Confidence Group Performance

On `full_gsm8k.npz`, layer 14, OOF group results show that combining hidden geometry with anchor/uncertainty features gives a stronger detector:

| Feature Group | OOF AUROC |
|---|---:|
| `static` | 0.779 |
| `anchor_uncertainty` | 0.811 |
| `dynamic_online` | 0.782 |
| `offline_recovery` | 0.777 |

The paper should use this as evidence for confidence-geometry complementarity: hidden geometry is strong by itself, and the best current operational group combines geometry, anchor drift, and uncertainty.

### 3. Online Monitoring Signal

On `full_gsm8k.npz`, layer 14, online alarm evaluation gives usable early-warning behavior:

| Detector | FPR | Recall | Median Delay | Early Rate |
|---|---:|---:|---:|---:|
| single-threshold alarm | 0.063 | 0.307 | 0.0 | 0.095 |
| single-threshold alarm | 0.126 | 0.346 | 0.0 | 0.141 |
| single-threshold alarm | 0.221 | 0.498 | 0.0 | 0.186 |
| CUSUM-style alarm | 0.221 | 0.507 | 0.0 | 0.173 |

For the paper, the deployable framing should emphasize **single-forward monitoring**.  The stronger story is not cumulative scoring, but using hidden-state health as a plug-in signal during generation.

### 4. Directional Sphere Geometry

The sphere audit supports that the signal is not merely hidden-vector norm:

| Feature Group | OOF AUROC | AUPR |
|---|---:|---:|
| `norm_only` | 0.727 | 0.334 |
| `raw_magnitude_geometry` | 0.779 | 0.403 |
| `unit_sphere` | 0.776 | 0.397 |
| `unit_plus_entropy` | 0.786 | 0.428 |
| `anchor_uncertainty_like` | 0.786 | 0.429 |

Key increments:

| Comparison | AUROC Increment | Bootstrap CI |
|---|---:|---|
| `raw_magnitude_geometry` over `norm_only` | +0.052 | [+0.029, +0.082] |
| `unit_sphere` over `norm_only` | +0.049 | [+0.024, +0.080] |
| `unit_plus_entropy` over `unit_sphere` | +0.010 | [-0.001, +0.021] |

Interpretation: the geometry signal is largely directional.  This supports a hypersphere/manifold-health view of hidden states.

### 5. Same-Problem Multisampling Evidence

Same-problem multisampling controls for problem difficulty.  The cleanest policy is `answer_format_ok`, which filters final-answer format failures.

| Dataset | Samples | Correct | Error | Contrastive Problems | Error-Correct Pairs |
|---|---:|---:|---:|---:|---:|
| `gsm8k_v2_5shot.npz` | 2035 | 1756 | 279 | 94 | 931 |
| `gsm8k_v2_custom.npz` | 3452 | 2920 | 532 | 147 | 2868 |

The most robust same-problem geometry signal is cloud spread:

| Dataset | Signal | Same-Problem AUROC | Cross-Problem AUROC | Error Median | Correct Median |
|---|---|---:|---:|---:|---:|
| 5shot | `cloud_spread_max` | 0.634 | 0.826 | 0.406 | 0.371 |
| custom | `cloud_spread_late` | 0.659 | 0.769 | 0.317 | 0.291 |
| custom | `cloud_spread_max` | 0.639 | 0.786 | 0.392 | 0.362 |
| custom | `ae_deep_late` | 0.580 | 0.565 | 210.917 | 210.324 |

Interpretation: even after controlling for problem identity, incorrect samples are more diffuse in hidden geometry.  This is weaker than cross-problem detection, but it is more credible as a reasoning-trajectory signal.

### 6. Hypergraph Branch

The hypergraph branch reads token hidden states through a structured token-step-layer graph.  On `full_gsm8k.npz` with layers `[10, 14, 18, 22]`, causal construction, and hidden diagnostic features:

| Metric | Value |
|---|---:|
| OOF node AUROC | 0.6905 |
| OOF node AUPR | 0.3990 |
| OOF step AUROC | 0.7602 |
| OOF step AUPR | 0.3602 |
| OOF graph AUROC | 0.6778 |
| OOF graph AUPR | 0.6831 |
| First-error localization top1 | 0.8539 |
| Expected localization top1 | 0.4629 |

Interpretation: token-level hidden relations contain enough signal for a structured neural reader to recover step-level failure and strong localization.  This branch should be presented as a **relational readout** of the same constrained-geometry phenomenon, not as the core lightweight deployment module.

## Proposed Method Section

### Constrained Geometry Health

For each reasoning step, compute a hidden token cloud:

```text
H_t^l = [h_{t,1}^l, ..., h_{t,n_t}^l]
```

Normalize token directions and compute:

```text
resultant_t^l = || mean_i normalize(h_{t,i}^l) ||
spread_t^l = 1 - resultant_t^l
```

Then aggregate across selected layers and optionally combine with:

- activation entropy / participation ratio;
- boundary logit entropy;
- prompt-anchor or question-vector alignment when available.

This gives a single-forward monitor:

```text
health_t = f(spread_t, active_dim_t, uncertainty_t, anchor_t)
```

### Difficulty-Controlled Evaluation

The paper should use two complementary evaluation settings:

1. **ProcessBench first-error detection**: step-level labels allow first-error localization and online alarm analysis.
2. **Same-problem multisampling**: multiple sampled answers per question control problem difficulty and test whether geometry separates correct/incorrect reasoning paths for the same problem.

Same-problem paired AUROC should be a required metric:

```text
P(score(error sample) > score(correct sample) | same question)
```

### Structured Hypergraph Readout

Construct a hypergraph where:

- nodes are token or step hidden units;
- hyperedges connect tokens in the same step, adjacent steps, and selected layers;
- message passing predicts token/step/graph-level failure risk.

This branch demonstrates that relational hidden-state structure carries failure information beyond raw text.  It can be reported as a stronger but heavier reader, while the main paper emphasizes lightweight scalar monitors for plug-in deployment.

### Low-Rank Transition Tube

The next method component should directly instantiate the constrained-manifold story:

```text
delta_t = x_{t+1} - x_t
```

Fit a low-rank tube from correct-chain transitions:

```text
delta_t ~= mu + U_k z_t
```

Then score a candidate chain by:

```text
off_tube_residual = || delta_t - Proj_U(delta_t - mu) ||
rank_energy = number of tube directions needed to explain transition energy
```

This is the cleanest bridge from empirical spread signals to the theory of constrained inference manifolds.

## Theory Support

### Constrained Inference Manifolds

The theory frame is that reasoning unfolds inside a constrained inference manifold.  Correct reasoning is not simply low-dimensional; it is low-dimensional while preserving sufficient information volume.  Our spread/resultant results instantiate the "constraint" side: correct steps have higher resultant and lower spread, while incorrect steps relax into more diffuse hidden states.

### Intrinsic Dimensionality of Reasoning Chains

Effective reasoning chains can reduce intrinsic dimensionality by making the computational path more compressible.  Our participation-ratio and activation-entropy results provide a hidden-state analogue: incorrect samples activate slightly broader dimensions, especially in deep layers.

### Spectral Geometry of Thought

The external spectral-geometry literature motivates token/layer spectral monitoring and step-boundary punctuation.  The paper should borrow the experimental structure rather than copy `alpha`: cross-layer, token/step-native geometry, and correctness prediction from internal representation dynamics.

### Hypergraph Geometry

A hypergraph view gives a mathematically natural structure for token-step-layer relations.  The hypergraph branch can be framed as learning a higher-order relational readout over the same hidden geometry that scalar spread only summarizes.

## Suggested Paper Organization

### Introduction

Frame the problem as real-time detection of reasoning failures without relying on textual self-explanation.  State the central claim:

> Reasoning failures are visible as degradation of constrained hidden-state geometry.

### Related Work

Group prior work into:

- entropy/logit confidence;
- spectral geometry and intrinsic dimensionality;
- process supervision and step-level verification;
- graph/hypergraph neural readers.

### Method

Introduce:

1. Constrained Geometry Health;
2. same-problem paired evaluation;
3. online single-forward monitoring;
4. hypergraph relational readout;
5. low-rank transition tube as the theoretical extension.

### Experiments

Recommended main tables:

1. ProcessBench step-level AUROC.
2. OOF group performance and online alarms.
3. Same-problem paired AUROC.
4. Hypergraph branch performance.
5. Ablation by geometry / uncertainty / anchor.

### Discussion

The strongest current claim:

```text
Hidden geometry provides a reproducible, single-forward signal of reasoning health.
Incorrect reasoning paths are more diffuse and less constrained, and structured
token-level readers can recover step-level failures from hidden-state relations.
```

## Current Paper-Ready Claims

1. **Geometry works**: hidden token-cloud spread reaches AUROC 0.772 on GSM8K first-error detection.
2. **Geometry and confidence complement each other**: the `anchor_uncertainty` group reaches OOF AUROC 0.811.
3. **The signal survives difficulty control**: same-problem paired AUROC reaches 0.634 on 5shot and 0.659 on custom prompts for cloud spread.
4. **The signal is directional, not merely norm-based**: unit-sphere geometry reaches OOF AUROC 0.776 and improves over norm-only by +0.049.
5. **Structured token relations are informative**: hypergraph token HGN reaches OOF step AUROC 0.7602 and first-error top1 0.8539.
6. **The work supports a plug-in monitoring direction**: online alarms reach recall 0.498 at FPR 0.221 with median delay 0 on GSM8K L14.

## Next Paper-Critical Experiments

1. Treat the `within_problem_regime_hsmm_audit.py` 40-problem smoke result as negative: HSMM `0.538` / censor80 `0.506` vs static `mean:cloud_spread=0.682`.
2. Run `within_problem_path_kernel_audit.py` to test whether any shape-only trajectory information exists after same-problem centering and per-chain level/trend removal.
3. If shape-only conditional MMD or witness scores fail, stop adding latent-state complexity and prioritize richer extraction channels.
4. Re-extract same-problem samples with `sv_out_committal`, `sv_tok_entropy`, and `sv_tok_committal` only after the existing-channel shape model shows a non-endpoint same-problem signal.
5. Add prompt-anchor hidden banks so the anchor signal is semantic rather than a single question-vector cosine.
6. Build one minimal intervention only after the detector survives same-problem and endpoint-control audits.
