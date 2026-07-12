# Reasoning Fails Where Step Flow Breaks

- **Local PDF filename**: `Reasoning Fails Where Step Flow Breaks.pdf`
- **Slug**: `reasoning-fails-where-step-flow-breaks`
- **Pages**: 17
- **Approx Words**: 11241
- **Auto Tags**: dynamics;faithfulness;hallucination;step-level
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.623127

## Keyword Profile

- `flow`: 110
- `chain of thought`: 12
- `faithful`: 8
- `causal`: 6
- `trajectory`: 2
- `hidden state`: 2
- `transition`: 1
- `hallucination`: 1
- `dimension`: 1

## Abstract / Opening Summary

Large reasoning models (LRMs) that gen- erate long chains of thought now perform well on multi-step math, science, and cod- ing tasks. However, their behavior is still unstable and hard to interpret, and existing analysis tools struggle with such long, struc- tured reasoning traces. We introduce Step- Saliency, which pools attention–gradient scores into step-to-step maps along the question– thinking–summary trajectory. Across sev- eral models, Step-Saliency reveals two re- curring information-flow failures: Shallow Lock-in, where shallow layers over-focus on the current step and barely use earlier con- text, and Deep Decay, where deep layers gradually lose saliency on the thinking seg- ment and the summary increasingly attends to itself and the last few steps. Motivated by these patterns, we propose StepFlow, a saliency-inspired test-time intervention that ad- justs shallow saliency patterns measured by Step-Saliency via Odds-Equal Bridge and adds a small step-level residual in deep layers via Step Momentum Injection. StepFlow improves accuracy on math, science, and coding tasks across multiple LRMs without retraining, in- dicating that repairing information flow can recover part of their missing reasoning perfor- mance. Code is available at https://github. com/XiaoyuXu-Vincent/step-saliency. 1

## Method / Algorithms Extract

Guided by the Step-Saliency patterns in §3.3, we design StepFlow, a test-time intervention with two components: Odds-Equal Bridge (OEB) for shal- low layers and Step Momentum Injection (SMI) for deep layers. 4.1 Odds-Equal Bridge (OEB) OEB aims to avoid a situation where almost all influence mass sits on the current thinking step and its neighbours, while earlier context is ignored. Group-wise proxy target. For a fixed query po- sition t in one head and one layer, let pt be the causal attention distribution over past tokens for this head. Step-Saliency uses attention–gradient products for offline diagnosis, while OEB uses pt as a lightweight proxy during decoding to enforce a minimum mass on the bridge region. Using the segmentation from Step-Saliency, we split the keys into three disjoint sets: the current segment S, a bridge segment B that represents earlier context we want to preserve (e.g., the question while generat- ing analysis, or the analysis while generating the summary), and all remaining tokens O. We define the current group masses pt(g) = X k∈g pt(k), g ∈{S, B, O}. (5) We keep pt(O) fixed and set a soft lower bound on the bridge mass: τB = min s |B| |B| + |S|, τmax ! , τS = 1 −pt(O) −τB. (6) We apply OEB only when the bridge mass falls be- low the bound, pt(B) < τB; otherwise we leave the logits unchanged. This schedule keeps the bridge on the same order of magnitude as the current seg- ment instead of letting its mass shrink to nearly zero. Intuitively, the lower bound grows with the relative size of the bridge region, the square-root dampens extreme length effects, and τmax caps the intervention so OEB cannot dominate the attention distribution. A single scalar τmax is chosen per model; we show in Appendix B.1 that accuracy is robust across a wide range of τmax values. KL projection on logits. Let z denote the atten- tion logits (before the causal-mask softmax) for query position t in a given layer/head, so that pt = softmax(z). When pt(B) < τB, we seek a new distribution qt that stays as close as possible to pt in KL divergence while enforcing qt(O) = pt(O) and qt(B) = τB, and hence qt(S) = τS. This gives a small projection problem arg minqt KL(qt ∥pt) under linear constraints on group totals, which can be viewed as a constrained Bregman (KL) projec- tion (Banerjee et al., 2005). Under the softmax parameterization, the solution reduces to a simple group-wise shift of the scores: z′ k = zk + λg, k ∈g, g ∈{S, B}, (7) ===== PAGE 6 / 17 ===== where, when the constraint is active, λB = log τB pt(B) and λS = log τS pt(S). (8) Scores in O are le...

## Experiments / Evidence Extract

Model. We focus on open-weight large reason- ing models that emit explicit chain-of-thought. Our main backbones are DeepSeek-R1-Distill- Qwen (7B/14B/32B) (Guo et al., 2025), GPT- OSS-20B (Agarwal et al., 2025), and QwQ-32B- Preview (Team, 2025). Evaluation. We evaluate on six challenging benchmarks: AIME24, AIME25, AMC23, MATH-500 (Hendrycks et al., 2021), GPQA- Diamond (Rein et al., 2024), and Live- CodeBench (Jain et al., 2024). They are widely used to test multi-step reasoning. We use the same decoding setup for all models and report accuracy (Appendix C.3). To make the results more stable under random sampling, we average over 16 sampled solutions per problem for AIME24/25 and AMC23, and 8 for GPQA- Diamond; MATH-500 and LiveCodeBench use the standard single-sample setting. All methods are applied to each sample in one pass (no multi-pass voting). All baselines and StepFlow share identical decoding hyperparameters, stop conditions, and answer extraction rules (Appendix C.3). Baselines. We compare StepFlow against prompt- only baselines (Plan-and-Solve (PS+) (Wang et al., 2023a) and Hint-Infer (Round1) (Li et al., 2025b)), decode-level baselines (Budget Forcing (S1) (Muennighoff et al., 2025)), internal interven- ===== PAGE 7 / 17 ===== Table 1: Accuracy (%) on six benchmarks (columns) across multiple backbones.

## Conclusion / Discussion Extract

Causal status of the diagnostic. StepFlow’s gains exhibit three forms of specificity— complementary OEB/SMI profiles across benchmarks (Table 2), optimal performance only at the predicted layer bands (Table 4), and selective correction of propagation errors at 5–7× the rate of conceptual errors (Appendix D)—which rule out a generic regularization account. Nevertheless, the causal link between the diagnosed saliency patterns and the observed improvements remains suggestive rather than formally proved. Table 5: Compute-normalized comparison on AIME 24/25 (averaged accuracy, %) for R1-Distill-32B.

## Problem

TBD in close-reading pass.

## Core Hypothesis

TBD in close-reading pass.

## Relation To Our Project

- Hidden-state geometry:
- Manifold / Riemannian / topology:
- Temporal dynamics / online detection:
- Faithful CoT / process faithfulness:
- Error awareness / self-correction:
- Length/position/confidence proxy risk:

## What Gap Remains For Us

TBD in synthesis pass.

## Useful Quotes / Exact Pointers

TBD: add short excerpts with page markers from `../texts/reasoning-fails-where-step-flow-breaks.txt`.
