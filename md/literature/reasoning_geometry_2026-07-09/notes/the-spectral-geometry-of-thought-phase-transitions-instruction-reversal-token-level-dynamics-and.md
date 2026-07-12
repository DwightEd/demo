# The Spectral Geometry of Thought: Phase Transitions, Instruction Reversal, Token-Level Dynamics, and Perfect Correctness Prediction in How Transformers Reason

- **Local PDF filename**: `The Spectral Geometry of Thought Phase Transitions, Instruction Reversal, Token-Level Dynamics, and Perfect Correctness Prediction in How Transformers Reason.pdf`
- **Slug**: `the-spectral-geometry-of-thought-phase-transitions-instruction-reversal-token-level-dynamics-and`
- **Pages**: 26
- **Approx Words**: 9040
- **Auto Tags**: geometry;dynamics;faithfulness
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.628016

## Keyword Profile

- `spectral`: 191
- `phase`: 27
- `transition`: 18
- `dimension`: 16
- `geometry`: 13
- `hidden state`: 8
- `chain of thought`: 8
- `manifold`: 6
- `causal`: 5
- `geometric`: 4
- `probe`: 4
- `trajectory`: 3

## Abstract / Opening Summary

We discover that large language models exhibit spectral phase transitions in their hidden activation spaces when engaging in reasoning versus factual recall. Through systematic spectral analysis across 11 models spanning 5 architecture families (Qwen, Pythia, Phi, Llama, DeepSeek-R1), we identify seven fundamental phenomena: (1) Reasoning Spectral Compression—9/11 models show significantly lower α for reasoning (p < 0.05), with the effect size correlating with model capability; (2) Instruction Tuning Spectral Reversal—base models show reasoning α < factual α (compression), while instruction-tuned models of the same architecture reverse this relationship, demonstrating that instruction tuning fundamentally reorganizes how models structure representations for reasoning; (3) Architecture-Dependent Generation Taxonomy—prompt-to-response shifts partition into three categories: expansion (Qwen/Phi instruct, ∆α = −0.46 ± 0.18), compression (Pythia/Llama, ∆α = +0.40 ± 0.07), and equilibrium (DeepSeek-R1, ∆α ≈0); (4) Spectral Scaling Law—αreasoning ∝−0.074 ln N across 4 Qwen base models (R2 = 0.46); (5) Token-Level Spectral Cascade—per-token alpha tracking during generation reveals that adjacent layers have highly synchronized spectral dynamics (ρ = 0.84 at distance 9), but this synchronization decays exponentially with layer distance (ρ ∼e−d/19.8), with reasoning tasks showing systematically lower cross-layer coupling than factual tasks (∆ρ = −0.19 for distant layers); (6) Reasoning Step Spectral Punctuation—phase transition signatures in the alpha gradient coincide with reasoning step boundaries (“Step 1:”, new paragraphs, “therefore”), suggesting that spectral analysis can identify the micro-structure of thought; (7) Perfect Spectral Correctness Prediction—spectral α alone achieves AUC = 1.000 (Qwen2.5-7B, late layers) and mean AUC = 0.893 across 6 models in predicting whether a model will answer correctly before the final answer is generated, demonstrating that reasoning quality is legible in the geometry of computation. Together, these findings establish a comprehensive spectral theory of reasoning in transformers, revealing that the geometry of thought is universal in direction, architecture-specific in dynamics, and predictive of outcome. 1

## Method / Algorithms Extract

3.1 Spectral Analysis of Activations Given a transformer with L layers, let H(ℓ) ∈RT ×d denote the hidden state matrix at layer ℓ, where T is the sequence length and d is the hidden dimension. We compute the singular value decomposition: H(ℓ) = UΣV⊤, Σ = diag(σ1, σ2, . . . , σmin(T,d)) (1) where σ1 ≥σ2 ≥. . . ≥0 are the singular values in decreasing order. Spectral Alpha (α). We fit a power-law model σk ∝k−α via log-log linear regression on the ordered singular values. Higher α indicates faster spectral decay (concentrated representations where variance is dominated by a few dimensions); lower α indicates slower decay (distributed representations where variance is spread across many dimensions). Formally: α = − PK k=1(ln k −ln k)(ln σk −ln σ) PK k=1(ln k −ln k)2 (2) where K = min(T, d) and overlines denote means. We verified that the power-law fit is appropriate for these distributions (mean R2 > 0.85 across all models and layers; see Appendix B.4). Prompt-Response Decomposition. We separately analyze H(ℓ) prompt ∈RTp×d and H(ℓ) response ∈RTr×d, enabling us to track the spectral transition from input processing to generation. The prompt-response delta ∆αP→R = αresponse−αprompt quantifies how spectral structure changes during generation. 3 ===== PAGE 4 / 26 ===== Table 1: Model inventory: 11 models across 5 architecture families and 4 training paradigms. Model Family Params Layers Type Norm Attention Qwen2.5-0.5B Qwen 0.5B 24 Base RMSNorm GQA Qwen2.5-3B Qwen 3B 36 Base RMSNorm GQA Qwen2.5-7B Qwen 7B 28 Base RMSNorm GQA Qwen2.5-1.5B-Instruct Qwen 1.5B 28 Instruct RMSNorm GQA Qwen2.5-3B-Instruct Qwen 3B 36 Instruct RMSNorm GQA DeepSeek-R1-1.5B DeepSeek-R1 1.5B 28 Reasoning RMSNorm GQA Pythia-1B Pythia 1B 16 Base LayerNorm MHA Pythia-2.8B Pythia 2.8B 32 Base LayerNorm MHA Phi-2 Phi 2.7B 32 Base LayerNorm MHA Phi-3.5-mini-instruct Phi 3.8B 32 Instruct RMSNorm GQA TinyLlama-1.1B-Chat Llama 1.1B 22 Chat RMSNorm GQA Token-Level Dynamics. For fine-grained temporal analysis, we compute α over a sliding window of w = 10 tokens at each generation step, yielding a per-token spectral trajectory α(t, ℓ) across layers and time. The window captures local spectral structure while maintaining sufficient singular values for reliable estimation. Gradient analysis ∇tα(t, ℓ) = α(t + 1, ℓ) −α(t, ℓ) reveals spectral transition events at reasoning step boundaries. 3.2 Statistical Analysis For each comparison (reasoning vs. factual, prompt vs. response), we use the Welch two-sample t-test at significance level αstat = 0.05. For the spectral scaling law, we fit ∆α = a ln N + b via ...

## Experiments / Evidence Extract

4.1 Finding 1: Universal Reasoning Spectral Compression Table 2 and Figure 1 present our central finding: 9 out of 11 models show statistically significant differences between reasoning and factual task spectral profiles. When examining the full activation spectrum (∆α), the majority show reasoning spectral compression (lower α for reasoning). The two exceptions—Qwen instruct models—show the opposite pattern, which leads directly to our second finding. Response-only analysis (∆αR) isolates the generation phase from prompt effects. Interestingly, in the response-only comparison, the Qwen base models show positive ∆αR, meaning their reasoning responses actually have higher alpha than factual responses. This apparent contradiction with the overall ∆α arises because the large negative prompt-to-response shift (Finding 3) affects both task types differently. Effect size. The magnitude of |∆α| varies substantially across models: from 0.009 (Phi-3.5-I, non-significant) to 0.464 (Qwen2.5-7B, p < 10−65). Within the Qwen base family, effect size grows with model capacity: 0.219 → 0.318 →0.464 for 0.5B →3B →7B, connecting to Finding 4 (spectral scaling law). 5 ===== PAGE 6 / 26 ===== Table 2: Reasoning vs. Factual spectral α across 11 models. ∆α = Reasoning −Factual (negative = more distributed for reasoning). Response-only ∆αR controls for prompt effects. Model Type αR αF ∆α ∆αR p Qwen Base (reasoning = more distributed) Qwen2.5-0.5B Base 1.159 1.481 −0.219 +0.287 < 10−9 Qwen2.5-3B Base 0.985 1.398 −0.318 +0.301 < 10−5 Qwen2.5-7B Base 0.832 1.512 −0.464 +0.221 < 10−5 Qwen Instruct (reasoning = more concentrated or mixed) Qwen2.5-1.5B-I Instruct 0.946 1.685 +0.206 +0.307 < 10−28 Qwen2.5-3B-I Instruct 0.949 1.409 +0.121 +0.291 < 10−10 Other Architectures DS-R1-1.5B Reasoning 1.415 1.402 −0.291 −0.318 < 10−8 Pythia-1B Base 1.836 1.347 −0.096 −0.118 0.121 Pythia-2.8B Base 1.584 1.217 −0.130 −0.163 0.001 Phi-2 Base 1.036 1.216 −0.106 −0.124 < 10−7 Phi-3.5-I Instruct 0.937 1.536 +0.009 +0.019 0.739 TinyLlama-Chat Chat 1.478 1.132 −0.119 −0.059 0.004 4.2 Finding 2: Instruction Tuning Spectral Reversal Our most striking discovery is that instruction tuning reverses the spectral signature of reasoning. Comparing matched base-instruct pairs: • Qwen2.5-3B Base: ∆α = −0.318 (reasoning = more distributed) • Qwen2.5-3B Instruct: ∆α = +0.121 (reasoning = more concentrated) • Reversa...

## Conclusion / Discussion Extract

5.1 A Spectral Theory of Reasoning Our seven findings together constitute a coherent spectral theory of reasoning in transformers: 1. The Spectral Reasoning Hypothesis: Effective reasoning requires activating higher-dimensional subspaces of the representation manifold, measurable as decreased spectral α. This is universal across architectures. 2. The Training Paradigm Principle: Instruction tuning reorganizes how models deploy spectral resources for reason- ing. Base models use broad, diffuse representations; instruction-tuned models use focused, efficient representations. 3. The Spectral Cascade Model: Information propagates through the network with exponentially decaying spectral synchronization (τ ≈20 layers), creating local coherence zones. Reasoning decouples distant zones. 4. Punctuated Spectral Equilibrium: Reasoning proceeds through a series of spectral phase transitions at step boundaries, with stable spectral configurations within steps. 5. The Correctness Legibility Principle: The spectral geometry of activations encodes whether reasoning is succeeding or failing—with perfect discriminability (AUC = 1.000) at individual layers. 5.2

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

TBD: add short excerpts with page markers from `../texts/the-spectral-geometry-of-thought-phase-transitions-instruction-reversal-token-level-dynamics-and.txt`.
