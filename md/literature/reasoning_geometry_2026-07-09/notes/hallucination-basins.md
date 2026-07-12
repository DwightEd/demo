# Hallucination Basins: A Dynamic Framework for Understanding and Controlling LLM Hallucinations

- **Local PDF filename**: `Hallucination Basins.pdf`
- **Slug**: `hallucination-basins`
- **Pages**: 26
- **Approx Words**: 10465
- **Auto Tags**: geometry;dynamics;uncertainty;hallucination;emergence
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.613308

## Keyword Profile

- `hallucination`: 113
- `hidden state`: 32
- `geometric`: 21
- `trajectory`: 17
- `latent`: 17
- `entropy`: 16
- `dimension`: 15
- `geometry`: 14
- `causal`: 13
- `manifold`: 7
- `spectral`: 4
- `probe`: 3

## Abstract / Opening Summary

Large language models (LLMs) hallucinate: they produce fluent outputs that are factually incor- rect. We present a geometric dynamical systems framework in which hallucinations arise from task-dependent basin structure in latent space. Us- ing autoregressive hidden-state trajectories across multiple open-source models and benchmarks, we find that separability is strongly task-dependent rather than universal: factoid settings can show clearer basin separation, whereas summarization and misconception-heavy settings are typically less stable and often overlap. We formalize this behavior with task-complexity and multi-basin theorems, characterize basin emergence in L- layer transformers, and show that geometry-aware steering can reduce hallucination probability with- out retraining.

## Method / Algorithms Extract

Linearly interpolate factual hidden states toward basin centroid: hα = (1 −α)hfact + αµhall for α ∈[0, 1]. Train logistic classifier on factual/hall, measure P(hall|hα). 9. Discussion and Remarks When Basins Don’t Form Table 3 reveals a systematic trend of failures in basin formations. Notice that in Truth- 7 ===== PAGE 8 / 26 ===== HALLUCINATION BASINS Table 3. Centroid and Mahalanobis (Maha) AUROC reported with 95% CI. Note: Lay indicates Layer with highest AUROC, N indicates number of data samples, and B? indicates whether a basin exists or not. For larger models, due to computational constraints, this was limited to N = 1000. Model Data Lay Centroid (95% CI) Maha (95% CI) N B? gemma-2-2b FEVER L14 0.515 (0.489, 0.548) 0.514 (0.487, 0.535) 9999 × gemma-2-2b HaluEval qa L14 0.727 (0.703, 0.744) 0.725 (0.710, 0.745) 20000 ✓ gemma-2-2b HaluEval summ L20 0.508 (0.491, 0.519) 0.479 (0.457, 0.502) 20000 × gemma-2-2b MuSiQue L26 0.912 (0.894, 0.932) 0.926 (0.909, 0.944) 4834 ✓ gemma-2-2b TruthfulQA L14 0.607 (0.547, 0.662) 0.597 (0.535, 0.651) 1580 × llama-3.2-1b FEVER L8 0.670 (0.641, 0.700) 0.680 (0.659, 0.702) 9999 × llama-3.2-1b HaluEval qa L3 0.983 (0.976, 0.988) 0.984 (0.980, 0.988) 20000 ✓ llama-3.2-1b HaluEval summ L10 0.681 (0.666, 0.697) 0.674 (0.659, 0.690) 20000 × llama-3.2-1b MuSiQue L1 1.000 (1.000, 1.000) 1.000 (1.000, 1.000) 4834 ✓ llama-3.2-1b TruthfulQA L12 0.741 (0.662, 0.800) 0.724 (0.685, 0.777) 1580 ✓ llama-3.2-3b FEVER L12 0.702 (0.671, 0.725) 0.711 (0.686, 0.731) 9999 ✓ llama-3.2-3b HaluEval qa L3 0.986 (0.982, 0.990) 0.985 (0.981, 0.990) 20000 ✓ llama-3.2-3b HaluEval summ L21 0.669 (0.654, 0.687) 0.665 (0.648, 0.683) 20000 × llama-3.2-3b MuSiQue L3 1.000 (1.000, 1.000) 1.000 (1.000, 1.000) 4834 ✓ llama-3.2-3b TruthfulQA L12 0.771 (0.716, 0.833) 0.794 (0.751, 0.839) 1580 ✓ qwen-2.5-1.5b FEVER L18 0.728 (0.704, 0.748) 0.735 (0.719, 0.757) 9999 ✓ qwen-2.5-1.5b HaluEval qa L24 0.984 (0.979, 0.989) 0.983 (0.980, 0.988) 20000 ✓ qwen-2.5-1.5b HaluEval summ L18 0.663 (0.650, 0.683) 0.664 (0.648, 0.682) 20000 × qwen-2.5-1.5b MuSiQue L3 1.000 (1.000, 1.000) 1.000 (1.000, 1.000) 4834 ✓ qwen-2.5-1.5b TruthfulQA L21 0.738 (0.671, 0.803) 0.751 (0.705, 0.803) 1580 ✓ llama-3.1-8b HaluEval qa L0 0.571 (0.503, 0.731) 0.549 (0.503, 0.705) 1000 × llama-3.1-8b TruthfulQA L25 0.944 (0.899, 0.975) 0.958 (0.921, 0.987) 1000 ✓ mistral-7b-v0.3 HaluEval qa L24 0.704 (0.578, 0.823) 0.545 (0.503, 0.701) 1000 × mistral-7b-v0.3 TruthfulQA L17 0.939 (0.893, 0.975) 0.958 (0.923, 0.985) 1000 ✓ Figure 1. Task-Dependent Basin Geometry. Llama-3.2-3b’s performance on var...

## Experiments / Evidence Extract

Experiments We outline our validation protocol for the theoretical results. (1) validation of basin existence with a quantifiable geo- metric separation, (2) geometric features enable efficient detection without requiring sampling. View the experimen- tal protocol in Appendix F.2. 8.1. Experimental Design and Setup Models. To demonstrate generalizability and scales we evaluate on: Llama 3.2-1B/3B (Meta AI, 2024), Gemma- 2-2B (Riviere et al., 2024) and Qwen2-1.5B (Yang et al., 2025). Datasets. We use four diverse hallucination benchmarks: HaluEval (Li et al., 2023), MuSiQue (Trivedi et al., 2022), FEVER (Thorne et al., 2018), and TruthfulQA (Lin et al., 2022). Hidden State Extraction We use autoregressive decoding trajectories and extract final-token hidden states layerwise in a 70/30 stratified split with seed = 42. 8.2. Task-Dependent Basin Formation Hypothesis: We test whether basin geometry under au- toregressive decoding remains task-dependent: factoid set- tings should be more separable, while generation and mis- conception settings should show weaker or overlapping structure. Table 3 and Figure 1 summarize the evidence. 8.3. Causality: Pushing Factual →Basins Method Linearly interpolate factual hidden states toward basin centroid: hα = (1 −α)hfact + αµhall for α ∈[0, 1]. Train logistic classifier on factual/hall, measure P(hall|hα). 9. Discussion and Remarks When Basins Don’t Form Table 3 reveals a systematic trend of failures in basin formations. Notice that in Truth- 7 ===== PAGE 8 / 26 ===== HALLUCINATION BASINS Table 3. Centroid and Mahalanobis (Maha) AUROC reported with 95% CI. Note: Lay indicates Layer with highest AUROC, N indicates number of data samples, and B? indicates whether a basin exists or not. For larger models, due to computational constraints, this was limited to N = 1000. Model Data Lay Centroid (95% CI) Maha (95% CI) N B? gemma-2-2b FEVER L14 0.515 (0.489, 0.548) 0.514 (0.487, 0.535) 9999 × gemma-2-2b HaluEval qa L14 0.727 (0.703, 0.744) 0.725 (0.710, 0.745) 20000 ✓ gemma-2-2b HaluEval summ L20 0.508 (0.491, 0.519) 0.479 (0.457, 0.502) 20000 × gemma-2-2b MuSiQue L26 0.912 (0.894, 0.932) 0.926 (0.909, 0.944) 4834 ✓ gemma-2-2b TruthfulQA L14 0.607 (0.547, 0.662) 0.597 (0.535, 0.651) 1580 × llama-3.2-1b FEVER L8 0.670 (0.641, 0.700) 0.680 (0.659, 0.702) 9999 × llama-3.2-1b HaluEval qa L3 0.983 (0.976, 0.988) 0.984 (0.980, 0.988) 20000...

## Conclusion / Discussion Extract

Discussion and Remarks When Basins Don’t Form Table 3 reveals a systematic trend of failures in basin formations. Notice that in Truth- 7 ===== PAGE 8 / 26 ===== HALLUCINATION BASINS Table 3. Centroid and Mahalanobis (Maha) AUROC reported with 95% CI. Note: Lay indicates Layer with highest AUROC, N indicates number of data samples, and B? indicates whether a basin exists or not. For larger models, due to computational constraints, this was limited to N = 1000. Model Data Lay Centroid (95% CI) Maha (95% CI) N B? gemma-2-2b FEVER L14 0.515 (0.489, 0.548) 0.514 (0.487, 0.535) 9999 × gemma-2-2b HaluEval qa L14 0.727 (0.703, 0.744) 0.725 (0.710, 0.745) 20000 ✓ gemma-2-2b HaluEval summ L20 0.508 (0.491, 0.519) 0.479 (0.457, 0.502) 20000 × gemma-2-2b MuSiQue L26 0.912 (0.894, 0.932) 0.926 (0.909, 0.944) 4834 ✓ gemma-2-2b TruthfulQA L14 0.607 (0.547, 0.662) 0.597 (0.535, 0.651) 1580 × llama-3.2-1b FEVER L8 0.670 (0.641, 0.700) 0.680 (0.659, 0.702) 9999 × llama-3.2-1b HaluEval qa L3 0.983 (0.976, 0.988) 0.984 (0.980, 0.988) 20000 ✓ llama-3.2-1b HaluEval summ L10 0.681 (0.666, 0.697) 0.674 (0.659, 0.690) 20000 × llama-3.2-1b MuSiQue L1 1.000 (1.000, 1.000) 1.000 (1.000, 1.000) 4834 ✓ llama-3.2-1b TruthfulQA L12 0.741 (0.662, 0.800) 0.724 (0.685, 0.777) 1580 ✓ llama-3.2-3b FEVER L12 0.702 (0.671, 0.725) 0.711 (0.686, 0.731) 9999 ✓ llama-3.2-3b HaluEval qa L3 0.986 (0.982, 0.990) 0.985 (0.981, 0.990) 20000 ✓ llama-3.2-3b HaluEval summ L21 0.669 (0.654, 0.687) 0.665 (0.648, 0.683) 20000 × llama-3.2-3b MuSiQue L3 1.000 (1.000, 1.000) 1.000 (1.000, 1.000) 4834 ✓ llama-3.2-3b TruthfulQA L12 0.771 (0.716, 0.833) 0.794 (0.751, 0.839) 1580 ✓ qwen-2.5-1.5b FEVER L18 0.728 (0.704, 0.748) 0.735 (0.719, 0.757) 9999 ✓ qwen-2.5-1.5b HaluEval qa L24 0.984 (0.979, 0.989) 0.983 (0.980, 0.988) 20000 ✓ qwen-2.5-1.5b HaluEval summ L18 0.663 (0.650, 0.683) 0.664 (0.648, 0.682) 20000 × qwen-2.5-1.5b MuSiQue L3 1.000 (1.000, 1.000) 1.000 (1.000, 1.000) 4834 ✓ qwen-2.5-1.5b TruthfulQA L21 0.738 (0.671, 0.803) 0.751 (0.705, 0.803) 1580 ✓ llama-3.1-8b HaluEval qa L0 0.571 (0.503, 0.731) 0.549 (0.503, 0.705) 1000 × llama-3.1-8b TruthfulQA L25 0.944 (0.899, 0.975) 0.958 (0.921, 0.987) 1000 ✓ mistral-...

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

TBD: add short excerpts with page markers from `../texts/hallucination-basins.txt`.
