# Geometry of Reason: Spectral Signatures of Valid Mathematical Reasoning

- **Local PDF filename**: `Geometry of Reason Spectral Signatures of Valid Mathematical Reasoning.pdf`
- **Slug**: `geometry-of-reason-spectral-signatures-of-valid-mathematical-reasoning`
- **Pages**: 30
- **Approx Words**: 14036
- **Auto Tags**: geometry;dynamics;faithfulness;uncertainty;hallucination
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.612538

## Keyword Profile

- `spectral`: 151
- `geometry`: 44
- `entropy`: 40
- `hallucination`: 19
- `probe`: 12
- `hidden state`: 9
- `topolog`: 8
- `manifold`: 5
- `geometric`: 4
- `causal`: 4
- `dimension`: 4
- `flow`: 4

## Abstract / Opening Summary

Verifying whether a language model is genuinely reasoning or pattern-matching remains an open problem: learned verifiers are expensive, and output-based heuristics are brittle. We show that valid mathematical reasoning induces a mea- surable, training-free spectral signature in trans- former attention. By treating each attention matrix as a weighted token graph, we extract four diag- nostics: Fiedler value, High-Frequency Energy Ratio (HFER), spectral entropy, and smoothness, that require no learned parameters. Experiments across seven models from four architectural fam- ilies yield effect sizes up to Cohen’s d = 3.30 (p < 10−116), enabling 85–96% single-threshold classification accuracy. Two findings sharpen the interpretation. First, Platonic validity: the spec- tral signal tracks logical coherence rather than compiler acceptance, proofs rejected for time- outs or missing imports are correctly classified as valid, a distinction confirmed by a manual au- dit (κ = 0.82, n = 51). Second, architectural determinism: Sliding Window Attention shifts the discriminative feature from HFER to smooth- ness (d = 2.09, p < 10−48), showing that atten- tion design governs which spectral channel en- codes reasoning quality. Causal ablation confirms the signature traces induction-head circuits. The method generalises to informal chain-of-thought (d = 0.78, p < 10−3), and in proof search, HFER reranking improves Best-of-16 Pass@1 by +4.4– 6.6%, matching 98% of the AUC of fully super- vised probes with zero labels. Spectral graph anal- ysis is a principled, architecture-aware primitive for reasoning verification. * 1Devoteam, Levallois-Perret, France. Correspondence to: Valentin No¨el <valentin.noel@devoteam.com>. Proceedings of the 43 rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s). * vcnoel/geometry-of-reason

## Method / Algorithms Extract

Full Balanced Majority Class 76.6% 50.0% Random Forest 74.5% – Spectral Threshold 77.1% 68.4% 4.6. Downstream Utility and Supervised Baselines Best-of-N Proof Search. We apply HFER as a zero-shot reranker in Best-of-N (N=16, T=0.7) proof search on MiniF2F. Table 4 reports the full comparison across four reranking strategies on Llama-3.1-8B: The AUC–Pass@1 inversion arises because log-probability is blind to confident hallucinations: structurally incoher- ent proofs that are nonetheless fluent. HFER penalises ex- actly those cases. Cross-model replication confirms the gain scales with spectral separation: on Phi-3.5-mini (d=3.30), HFER achieves 37.8% vs. log-probability’s 31.2% (+6.6%), compared to +4.4% on Llama-3.1-8B (d=3.00). Comparison to Supervised Probing. We compare against Table 4. Best-of-16 proof search reranker comparison (N=16, T=0.7; Llama-3.1-8B, MiniF2F). HFER surpasses log-probability on Pass@1 despite lower AUC, penalising confident hallucinations the ensemble confirms as orthogonally complementary. Reranker Pass@1 AUC-ROC Random 22.4% – Token Entropy 30.4% 0.971 Log-Prob 29.8% 0.979 HFER (ours) 34.2% 0.962 Ensemble (ZLP−ZHFER) 37.1% 0.988 the supervised hallucination probe of Obeso et al. (2026), trained on Llama-3.1-8B Layer 16 hidden states: Table 5. Supervised vs. unsupervised. With only 50 calibration examples, HFER achieves 98% of the fully supervised upper bound (91.8%±2.4% accuracy).

## Experiments / Evidence Extract

Experiments across seven models from four architectural fam- ilies yield effect sizes up to Cohen’s d = 3.30 (p < 10−116), enabling 85–96% single-threshold classification accuracy. Two findings sharpen the interpretation. First, Platonic validity: the spec- tral signal tracks logical coherence rather than compiler acceptance, proofs rejected for time- outs or missing imports are correctly classified as valid, a distinction confirmed by a manual au- dit (κ = 0.82, n = 51). Second, architectural determinism: Sliding Window Attention shifts the discriminative feature from HFER to smooth- ness (d = 2.09, p < 10−48), showing that atten- tion design governs which spectral channel en- codes reasoning quality. Causal ablation confirms the signature traces induction-head circuits. The method generalises to informal chain-of-thought (d = 0.78, p < 10−3), and in proof search, HFER reranking improves Best-of-16 Pass@1 by +4.4– 6.6%, matching 98% of the AUC of fully super- vised probes with zero labels. Spectral graph anal- ysis is a principled, architecture-aware primitive for reasoning verification. * 1Devoteam, Levallois-Perret, France. Correspondence to: Valentin No¨el <valentin.noel@devoteam.com>. Proceedings of the 43 rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s). * vcnoel/geometry-of-reason 1. Introduction The remarkable performance of large language models (LLMs) on mathematical reasoning tasks (Lewkowycz et al., 2022; Trinh et al., 2024; Chervonyi et al., 2025; Azerbayev et al., 2023) has intensified interest in understanding and verifying the computational mechanisms underlying their outputs. When a model generates a mathematical proof, practitioners face a fundamental, epistemological, challenge: determining whether the output reflects genuine logical reasoning or sophisticated pattern matching that produces plausible-looking but potentially flawed arguments. Re- cent evaluations reinforce this concern: even frontier mod- els achieve below 25% on olympiad-level proofs (Petrov et al., 2025), suggesting a “reasoning illusion” where suc- cess may stem from pattern matching rather than genuine insight (Kuang et al., 2025). This challenge is particularly acute in high-stakes applications such as automated theo- rem proving (Polu & Sutskever, 2020; Yang et al., 2024; Ospanov et al., 2025), mathem...

## Conclusion / Discussion Extract

Conclusion We have introduced a training-free method for detecting valid mathematical reasoning through spectral analysis of transformer attention. Our experiments across seven mod- els from four architectural families establish that: (1) the spectral signature is universal (pMW < 10−47, pt < 10−75) and robust across all difficulty strata (d ≥1.31); (2) effect sizes are exceptionally large (up to d = 3.30); (3) single- threshold classification achieves 85.9–95.6% accuracy; (4) the method detects logical coherence rather than compiler acceptance (“Platonic validity”); (5) HFER reranking im- proves Best-of-16 Pass@1 by +4.4–6.6%, achieving 98% of fully supervised AUC with zero labels; and (6) Lanc- zos acceleration reduces eigendecomposition to O(kN 2), enabling real-time use at 32k-token contexts. These findings open several directions for future work: theo- retical analysis of why the spectral signature emerges, exten- sion to natural language reasoning, integration with proof assistants for real-time feedback, and investigation of other architectural features (grouped-query attention, mixture-of- experts) that may affect spectral properties. More broadly, our work demonstrates that interpretability methods grounded in classical mathematical frameworks, here, spectral graph theory, can yield practical tools for understanding and verifying neural network reasoning. As language models are deployed in increasingly high-stakes reasoning applications, such principled verification methods become essential for ensuring reliability and safety. 8. Limitations Scope. Validation is scoped to formalized Lean 4 reasoning on MiniF2F. Informal chain-of-thought yields a substantially weaker signal (d=0.78 vs. d>1.3), and claims do not extend to unstructured text. Model-specific calibration. Optimal thresholds are architecture-specific; Sliding Window Attention shifts the dominant feature from HFER to smoothness, requiring per- model tuning. Diagnostic, not causal. The method is a correlation-based diagnostic. A mechanistic account of why the signature emerges and how to exploit it for targeted reasoning im- provement remains future work. Computational cost. Full eigendecomposit...

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

TBD: add short excerpts with page markers from `../texts/geometry-of-reason-spectral-signatures-of-valid-mathematical-reasoning.txt`.
