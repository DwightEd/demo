# ===== PAGE 1 / 33 ===== Published as a conference paper at ICLR 2026

- **Local PDF filename**: `15744_TokUR_Token_Level_Uncert.pdf`
- **Slug**: `15744-tokur-token-level-uncert`
- **Pages**: 33
- **Approx Words**: 22276
- **Auto Tags**: uncertainty
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.592164

## Keyword Profile

- `entropy`: 25
- `hallucination`: 15
- `chain of thought`: 7
- `geometry`: 1
- `topolog`: 1
- `transition`: 1
- `faithful`: 1
- `probe`: 1

## Abstract / Opening Summary

While Large Language Models (LLMs) have demonstrated impressive capabili- ties, their output quality remains inconsistent across various application scenarios, making it difficult to identify trustworthy responses, especially in complex tasks requiring multi-step reasoning. In this paper, we propose a Token-level Uncertainty estimation framework for Reasoning (TokUR) that enables LLMs to self-assess and self-improve their responses in mathematical reasoning. Specifically, we in- troduce low-rank random weight perturbation during LLM decoding to generate predictive distributions for token-level uncertainty estimation, and we aggregate these uncertainty quantities to capture the semantic uncertainty of generated re- sponses. Experiments on mathematical reasoning datasets of varying difficulty demonstrate that TokUR exhibits a strong correlation with answer correctness and model robustness, and the uncertainty signals produced by TokUR can be leveraged to enhance the model’s reasoning performance at test time. These results highlight the effectiveness of TokUR as a principled and scalable approach for improving the reliability and interpretability of LLMs in challenging reasoning tasks. The source code is avaliable at https://github.com/Wang-ML-Lab/TokUR. 1

## Method / Algorithms Extract

ISO? MATH500 GSM8K DeepScaleR AUROC AUPRC ACC∗ AUROC AUPRC ACC∗ AUROC AUPRC ACC∗ Llama-3.2-1B-Instruct CoT (Lower-Bound) - - - 25.60±0.00 - - 44.43±0.00 - - 14.25±0.00 SE ✗ 47.29±3.81 25.71±2.33 24.13±4.42 50.64±4.44 45.09±0.72 42.62±0.16 46.30±0.21 12.94±0.23 12.58±0.49 SAR ✗ 44.57±2.04 24.03±2.53 21.07±1.62 50.28±0.97 43.24±0.89 43.95±0.77 43.14±1.42 12.34±0.35 11.14±0.47 UEcc ✗ 48.75±1.05 25.79±1.83 25.20±0.33 49.05±0.46 60.02±0.44 59.62±0.22 48.68±0.24 13.77±0.29 14.23±0.45 UDeg ✗ 60.57±2.31 36.32±2.59 30.93±0.94 66.60±0.36 75.72±0.36 71.99±0.39 56.88±0.54 18.04±0.63 16.50±0.39 P(True) ✓ 54.38±1.20 26.39±1.26 27.60±1.18 56.64±0.04 48.22±0.03 48.92±0.00 59.58±0.43 17.48±0.25 17.52±0.50 LLM-Check ✓ 56.41±0.96 27.01±1.22 31.33±1.29 71.01±0.02 61.29±0.08 59.54±0.00 55.76±0.48 14.55±0.26 17.30±0.51 INSIDE ✓ 55.71±4.69 28.82±4.05 29.20±4.33 53.66±0.92 46.03±0.23 45.79±1.25 54.73±0.82 15.50±0.48 16.30±0.35 PE ✓ 57.08±0.89 26.88±1.05 31.33±0.82 71.21±0.03 61.61±0.08 59.85±0.00 56.09±0.46 14.74±0.23 17.33±0.92 LL ✓ 55.41±0.54 25.88±0.87 29.87±0.82 69.01±0.03 58.51±0.09 57.38±0.00 53.84±0.47 13.93±0.23 16.83±0.48 Self-Certainty ✓ 71.17±0.30 48.37±0.50 38.13±0.61 73.41±0.00 68.38±0.00 61.38±0.00 71.93±0.04 33.81±0.08 21.76±0.04 DeepConf ✓ 71.77±0.12 46.00±0.42 39.87±0.46 75.70±0.00 69.72±0.00 62.77±0.00 71.65±0.04 29.99±0.05 22.00±0.04 TokUR (TU, Ours) ✓ 80.64±0.29 56.79±0.74 44.67±0.46 75.07±0.05 70.29±0.07 62.31±0.00 83.55±0.02 47.56±0.04 25.71±0.02 TokUR (AU, Ours) ✓ 80.61±0.27 56.73±0.75 44.67±0.46 75.03±0.06 70.22±0.05 62.21±0.18 83.52±0.02 47.48±0.05 25.71±0.02 TokUR (EU, Ours) ✓ 79.74±0.21 56.64±0.41 44.13±0.83 71.79±0.80 66.40±1.02 59.74±1.00 82.87±0.32 46.76±0.38 25.52±0.11 Llama-3.1-8B-Instruct CoT (Lower-Bound) - - - 48.60±0.00 - - 85.69±0.00 - - 24.86±0.00 SE ✗ 62.93±0.90 55.21±1.04 55.73±0.83 55.61±3.36 87.16±1.14 86.77±1.01 67.68±0.94 35.18±1.00 35.55±0.37 SAR ✗ 69.42±2.19 63.74±3.03 59.20±1.06 60.16±2.22 89.24±0.74 87.99±0.81 73.01±0.28 42.89±0.65 37.51±0.12 UEcc ✗ 50.23±2.23 49.48±2.44 49.60±2.04 47.47±2.15 84.69±0.89

## Experiments / Evidence Extract

This section presents practical applications of our TokUR for LLM reasoning. For additional experimental results, please refer to Appendix E. Datasets. We run our main experiments on three mathematical reasoning benchmarks of varying difficulty levels: GSM8K (Cobbe et al., 2021) (grade-school arithmetic problems), MATH500 (Light- man et al., 2023) (challenging high school/college mathematics competition problems), and 5,000- example subset of DeepScaleR (Luo et al., 2025) (high-difficulty problems from diverse sources). For these complex math problems, LLMs often need to perform multi-step reasoning (Wei et al., 2022b; Yao et al., 2023; Zhou et al., 2023) to reach the final answer. These tasks inherently involve long-form generation, therefore well-suited for evaluation of uncertainty estimation methods. To assess the generalization of TokUR beyond mathematical reasoning, we further evaluate TokUR on five non-math long-form generation tasks, spanning logical reasoning, code generation, and truthfulness evaluation. For logical reasoning, we use three tasks from Reasoning Gym (Stojanovski et al., 2025): Zebra Puzzles, Leg Counting, and Color-Cube Rotation. For code generation, we evaluate on the HumanEval (Chen, 2021) benchmark, a widely adopted standard for functional code synthesis. For truthfulness, we use the FactScore (Min et al., 2023) dataset, which measures factual consistency by decomposing generated outputs into atomic facts; we follow prior work and use GPT-5-mini as both the fact annotator and judge. Models. We evaluate our TokUR using models from two open-source LLM families: Llama (3.2-1B-Instruct and 3.1-8B-Instruct) (Grattafiori et al., 2024) and Qwen (2.5-3B-Instruct and 2.5-7B-Instruct) (Team et al., 2024). These models represent recent advances in open-source instruction tuning and provide a practical balance between capability and efficiency. Their differing model scales and architectural families further enable us to examine the consistency of uncertainty estimation across both model sizes and model types. Implementation of our TokUR. We estimate token-level uncertainties by applying random pertur- bations as in Eqn. 17 to the query and key weight matrices (W Q, W K) (Vaswani et al., 2017) in all the attention layers of LLMs (Hu et al., 2022; Yang et al., 2023; Wang et al., 2024; Shi et al., 2024). For more details, please refer to Appendi...

## Conclusion / Discussion Extract

conclusions are the authors’ own. B ALGORITHM DETAILS Algorithm 1 Low-Rank Weight Perturbation as Approximation of Weight Posterior. 1: Input 2: The base model policy p(y|x); 3: The set of weight matrices to be Bayesianized {W k 0 }N k=1; 4: rank of noise matrix r′; 5: The perturbation strength σq. 6: for i = 1 to N do 7: U, diag(d), V ⊤←SVD(W k 0 ). ▷Eqn. 16 8: U ′ ←the first r′ columns of matrix U. 9: Sample noise matrix ϵ ∈Rn×r′: ϵij ∼N(0, σq). 10: Perturb the weight matrix: W k ←W k 0 + U ′ϵ⊤. ▷Eqn. 17 11: Get the weight posterior: q(vec(W k)|σq). ▷Eqn. 18 12: end for 13: Output: The overall approximate posterior: q(θ|σq) ←Q k q(vec(W k)|σq) C PROOF OF PROPOSITIONS Lemma C.1 (Definition of Conditional Entropy (Cover, 1999)). Give (y, x) ∼p(y, x), the conditional entropy H(y|x) is defined as H(y|x) = X x∈X p(x)H(y|x) = Ex∼p(x)[H(y|x)]. (20) 17 ===== PAGE 18 / 33 ===== Published as a conference paper at ICLR 2026 Algorithm 2 Particle Filtering for Inference-Time Scaling (Puri et al., 2025) 1: Input 2: The number of particles N; 3: A reward model br; 4: A LLM pM and a prompt c. 5: Initialize N particles {xi 1 ∼pM(·|c)}N i=1. 6: t ←1. 7: while not all particles stop do 8: Update rewards w = {br(x(1) 1:t), br(x(2) 1:t), . . . , br(x(N) 1:t )}. 9: Compute softmax distribution θ = softmax(w). 10: Sample indices {j(i) t }N i=1 ∼Pt(j = i) = θi. 11: Update the set of particles as {x(j(I) t ) 1:t }N i=1. 12: Transition {xi t+1 ∼pM(·|c, x(i) 1:t)}N i=1. 13: t ←t + 1. 14: end while 15: Output: The set of particles in the end. Lemma C.2 (Chain rule of Conditional Entropy (Cover, 1999)). Let X and Y be two random variables, then the conditional entropy of the joint distribution H(X, Y ) can be decomposed as: H(X, Y ) = H(X) + H(Y |X) (21) Lemma C.1 (Cover, 1999) reveals the relationship between conditional entropy H(y|x) and the entropy derived from conditional probability distributions. Lemma C.2 lays the foundation for estimating the uncertainties of sequences. The two lemmas together give us the following proposition. Proposition C.1 (Decomposition of Query-Level Uncertainty, Eqn. 4). Suppose that we have an input sequence x and a model policy p(y|x). The sequence-level...

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

TBD: add short excerpts with page markers from `../texts/15744-tokur-token-level-uncert.txt`.
