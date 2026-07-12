# An Isotropic Approach to Efficient Uncertainty Quantification with Gradient Norms

- **Local PDF filename**: `An Isotropic Approach to Efficient Uncertainty.pdf`
- **Slug**: `an-isotropic-approach-to-efficient-uncertainty`
- **Pages**: 30
- **Approx Words**: 15437
- **Auto Tags**: geometry;uncertainty
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.601109

## Keyword Profile

- `entropy`: 20
- `curvature`: 8
- `spectral`: 8
- `hallucination`: 6
- `dimension`: 4
- `trajectory`: 3
- `faithful`: 1

## Abstract / Opening Summary

Existing methods for quantifying predictive uncertainty in neural networks are either computationally intractable for large language models or require access to training data that is typically unavailable. We derive a lightweight alternative through two approximations: a first-order Taylor expansion that expresses uncertainty in terms of the gradient of the prediction and the parameter covariance, and an isotropy assumption on the parameter covariance. Together, these yield epistemic uncertainty as the squared gradient norm and aleatoric uncertainty as the Bernoulli variance of the point prediction, from a single forward-backward pass through an unmodified pretrained model. We justify the isotropy assumption by showing that covariance estimates built from non-training data introduce structured distortions that isotropic covariance avoids, and that theoretical results on the spectral properties of large networks support the approximation at scale. Validation against reference Markov Chain Monte Carlo estimates on synthetic problems shows strong correspondence that improves with model size. We then use the estimates to investigate when each uncertainty type carries useful signal for predicting answer correctness in question answering with large language models, revealing a benchmark-dependent divergence: the combined estimate achieves the highest mean AUROC on TruthfulQA, where questions involve genuine conflict between plausible answers, but falls to near chance on TriviaQA’s factual recall, suggesting that parameter-level uncertainty captures a fundamentally different signal than self-assessment methods.

## Method / Algorithms Extract

TriviaQA TruthfulQA P(True) 0.69 ± 0.06 0.55 ± 0.06 Sem. Entropy 0.55 ± 0.03 0.54 ± 0.08 Na¨ıve Entropy 0.52 ± 0.04 0.51 ± 0.08 Aleatoric 0.60 ± 0.04 0.60 ± 0.09 Epistemic 0.52 ± 0.07 0.55 ± 0.07 Epi. & Alea. 0.61 ± 0.05 0.63 ± 0.08 Table 2: AUROC (mean ± std over 300 bootstrap runs, 4 LLMs) for predicting answer correctness. Higher is better; 0.50 is chance. Best per column in bold. Table 2 reports the AUROC scores averaged across models. On TriviaQA, P(True) (0.69) dominates. On TruthfulQA, the pattern reverses: the com- bined estimate achieves 0.63, the highest score on this benchmark, significantly outperforming P(True) (0.55) and the entropy baselines (p < 0.01 after Benjamini–Hochberg (BH) correction; Ap- pendix G.7). This divergence between the two benchmarks is the most instructive finding. TriviaQA tests factual recall, where a model may be equally confident in correct and incorrect answers, so uncertainty and correctness are largely independent. TruthfulQA targets common misconceptions with genuinely ambiguous answer spaces, creating both inherent output ambiguity and epistemic conflict between popular and truthful answers. In this setting, the aleatoric estimate (0.60) reflects output-level hedging, the epistemic estimate (0.55) captures parameter-level sensitivity, and their combination (0.63) outperforms all baselines. P(True) loses its advantage because the model’s self-assessment is precisely what TruthfulQA is designed to ===== PAGE 9 / 30 ===== An Isotropic Approach to Efficient Uncertainty Quantification with Gradient Norms 9 defeat, while aleatoric and epistemic uncertainty carry genuinely useful signal, suggesting that these uncertainty types are most informative when the task involves conflict between plausible parameterizations rather than factual memorization. The epistemic estimate and P(True) are only weakly correlated (Spearman ρ ≈−0.2 on both benchmarks; Appendix G.8), confirming they capture largely distinct signal. The per-model breakdown (Table 11 in Appendix G.5) reveals substantial model-level variation, but several trends are consistent. The benchmark divergence is universal: on TruthfulQA, the combined estimate is at least on par with the best baseline for every model, while on TriviaQA the reverse holds for three of four models. The relative utility of aleatoric and epistemic uncertainty is model-dependent: both Llama models favor aleatoric uncertainty on TruthfulQA, while OLMo and Phi-4 favor epistemic. Models from the same family behaving alike suggests training data as a driver—models that have seen more relevant dat...

## Experiments / Evidence Extract

Experiments We first validate ∥g∥2 against MCMC estimates on synthetic problems (Section 4.1), then investigate the utility of aleatoric and epistemic uncertainty for predicting answer correctness in LLM question answering (Section 4.2). 4.1. Validation Linear XOR Rings Epistemic (GN) r 0.95 0.65 0.86 ρ 0.99 0.68 0.44 Epistemic (LA) r 0.95 0.68 0.86 ρ 0.99 0.70 0.46 Aleatoric r 0.99 0.76 0.95 ρ 1.00 0.74 0.58 (a) Binary classification Clusters Spirals Rings Epistemic (GN) r 0.86 0.76 0.88 ρ 0.97 0.91 0.97 Aleatoric r 0.95 0.96 0.96 ρ 0.99 0.97 0.98 (b) Multiclass classification Linear Nonlin. Epistemic (GN) r 0.98 0.73 ρ 0.99 0.81 Epistemic (LA) r 1.00 0.93 ρ 1.00 0.97 (c) Regression Table 1: Pearson (r) and Spearman (ρ) correlations between our estimates and MCMC estimates. GN: gradient norm ∥g∥2; LA: Laplace g⊤H−1g. Aleatoric: p(yc | x, θ∗)(1 −p(yc | x, θ∗)). We compare our estimates directly against the quantity they are designed to approximate, rather than using out-of-distribution (OOD) detection as a proxy. OOD detection assumes that inputs far from the training data produce high epistemic uncertainty, but Bayesian epistemic uncertainty depends on ===== PAGE 7 / 30 ===== An Isotropic Approach to Efficient Uncertainty Quantification with Gradient Norms 7 (a) Epistemic, MCMC (b) Epistemic, GN (c) Aleatoric, MCMC (d) Aleatoric, point est. Figure 2: Multiclass spirals uncertainty maps. Left two panels: epistemic uncertainty (MCMC vs. gradient norm ∥g∥2). Right two panels: aleatoric uncertainty (MCMC vs. point estimate). All maps are individually normalized to [0, 1]. Additional problems in Appendix D. the space of plausible parameterizations, not on distance from training data alone—a linear classifier, for instance, cannot exhibit high epistemic uncertainty far from its boundary regardless of how distant the input is from any training point. This disconnect has been observed in practice (Ulmer et al., 2020), so failures on OOD benchmarks may reflect a mismatch between the validation assumption and the quantity being measured rather than a deficiency of the method. On synthetic problems where the parameter count permits full posterior inference, we use Hamiltonian Monte Carlo (HMC) (Betancourt, 2018) with dual-averaging step-size adaptation (Nesterov, 2009; Hoffman and Gelman, 2014) to compute Varθ[p(yc | x, θ)] and measure how well ∥g∥2 tracks it, using P...

## Conclusion / Discussion Extract

Conclusion By approximating uncertainty via a first-order Taylor expansion under isotropic covariance, we reduce epistemic uncertainty to the squared norm of the prediction gradient and aleatoric uncertainty to the Bernoulli variance of the point prediction, giving a complete uncertainty decomposition from a single forward-backward pass through an unmodified pretrained model. Validation against reference MCMC estimates on synthetic problems shows strong correspondence in classification (Spearman ρ of 0.44–0.99 across settings), with an improving trend at larger model sizes that supports the isotropy assumption. The downstream question answering experiments reveal that uncertainty estimates are most informative when the model faces genuine conflict between plausible parameterizations (as on TruthfulQA, where at least one uncertainty estimate exceeds all baselines for every model), rather than when correctness depends on factual memorization, though the relative utility of aleatoric and epistemic uncertainty varies substantially between models. More broadly, the near-chance epistemic AUROC on TriviaQA suggests that epistemic uncertainty may not be as useful for hallucination detection as previously assumed (Xiao and Wang, 2021; Han et al., 2025; Park et al., 2026; Liu et al., 2026), since factual errors need not coincide with parameter-level disagreement; gradient-based uncertainty captures a complementary signal to self-assessment methods like P(True), with the two excelling on fundamentally different question types. More generally, even when the Bayesian calibration of the squared gradient norm is approximate, it retains a meaningful ranking of inputs as a measure of local sensitivity to parameter perturbations. Limitations. The estimates are on the scale of squared gradient norms, which lack intuitive interpretation and do not generalize across model architectures: training an answer correctness classifier on the uncertainty estimates from three models and evaluating on the fourth yields chance-or- below performance, with the relationship between gradient norm and correctness occasionally inverting on the held-out model, even after normalizing by the squared pa...

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

TBD: add short excerpts with page markers from `../texts/an-isotropic-approach-to-efficient-uncertainty.txt`.
