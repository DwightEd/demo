# Limited Reasoning Space: The cage of long-horizon reasoning in LLMs

- **Local PDF filename**: `Limited Reasoning Space.pdf`
- **Slug**: `limited-reasoning-space`
- **Pages**: 26
- **Approx Words**: 14119
- **Auto Tags**: dynamics;faithfulness;uncertainty;hallucination
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.618574

## Keyword Profile

- `entropy`: 80
- `trajectory`: 29
- `transition`: 17
- `spectral`: 14
- `hallucination`: 11
- `phase`: 8
- `geometric`: 5
- `hidden state`: 5
- `chain of thought`: 5
- `manifold`: 4
- `latent`: 4
- `dimension`: 4

## Abstract / Opening Summary

The test-time compute strategy, such as Chain- of-Thought (CoT), has significantly enhanced the ability of large language models to solve complex tasks like logical reasoning. However, empirical studies indicate that simply increasing the com- pute budget can sometimes lead to a collapse in test-time performance when employing typical task decomposition strategies such as CoT. This work hypothesizes that reasoning failures with larger compute budgets stem from static plan- ning methods, which hardly perceive the intrin- sic boundaries of LLM reasoning. We term it as the Limited Reasoning Space hypothesis and per- form theoretical anaylsis through the lens of a non- autonomous stochastic dynamical system. This insight suggests that there is an optimal range for compute budgets; over-planning can lead to redun- dant feedback and may even impair reasoning ca- pabilities. To exploit the compute-scaling benefits and suppress over-planning, this work proposes Halo, a model predictive control framework for LLM planning. Halo is designed for long-horizon tasks with reason-based planning and crafts an entropy-driven dual controller, which adopts a Measure-then-Plan strategy to achieve control- lable reasoning. Experimental results demonstrate that Halo outperforms static baselines on com- plex long-horizon tasks by dynamically regulating planning at the reasoning boundary.

## Method / Algorithms Extract

MATHEMATICAL & SYMBOLIC REASONING LONG-CONTEXT STABILITY & RETRIEVAL GSM8K (TIER 1) MATH (TIER 1) OMNI (TIER 2) RULER (TIER 2) LONGBENCH INFBENCH LRA-L2 SR↑ RTO SR↑ RTO SR↑ RTO SR↑ RTO SR↑ RTO SR↑RTO SR↑RTO STANDARD COT 82.4 1.00 10.8 1.00 12.5 1.00 18.2 1.00 35.6 1.00 28.9 1.00 15.3 1.00 COT-SC (k=10) 86.1 2.15 14.2 2.15 15.8 2.15 21.4 2.15 39.2 2.15 32.5 2.15 18.7 2.15 ADACOT 88.5 1.45 19.3 1.45 21.5 1.45 25.1 1.45 42.8 1.45 36.7 1.45 22.1 1.45 TOT (b=5,d=3) 87.4 3.50 19.1 3.50 21.3 3.50 25.7 3.50 41.5 3.50 35.2 3.50 21.5 3.50 COT-VALVE 88.9 0.85 36.4 0.85 40.2 0.85 65.8 0.85 58.3 0.85 51.6 0.85 38.9 0.85 HALO (OURS) 89.2 1.29 38.5 1.29 42.7 1.29 76.4 1.29 62.5 1.29 55.8 1.29 42.3 1.29 Table 3. Cross-Model Robustness on Omni-MATH. Halo demon- strates consistent gains across varying scales. Rel. Gain de- notes improvement over AdaCoT. Notably, Halo pushes the SOTA Qwen2.5-Math to 91.3%, breaking the static inference ceiling. BACKBONE COT TOT ADACOT HALO GAIN Small & Efficient Models LLAMA-3.1-8B 42.5 48.1 49.3 56.8 +15% MISTRAL-V0.3 44.2 49.5 51.0 57.4 +13% GEMMA-2-27B 58.1 62.4 63.8 69.2 +8.5% MoE Architectures MIXTRAL-8X7B 55.4 59.2 60.5 66.7 +10% DEEPSEEK-V2-LITE 52.3 57.8 58.9 65.1 +11% Large & SOTA Models LLAMA-3.1-70B 76.8 79.5 80.2 83.4 +4.0% QWEN2.5-72B 82.4 84.1 84.8 87.2 +2.8% QWEN2.5-MATH 88.0 89.2 89.5 91.3 +2.0% tion exactly when the model is about to degenerate, avoiding both premature interruption and delayed intervention. 4.4. Parameter Sensitivity We evaluate the robustness of Halo with respect to its two hyperparameters: the tolerance threshold Ψ and the en- tropy sensitivity α. Table 4 reports the Success Rate (SR) on Omni-MATH using the LLaMA-3-8B model. The re- sults indicate that Halo maintains stable performance for Ψ ∈[4.0, 6.0]. Lower thresholds (Ψ < 3.0) lead to ex- cessive interventions which interrupt valid reasoning steps, while higher thresholds (Ψ > 8.0) delay the necessary rec- tification. Regarding α, values in the range [0.7, 1.0] yield consistent gains. Notably, across all tested configurations, Halo outperforms the AdaCoT baseline (49.3%), suggesting that the improvements strictly originate from the entropy- driven mechanism rather than specific parameter tuning. Logic Puzzles Vector Algebra Geometry Proof Combina- torics Adv. Calculus Differential Eq. Number Theory Figure 5. Distributional Alignment of Reasoning Boundaries. We compare the step index of baseline failures (Red) and Halo in- terventions (Blue) across seven reasoning domains. The tight clustering of data points indicates that Halo consistently identifies...

## Experiments / Evidence Extract

Experiments To empirically validate the Limited Reasoning Space hy- pothesis and evaluate the efficacy of Halo, we design a comprehensive evaluation protocol centered on two core question: (1) Can the entropy-based observer reliably detect the boundary of the Limited Reasoning Space and trigger the Halo intervention with temporal precision? (2) Can Halo effectively extend the reasoning horizon N ∗, translating ex- tended reasoning chains into progressively higher accuracy without incurring prohibitive computational costs? 4.1. Experimental Setup Benchmark and Baselines. Guided by the theoretical Crit- ical Reasoning Horizon N ∗(Eq. 7), we stratify our evalua- tion into two tiers: (1) Tier 1 (Within-Capacity, D < N ∗): We use GSM8K (Cobbe et al., 2021) and MATH (Easy) (Hendrycks et al., 2021) to verify that Halo maintains ef- ficiency in stable regimes. (2) Tier 2 (Beyond-Capacity, D ≫N ∗): We employ Omni-MATH (Gao et al., 2024) and RULER (Hsieh et al., 2024) to stress-test the "Rea- soning Collapse." We compare Halo against 8 baselines across three paradigms: Open-Loop Generation (Standard CoT), Search-Based Optimization (CoT-SC (Wang et al., 2022), ToT (Yao et al., 2024), GoT (Besta et al., 2024)), and Adaptive Strategies (AdaCoT (Pan et al., 2023), CoT-Valve (Ma et al., 2025)). Detailed dataset statistics and baseline hyperparameters are provided in Appendix F. Metrics and Backbones. Beyond standard Success Rate (SR), we introduce Rectification Success Rate (RSR) to measure controller precision and Relative Token Overhead (RTO) to quantify efficiency compared to standard CoT. We Algorithm 1 Halo: Horizon-Aware Logical Optimization Input: Prompt Q; LLM Policy πθ; Stability Threshold Ψ; Dynamics params (α, β). Output: Final Reasoning Chain Cfinal. 1 Initialize Context C0 ←[Q] Initialize Cumulative Uncer- tainty Ω0 ←0 t ←0 2 while not IsFinished(Ct) do /* Phase 1: The Observer (Dynamics Estimation) */ 3 Compute attention matrix At from forward pass πθ(Ct) Calculate mean attention entropy Ht via Eq. (8) Esti- mate instantaneous drift: ˆλt ←β + α · Ht /* Phase 2: The Controller (State Integration) */ 4 Update accumulated uncertainty: Ωt ←Ωt−1 + ˆλt // Eq. (9) 5 if CheckStability(Ωt ≥Ψ) then /* Critical Regime: Trigger Actuator (Reset) */ // 1. Manifold Projection via Semantic Compression (Eq. 10) 6 ¯st ←LLM(Promptcompress ⊕Ct) // 2. Dynamics Interrupt (History ...

## Conclusion / Discussion Extract

Conclusion In this work, we formulate long-horizon reasoning as a Non-autonomous Stochastic Dynamical System , attributing reasoning failures to the exponential error accumulation that defines a Limited Reasoning Space. To overcome this intrin- sic boundary, we propose Halo, a Model Predictive Control (MPC) framework that shifts the paradigm from open-loop generation to a regulated Measure-then-Plan strategy. By leveraging Attention Entropy to detect stability drifts and executing semantic state resets, Halo effectively mitigates noise dominance. Our experiments on Omni-MATH and RULER demonstrate that Halo significantly extends the ef- fective reasoning horizon, achieving superior accuracy with substantially lower computational cost compared to static decomposition methods. 8 ===== PAGE 9 / 26 ===== Limited Reasoning Space: The cage of long-horizon reasoning in LLMs 1 2 4 8 12 16 20 24 Decomposition Depth (D) 50 100 200 400 600 800 1000 1500 Reasoning Width (W, avg tokens/step) 0.92 0.91 0.92 0.91 0.82 0.74 0.65 0.48 0.90 0.91 0.87 0.74 0.47 0.15 0.05 0.04 0.88 0.88 0.73 0.16 0.05 0.07 0.03 0.05 0.87 0.75 0.16 0.02 0.08 0.08 0.07 0.04 0.84 0.47 0.05 0.03 0.05 0.02 0.07 0.04 0.73 0.16 0.05 0.05 0.03 0.08 0.07 0.08 0.64 0.07 0.06 0.08 0.03 0.03 0.02 0.04 0.21 0.07 0.04 0.04 0.05 0.03 0.07 0.02 Empirical Reasoning Volume on Omni-MATH Capacity Boundary 0.2 0.4 0.6 0.8 Success Rate (SR) Figure 6. Stability Phase Transition Analysis. The heatmap illustrates the Success Rate (SR) as a function of Reasoning Length N. The dashed line represents the theoretical stability boundary N ∗. The stochastic fluctuations in the red Divergence Regime indicate the collapse of semantic consistency as the accumulated uncertainty dominates the trajectory. Table 4. Sensitivity Analysis on Omni-MATH (LLaMA-3-8B). Success Rate (SR, %) with varying Tolerance Thresholds (Ψ) and Sensitivity (α). The performance remains robust across a wide range of configurations compared to the AdaCoT baseline (49.3%). SENSITIVITY (α) TOLERANCE THRESHOLD (Ψ) Ψ = 2.0 Ψ = 4.0 Ψ = 5.0 Ψ = 6.0 Ψ = 8.0 0.5 51.2 52.8 53.1 52.5 50.4 0.85 (OURS) 53.5 56.5 56.8 56.2 51.9 1.0 52.8 55.9 56.4 55.7 51.2 1.2 50.1 54.2 54...

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

TBD: add short excerpts with page markers from `../texts/limited-reasoning-space.txt`.
