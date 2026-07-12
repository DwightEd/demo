# Entropy trajectory shape predicts LLM reasoning reliability: A diagnostic study of uncertainty dynamics in chain-of-thought

- **Local PDF filename**: `ENTROPY TRAJECTORY SHAPE PREDICTS LLM REASONING.pdf`
- **Slug**: `entropy-trajectory-shape-predicts-llm-reasoning`
- **Pages**: 18
- **Approx Words**: 11462
- **Auto Tags**: dynamics;faithfulness;uncertainty;step-level
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.611138

## Keyword Profile

- `entropy`: 97
- `trajectory`: 61
- `chain of thought`: 11
- `transition`: 5
- `probe`: 3
- `dimension`: 1

## Abstract / Opening Summary

Chain-of-thought (CoT) reasoning improves LLM accuracy on complex tasks, yet reliable methods for detecting reasoning failures without expensive multi-sample approaches remain elusive. We study whether the shape of uncertainty dynamics across reasoning steps—captured cheaply by sampling a handful of answer completions at each step—predicts whether the final answer is correct. We introduce the concept of entropy-trajectory monotonicity: a chain is monotone if its per-step answer-distribution entropy decreases at every step, reflecting consistent uncertainty reduction. On GSM8K (n=300) with Qwen2.5-7B-Instruct, monotone chains achieve 68.8% accuracy versus 46.8% for non-monotone chains—a gap of +21.9 percentage points (Fisher’s exact p=0.0005; OR= 2.50). Critically, the scalar total entropy reduction is not predictive (ρ=−0.06, p=0.31), revealing a shape-over-magnitude dissociation: it is whether entropy decreases at every step, not how much it drops, that predicts correctness. The dissociation extends to the graded signal: in full-scale runs, increasing violation count consistently reduces accuracy on both GSM8K and MATH-500. Beyond the 300-problem pilot, the effect persists at larger scale. On full GSM8K (n=1319), monotone chains reach 93.2% accuracy versus 81.7% for non-monotone chains (+11.5 pp). On MATH-500 (n=500), monotone chains reach 63.7% versus 30.4% (+33.3 pp). Across both datasets, violation count is negatively correlated with correctness (Spearman ρ = −0.198 on GSM8K and ρ = −0.381 on MATH-500). We further show that token log-probability confidence worsens in calibration with step depth (ECE: 0.186 →0.312 from step 0 to step 7), and that entropy-trajectory monotonicity achieves +5.8 pp at 73.7% coverage, outperforming all scalar baselines including final-step entropy (+2.2 pp) and scalar coherence (−0.6 pp, worse than random) at ≈1,500 tokens/question— one-eighth the cost of 40- chain self-consistency. At matched answered-set coverage, SC@3/SC@5 can be slightly higher than monotonicity ranking, so our claim is not dominance over SC voting, but a cheap and interpretable single-chain triage signal with favorable accuracy-cost trade-offs. The initial pilot findings further replicate on a second model (Mistral-7B-Instruct-v0.3, n=300), where monotone chains reach 72.3% vs. 37.6% for non-monotone chains (+34.7 pp; OR= 4.33). Structural properties of uncertainty trajectories are thus more informative than aggregate magnitude measures across model families. Current evidence is strongest on numeric/discrete-answer tasks; extending to open-domain free...

## Method / Algorithms Extract

Accuracy Avg tokens / problem Coverage SC@3 66.0% 831.4 100% SC@5 (near-equal budget) 65.3% 1385.7 100% ESC-sim (min 2 chains) 66.3% 673.6 100% Our selective (monotonicity) 68.8% ≈1500 73.7% ESC stops at 2 chains for 234/300 problems (78.0%), at 3 chains for 46/300 (15.3%), and uses all 5 chains for 20/300 (6.7%), yielding an average stop point of 2.35 chains. This confirms the expected efficiency advantage of ESC in full-coverage operation. A.20 Coverage-Aware Self-Consistency Curves To match the selective-prediction target directly, we build coverage-aware SC rankings from the same sc_baseline/per_problem.json: for each problem, SC confidence is defined as vote agreement (majority- vote fraction) among the sampled full-chain answers, and problems are ranked by this score. We compute curves for both SC@3 and SC@5 and evaluate answered-set accuracy as coverage increases.

## Experiments / Evidence Extract

are computed by figures/compute_difficulty_control.py and summarized in figures/difficulty_control.json. The monotonicity coefficient remains positive and significant under this control (coef = 0.861, OR = 2.37, bootstrap p ≈0.003, 95% CI for OR [1.43, 3.98]), while chain length and question length are near-null in this specification. SC@3 agreement is also significant (OR = 1.70, p ≈0.003), suggesting both signals contribute complementary information. Table 8: Difficulty-proxy controlled logistic regression on GSM8K (n=300). Bootstrap-based uncertainty estimates are used for the fallback solver. Variable Coef. Odds ratio p-value monotone 0.861 2.367 0.003 chain len z 0.020 1.020 0.880 question len z 0.019 1.019 0.933 sc3 agreement z 0.528 1.696 0.003 As an additional stratified check, we group items by SC@3 agreement level (1/3, 2/3, and 1) and recompute the monotone/non-monotone gap within each stratum. The gap remains positive in all three groups (from +18.2 pp to +23.5 pp), with weighted average +21.5 pp. This supports the interpretation that monotonicity is not reducible to a single difficulty proxy. A.9 ε-Tolerance Ablation Table 9 reports the monotonicity rate, monotone accuracy, non-monotone accuracy, and the accuracy gap for ε ∈ {0.000, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200}. Table 9: Sensitivity of the monotonicity result to ε. Accuracy gap = monotone accuracy −non-monotone accuracy. Results are essentially identical across ε ∈[0, 0.10], confirming that the choice ε=0.01 is not critical. ε Mono. rate Mono. acc Non-mono. acc Gap 0.000 0.737 0.688 0.468 +21.9 pp 0.005 0.737 0.688 0.468 +21.9 pp 0.010 0.737 0.688 0.468 +21.9 pp 0.020 0.737 0.688 0.468 +21.9 pp 0.050 0.737 0.688 0.468 +21.9 pp 0.100 0.737 0.688 0.468 +21.9 pp 0.200 0.740 0.685 0.474 +21.0 pp The result is remarkably stable: the +21.9 pp gap is unchanged for all ε ≤0.10, and the gap shrinks to only +21.0 pp at ε=0.20. This stability arises because entropy jumps in non-monotone chains tend to be larger than 0.20 nats; the ε threshold only matters for very small fluctuations, which are rare in practice. A.10 Step Exclusion Statistics Steps with fewer than 2 parseable numerical answers are excluded from the entropy trajectory (see Section A.3). In our n=300 evaluation, zero steps were excluded: all 1,474 nominal steps across all chains produced at least 2 parseable answers. The step-exc...

## Conclusion / Discussion Extract

We studied whether the shape of answer-distribution entropy dynamics across reasoning steps predicts the correctness of chain-of-thought outputs. On GSM8K with Qwen2.5-7B-Instruct, we found a clear shape-over-magnitude dissociation: binary entropy-trajectory monotonicity is a significant predictor of correctness (OR= 2.50, Fisher’s p=0.0005, +21.9 pp accuracy gap), while the scalar total entropy drop is not (ρ=−0.06, p=0.31). Token log-probability confidence worsens in calibration from the first to the last reasoning step, and monotonicity-based selective prediction achieves +5.8 pp accuracy at 73.7% coverage. This directional signal also persists in larger runs: on full GSM8K (n=1319), monotone chains achieve 93.2% vs. 81.7% for non-monotone chains (+11.5 pp), and on MATH-500 (n=500), 63.7% vs. 30.4% (+33.3 pp). The same directional effect replicates on Mistral-7B-Instruct-v0.3 with an even larger gap (+34.7 pp; OR= 4.33), supporting cross-family robustness. Limitations. Despite larger-scale runs and second-model replication, this study remains limited in breadth. MATH- 500 (n=500) provides encouraging cross-dataset evidence within the math domain (monotone +33.3 pp; Section A.16), but broader task diversity is still needed. On GSM8K, replication on Mistral-7B-Instruct-v0.3 confirms the core signal (Section A.17), but broader model coverage (e.g., Phi-3, Gemma, Llama variants) is still needed. Temperature ablations (τ ∈{0.3, 0.5, 0.7, 1.0}) confirm that the +21.9 pp gap is robust across sampling temperatures (Section A.1). Threshold robustness and confounder control analyses (Sections A.7 and A.9) show that the +21.9 pp gap is unchanged for all ε ≤0.10 and that partial correlation controlling for chain length remains significant (r=0.179, p=0.0018). Problem difficulty and other confounders have not been controlled. The step-level ECE trend is a descriptive observation underpowered for formal inference at n=8 step bins. The monotonicity signal has a 31.2% false-positive rate, limiting its precision as a standalone correctness certificate; the graded violation count provides additional resolution (Section A.12). The calibration analysis uses final-answer correctn...

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

TBD: add short excerpts with page markers from `../texts/entropy-trajectory-shape-predicts-llm-reasoning.txt`.
