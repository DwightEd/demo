# Hidden Error Awareness in Chain-of-Thought Reasoning: The Signal Is Diagnostic, Not Causal

- **Local PDF filename**: `Hidden Error Awareness in Chain-of-Thought Reasoning-The Signal Is Diagnostic, Not Causal.pdf`
- **Slug**: `hidden-error-awareness-in-chain-of-thought-reasoning-the-signal-is-diagnostic-not-causal`
- **Pages**: 9
- **Approx Words**: 4926
- **Auto Tags**: dynamics;faithfulness;uncertainty
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.614094

## Keyword Profile

- `probe`: 68
- `hidden state`: 20
- `causal`: 18
- `patching`: 11
- `faithful`: 7
- `chain of thought`: 5
- `trajectory`: 1
- `latent`: 1

## Abstract / Opening Summary

Chain-of-thought (CoT) prompting assumes that generated reasoning reflects a model’s internal computation. We show this assumption is wrong in a specific, measurable way: models inter- nally detect their own reasoning errors but out- wardly express confidence in them. A linear probe on hidden states predicts trace correctness with 0.95 AUROC—from the very first reasoning step (0.79)—while verbalized confidence for wrong traces is 4.55/5, nearly identical to correct ones (4.87/5). A text-surface classifier achieves only 0.59 on the same data, confirming a 0.20-point gap invisible in the generated text. This hidden error awareness holds across three model fami- lies (Qwen, Llama, Phi), 1.5B–72B parameters, and RL-trained reasoning models (DeepSeek-R1, 0.852 AUROC). The natural question is whether this signal can fix the errors it detects. It cannot. Four interventions—activation steering, probe- guided best-of-N, self-correction, and activation patching—all fail; patching destroys output co- herence entirely. The signal is diagnostic, not causal: a readout of computation quality, not a lever to redirect it. This delineates a boundary for mechanistic interpretability: error representations during reasoning are fundamentally different from the factual knowledge representations that prior work has successfully edited.

## Method / Algorithms Extract

AUROC Cost Hidden State Probe (ours) 0.953 1 fwd pass Self-Consistency (N=5) 0.823 5× gen CCS (Burns et al., 2023) 0.718 1 fwd pass P(True) 0.721 1 query Verbalized Confidence 0.674 1 query Sequence Log-prob 0.676 free Table 2. Error detection baselines (Qwen2.5-3B, n=200). Model Mixed Correct Wrong d Qwen2.5-3B 18 0.278 0.481 0.55 Qwen2.5-7B 13 0.154 0.391 0.63 Qwen2.5-14B 18 0.454 0.502 0.13 Llama-3.1-8B 24 0.363 0.561 0.69 Table 3. Within-problem difficulty control. Mixed = problems with both correct and wrong traces; d = Cohen’s d. Difficulty control. A key concern is confounding with problem difficulty. For 50 problems, we generate 5 traces each (temperature = 0.7) and compare probe scores for correct vs. wrong traces on the same problem (Table 3). Wrong traces score significantly higher (p < 0.05, d = 0.55–0.83), confirming trace-level detection. Layer analysis. Figure 2 shows the error signal concen- trates in the upper layers (70–85% depth), suggesting en- coding in abstract, higher-level representations rather than early syntactic processing. 4. The Signal Is Early and Hidden 4.1. First-step prediction The first-step hidden state alone achieves 0.787 AUROC— 98% of full-trace performance (Table 4). For DS-R1-7B, first-step AUROC is 0.686, still well above chance. The model “knows” from Step 1 whether its reasoning will suc- ceed, yet proceeds to generate multiple incorrect steps. Two temporal regimes. Step-level probe scores (Figure 3) reveal distinct dynamics. Qwen2.5-3B shows front-loaded awareness: maximum separation at Step 1 (gap = 0.41), sug- gesting early commitment to a flawed trajectory. Qwen2.5- 7B shows accumulating awareness: separation grows from 0.11 to 0.38 as error evidence builds. Neither pattern is con- sistent with the probe merely reading post-hoc text quality. 4.2. Textual indistinguishability If the signal is in hidden states, is it also in the text? We formalize this with a textual indistinguishability test: train a TF-IDF + logistic regression classifier on first-step text and compare with the hidden-state probe. Let stext and shidden denote the AUROC of the text and 3 ===== PAGE 4 / 9 ===== Title Suppressed Due to Excessive Size 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 Relative Layer Position 0.5 0.6 0.7 0.8 0.9 1.0 CV AUROC Probe Error Detection Across Models and Layers Qwen2.5-3B Qwen2.5-1.5B Phi-3.5-mini Figure 2. Probe AUROC across layers. The error signal peaks in the upper third of the network across all model families. Hidden State Used AUROC 95% CI First step only 0.787 [0.634, 0.840] Last step only 0.754 – Max score acros...

## Experiments / Evidence Extract

experiments extend causal mediation analysis (Meng et al., 2022; Vig et al., 2020) from factual recall to multi-step reasoning, revealing that the causal structure of error representations differs qualitatively from that of factual associations. 3. The Error Signal Exists 3.1. Method Given a model M with L layers, we extract the hidden state h(l) T ∈Rd at layer l and the last token position T of a complete CoT trace. We train a logistic regression probe: p(error | h) = σ(w⊤h + b) (1) where h is standardized (zero mean, unit variance per dimen- sion) and the label y = 1 if the trace reaches an incorrect answer. The training objective is regularized log-likelihood: L = −1 N N X i=1 [yi log ˆpi + (1−yi) log(1−ˆpi)] + 1 2C ∥w∥2 (2) with C = 0.1. Training uses 100 MATH-500 problems (Hendrycks et al., 2021), disjoint from all evaluation sets. We select the best layer by 5-fold cross-validated AUROC; no other hyperparameters are tuned. Algorithm 1 summa- rizes the full pipeline. 2 ===== PAGE 3 / 9 ===== Title Suppressed Due to Excessive Size Algorithm 1 Hidden Error Awareness Probe Require: Model M, training set Dtrain, candidate layers L Ensure: Best probe (f ∗, l∗) 1: for each problem xi ∈Dtrain do 2: ri ←GenerateCoT(M, xi) // greedy 3: yi ←1[answer(ri) ̸= ref(xi)] 4: for each layer l ∈L do 5: h(l) i ←ExtractHidden(M, xi ⊕ri, l) 6: end for 7: end for 8: for each layer l ∈L do 9: X(l) ←Standardize([h(l) 1 , . . . , h(l) N ]) 10: al ←CV-AUROC(LogReg(X(l), y), k=5) 11: end for 12: l∗←arg maxl al 13: f ∗←LogReg(X(l∗), y) // fit on all data 14: return (f ∗, l∗) Model Type Acc Lyr CV Eval Qwen2.5-1.5B std .35 27 .918 .724 Qwen2.5-3B std .53 27 .953 .956 Phi-3.5-mini std .39 8 .936 – Qwen2.5-7B std .62 16 .669 .737 Llama-3.1-8B std .46 16 .703 .811 Qwen2.5-14B std .65 39 .762 – Qwen2.5-32B std .53 32 .956 – Qwen2.5-72B std .41 64 .977 – DS-R1-7B reas. .76 12 .884 .852 Table 1. Probe error detection. CV: 5-fold cross-validated AU- ROC on 100 training problems; Eval: held-out AUROC on 200 separate problems. DS-R1-7B = DeepSeek-R1-Distill-Qwen-7B (RL-distilled reasoning model). 3.2. Results across models and scales Table 1 shows that the probe achieves >0.9 AUROC for five of nine models—including at 72B scale (0.977). The signal is weaker at intermediate scales (7B: 0.669, 14B: 0.762) but recovers at 32B and 72B, suggesting the dip reflects training dynamics rather than a ...

## Conclusion / Discussion Extract

conclusion: where they show the signal aids selection, we show it cannot improve the reasoning itself. We further establish a 0.20 AUROC gap between hidden-state and text- surface classifiers that verification-focused work does not measure. Self-correction and causal analysis. Huang et al. (2024) find that intrinsic self-correction remains limited. Our activa- tion patching experiments extend causal mediation analysis (Meng et al., 2022; Vig et al., 2020) from factual recall to multi-step reasoning, revealing that the causal structure of error representations differs qualitatively from that of factual associations. 3. The Error Signal Exists 3.1. Method Given a model M with L layers, we extract the hidden state h(l) T ∈Rd at layer l and the last token position T of a complete CoT trace. We train a logistic regression probe: p(error | h) = σ(w⊤h + b) (1) where h is standardized (zero mean, unit variance per dimen- sion) and the label y = 1 if the trace reaches an incorrect answer. The training objective is regularized log-likelihood: L = −1 N N X i=1 [yi log ˆpi + (1−yi) log(1−ˆpi)] + 1 2C ∥w∥2 (2) with C = 0.1. Training uses 100 MATH-500 problems (Hendrycks et al., 2021), disjoint from all evaluation sets. We select the best layer by 5-fold cross-validated AUROC; no other hyperparameters are tuned. Algorithm 1 summa- rizes the full pipeline. 2 ===== PAGE 3 / 9 ===== Title Suppressed Due to Excessive Size Algorithm 1 Hidden Error Awareness Probe Require: Model M, training set Dtrain, candidate layers L Ensure: Best probe (f ∗, l∗) 1: for each problem xi ∈Dtrain do 2: ri ←GenerateCoT(M, xi) // greedy 3: yi ←1[answer(ri) ̸= ref(xi)] 4: for each layer l ∈L do 5: h(l) i ←ExtractHidden(M, xi ⊕ri, l) 6: end for 7: end for 8: for each layer l ∈L do 9: X(l) ←Standardize([h(l) 1 , . . . , h(l) N ]) 10: al ←CV-AUROC(LogReg(X(l), y), k=5) 11: end for 12: l∗←arg maxl al 13: f ∗←LogReg(X(l∗), y) // fit on all data 14: return (f ∗, l∗) Model Type Acc Lyr CV Eval Qwen2.5-1.5B std .35 27 .918 .724 Qwen2.5-3B std .53 27 .953 .956 Phi-3.5-mini std .39 8 .936 – Qwen2.5-7B std .62 16 .669 .737 Llama-3.1-8B std .46 16 .703 .811 Qwen2.5-14B std .65 39 .762 – Qwen2.5-32B std .53 32 .95...

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

TBD: add short excerpts with page markers from `../texts/hidden-error-awareness-in-chain-of-thought-reasoning-the-signal-is-diagnostic-not-causal.txt`.
