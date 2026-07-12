# LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals

- **Local PDF filename**: `LLMReasoning as Trajectories.pdf`
- **Slug**: `llmreasoning-as-trajectories`
- **Pages**: 16
- **Approx Words**: 9184
- **Auto Tags**: geometry;dynamics;faithfulness;step-level
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.619317

## Keyword Profile

- `trajectory`: 70
- `probe`: 36
- `geometry`: 25
- `geometric`: 12
- `transition`: 11
- `chain of thought`: 9
- `faithful`: 5
- `causal`: 5
- `dimension`: 5
- `hidden state`: 4
- `entropy`: 3
- `manifold`: 1

## Abstract / Opening Summary

This work characterizes large language models’ chain-of-thought generation as a structured tra- jectory through representation space. We show that mathematical reasoning traverses function- ally ordered, step-specific subspaces that be- come increasingly separable with layer depth. This structure already exists in base models, while reasoning training primarily accelerates convergence toward termination-related sub- spaces rather than introducing new represen- tational organization. While early reasoning steps follow similar trajectories, correct and incorrect solutions diverge systematically at late stages. This late-stage divergence enables mid-reasoning prediction of final-answer cor- rectness with ROC–AUC up to 0.87. Further- more, we introduce trajectory-based steering, an inference-time intervention framework that enables reasoning correction and length control based on derived ideal trajectories. Together, these results establish reasoning trajectories as a geometric lens for interpreting, predicting, and controlling LLM reasoning behavior.1 1

## Method / Algorithms Extract

Best-layer AUC Step-count only 0.649 ± 0.021 LogitLens (best config) 0.765 ± 0.027 Trajectory features (ours) 0.852 ± 0.039 Table 3: Correctness predictor comparison. Trajec- tory features substantially outperform both a step-count- only baseline and logit-level features (entropy, answer- marker token rank, top-1 probability) at step boundaries. Values report best-layer AUC averaged over three seeds. tion 4.1, these findings provide evidence that trajec- tory divergence provides a concrete mid-reasoning signal for final-answer correctness. Trajectory features outperform logit-level and length-based baselines. To rule out surface-level confounds, we compare against step-count-only and logit-lens baselines (Table 3). A classifier us- ing only the number of reasoning steps as a feature achieves AUC 0.649±0.021, indicating that length carries some signal but falls well below trajectory- based prediction. Logit-lens features at step bound- aries (entropy, answer-marker token rank, and top-1 probability) achieve a best AUC of 0.765 ± 0.027. Trajectory features (0.852 ± 0.039) substantially outperform both. Under length-balanced resam- pling, trajectory features still achieve AUC 0.847± 0.006, only ∼0.02 below the original, confirming that the signal is not driven by length. 4.3 Toward Error-targeted Inference-time Interventions We next explore how this signal can be used to enhance inference-time intervention methods, with the goal of intervening only when an impending failure is detected. This aims to mitigate the over- thinking drawback of unconditional interventions, which can unintentionally degrade originally cor- rect reasoning (Ghosal et al., 2025; Wang et al., 2025; Zhao et al., 2025). Here, we evaluate two common classes of inference-time interventions: test-time scaling and activation steering. Premise. Test-time scaling methods modify gen- eration by inserting tokens directly into the model’s ongoing output stream. When the model is about to generate the final answer marker, we instead inject additional tokens that encourage further checking.4 These injected cues encourage the model to extend 4Specifically, we tested: Wait – “Wait, let me double check.”, Hmm – “Hmm, let me think about this more carefully.”, and Check – “Let me double-check this step by step.” ===== PAGE 7 / 16 ===== Intervention Always Gated Gain vs Always Step −1.59 +0.91 +2.50 Check −11.70 +0.23 +11.90 Wait −30.50 −0.68 +29.80 Hmm −36.00 −0.61 +35.41 Prolong (Last) +0.45 +0.76 +0.31 Prolong (Mid) +0.38 +0.76 +0.38 Table 4: Unconditional vs. error-targeted interven- tions on GSM8K ...

## Experiments / Evidence Extract

evaluation pair. Unlike Table 1, Step 1 is included, showing near-perfect transferability across models. Category Step 1 Step 2 Step 3 Step 4 Step 5 Answer Step X: 0.96 (0.98 @ L1) 0.83 (0.85 @ L19) 0.81 (0.84 @ L28) 0.72 (0.83 @ L10) 0.71 (0.88 @ L30) 0.86 (0.93 @ L20) Non-Step X: 0.85 (0.86 @ L1) 0.82 (0.85 @ L18) 0.79 (0.84 @ L7) 0.77 (0.85 @ L10) 0.75 (0.88 @ L25) 0.86 (0.92 @ L10) Numbered list 0.82 (0.91 @ L1) 0.83 (0.89 @ L18) 0.81 (0.83 @ L30) 0.75 (0.85 @ L10) 0.73 (0.89 @ L24) 0.86 (0.95 @ L10) \n\n paragraphs 0.82 (0.82 @ L1) 0.75 (0.81 @ L14) 0.79 (0.83 @ L30) 0.82 (0.86 @ L10) 0.81 (0.91 @ L22) 0.81 (0.96 @ L23) \n lines 0.87 (0.87 @ L0) 0.82 (0.87 @ L18) 0.77 (0.86 @ L0) 0.75 (0.86 @ L10) 0.79 (0.90 @ L30) 0.89 (0.94 @ L10) Single block 0.87 (0.88 @ L1) 0.87 (0.87 @ L18) 0.81 (0.87 @ L30) 0.74 (0.85 @ L30) 0.67 (0.85 @ L31) 0.87 (0.94 @ L6) Table 7: Per-category freeform probe transfer accuracy. Each cell reports average accuracy across 32 layers, with the best single-layer accuracy and corresponding layer in parentheses. Probes trained on fixed-form Step X: activations transfer to all freeform format categories, including those with no Step markers. Correctness predictors. We implement single- layer logistic regression (nn.Linear(d, 1)) in PyTorch with binary cross-entropy loss and ℓ2 regu- larization via the Adam optimizer’s weight_decay parameter (set to 1/C). Training uses learning rate 0.01, batch size 32, a maximum of 1,000 epochs, and early stopping with patience 50 on validation loss. The regularization strength C is selected via 5- fold stratified cross-validation (StratifiedKFold) over C ∈ {0.001, 0.01, 0.1, 1.0, 10.0, 100.0}. Data is split 90/10 (stratified) for final training and evaluation. For feature sets requiring dimensional- ity reduction, PCA with ncomponents = 128 is fit on the training split only. Activation steering (Prolong/Shorten). Steer- ing directions are computed per-layer as the mean difference between termination-preceding and step- preceding activations on the training split (Sec- tion 4.3). During decoding, steering is applied additively at the token position immediately pre- ceding the final answer marker. LAST intervenes on the final 5 layers (layers 27–31); MID intervenes on 5 layers centered at layer 15 (layers 13–17). The coefficient |α| controls the steering magnitude. Trajectory-based steering. The ideal t...

## Conclusion / Discussion Extract

6.1 Conclusion This work advances a geometric perspective on LLM reasoning by demonstrating that multi-step reasoning unfolds along structured trajectories in representation space. Intermediate reasoning states occupy step-specific regions that become increas- ingly linearly separable at deeper layers, and this organization is already present in Base models; rea- soning distillation primarily reshapes the depth at which convergence occurs rather than introducing new representational structure. Building on this, we show that late-step geom- etry provides an actionable signal for predicting final-answer correctness prior to answer emission, enabling more selective, error-targeted inference- time interventions that mitigate the overthink- ing associated with unconditional test-time scal- ing. Beyond correctness, reasoning trajectories can be causally manipulated to control reasoning length by steering activations toward or away from termination-related regions. These results advance the view of reasoning trajectories as a unifying abstraction for interpreting, predicting, and influ- encing LLM reasoning behavior. 6.2 Future Work

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

TBD: add short excerpts with page markers from `../texts/llmreasoning-as-trajectories.txt`.
