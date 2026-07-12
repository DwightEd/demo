# Truth as a Trajectory: What Internal Representations Reveal About Large Language Model Reasoning

- **Local PDF filename**: `Truth as a Trajectory.pdf`
- **Slug**: `truth-as-a-trajectory`
- **Pages**: 15
- **Approx Words**: 9684
- **Auto Tags**: geometry;dynamics;faithfulness
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.631762

## Keyword Profile

- `trajectory`: 68
- `probe`: 54
- `geometric`: 22
- `curvature`: 16
- `geometry`: 13
- `dimension`: 5
- `manifold`: 4
- `hidden state`: 4
- `hallucination`: 3
- `latent`: 3
- `causal`: 3
- `flow`: 3

## Abstract / Opening Summary

Existing explainability methods for Large Lan- guage Models (LLMs) typically treat hidden states as static points in activation space, as- suming that correct and incorrect inferences can be separated using representations from an individual layer. However, these activations are saturated with polysemantic features, lead- ing to linear probes learning surface-level lex- ical patterns rather than underlying reasoning structures. We introduce Truth as a Trajec- tory (TaT), which models the transformer in- ference as an unfolded trajectory of iterative refinements, shifting analysis from static acti- vations to layer-wise geometric displacement. By analyzing displacement of representations across layers, TaT uncovers geometric invari- ants that distinguish valid reasoning from spu- rious behavior. We evaluate TaT across dense and Mixture-of-Experts (MoE) architectures on benchmarks spanning commonsense reason- ing, question answering, and toxicity detection. Without access to the activations themselves and using only changes in activations across layers, we show that TaT effectively mitigates reliance on static lexical confounds, outper- forming conventional probing, and establishes trajectory analysis as a complementary perspec- tive on LLM explainability. 1

## Method / Algorithms Extract

Evaluation Dataset Avg ID Acc. OOD Avg. ARC-C ARC-E OpenQA BoolQ Hellaswag CosQA SiQA ComQA StoryCloze Zero-shot Accuracy 50.1 78.5 62.4 74.7 57.4 81.7 (26.1) 65.2 (48.0) 62.4 78.8 67.9 (59.8) - - Few-shot Accuracy 65.3 84.3 67.0 83.8 76.8 82.0 (44.5) 67.7 (60.8) 71.1 83.1 75.7 (70.7) - - ARC-C Linear Probe 75.32 80.09 78.60 55.48 70.35 73.34 65.47 74.59 66.01 71.03 75.32 70.49 TaT (Ours) 82.17 85.31 73.60 96.91 73.89 74.24 75.49 72.48 82.58 79.63 82.17 79.31 ARC-E Linear Probe 75.55 83.99 80.95 58.82 71.07 66.83 66.60 78.55 69.62 72.44 83.99 71.00 TaT (Ours) 73.81 89.10 77.20 78.56 79.19 75.08 60.85 71.09 94.98 77.76 89.10 76.34 OpenQA Linear Probe 66.42 69.44 83.15 56.79 70.26 61.53 65.47 73.01 57.94 67.11 83.15 65.11 TaT (Ours) 78.41 87.12 90.80 89.85 56.50 76.42 69.70 75.02 81.64 78.38 90.80 76.83 BoolQ Linear Probe 44.51 46.61 46.30 83.65 44.16 61.51 49.20 53.22 59.45 54.29 83.65 50.62 TaT (Ours) 53.50 62.75 54.20 85.05 33.08 51.56 50.20 47.50 71.41 56.58 85.05 53.02 Hellaswag Linear Probe 52.92 58.60 58.35 58.24 88.64 71.17 60.39 63.34 77.61 65.47 88.64 62.58 TaT (Ours) 65.96 74.92 65.80 64.22 92.46 66.40 55.27 62.82 95.75 71.51 92.46 68.89 ComQA Linear Probe 71.72 76.27 77.05 50.31 73.49 74.18 66.38 81.98 60.72 70.23 81.98 68.77 TaT (Ours) 60.92 80.05 73.80 68.72 59.51 64.89 60.18 77.56 88.51 70.46 77.56 69.57 CosQA Linear Probe 70.09 75.27 78.10 66.50 60.73 81.64 67.69 77.34 57.36 70.52 81.64 69.13 TaT (Ours) 69.71 83.59 74.60 77.77

## Experiments / Evidence Extract

We present a comprehensive evaluation of the Truth as a Trajectory (TaT) framework to assess its ability to distinguish valid reasoning from spurious cor- relations. Our experiments span a diverse set of domains, including commonsense reasoning, read- ing comprehension, factuality, and toxicity detec- tion, across both dense (Llama-3.1-8B, Qwen2.5- 14B/32B) and Mixture-of-Experts (Qwen2.5-30B MoE) architectures. We compare the performance of our trajectory-based classifier against standard static linear probing baselines and the underlying frozen language model’s intrinsic zero-shot and few-shot capabilities (which we refer to as the base model). For implementation details we refer to Appendix C. 5.1 Are Reasoning Trajectories Generalizable? We investigate whether the geometric signature of valid reasoning is consistent across different tasks. If TaT captures a fundamental structural invariant of "truth" (or validity) rather than task-specific con- founds, a trajectory classifier trained on one dataset should generalize to unseen datasets without fine- tuning on the unseen task. Setup We evaluate on a suite of reasoning bench- marks: ARC-Easy (ARC-E), ARC-Challenge (ARC-C) (Clark et al., 2018), BoolQ (Clark et al., 2019), Hellaswag (Zellers et al., 2019), Open- BookQA (OpenQA) (Mihaylov et al., 2018), Sto- ryCloze (Mostafazadeh et al., 2016), Common- senseQA (ComQA) (Talmor et al., 2019), Cos- mosQA (CosQA) (Huang et al., 2019), and So- cialIQA (SiQA) (Sap et al., 2019). For each dataset, we train a TaT classifier (using layer-wise displace- ment) and a linear probe (mid layer probe) on the training split and evaluate them on all other datasets’ evaluation splits. Because probe perfor- mance can be highly layer-dependent, we use the middle layer as a standard choice, and report a sweep over mid-to-late layers in Appendix D.

## Conclusion / Discussion Extract

We introduced Truth as a Trajectory (TaT), a frame- work that reframes LLM explainability from static 9 ===== PAGE 10 / 15 ===== Table 7: Computational overhead of the LSTM classifier compared to the base LLaMA 3.1-8B model. ∗Infer- ence overhead was computed in the simplest case of extracting all activations from all tokens across all lay- ers and then passing them through the LSTM model separately. However, in a realistic deployment scenario, the sequential classifier would be embedded within each layer of the model and would cause a negligible amount of inference overhead. Metric LLaMA 3.1-8B (fp16) LSTM Classifier Overhead Parameters 8.0B 4.76M 0.06% Inference Time (ms) 64.0 10.5 16%∗ Model Memory (MB) ∼15,000 18.1 0.12% layer-wise analysis to a dynamic geometric per- spective. By modeling the displacement of acti- vations across layers, TaT mitigates reliance on static lexical confounds and isolates the structural evolution of reasoning. Our results demonstrate that this trajectory-based approach yields trans- ferable classifiers that generalize across diverse reasoning benchmarks and architectures, signifi- cantly outperforming static linear probes and in- trinsic model baselines. Furthermore, in toxicity detection, TaT robustly distinguishes between toxic intent and benign vocabulary. Overall, these find- ings suggest that the geometry of inference offers a task-agnostic, invariant signature of inference validity, paving the way for more reliable and trans- ferable methods for monitoring and interpreting Large Language Models. 7 Future Directions Our current formulation positions TaT primarily as a validity detector, i.e., given a prompt and a candi- date continuation, it predicts whether the model’s internal inference trajectory is consistent with a correct choice. A natural next step is to transition TaT from detection to an interpretability tool. Iden- tifying where in the token×layer computation a candidate begins to diverge from a valid trajectory, and which mechanisms drive this divergence. One promising direction is to couple TaT with causal and circuit-level analysis. Because each dis- placement vector can be mapped back to a specific token positi...

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

TBD: add short excerpts with page markers from `../texts/truth-as-a-trajectory.txt`.
