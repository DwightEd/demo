# GeoFaith: A Spatio-Temporal Dual View of Faithful Chain-of-Thought

- **Local PDF filename**: `GeoFaith-A Spatio-Temporal Dual View of Faithful Chain-of-Thought.pdf`
- **Slug**: `geofaith-a-spatio-temporal-dual-view-of-faithful-chain-of-thought`
- **Pages**: 30
- **Approx Words**: 15500
- **Auto Tags**: geometry;dynamics;faithfulness;uncertainty;step-level
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.611864

## Keyword Profile

- `faithful`: 256
- `entropy`: 76
- `geometric`: 47
- `latent`: 39
- `manifold`: 36
- `trajectory`: 33
- `dimension`: 31
- `chain of thought`: 24
- `geometry`: 19
- `hidden state`: 18
- `hallucination`: 6
- `riemann`: 4

## Abstract / Opening Summary

Chain-of-Thought (CoT) reasoning has ad- vanced large language models (LLMs), but outcome-based supervision leads to pervasive post-hoc rationalization, producing plausible yet unfaithful reasoning chains. Most prior faithfulness assessment methods are either un- scalable, expensive, or unreliable. We propose GeoFaith, a spatio-temporal framework that leverages latent geometric structure and en- tropy dynamics to diagnose and enforce faithful reasoning. We develop a scalable bootstrap- ping pipeline expanding step-level annotations from 1k to 20k samples across four domains, train an 8B faithfulness detector outperforming GPT-5 on standard benchmarks, and design a faithfulness-aware reinforcement learning framework jointly optimizing outcome correct- ness, process faithfulness, and trajectory consis- tency. Experiments show the proposed method achieves superior performance on both faith- fulness detection and downstream reasoning, producing shorter, more interpretable chains without sacrificing accuracy. Our code will be made available publicly. 1

## Method / Algorithms Extract

To assess and enforce the faithfulness of CoT rea- soning, we propose GeoFaith, a two-stage frame- work for faithfulness supervision and optimization. Since the proposed geometric and entropy-based signals are costly and cumbersome to apply directly during inference, we use them as weak supervision signals to bootstrap a scalable step-level detector through inter-group geometric mining, intra-group entropy-based refinement, and iterative bootstrap- ping. GeoFaith further incorporates this detector into a faithfulness-aware reinforcement learning objective that jointly optimizes outcome correct- ness, process faithfulness, entropy regularity, and manifold consistency. 3.1 Scalable Detector Construction Faithful reasoning supervision is difficult to scale because the model’s internal reasoning process is only partially observable through generated trajec- tories. We therefore construct a scalable detector framework consisting of (1) inter-group geomet- ric mining, (2) intra-group step refinement, and (3) iterative bootstrapping for large-scale annotation expansion (Figure 8). Inter-group geometric mining. For each query x, we sample N reasoning rollouts {τi}N i=1 from the policy model under stochastic decoding. Each tra- jectory is mapped to the latent manifold to compute its geometric features. To characterize geometric abnormality, we define the information-geometric contrast C(τ) = ¯dFR(τ)/ exp( ¯U(τ)). (7) Here, ¯dFR(τ) and ¯U(τ) denote the average step- wise Fisher–Rao distance and uncertainty along trajectory τ, respectively. Since U(zt) < 0 in the latent space, the exponential term exp( ¯U(τ)) acts as a nonlinear contrast modulator: low-uncertainty trajectories enhance the contrast, whereas collapsed uncertainty suppresses it. We then cluster trajec- tories in the joint feature space of distortion ratio ρ(τ) and information-geometric contrast C(τ) us- ing density-aware clustering. Intra-group step refinement. Within suspicious trajectory groups, we perform step-level refinement using entropy dynamics and a lightweight detector. For each reasoning step ct, we combine detector predictions and entropy patterns into a unified con- 5 ===== PAGE 6 / 30 ===== fidence score: st = αsdet(ct) + (1 −α)stemp(ct), (8) where sdet(ct) denotes the confidence score pro- duced by the faithfulness detector, and stemp(ct) measures the local stability of entropy dynamics by penalizing abnormal patterns. Since entropy sig- nals provide only an indirect proxy for faithfulness, we set α to 0.7. Seed-set validation of entropy– detector fusion is provided in Appendix D.2. We ...

## Experiments / Evidence Extract

4.1 Setups Datasets and evaluation metrics. We eval- uate on two suites. (i) Faithfulness detec- tion: our own benchmark spanning mathemati- cal, logical, factual, and agentic reasoning (Ta- ble 1), plus existing testbeds RAGTruth (Niu et al., 2024), FCGPT (Wang et al., 2024b), Pro- cessBench (Zheng et al., 2025) and FaithCoT- Bench (Shen et al., 2025). (ii) Reasoning genera- tion: AMC23, LogiQA (Liu et al., 2020), 2Wiki- MultihopQA (Ho et al., 2020), and GPQA-D (Rein et al., 2023). We strictly separate detector training data from downstream RL evaluation benchmarks. For detection, we report per-domain F1 for faithful (FF1) and unfaithful (UF1) classes. For genera- tions, average faithfulness scores are evaluated by our detector as well as GPT-5 and DeepSeek-V3.2. Baselines. For faithfulness detection, we com- pare against popular closed-source models and open-source models, as well as specialized de- tectors: HHEM2.1, FaithLens (Si et al., 2025), and LogicReward (Xu et al., 2025). For reason- ing generation, we compare against standard RL training (GRPO) (Shao et al., 2024), knowledge- grounded methods (KnowRL (Ren et al., 2025) and TruthRL (Wei et al., 2025)), and the trajectory hal- lucination suppression baseline THS (Gui et al., 2026). 4.2

## Conclusion / Discussion Extract

In this work, we study chain-of-thought faith- fulness of large language models from a spatio- temporal perspective, showing that faithful and un- faithful reasoning exhibit distinct manifold geome- try and entropy dynamics. Building on these obser- vations, we develop a scalable detector framework and a faithfulness-aware reinforcement learning. 8 ===== PAGE 9 / 30 ===== Our results demonstrate that internal representa- tion geometry and temporal uncertainty provide effective signals for understanding and improving reasoning faithfulness. Limitations While our work yields encouraging results, sev- eral aspects warrant further investigation and refine- ment. First, due to computational constraints, our experimental comparisons with state-of-the-art rea- soning optimization methods have so far focused on models of moderate scale. Extending these eval- uations to frontier-scale LLMs would be a natural and valuable next step, which we hope to explore in future work. Second, we openly acknowledge a gap between our supervision signal and the underlying notion of faithfulness: our detector is trained on human-observable step labels, while CoT faithful- ness concerns whether the generated trace reflects the model’s internal computation. Step-level coun- terfactual verification is a more direct alternative, but replacing reasoning steps and measuring their causal effect is computationally prohibitive, scaling as O(T 2). We therefore adopt a scalable bootstrap- ping pipeline that uses spatio-temporal geometric and entropy signals to filter data before training the detector. This design improves scalability, but it re- mains an approximate substitute for direct internal verification and may accumulate annotation noise over iterative self-training. References Alessio Ansuini, Alessandro Laio, Jakob H Macke, and Davide Zoccolan. 2019. Intrinsic dimension of data representations in deep neural networks. In NeurIPS. Iván Arcuschin, Jett Janiak, Robert Krzyzanowski, Senthooran Rajamanoharan, Neel Nanda, and Arthur Conmy. 2025. Chain-of-thought reasoning in the wild is not always faithful. arXiv preprint arXiv:2503.08679. Georgios Arvanitidis, Lars Kai Hansen, and Søren Hauberg...

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

TBD: add short excerpts with page markers from `../texts/geofaith-a-spatio-temporal-dual-view-of-faithful-chain-of-thought.txt`.
