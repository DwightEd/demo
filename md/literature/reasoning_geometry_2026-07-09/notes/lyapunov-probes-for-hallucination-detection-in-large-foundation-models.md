# Lyapunov Probes for Hallucination Detection in Large Foundation Models

- **Local PDF filename**: `Lyapunov Probes for Hallucination Detection in Large Foundation Models.pdf`
- **Slug**: `lyapunov-probes-for-hallucination-detection-in-large-foundation-models`
- **Pages**: 11
- **Approx Words**: 7636
- **Auto Tags**: dynamics;faithfulness;uncertainty;hallucination;emergence
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.619971

## Keyword Profile

- `probe`: 77
- `hallucination`: 55
- `hidden state`: 11
- `transition`: 6
- `dimension`: 4
- `entropy`: 2
- `geometric`: 1
- `trajectory`: 1
- `latent`: 1

## Abstract / Opening Summary

We address hallucination detection in Large Language Models (LLMs) and Multimodal Large Language Mod- els (MLLMs) by framing the problem through the lens of dynamical systems stability theory. Rather than treating hallucination as a straightforward classification task, we conceptualize (M)LLMs as dynamical systems, where fac- tual knowledge is represented by stable equilibrium points within the representation space. Our main insight is that hallucinations tend to arise at the boundaries of knowl- edge—transition regions separating stable and unstable zones. To capture this phenomenon, we propose Lyapunov Probes: lightweight networks trained with derivative-based stability constraints that enforce a monotonic decay in con- fidence under input perturbations. By performing system- atic perturbation analysis and applying a two-stage train- ing process, these probes reliably distinguish between sta- ble factual regions and unstable, hallucination-prone re- gions. Experiments on diverse datasets and models demon- strate consistent improvements over existing baselines.

## Method / Algorithms Extract

TriviaQA PopQA CoQA MMLU Llama-2-7B Verbalized [43] 58.37 20.13 51.52 28.99 Surrogate [50] 57.14 18.42 53.29 30.77 Seq. Prob. [25] 63.76 20.22 49.63 29.41 Probe [37] 75.99 61.87 72.25 33.60 Ours 83.09 63.37 76.13 33.79 Llama-3-8B Verbalized [43] 64.72 21.23 52.40 54.09 Surrogate [50] 66.02 18.26 53.50 52.43 Seq. Prob. [25] 70.72 27.02 50.35 57.48 Probe [37] 78.82 60.77 80.67 79.26 Ours 86.46 67.08 81.28 80.00 Qwen-3-4B Verbalized [43] 43.54 12.04 62.56 67.61 Surrogate [50] 39.70 10.08 59.54 65.43 Seq. Prob. [25] 49.82 13.79 61.13 71.70 Probe [37] 74.47 64.41 88.30 87.35 Ours 79.47 64.02 89.01 87.48 Falcon-7B Verbalized [43] 35.18 13.76 35.54 23.90 Surrogate [50] 37.23 16.53 34.83 23.78 Seq. Prob. [25] 40.00 15.13 37.13 24.98 Probe [37] 63.27 60.48 65.36 24.79 Ours 65.52 61.23 66.03 25.11 • Semantic Perturbations: These include controlled vari- ations such as substitution of words from the same gram- matical class, insertion of random tokens, and adjustment of sentence structure. They ensure the probe learns to distinguish between cases where core factual content re- mains stable amid linguistic shifts and those where such variations alter the underlying truth. • Representational Perturbations: These involve direct modifications to the hidden states by injecting Gaussian noise. Such perturbations simulate small, random fluc- tuations in the model’s internal representations, which are designed to systematically push the representation to- wards and across knowledge boundaries. For each input, we construct a sequence of perturbations δ1, . . . , δK with controlled, incremental magnitudes, where the intensity of both semantic and representational pertur- bations gradually increases. We calculate δ as the cosine similarity between the unperturbed representation h and the perturbed representation hδ: δ = 1 −cos(h, hδ). This bal- ances the need to make stability transitions observable while preserving the underlying semantics of the input. Training proceeds in two stages. In the first stage, the probe is trained using binary cross-entropy loss to distin- guish factual from non-factual outputs. In the second stage, the Lyapunov constraint loss is gradually introduced with increasing weight λ, enforcing monotonic confidence de- cay as perturbations intensify. This approach ensures stable optimization while establishing desired stability properties.

## Experiments / Evidence Extract

Experiments on diverse datasets and models demon- strate consistent improvements over existing baselines. 1. Introduction Large Language Models (LLMs) [1, 3, 5, 44] and Multi- modal Large Language Models (MLLMs) [7, 12, 47, 52] have demonstrated remarkable capabilities across diverse tasks [54], yet their tendency to generate factually incorrect content—commonly referred to as hallucinations—poses critical challenges for deployment in high-stakes domains such as healthcare, legal reasoning, and financial analy- sis [8, 27, 34, 35, 37]. These hallucinations manifest as *Corresponding author (zhaoxinf@buaa.edu.cn). Representation Space 0 0 1 1 1 2 1 2 (a) Known Region (b) UnKnown Region (c) Hallucination Region Stable Domain 0 Unstable Domain Knowledge Boundary Perturbation Instability Figure 1. Illustration of representation space partition in large models. We define that the data (representation) space can be divided into three regions: (a) stable known regions, (b) stable unknown regions, and (c) unstable knowledge boundary regions. Hallucinations primarily emerge in the unstable boundary regions. plausible-sounding but factually unsupported statements, undermining trust and limiting practical applications. Current hallucination detection approaches fall into two main paradigms: external verification methods that com- pare outputs against knowledge bases, and internal feature- based methods that train classifiers on model represen- tations or token probabilities [16, 46]. However, these approaches suffer from fundamental limitations. Exter- nal methods require comprehensive, continuously updated fact repositories that are expensive and limited in cover- age [21, 30, 39, 48, 53]. Internal methods lack theoret- ical grounding and fail to capture the underlying mecha- 1 arXiv:2603.06081v1 [cs.CV] 6 Mar 2026 ===== PAGE 2 / 11 ===== nisms that give rise to hallucinations [4, 9]. Most criti- cally, existing methods treat hallucination detection as stan- dard binary classification without addressing the fundamen- tal question of why and where hallucinations occur in the model’s knowledge space. We propose that the key to understanding and detect- ing hallucinations lies in recognizing the dynamical nature of Large Language Models and their knowledge bound- aries. Our central hypothesis is that, as shown in Figure 1, hallucinations are not randomly distributed erro...

## Conclusion / Discussion Extract

Conclusion This paper proposes a simple yet effective approach to hal- lucination detection in (Multimodal) Large Language Mod- els by rethinking the problem through the perspective of dy- namical systems stability. Instead of viewing hallucination as a simple classification task, we conceptualize (M)LLMs as systems where factual knowledge resides at stable equi- librium points within the representation space, and halluci- nations emerge in transitional regions near instability. To address this, we introduce Lyapunov Probes—lightweight models trained with stability-driven constraints that enforce monotonic confidence decay under controlled input pertur- bations. Using a systematic perturbation framework and a two-stage training process, these probes effectively dif- ferentiate between stable, reliable knowledge and unstable, hallucination-prone regions. Extensive experiments across diverse datasets and architectures demonstrate the robust- ness and effectiveness of our method. Acknowledgements This work was supported by the National Natural Science Foundation of China under Grant No. 62441617. It was also supported by the Postdoctoral Fellowship Program and China Postdoctoral Science Foundation under Grant No. 2024M764093 and Grant No. BX20250485, the Beijing Natural Science Foundation under Grant No. 4254100, and by Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing. It was also supported by the Young Elite Scientists Sponsorship Program of the Beijing High Innovation Plan (NO. 20250860). References [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ah- mad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1 [2] Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Al- shamsi, Alessandro Cappelli, Ruxandra Cojocaru, M´erouane Debbah, ´Etienne Goffinet, Daniel Hesslow, Julien Launay, Quentin Malartic, et al. The falcon series of open language models. arXiv preprint arXiv:2311.16867, 2023. 5 [3] Anthropic. The Claude 3 model family: Opus, Sonnet, Haiku. 2024. 1 [4] Amos Azaria and Tom Mitchell. The internal state of an ...

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

TBD: add short excerpts with page markers from `../texts/lyapunov-probes-for-hallucination-detection-in-large-foundation-models.txt`.
