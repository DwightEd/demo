# ===== PAGE 1 / 47 ===== Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 25580–25626

- **Local PDF filename**: `Two Pathways to Truthfulness-On the Intrinsic.pdf`
- **Slug**: `two-pathways-to-truthfulness-on-the-intrinsic`
- **Pages**: 47
- **Approx Words**: 15469
- **Auto Tags**: dynamics;uncertainty;hallucination;emergence
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.632362

## Keyword Profile

- `hallucination`: 36
- `probe`: 29
- `flow`: 21
- `patching`: 14
- `latent`: 2
- `entropy`: 1
- `hidden state`: 1

## Abstract / Opening Summary

Despite their impressive capabilities, large lan- guage models (LLMs) frequently generate hal- lucinations. Previous work shows that their in- ternal states encode rich signals of truthfulness, yet the origins and mechanisms of these signals remain unclear. In this paper, we demonstrate that truthfulness cues arise from two distinct in- formation pathways: (1) a Question-Anchored pathway that depends on question-answer in- formation flow, and (2) an Answer-Anchored pathway that derives self-contained evidence from the generated answer itself. First, we val- idate and disentangle these pathways through attention knockout and token patching. After- wards, we uncover notable and intriguing prop- erties of these two mechanisms. Further ex- periments reveal that (1) the two mechanisms are closely associated with LLM knowledge boundaries; and (2) internal representations are aware of their distinctions. Finally, building on these insightful findings, two applications are proposed to enhance hallucination detection performance. Overall, our work provides new insight into how LLMs internally encode truth- fulness, offering directions for more reliable and self-aware generative systems.1 1

## Method / Algorithms Extract

Llama-3-8B Mistral-7B-v0.3 PopQA TriviaQA HotpotQA NQ PopQA TriviaQA HotpotQA NQ P(True) 55.85 49.92 52.14 53.27 45.49 47.61 57.87 52.79 Logits-mean 74.52 60.39 51.94 52.63 69.52 66.76 55.45 57.88 Logits-min 85.36 70.89 61.28 56.50 87.05 77.33 68.08 54.40 Probing Baseline 88.71 77.58 82.23 70.20 87.39 81.74 83.19 73.60 MoP-RandomGate 75.52 69.17 79.88 66.56 79.81 70.88 72.23 61.19 MoP-VanillaExperts 89.11 78.73 84.57 71.21 88.53 80.93 82.93 73.77 MoP 92.11 81.18 85.45 74.64 91.66 83.57 85.82 76.87 PR 94.01 83.13 87.81 79.10 93.09 84.36 89.03 79.09 Table 3: Comparison of hallucination detection performance (AUC). Full results in Appendix H. internal-based baselines, including (1) P(True) (Ka- davath et al., 2022), (2) uncertainty-based metrics (Aichberger et al., 2024; Xue et al., 2025a), and (3) probing classifiers (Chen et al., 2024; Orgad et al., 2025). Results are averaged over three ran- dom seeds. Additional implementation details are provided in Appendix B.5 and B.6.

## Experiments / Evidence Extract

We demonstrate Kernel Density Esti- mation results of the saliency scores on Trivi- aQA (Joshi et al., 2017) and Natural Questions (Kwiatkowski et al., 2019) datasets. As shown in Figure 1, probability densities reveal a clear bi- modal distribution: for all examined information types originating from the question, the probability mass concentrates around two peaks, one near zero saliency and another at a substantially higher value. The near-zero peak suggests that, for a substantial subset of samples, the question-to-answer informa- tion flow contributes minimally to hallucination detection, whereas the higher peak reflects strong dependence on such flow. Hypothesis These observations lead to the hy- pothesis that there are two distinct mechanisms of 25582 ===== PAGE 4 / 47 ===== 0 10 20 30 Layer 80 60 40 20 0 P Llama-3-8B 0 20 40 60 80 Layer 80 60 40 20 0 P Llama-3-70B 0 10 20 30 Layer 80 60 40 20 0 P Mistral-7B-v0.3 Q-Anchored (PopQA) A-Anchored (PopQA) Q-Anchored (TriviaQA) A-Anchored (TriviaQA) Q-Anchored (HotpotQA) A-Anchored (HotpotQA) Q-Anchored (NQ) A-Anchored (NQ) Figure 2: ∆P under attention knockout. The layer axis indicates the Transformer layer on which the probe is trained. Shaded regions indicate 95% confidence intervals. Full results in Appendix C. internal truthfulness encoding for hallucination detection: (1) one characterized by strong reliance on the key question-to-answer information from the exact question tokens, and (2) one in which truth- fulness encoding is largely independent of the ques- tion. We validate the proposed hypothesis through further experiments in the next section. 3.2 Disentangling Information Mechanisms We hypothesize that the internal truthfulness en- coding operates through two distinct information flow mechanisms, driven by the attention modules within Transformer blocks. To validate the hypothe- sis, we first block information flows associated with the exact question tokens and analyze the resulting changes in the probe’s predictions. Subsequently, we apply a complementary technique, called token patching, to further substantiate the existence of these two mechanisms. Finally, we demonstrate that the self-contained information from the LLM- generated answer itself drives the truthfulness en- coding for the A-Anchored type. 3.2.1

## Conclusion / Discussion Extract

We investigate how LLMs encode truthfulness, re- vealing two complementary pathways: a Question- Anchored pathway relying on question-answer flow, and an Answer-Anchored pathway extract- ing self-contained evidence from generated outputs. Analyses across datasets and models highlight their ties to knowledge boundaries and intrinsic self- awareness. Building on these insights, we further propose two applications to improve hallucination detection. Overall, our findings not only advance mechanistic understanding of intrinsic truthfulness 25587 ===== PAGE 9 / 47 ===== encoding but also offer practical applications for building more reliable generative systems. Limitations While this work provides a systematic analysis of intrinsic truthfulness encoding mechanisms in LLMs and demonstrates their utility for hallucina- tion detection, one limitation is that, similar to prior work on mechanistic interpretability, our analyses and pathway-aware applications assume access to internal model representations. Such access may not always be available in strictly black-box set- tings. In these scenarios, additional engineering or alternative approximations may be required for practical deployment, which we leave for future work. Ethics Statement Our work presents minimal potential for negative societal impact, primarily due to the use of publicly available datasets and models. This accessibility inherently reduces the risk of adverse effects on individuals or society. Acknowledgments This work was supported by Beijing Natural Science Foundation (L253020) & the Academic Research Projects of Beijing Union University (NO.ZK10202405). References Lukas Aichberger, Kajetan Schweighofer, Mykyta Ielanskyi, and Sepp Hochreiter. 2024. Semanti- cally diverse language generation for uncertainty estimation in language models. arXiv preprint arXiv:2406.04306. Lukas Aichberger, Kajetan Schweighofer, Mykyta Ielan- skyi, and Sepp Hochreiter. 2025. Improving uncer- tainty estimation through semantically diverse lan- guage generation. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net. Ge Bai, Jie Liu, Xingyuan Bu, Yanc...

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

TBD: add short excerpts with page markers from `../texts/two-pathways-to-truthfulness-on-the-intrinsic.txt`.
