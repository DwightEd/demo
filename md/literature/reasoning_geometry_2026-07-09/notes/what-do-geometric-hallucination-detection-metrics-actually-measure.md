# What do Geometric Hallucination Detection Metrics Actually Measure?

- **Local PDF filename**: `What do Geometric Hallucination Detection Metrics Actually Measure.pdf`
- **Slug**: `what-do-geometric-hallucination-detection-metrics-actually-measure`
- **Pages**: 10
- **Approx Words**: 5321
- **Auto Tags**: geometry;uncertainty;hallucination
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.633564

## Keyword Profile

- `hallucination`: 87
- `geometric`: 38
- `entropy`: 14
- `hidden state`: 8
- `probe`: 5
- `dimension`: 4
- `geometry`: 1
- `causal`: 1

## Abstract / Opening Summary

Hallucination remains a barrier to deploying gen- erative models in high-consequence applications. This is especially true in cases where external ground truth is not readily available to validate model outputs. This situation has motivated the study of geometric signals in the internal state of an LLM that are predictive of hallucination and require limited external knowledge. Given that there are a range of factors that can lead model output to be called a hallucination (e.g., irrele- vance vs incoherence), in this paper we ask what specific properties of a hallucination these geomet- ric statistics actually capture. To assess this, we generate a synthetic dataset which varies distinct properties of output associated with hallucination. This includes output correctness, confidence, rel- evance, coherence, and completeness. We find that different geometric statistics capture different types of hallucinations. Along the way we show that many existing geometric detection methods have substantial sensitivity to shifts in task do- main (e.g., math questions vs. history questions). Motivated by this, we introduce a simple normal- ization method to mitigate the effect of domain shift on geometric statistics, leading to AUROC gains of +34 points in multi-domain settings.

## Method / Algorithms Extract

methods have substantial sensitivity to shifts in task do- main (e.g., math questions vs. history questions). Motivated by this, we introduce a simple normal- ization method to mitigate the effect of domain shift on geometric statistics, leading to AUROC gains of +34 points in multi-domain settings. 1. Introduction Developing efficient and effective methods for detecting hallucinations is currently a major need for the larger goal of achieving reliable and responsible generative models (Huang et al., 2025). Proposed approaches come in a range of flavors, from methods that validate large langauge model’s (LLM’s) output using external data sources (Lewis 1Pacific Northwest National Laboratory 2University of Penn- sylvania 3Colorado State University 4University of Texas, El Paso 5Laboratory for Advanced Cybersecurity Research, National Secu- rity Agency 6University of Washington. Correspondence to: Eric Yeats <eric.yeats@pnnl.gov>. Published at ICML 2025 Workshop on Reliable and Responsible Foundation Models. Copyright 2025 by the author(s). et al., 2020; Asai et al., 2023), to methods that validate us- ing a set of distinct judge LLMs (Jacovi et al., 2025; Zheng et al., 2023), to methods that use the consistency of a par- ticular output as a signal for factual accuracy (Chen et al., 2024; Manakul et al., 2023). One family of methods that is particularly attractive is those that use signals extracted from model internals such as token representations in the residual stream or attention maps to detect when output is likely to be a hallucination. Though sometimes requiring labeled data upfront (Azaria & Mitchell, 2023; Orgad et al., 2024), these methods are often compute efficient to run and do not require external knowledge at inference time. Since model internals tend to be high-dimensional and not intrinsically interpretable, it has been common to leverage geometric or information-theoretic statistics in detection frameworks (Sriramanan et al., 2024; Du et al., 2024; Yin et al., 2024). However, hallucinations can come in many forms and be characterized in several ways (Huang et al., 2025). What aspect of a hallucination a particular method is flagging re- mains mostly unexplored. In this paper we try to answer the question: what characteristics of a hallucination do popular geometric detection methods actually capture? Specifically, we programmatically generate user prompts and model re- sponses which exhibit different properties of a hallucination within different domains. We then run these prompts and re- sponses through the model and extract hidden stat...

## Experiments / Evidence Extract

We augment each of the statistics with f ∗(·) and record their scores on all. Figure 2a depicts the distri- butions of HS-Norm scores for baseline responses and for level-1 incorrectness hallucinations (this should be com- pared with Figure 1). The correct responses for each of the domains are aligned on the bottom half of the chart at layer 30, while the incorrect response distributions are aligned on the top half of the chart at layer 30. Figure 2b depicts the normalized distributions of HS-Norm scores on all for correct responses and incorrect responses (levels 1-3). We observe that HS-Norm separates incorrect from correct well in the later stages of the network. Figure 2c depicts the AUROC for each statistic on all (level 1 hallucinations). The aligned domains lead to significantly improved detector performance. HS-Norm and ME-Norm achieve AUROCs of 0.96 (40 point gain) at layer 30, while AS-Norm achieves an AUROC of 0.89 (34 point gain) at layer 31.

## Conclusion / Discussion Extract

Conclusion We design a multi-domain dataset of hallucinations with various properties and severities to try to answer the ques- tion “What do geometric hallucination detection metrics actually measure?” We find that all geometric statistics are correlated with incorrectness, but different statistics respond to different hallucination characteristics. Additionally, we find that domain shift impairs the detection performance of the statistics on incorrectness. We mitigate domain shift with a simple normalization technique, leading to 34 to 40 point AUROC gains for each statistic in multi-domain hallucination detection scenarios. 4 ===== PAGE 5 / 10 ===== What do Geometric Hallucination Detection Metrics Actually Measure? References Asai, A., Wu, Z., Wang, Y., Sil, A., and Hajishirzi, H. Self- rag: Learning to retrieve, generate, and critique through self-reflection. In The Twelfth International Conference on Learning Representations, 2023. Azaria, A. and Mitchell, T. The internal state of an LLM knows when it’s lying. arXiv preprint arXiv:2304.13734, 2023. Chen, C., Liu, K., Chen, Z., Gu, Y., Wu, Y., Tao, M., Fu, Z., and Ye, J. Inside: LLMs’ internal states re- tain the power of hallucination detection. arXiv preprint arXiv:2402.03744, 2024. Du, X., Xiao, C., and Li, S. Haloscope: Harnessing unla- beled LLM generations for hallucination detection. Ad- vances in Neural Information Processing Systems, 37: 102948–102972, 2024. Farquhar, S., Kossen, J., Kuhn, L., and Gal, Y. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630, 2024. Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., Chen, Q., Peng, W., Feng, X., Qin, B., et al. A survey on hallucination in large language models: Principles, taxon- omy, challenges, and open questions. ACM Transactions on Information Systems, 43(2):1–55, 2025. Jacovi, A., Wang, A., Alberti, C., Tao, C., Lipovetz, J., Olszewska, K., Haas, L., Liu, M., Keating, N., Bloniarz, A., et al....

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

TBD: add short excerpts with page markers from `../texts/what-do-geometric-hallucination-detection-metrics-actually-measure.txt`.
