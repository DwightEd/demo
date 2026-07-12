# Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs

- **Local PDF filename**: `TowardsLong-HorizonInterpretability.pdf`
- **Slug**: `towardslong-horizoninterpretability`
- **Pages**: 26
- **Approx Words**: 15184
- **Auto Tags**: dynamics;faithfulness
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.629987

## Keyword Profile

- `faithful`: 81
- `flow`: 26
- `causal`: 14
- `dimension`: 7
- `chain of thought`: 5
- `phase`: 1
- `transition`: 1
- `hidden state`: 1

## Abstract / Opening Summary

Token attribution methods provide intuitive ex- planations for language model outputs by iden- tifying causally important input tokens. How- ever, as modern LLMs increasingly rely on ex- tended reasoning chains, existing schemes face two critical challenges: (1) efficiency bottleneck, where attributing a target span of M tokens within a context of length N requires O(M · N) op- erations, making long-context attribution pro- hibitively slow; and (2) faithfulness drop, where intermediate reasoning tokens absorb attribution mass, preventing importance from propagating back to the original input. To address these, we introduce FLASHTRACE, an efficient multi- token attribution method that employs span-wise aggregation to compute attribution over multi- token targets in a single pass, while maintain- ing faithfulness. Moreover, we design a recur- sive attribution mechanism that traces importance through intermediate reasoning chains back to source inputs. Extensive experiments on long- context retrieval (RULER) and multi-step rea- soning (MATH, MorehopQA) tasks demonstrate that FLASHTRACE achieves over 130× speedup over existing baselines while maintaining supe- rior faithfulness. We further analyze the dynam- ics of recursive attribution, showing that even a single recursive hop improves faithfulness by trac- ing importance through the reasoning chain. Our code is available at https://github.com/ wbopan/flashtrace. 1Department of Computer Science, City University of Hong Kong, Hong Kong SAR, China 2Harbin Institute of Technol- ogy, Harbin, China. Correspondence to: Haining Yu <yuhain- ing@hit.edu.cn>. Proceedings of the 43 rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s). ... ... Input (I) Reasoning Tokens (T) Output (O) Complexity Slow & Inefficient Reasoning Tokens Absorb Importance ... ... Input (I) Reasoning Tokens (T) Output (O) Naive Token-by-Token Attribution FlashTrace: Recursive & Span-wise K-hop Span-wise Attribution Importance Trace Through Reasoning Direct COT 0.0 0.2 0.4 Attr. Weight (%) 19% 7% Direct COT 0.0 0.2 Recovery Rate (%) 26% 9% 100 2,000 5,000 10,000 0 200 Time (s) Baseline FlashTrace Figure 1. Motivation for FLASHTRACE. Top: Naive token-by- token attribution requires expensive per-token computation, while FLASHTRACE performs efficient span-wise recursive attribution. Bottom: (a) With extended reasoning, attribution weight on rea- soning tokens increases significantly (from approximately 80% to over 90%); (b) This causes recovery rate of ground-truth input tokens to dro...

## Method / Algorithms Extract

mq q2 mq q4 mq q8 mv v2 mv v4 mv v8 h2 c3 h4 c1 h6 c1 h10 c1 (1024) Perturbation 0.391 0.090 0.010 0.255 0.161 0.080 0.060 0.027 0.051 0.011 0.329 REAGENT 0.244 0.085 0.005 0.180 0.156 0.074 0.045 0.023 0.050 0.014 0.222 Recovery Rate (% ↑) CLP 0.399 0.086 0.008 0.207 0.146 0.073 0.130 0.038 0.063 0.020 0.335 IFR 0.471 0.328 0.012 0.575 0.452 0.179 0.136 0.253 0.202 0.155 0.268 AttnLRP 0.215 0.204 0.076 0.254 0.243 0.159 0.212 0.229 0.202 0.173 0.189 FLASHTRACE 0.483 0.413 0.075 0.556 0.516 0.204 0.698 0.755 0.659 0.514 0.384 Perturbation 0.095 0.239 0.499 0.134 0.186 0.351 0.384 0.354 0.458 0.466 0.133 REAGENT 0.117 0.260 0.487 0.188 0.211 0.369 0.438 0.397 0.486 0.495 0.145 Faithfulness (RISE ↓) CLP 0.098 0.253 0.510 0.156 0.217 0.393 0.374 0.328 0.423 0.451 0.101 IFR 0.075 0.115 0.371 0.069 0.073 0.205 0.161 0.102 0.125 0.153 0.074 AttnLRP 0.196 0.263 0.377 0.140 0.193 0.285 0.319 0.324 0.338 0.357 0.155 FLASHTRACE 0.068 0.113 0.352 0.069 0.070 0.183 0.132 0.110 0.122 0.143 0.033 Perturbation 0.144 0.327 0.709 0.187 0.244 0.458 0.551 0.517 0.684 0.701 0.220 REAGENT 0.197 0.357 0.694 0.276 0.291 0.494 0.668 0.603 0.745 0.741 0.235 Faithfulness (MAS ↓) CLP 0.166 0.320 0.657 0.216 0.280 0.511 0.490 0.420 0.565 0.597 0.190 IFR 0.140 0.177 0.460 0.134 0.142 0.275 0.231 0.148 0.173 0.201 0.166 AttnLRP 0.326 0.451 0.602 0.229 0.325 0.475 0.521 0.548 0.572 0.592

## Experiments / Evidence Extract

experiments on long- context retrieval (RULER) and multi-step rea- soning (MATH, MorehopQA) tasks demonstrate that FLASHTRACE achieves over 130× speedup over existing baselines while maintaining supe- rior faithfulness. We further analyze the dynam- ics of recursive attribution, showing that even a single recursive hop improves faithfulness by trac- ing importance through the reasoning chain. Our code is available at https://github.com/ wbopan/flashtrace. 1Department of Computer Science, City University of Hong Kong, Hong Kong SAR, China 2Harbin Institute of Technol- ogy, Harbin, China. Correspondence to: Haining Yu <yuhain- ing@hit.edu.cn>. Proceedings of the 43 rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s). ... ... Input (I) Reasoning Tokens (T) Output (O) Complexity Slow & Inefficient Reasoning Tokens Absorb Importance ... ... Input (I) Reasoning Tokens (T) Output (O) Naive Token-by-Token Attribution FlashTrace: Recursive & Span-wise K-hop Span-wise Attribution Importance Trace Through Reasoning Direct COT 0.0 0.2 0.4 Attr. Weight (%) 19% 7% Direct COT 0.0 0.2 Recovery Rate (%) 26% 9% 100 2,000 5,000 10,000 0 200 Time (s) Baseline FlashTrace Figure 1. Motivation for FLASHTRACE. Top: Naive token-by- token attribution requires expensive per-token computation, while FLASHTRACE performs efficient span-wise recursive attribution. Bottom: (a) With extended reasoning, attribution weight on rea- soning tokens increases significantly (from approximately 80% to over 90%); (b) This causes recovery rate of ground-truth input tokens to drop substantially (from 26% to below 10%); (c) Naive multi-hop attribution scales poorly with reasoning length, while FLASHTRACE remains efficient even for 10K tokens. 1. Introduction While more and more high-stakes decisions are made by Large Language Model (LLM) agents (Novikov et al., 2025), interpreting their outputs becomes increasingly important. Token attribution methods offer an intuitive explanation approach (Achtibat et al., 2024; Ferrando & Voita, 2024). Given a specific output to explain, these methods calculate an importance distribution over all input tokens to identify those causally responsible for the generation. This provides principled leverage to both understand LLM behaviors and optimize context. However, as recent reasoning and agentic LLMs gen...

## Conclusion / Discussion Extract

Conclusion We introduced FLASHTRACE, an efficient multi-token attri- bution method that addresses the efficiency and faithfulness challenges of interpreting reasoning LLMs. By combining span-wise aggregation with recursive attribution, FLASH- TRACE achieves efficient span-wise attribution and traces importance through reasoning chains back to source inputs. Experiments across long-context retrieval, mathematical reasoning, and multi-hop QA demonstrate significant speedup while maintaining superior faithfulness, enabling scalable interpretability for modern agentic workflows. Acknowledgements This work was supported in part by the National Natu- ral Science Foundation of China under Grant 62302122, the National Key Research and Development Program of China under Grant 2025YFB3109803, the Heilongjiang Provincial Natural Science Foundation of China under Grant JQ2024F001, and the Hong Kong Research Grants Council under Grants C1043-24GF and RFS2425-1S01. Impact Statement This paper presents work whose goal is to advance the inter- pretability of large language models. By enabling efficient and faithful attribution of model outputs to their inputs, our method contributes to AI transparency and may help practitioners better understand, debug, and audit model be- havior in high-stakes applications. We believe improved interpretability is a positive step toward safer and more trustworthy AI systems. We do not foresee direct negative societal consequences specific to this work. References Achtibat, R., Hatefi, S. M. V., Dreyer, M., Jain, A., Wie- gand, T., Lapuschkin, S., and Samek, W. Attnlrp: Attention-aware layer-wise relevance propagation for transformers. In Proceedings of the 41st International Conference on Machine Learning (ICML), pp. 135–168, 2024. URL https://proceedings.mlr.press/ v235/achtibat24a.html. Chen, J., Li, X., Yu, L., Dou, D., and Xiong, H. Be- yond intuition: Rethinking token attributions inside trans- formers. Transactions on Machine Learning Research, 2022. URL https://openreview.net/forum? id=rm0zIzlhcX. Chen, Q., Qin, L., Liu, J., Peng, D., Guan, J., Wang, P., Hu, M., Zhou, Y., Gao, T., and Che, W. Towards reasoning era: A survey of long chain...

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

TBD: add short excerpts with page markers from `../texts/towardslong-horizoninterpretability.txt`.
