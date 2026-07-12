# Large Vision-Language Models Get Lost in Attention

- **Local PDF filename**: `Large Vision–Language Models Get Lost in Attention.pdf`
- **Slug**: `large-vision-language-models-get-lost-in-attention`
- **Pages**: 25
- **Approx Words**: 16199
- **Auto Tags**: geometry;dynamics
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.617002

## Keyword Profile

- `manifold`: 19
- `dimension`: 18
- `geometric`: 13
- `entropy`: 11
- `spectral`: 9
- `hallucination`: 9
- `geometry`: 8
- `flow`: 6
- `causal`: 5
- `probe`: 4
- `transition`: 3
- `hidden state`: 3

## Abstract / Opening Summary

Despite the rapid evolution of training paradigms, the decoder backbone of large vision–language models (LVLMs) remains fundamentally rooted in the residual-connection Transformer architec- ture. Therefore, deciphering the distinct roles of internal modules is critical for understanding model mechanics and guiding architectural op- timization.

## Method / Algorithms Extract

methods in neural language processing: A survey. Transactions of the Asso- ciation for Computational Linguistics, 7:49–72, 2019. 9 ===== PAGE 10 / 25 ===== Large Vision–Language Models Get Lost in Attention Belrose, N., Furman, Z., Smith, L., Halawi, D., Ostrovsky, I., McKinney, L., Biderman, S., and Steinhardt, J. Eliciting latent predictions from transformers with the tuned lens. arXiv preprint arXiv:2303.08112, 2023. Bengio, Y., Courville, A., and Vincent, P. Representation learning: A review and new perspectives. IEEE transac- tions on pattern analysis and machine intelligence, 35(8): 1798–1828, 2013. BT, R. et al. Studio encoding parameters of digital television for standard 4: 3 and wide-screen 16: 9 aspect ratios. International radio consultative committee international telecommunication union, Switzerland, CCIR Rep, 2011. Chen, L., Li, J., Dong, X., Zhang, P., Zang, Y., Chen, Z., Duan, H., Wang, J., Qiao, Y., Lin, D., et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37: 27056–27087, 2024. Conneau, A., Kruszewski, G., Lample, G., Barrault, L., and Baroni, M. What you can cram into a single vector: Probing sentence embeddings for linguistic properties. arXiv preprint arXiv:1805.01070, 2018. Cunningham, H., Ewart, A., Riggs, L., Huben, R., and Sharkey, L. Sparse autoencoders find highly inter- pretable features in language models. arXiv preprint arXiv:2309.08600, 2023. Deb, M. and Ogunfunmi, T. Information-theoretical analysis of a transformer-based generative ai model. Entropy, 27 (6):589, 2025. Du, X., Mo, F., Wen, M., Gu, T., Zheng, H., Jin, H., and Shi, J. Multi-turn jailbreaking large language models via attention shifting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 23814–23822, 2025. Dunefsky, J., Chlenski, P., and Nanda, N. Transcoders find interpretable llm feature circuits. Advances in Neural Information Processing Systems, 37:24375–24410, 2024. Eckart, C. and Young, G. The approximation of one matrix by another of lower rank. Psychometrika, 1(3):211–218, 1936. Edelman, E., Tsilivis, N., Edelman, B., Malach, E., and Goel, S. The evolution of statistical induction heads: In-context learning markov chains. Advances in neural information processing systems, 37:64273–64311, 2024. Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., et al. A mathematical framework for transformer circuits. Transformer Circuits Thread, 1(1):12, 2021. Elhelo, A. and Geva, M. Inferring func...

## Experiments / Evidence Extract

experiments across 15 state-of-the-art LVLMs spanning three dominant archi- tectures on a broad suite of multimodal benchmarks. Our analysis reveals two profound insights: first, we quantita- tively validate a sharp functional decoupling in Transformer residual stream computation: attention primarily performs entropic reconfiguration that preserves the existing represen- tation support, whereas FFNs dominate innovation by intro- ducing new semantic directions. Building on this division of labor, we further diagnose a systemic pathology in cur- rent LVLMs: decoder visual attention often fails to perform meaningful mixing over question-relevant visual evidence, and instead exhibits substantial redundancy, frequently get- ting lost in interaction patterns with limited contribution to informative updates. Our main contributions are summarized as follows: • Theoretical Framework: We propose a rigorous formal- ism based on the manifold hypothesis to define repre- sentational information. We introduce RID and MixIG as dual metrics to quantify the geometric and entropic impact of residual updates, offering a generalized tool for probing representation dynamics. • Module-level Interpretability: We provide a quantita- tive explanation of the distinct roles within Transformer blocks. We demonstrate that Attention and FFNs operate in orthogonal regimes—reconfiguration versus innova- tion—thereby substantiating the modularity hypothesis with geometric evidence. • Empirical Diagnostics: We uncover critical inefficien- cies in LVLM designs. Our results highlight that despite architectural scaling, current models suffer from severe in- formational redundancy in visual processing, suggesting that the integration of visual tokens is often computation- ally expensive yet informationally sparse. 2. Related work Interpretability of LLMs. A large body of work studies what information is encoded in LLM representations and where it appears in the network (Belinkov & Glass, 2019). Early work uses lightweight linear probes on intermediate hidden states (Conneau et al., 2018; Hewitt & Manning, 2019; Belrose et al., 2023). Subsequent decoding based efforts, such as the tuned lens, map hidden states to vocabu- lary distributions (Belrose et al., 2023). Alongside probing and decoding, sparse feature learning approaches, including transcoders (Dunefsky et al., 2024) and sparse autoencoder...

## Conclusion / Discussion Extract

Conclusion We propose a unified theoretical framework for assessing how residual-stream updates shape representations in large models. Applying it to LVLMs reveals a consistent module- level functional separation, where attention primarily sup- ports token-level reconfiguration while FFNs drive inno- vation, and further diagnoses a pervasive failure mode in current decoders: visual attention often misallocates inter- action away from question-relevant evidence. Motivated by this deficiency, we conduct a proof-of-concept intervention by replacing attention scores in selected layers with sim- ple predefined priors, and observe little to no degradation in capability, suggesting substantial redundancy in learned scoring. Beyond these specific findings, our framework and empirical protocol offer a general tool for evaluating residual-update mechanisms across model families and mo- tivate targeted attention-centric optimization. In conclusion, our framework turns LVLM residual updates into measurable innovation–reconfiguration dynamics and provides evidence that current Transformer-based LVLMs can get lost in attention. Future work includes extending the analysis to training-time dynamics and leveraging the ob- served redundancy to design more efficient attention mech- anisms or regularizers that preserve useful mixing while reducing unnecessary scoring. Impact Statement This paper presents work whose goal is to advance the field of Large Vision–Language Model Interpretability. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. References Abnar, S. and Zuidema, W. Quantifying attention flow in transformers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Associ- ation for Computational Linguistics, 2020. Absil, P.-A., Mahony, R., and Sepulchre, R. Optimization Algorithms on Matrix Manifolds. Princeton University Press, 2008. Agrawal, K. K., Mondal, A. K., Ghosh, A., and Richards, B. α-req: Assessing representation quality in self-supervised learning by measuring eigenspectrum decay. Advances in Neural Information Processing Systems, 35:17626– 17638, 2022. AI,...

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

TBD: add short excerpts with page markers from `../texts/large-vision-language-models-get-lost-in-attention.txt`.
