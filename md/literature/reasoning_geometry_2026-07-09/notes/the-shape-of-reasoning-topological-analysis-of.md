# The Shape of Reasoning: Topological Analysis of Reasoning Traces in Large Language Models

- **Local PDF filename**: `The Shape of Reasoning - Topological Analysis of.pdf`
- **Slug**: `the-shape-of-reasoning-topological-analysis-of`
- **Pages**: 19
- **Approx Words**: 8138
- **Auto Tags**: geometry;dynamics;faithfulness
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.626968

## Keyword Profile

- `topolog`: 55
- `entropy`: 12
- `geometric`: 5
- `geometry`: 4
- `faithful`: 4
- `chain of thought`: 3
- `latent`: 2
- `dimension`: 2
- `trajectory`: 1
- `phase`: 1
- `transition`: 1
- `causal`: 1

## Abstract / Opening Summary

Evaluating the quality of reasoning traces from large language models remains understudied, labor-intensive, and unreliable: current practice relies on expert rubrics, manual annotation, and slow pairwise judgments. Automated efforts are dominated by graph-based proxies that quantify structural connectivity but do not clarify what constitutes high-quality reasoning; such abstrac- tions can be overly simplistic for inherently com- plex processes. We introduce a topological data analysis (TDA)–based evaluation framework that captures the geometry of reasoning traces and en- ables label-efficient, automated assessment. In our empirical study, topological features yield substantially higher predictive power for assess- ing reasoning quality than standard graph metrics, suggesting that effective reasoning is better cap- tured by higher-dimensional geometric structures rather than purely relational graphs. We further show that a compact, stable set of topological fea- tures reliably indicates trace quality, offering a practical signal for future reinforcement learning algorithms.

## Method / Algorithms Extract

methods treat reasoning traces as instru- mental by aggregating multiple reasoning paths to obtain the most frequent final answer while discarding the traces themselves. Recent work has shifted attention to reasoning faithfulness: whether a model’s explicit reasoning aligns with its underlying decision process. Agarwal et al. (2024) demonstrated that LLMs often produce plausible but un- faithful explanations that do not reflect actual reasoning, and Nguyen et al. (2024) found that models can achieve correct answers through logically flawed or spurious rea- soning. These findings emphasize the need for metrics that evaluate reasoning process quality rather than final-answer correctness. To quantify reasoning structure, Xiong et al. (2025) in- troduced a reasoning-graph framework that maps LLM- generated reasoning traces into directed graphs, enabling analysis of properties such as branching and convergence. Their findings suggest that effective reasoning depends on a balanced pattern of exploration and convergence rather than on trace length alone, aligning with the observations of Su et al. (2025) that overly long reasoning often re- duces accuracy. Complementary work has explored au- tomated reasoning evaluation: Ton et al. (2025) proposed an information-theoretic approach measuring the contribution of each reasoning step, while Nguyen et al. (2024) grounded CoT reasoning in knowledge graphs to test factual and logi- cal validity. Together, these works advance the field toward systematic evaluation of reasoning traces. Efforts to improve reasoning quality have paralleled those to analyze it. Reinforcement learning methods, such as Graph- PRM (Peng et al., 2025), directly optimize reasoning steps rather than outcomes, using process-level rewards to align model behavior with logical norms. Structured prompting techniques such as Tree-of-Thoughts (Yao et al., 2023) and Graph-of-Thoughts (Besta et al., 2024) further demonstrate that the organization of reasoning whether tree, graph, or chain-based may critically influences problem-solving suc- cess. Collectively, these studies motivate the search for more robust and interpretable reasoning metrics, which our work seeks to advance through a geometric–topological formulation. 2.2. Topological Data Analysis Applied to LLMs Topological Data Analysis (TDA) offers a mathematical lens for understanding the shape of data by identifying invariant geometric structures such as connected compo- nents and holes. Recent work has applied TDA to study neural representations in large models. Gardinazzi et al. (2025) introdu...

## Experiments / Evidence Extract

experiments use the American Invitational Mathematics Examination (AIME) dataset because, to the best of our knowledge, it is the only publicly available corpus with step–by–step solution traces for non-trivial problems. While olympiad-level math of- fers rich reasoning chains, this focus restricts the diversity of reasoning styles and problem domains considered. Sur- veys on step-by-step reasoning evaluation note that existing resources tend to be either overly simple or confined to specialised domains (Lee & Hockenmaier, 2025). Relying on a single dataset therefore limits the generality of our findings. A clear avenue for future work is to curate or annotate additional datasets with explicit reasoning traces across domains such as commonsense reasoning, science, programming and real-world problem solving. Even small human-annotated corpora in other domains would allow us to test whether the topological patterns identified here persist beyond competitive mathematics. Topological features interpretation. Our features sum- marize the geometry of step embeddings, not the symbolic structure of reasoning trees. Holes in H1 and mergers in H0 arise from how steps are placed in the embedding space and from the distance used to build the Vietoris–Rips complex; they need not correspond to literal detours or merges in a human-readable proof. In our pipeline, both alignment and topology operate on sentence embeddings with cosine distance. Changing the embedder, the segmentation, or the metric can create or remove cycles and shift lifetimes without altering the underlying textual logic. Consequently, interpretations such as “H1 captures detours” or “H0 cap- tures clustering of ideas” are, at best, geometric proxies for reasoning structure rather than direct evidence of a particu- lar tree or graph motif. We therefore caution against reading persistence diagrams as faithful maps of the latent reasoning program and instead view them as embedding-dependent signals that correlate with alignment quality. For future work, we aim to ground topological events in interpretable operations, such as opening a branch, running a short check, and rejoining, while remaining graph-free, since explicit trace graphs are rarely available. Acknowledgements We would like to thank Infocomm Media Development Authority (IMDA) for providing us with the opportunity and funding to work on this res...

## Conclusion / Discussion Extract

conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of the Infocomm Media Development Authority, Singapore. References Agarwal, C., Tanneru, S. H., and Lakkaraju, H. Faith- fulness vs. plausibility: On the (un) reliability of ex- planations from large language models. arXiv preprint arXiv:2402.04614, 2024. Art of Problem Solving. AIME Problems and Solutions. https://artofproblemsolving.com/wiki/ index.php/AIME_Problems_and_Solutions, 2025. Accessed on 29 September 2025. Balderas, L., Lastra, M., and Benitez, J. M. A green ai methodology based on persistent homology for compress- ing bert. Applied Sciences, 15(1):390, 2025. Besta, M., Blach, N., Kubicek, A., Gerstenberger, R., Pod- stawski, M., Gianinazzi, L., Gajda, J., Lehmann, T., Niewiadomski, H., Nyczyk, P., et al. Graph of thoughts: Solving elaborate problems with large language models. In Proceedings of the AAAI conference on artificial intel- ligence, volume 38, pp. 17682–17690, 2024. Gamelin, T. W. and Greene, R. E. Introduction to topology. Courier Corporation, 1999. Gardinazzi, Y., Viswanathan, K., Panerai, G., Cazzaniga, A., Biagetti, M., et al. Persistent topological features in large language models. In Forty-second International Conference on Machine Learning, 2025. Lee, J. and Hockenmaier, J. Evaluating step-by-step reason- ing traces: A survey. arXiv preprint arXiv:2502.12289, 2025. 5 ===== PAGE 6 / 19 ===== The Shape of Reasoning: Topological Analysis of Reasoning Traces in Large Language Models Minegishi, G., Furuta, H., Kojima, T., Iwasawa, Y., and Matsuo, Y. Topology of reasoning: Understanding large reasoning models through reasoning graph properties. arXiv preprint arXiv:2506.05744, 2025. Nguyen, T., Luo, L., Shiri, F., Phung, D., Li, Y.-F., Vu, T., and Haffari, G. Direct evaluation of chain-of-thought in multi-hop reasoning with knowledge graphs. In Findings of the Association for Computational Linguistics ACL 2024, pp. 2862–2883, 2024. Peng, M., Chen, N., Suo, Z., and Li, J. Rewarding graph reasoning process makes llms more generalized reasoners. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data M...

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

TBD: add short excerpts with page markers from `../texts/the-shape-of-reasoning-topological-analysis-of.txt`.
