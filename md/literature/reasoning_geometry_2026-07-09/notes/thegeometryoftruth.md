# The Geometry of Truth: Layer-wise Semantic Dynamics for Hallucination Detection in Large Language Models

- **Local PDF filename**: `TheGeometryofTruth.pdf`
- **Slug**: `thegeometryoftruth`
- **Pages**: 27
- **Approx Words**: 8184
- **Auto Tags**: geometry;dynamics;faithfulness;uncertainty;hallucination
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.629099

## Keyword Profile

- `hallucination`: 73
- `trajectory`: 44
- `geometric`: 33
- `manifold`: 13
- `geometry`: 12
- `hidden state`: 9
- `dimension`: 8
- `entropy`: 6
- `latent`: 4
- `probe`: 2
- `flow`: 2
- `transition`: 1

## Abstract / Opening Summary

Large Language Models (LLMs) often produce fluent yet factually incorrect statements—a phenomenon known as hallucination—posing serious risks in high-stakes domains. We present Layer-wise Seman- tic Dynamics (LSD), a geometric framework for hallucination detection that analyzes the evolution of hidden-state semantics across transformer layers. Unlike prior methods that rely on multiple sampling passes or external verification sources, LSD operates intrinsically within the model’s representational space. Using margin-based contrastive learning, LSD aligns hidden activations with ground-truth em- beddings derived from a factual encoder, revealing a distinct separation in semantic trajectories: fac- tual responses preserve stable alignment, while hallucinations exhibit pronounced semantic drift across depth. Evaluated on the TruthfulQA and synthetic factual-hallucination datasets, LSD achieves an F1- score of 0.92, AUROC of 0.96, and clustering accuracy of 0.89, outperforming SelfCheckGPT and Semantic Entropy baselines while requiring only a single forward pass. This efficiency yields a 5–20× speedup over sampling-based methods without sacrificing precision or interpretability. LSD offers a scalable, model-agnostic mechanism for real-time hallucination monitoring and provides new insights into the geometry of factual consistency within large language models. Keywords: Hallucination Detection, Semantic Dynamics, Transformer Representations, Contrastive Learning, Geometric Analysis, Language Model Interpretability Contents 1

## Method / Algorithms Extract

Precision Recall F1-Score AUROC SelfCheckGPT 0.823 0.874 0.847 0.891 Semantic Entropy 0.798 0.826 0.812 0.864 Final-layer Probing 0.756 0.814 0.784 0.838 LSD (Ours) 0.920 0.922 0.922 0.959 Velocity and acceleration magnitudes are comparable across classes, suggesting that both factual and hallu- cinated trajectories exhibit similar dynamical rates of representational change, but diverge along orthogonal semantic directions. All alignment-based metrics are statistically significant (p < 0.0001) with very large effect sizes (Cohen’s d > 2.8), reinforcing LSD’s ability to serve as a stable, interpretable indicator of truthfulness within model internals. Table 2 provides detailed statistical comparison across all metrics. 4.4.3 Detection Performance Table 3 compares LSD with established hallucination detection baselines on the TruthfulQA benchmark. Results show that LSD achieves superior discriminative performance while maintaining computational effi- ciency. LSD achieves: • 7.5% absolute improvement in F1-score over SelfCheckGPT (0.922 vs. 0.847) • 6.8% improvement in AUROC over Semantic Entropy (0.959 vs. 0.891) • Balanced precision and recall (≈0.92), indicating both high sensitivity and conservatism in hallucination detection Unlike sampling-based methods such as SelfCheckGPT, which require generating multiple responses 15 ===== PAGE 16 / 27 ===== Table 4: Performance of LSD-based classifiers. Results are reported for the hybrid (TruthfulQA + Syn- thetic) configuration with 1000 samples. The Logistic Regression model achieves the highest F1 and AU- ROC, demonstrating strong linear separability of factual and hallucinatory trajectories in LSD space. Model F1 AUC-ROC Precision Recall LSD_LogisticRegression 0.9215 0.9591 0.920 0.922 LSD_RandomForest 0.8602 0.9510 0.861 0.859 LSD_GradientBoosting 0.8723 0.9475 0.870 0.874 LSD_Unsupervised 0.8920 — — — (typically 5–20 per query), LSD operates with a single forward pass through the model’s hidden layers. This yields a 5×–20× computational speedup while maintaining high accuracy, making LSD particularly suitable for real-time or large-scale hallucination monitoring in deployed LLMs. Table 4 summarizes the comparative performance of classifiers trained on LSD-derived features. The LSD_LogisticRegression model’s superior performance confirms that factual and hallucinatory trajectories are linearly separable in the LSD representation space, indicating that the layer-wise semantic manifold organizes truth-aligned representations into compact, low-entropy clusters. 4.5 Visualization of Semantic Dynamics Figure 2 pro...

## Experiments / Evidence Extract

experiments on synthetic pairs and TruthfulQA, we demonstrate significant separation between factual and hallucinated content (Cohen’s d = 2.97, p < 0.0001) across all transformer layers, achieving F1-score of 0.92, AUROC of 0.96, and clustering accu- racy of 0.89 (Section 4). 5. Ablation and Interpretability Analysis: We systematically evaluate component contributions and pro- vide interpretable visualizations revealing how semantic dynamics distinguish factual from hallucinated content (Section 5). 6. Open-source Implementation: We release a production-ready system enabling real-time hallucination detection with interpretable confidence estimates, facilitating reproducibility and practical deployment. 1.4 Paper Organization The remainder of this paper is structured as follows: Section 2 surveys related work in hallucination de- tection, internal representation analysis, and representation geometry. Section 3 formalizes the LSD frame- work, including problem formulation, architecture, and theoretical foundations. Section 4 presents experi- mental setup, datasets, and comprehensive results. Section 5 provides ablation studies and interpretability analysis. Section 6 discusses implications, limitations, and future directions. Section 7 concludes. 2 Related Work 2.1 Hallucination Detection in Language Models Hallucination detection has evolved through several methodological paradigms, each addressing different aspects of the problem while facing distinct limitations. 2.1.1 Consistency-based Approaches Consistency-based methods [2] operate on the principle that hallucinated content exhibits higher variance across multiple samples than factual content. SelfCheckGPT samples n outputs and measures consistency using BERTScore, question answering, or n-gram overlap. While effective, these methods incur O(n) computational cost per query, making them impractical for latency-sensitive applications. Furthermore, 4 ===== PAGE 5 / 27 ===== they assume that hallucinations manifest as inconsistencies, which may not hold when models consistently reproduce learned biases or systematic errors. 2.1.2 Retrieval-augmented Verification Retrieval-augmented techniques [4, 3] decompose generated text into atomic claims and verify each against external knowledge bases. FActScore achieves fine-grained factual precision by using InstructGPT to break down responses and check each fact in...

## Conclusion / Discussion Extract

24 6.1 Theoretical Implications . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24 6.2 Practical Applications . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25 6.3 Limitations and Future Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25 7

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

TBD: add short excerpts with page markers from `../texts/thegeometryoftruth.txt`.
