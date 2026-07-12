# ===== PAGE 1 / 28 ===== Published as a conference paper at ICLR 2026

- **Local PDF filename**: `VERIFYING CHAIN-OF-THOUGHT REASONING VIA ITS.pdf`
- **Slug**: `verifying-chain-of-thought-reasoning-via-its`
- **Pages**: 28
- **Approx Words**: 16175
- **Auto Tags**: geometry;dynamics;faithfulness;step-level;emergence
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.632938

## Keyword Profile

- `chain of thought`: 17
- `causal`: 16
- `probe`: 16
- `topolog`: 12
- `faithful`: 11
- `hidden state`: 9
- `dimension`: 9
- `flow`: 8
- `latent`: 7
- `entropy`: 5
- `trajectory`: 2
- `geometry`: 1

## Abstract / Opening Summary

Current Chain-of-Thought (CoT) verification methods predict reasoning correct- ness based on outputs (black-box) or activations (gray-box), but offer limited in- sight into why a computation fails. We introduce a white-box method: Circuit- based Reasoning Verification (CRV). We hypothesize that attribution graphs of correct CoT steps, viewed as execution traces of the model’s latent reasoning cir- cuits, possess distinct structural fingerprints from those of incorrect steps. By training a classifier on structural features of these graphs, we show that these traces contain a powerful signal of reasoning errors. Our white-box approach yields novel scientific insights unattainable by other methods. (1) We demonstrate that structural signatures of error are highly predictive, establishing the viability of verifying reasoning directly via its computational graph. (2) We find these sig- natures to be highly domain-specific, revealing that failures in different reasoning tasks manifest as distinct computational patterns. (3) We provide evidence that these signatures are not merely correlational; by using our analysis to guide tar- geted interventions on individual transcoder features, we successfully correct the model’s faulty reasoning. Our work shows that, by scrutinizing a model’s com- putational process, we can move from simple error detection to a deeper, causal understanding of LLM reasoning. 1

## Method / Algorithms Extract

Unlike in Process Reward Modeling (PRM), where the goal is limited to judging the correctness of a reasoning step, we take the perspective of a model developer interested in debugging reasoning failures in a specific model to which they have full access. We introduce Circuit-based Reasoning Verification (CRV), a method for detecting flawed reasoning by analyzing its structural fingerprint. 3.1 DATASET CURATION AND STEP-LEVEL ANNOTATION A prerequisite for developing our method is a dataset with reliable step-level correctness labels. Furthermore, our white-box methodology imposes a critical requirement that distinguishes our data needs from prior work. Since CRV analyzes the causal computational graph that produces a rea- soning step, we must capture the full internal state of our specific model during the generation process. Consequently, existing text-only datasets such as PRM800K (Lightman et al., 2024) and REVEAL (Jacovi et al., 2024), which provide static ‘(text, label)’ pairs and are designed for training black-box verifiers, are incompatible with our mechanistic approach. We must generate and label our own model’s CoT outputs to create the necessary ‘(text, label, computational trace)’ tuples for analysis. We therefore created a new benchmark covering both controlled synthetic tasks and the real-world GSM8K dataset (Cobbe et al., 2021). Synthetic Datasets (Boolean and Arithmetic). To study reasoning failures in a controlled environ- ment, we generated two datasets. The first involves evaluating complex boolean expressions, while the second involves multi-step arithmetic problems. The motivation for these datasets is the unam- 3 ===== PAGE 4 / 28 ===== Published as a conference paper at ICLR 2026 biguous ground truth: the correctness of any step in the reasoning chain (e.g., “15 + 7 = 22”) can be verified automatically by a simple parser and evaluator. This allows us to generate a large, labeled dataset for initial training and analysis. Furthermore, these tasks are intrinsically compositional, and the complexity of samples can be fully controlled. Further details are provided in Appendix A. Step-Level Annotation for GSM8K. Annotating a real-world dataset like GSM8K is challenging. To scale, we used a semi-automated process with a stronger LLM (e.g., Llama 3.3 70B Instruct) as an expert judge. For each CoT, the judge evaluated step correctness given the full problem context. We validated these labels through manual review of a substantial subset, yielding a high-fidelity dataset for real-world reasoning. Further details are provided in Appendix A. 3...

## Experiments / Evidence Extract

We conduct a series of experiments designed to validate the central hypothesis of our work: that the attribution graphs of reasoning steps contain a rich, structural signal of their correctness. Our evaluation is structured around three primary research questions. First, we investigate whether CRV’s white-box approach significantly outperforms a comprehensive suite of gray-box and black- box baselines in verification accuracy and test its robustness to domain shifts and increasing task difficulty (RQ1). Next, we analyze our trained models to identify which specific computational structures within the graph are most predictive of failure, moving from detection to mechanistic un- derstanding (RQ2). Finally, we conduct exploratory studies to assess if these mechanistic insights can be used to perform targeted, causal interventions that correct faulty reasoning (RQ3). 4.1

## Conclusion / Discussion Extract

In this work, we introduced CRV, a white-box methodology for studying the computational struc- ture of reasoning failures. By treating attribution graphs as execution traces of latent circuits, we showed that correct and incorrect reasoning leave distinct structural fingerprints. CRV revealed that these error signatures not only enable accurate verification but are also domain-specific, with fail- ures in different reasoning tasks manifesting as distinct patterns. Moreover, targeted interventions on transcoder features demonstrated that these signatures are causally implicated, allowing us to correct faulty reasoning. Together, these findings establish CRV as a proof-of-concept for mechanistic anal- ysis, showing that shifting from opaque activations to interpretable computational structure enables a causal understanding of how and why LLMs fail to reason correctly. ETHICS STATEMENT Our research yields insights into success and failure patterns in LLM reasoning. Such knowledge could theoretically be used for malicious purposes, such as designing adversarial attacks or engineer- ing more subtle, undetectable reasoning failures. However, the computationally intensive nature of CRV, which also requires deep expertise and white-box model access, positions it as a tool for deep scientific analysis rather than a scalable method for generating exploits. The primary and intended application of our work is defensive: by providing a scientific instrument for developers to diagnose why a model fails, we aim to accelerate the development of more robust, reliable, and safer AI sys- tems. We believe the benefits of enabling a deeper, causal understanding of AI failures for safety and alignment research significantly outweigh the risks of misuse. REPRODUCIBILITY STATEMENT We are committed to the reproducibility of our work. Our newly generated datasets with step-level labels and our trained transcoders are publicly available. We provide details on our experimental setup in Section 4.1. Comprehensive details are provided throughout the Appendix, including: our dataset construction, annotation prompts, and full data statistics (Appendices A.1, A.2, and A.5); our transcoder train...

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

TBD: add short excerpts with page markers from `../texts/verifying-chain-of-thought-reasoning-via-its.txt`.
