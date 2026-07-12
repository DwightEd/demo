# Effective Reasoning Chains Reduce Intrinsic Dimensionality

- **Local PDF filename**: `Effective Reasoning Chains Reduce Intrinsic Dimensionality.pdf`
- **Slug**: `effective-reasoning-chains-reduce-intrinsic-dimensionality`
- **Pages**: 20
- **Approx Words**: 11820
- **Auto Tags**: faithfulness
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.609070

## Keyword Profile

- `dimension`: 114
- `geometric`: 11
- `chain of thought`: 10
- `trajectory`: 3
- `manifold`: 1

## Abstract / Opening Summary

Chain-of-thought (CoT) reasoning and its variants have substantially improved the performance of language models on complex reasoning tasks, yet the precise mechanisms by which different strate- gies facilitate generalization remain poorly un- derstood. While current explanations often point to increased test-time computation or structural guidance, establishing a consistent, quantifiable link between these factors and generalization re- mains challenging. In this work, we identify in- trinsic dimensionality as a quantitative measure for characterizing the effectiveness of reasoning chains. Intrinsic dimensionality quantifies the minimum number of model dimensions needed to reach a given accuracy threshold on a given task. By keeping the model architecture fixed and varying the task formulation through different rea- soning strategies, we demonstrate that effective reasoning strategies consistently reduce the intrin- sic dimensionality of the task. Validating this on GSM8K with Gemma-3 1B and 4B, we observe a strong inverse correlation between the intrinsic dimensionality of a reasoning strategy and its gen- eralization performance on both in-distribution and out-of-distribution data. Our findings suggest that effective reasoning chains facilitate learning by better compressing the task using fewer pa- rameters, offering a new quantitative metric for analyzing reasoning processes.

## Method / Algorithms Extract

methods (Ze- likman et al., 2022; Chung et al., 2024) – has substantially improved the performance of large language models (LLMs) on reasoning tasks by generating textual rationales before 1UNC Chapel Hill 2Google DeepMind. *Work partially done during an internship at Google DeepMind. Correspondence to: Archiki Prasad <archiki@cs.unc.edu>. Preprint. February 11, 2026. final answers. Subsequent work has proposed numerous variations with different stylistic and strategic features, in- cluding code-based solutions (Gao et al., 2023; Chen et al., 2023), decomposition strategies (Zhou et al., 2023; Khot et al., 2023; Wang et al., 2023b), and extended reasoning with verification loops (Snell et al., 2024; Muennighoff et al., 2025). These variations represent different ways of communicating problem-solving strategies and structuring solutions – analogous to how humans adapt their commu- nication style to their interlocutor in dialogue (Pickering & Garrod, 2004; Giles et al., 1991). Empirical evidence shows different reasoning strategies yield varying performance across tasks (Zhou et al., 2024), consistent with the intuition that different solution approaches suit different problems or learners. Further, not all problems benefit from generating rationales prior to the answer (Sprague et al., 2025). This motivates an important research question: when and why is reasoning effective, and given different reasoning strategies, which is most effective for improving model per- formance? Existing explanations in prior work suffer from notable limitations. First, qualitative hypotheses about the importance of “structure” or relevance of a reasoning chains are not quantifiable (Wang et al., 2023a; Li et al., 2025). Consequently, these hypotheses are subject to interpreta- tion, limiting both their predictive capacity and the ability to offer a theoretically grounded explanation for what makes reasoning effective. On the other hand, prevalent quantita- tive measures are often associated with conflicting evidence. For example, the relationship between the length of reason- ing trajectories and the subsequent increased inference-time computational capacity remains unclear; while some works find clear gains (Muennighoff et al., 2025; Li et al., 2025), other work reports that shorter chains can be more effec- tive and that continuing to extend reasoning (e.g., via “wait” tokens) can yield degradation in performance (Wu et al., 2025; Marjanovi´c et al., 2025). Current approaches such as process reward models or correctness-based classifiers also require subjective specificati...

## Experiments / Evidence Extract

experiments to rigorously determine the optimal training duration and learning rate. We extended training runs up to 15,000 steps to empirically identify the point of convergence, observing that training accuracy and loss consistently plateaued well before our selected limits (8,000 steps for 1B and 6,000 steps for 4B). Additionally, we performed a comprehensive learning rate sweep over a logarithmic scale ranging from 1 × 10−2 to 1 × 10−6 (evaluating intermediate steps such as 1 × 10−2, 5 × 10−3, 1 × 10−3, . . . , 1 × 10−6). The final learning rates reported in the main text were selected based on the optimal balance of training stability and validation performance observed during this sweep. B. Size of Training and Test Splits Evaluation Splits. We evaluate all models on six test splits spanning both in-distribution (ID) and out-of-distribution (OOD) settings. The in-distribution evaluation uses the GSM8K test set, which contains 1.32K instances (Cobbe et al., 2021).1 Out-of-distribution evaluations include several GSM-based variants designed to stress different generalization axes: (i) GSM Symbolic (Main) (Mirzadeh et al., 2025),2 consisting of 5K instances generated from distinct symbolic template variations; (ii) GSM Symbolic P1 (Mirzadeh et al., 2025),3 a higher-difficulty symbolic split with 5K instances; and (iii) GSM Symbolic P2 (Mirzadeh et al., 2025),4 the most challenging symbolic split, containing 2.5K instances. We additionally evaluate on GSM-IC (Shi et al., 2023),5 for which we sample 5K instances from the m-step dataset augmented with irrelevant contextual information, and on GSM-Hard (Gao et al., 2023),6 which contains 1.32K instances featuring more challenging arithmetic. Together, these splits enable a systematic assessment of generalization across symbolic structure, difficulty, and robustness to distractors. 1Source: https://huggingface.co/datasets/openai/gsm8k/viewer/main/test 2Source: https://huggingface.co/datasets/apple/GSM-Symbolic/ 3Source: https://huggingface.co/datasets/apple/GSM-Symbolic/viewer/p1 4Source: https://huggingface.co/datasets/apple/GSM-Symbolic/viewer/p2 5Source: https://github.com/google-research-datasets/GSM-IC/blob/main/GSM-IC_mstep.json 6Source: https://huggingface.co/datasets/reasoning-machines/gsm-hard 12 ===== PAGE 13 / 20 ===== Effective Reasoning Chains Reduce Intrinsic Dimensionality Training Splits across...

## Conclusion / Discussion Extract

conclusions drawn from intrinsic dimen- sions are generally robust to the exact choice of threshold, holding across a wide range of thresholds. 3. Experimental Setup Datasets. We use the training split of the well-studied GSM8K dataset (Cobbe et al., 2021) comprising grade- school level math word problems. To measure models’ abilities at solving word problems in general, we evaluate the trained models on the (i) in-domain test split of GSM8K, as well as several stress test sets that measure out-of-domain generalization (ii) GSM-Symbolic (Mirzadeh et al., 2025), (iii) GSM-IC (Shi et al., 2023), and (iv) GSM-Hard (Gao et al., 2023). Mirzadeh et al. (2025) propose GSM-Symbolic to test robustness of models on a diverse set of questions sampled from symbolic perturbations to the question’s phras- 3 ===== PAGE 4 / 20 ===== Effective Reasoning Chains Reduce Intrinsic Dimensionality ing and varying difficulty via three different splits. Shi et al. (2023) find that performance of models on math word prob- lems is diminished in the presence of irrelevant sentences in the question which have no bearing on the solution. Finally, Gao et al. (2023) measure the numerical robustness and the ability to solve word problems involving more complex arithmetic. We use the test split of the GSM8K dataset to measure the in-distribution (ID) performance, and report the geometric mean of the 5 stress test sets as the out-of- distribution (OOD) performance. The overall performance is computed as the geometric mean across all the 6 test splits. We enumerate the size of various test splits in Appendix B. Reasoning Strategies. We evaluate intrinsic dimension- ality across a diverse set of reasoning strategies that vary in length, structure, and generation method. Our simplest baselines are No CoT, which outputs a direct answer with- out intermediate reasoning (Sprague et al., 2025), and No CoT with extra tokens, which appends filler text to isolate the effect of inference-time computation from reasoning quality. Using Gemma-3 27B, we generate three natural- language CoT variants: Very Short CoT, prompted for concise, equation-style reasoning (Nye et al., 2022); Short CoT, restricted to brief...

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

TBD: add short excerpts with page markers from `../texts/effective-reasoning-chains-reduce-intrinsic-dimensionality.txt`.
