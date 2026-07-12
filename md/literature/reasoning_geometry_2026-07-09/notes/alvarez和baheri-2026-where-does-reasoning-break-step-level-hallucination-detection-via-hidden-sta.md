# Where Does Reasoning Break? Step-Level Hallucination Detection via Hidden-State Transport Geometry

- **Local PDF filename**: `Alvarez和Baheri - 2026 - Where Does Reasoning Break Step-Level Hallucination Detection via Hidden-State Transport Geometry.pdf`
- **Slug**: `alvarez和baheri-2026-where-does-reasoning-break-step-level-hallucination-detection-via-hidden-sta`
- **Pages**: 14
- **Approx Words**: 8125
- **Auto Tags**: geometry;dynamics;uncertainty;hallucination;step-level
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.598272

## Keyword Profile

- `transition`: 25
- `hallucination`: 25
- `geometry`: 24
- `hidden state`: 16
- `geometric`: 13
- `trajectory`: 13
- `entropy`: 11
- `manifold`: 9
- `probe`: 7
- `dimension`: 5
- `latent`: 2
- `chain of thought`: 1

## Abstract / Opening Summary

Large language models hallucinate during multi- step reasoning, but most existing detectors operate at the trace level: they assign one confidence score to a full output, fail to localize the first error, and often require multiple sampled completions. We frame hallucination instead as a property of the hidden-state trajectory produced during a single forward pass. Correct reasoning moves through a stable manifold of locally coherent transitions; a first error appears as a localized excursion in transport cost away from this manifold. We op- erationalize this view with a label-conditioned teacher that builds a trace-specific contrastive PCA lens and scores each step with seven geomet- ric transition features, and a deployable BiLSTM student distilled from the teacher that operates on raw hidden states without inference-time la- bels. We prove that contrastive PCA is the opti- mal projection for a transport-separation objec- tive between first-error and correct states, and that single-pass first-error localization holds whenever the first error creates a positive transport margin over preceding correct transitions. On Process- Bench, PRM800K, HaluEval, and TruthfulQA, both models outperform entropy-based, probing- based, and attention-based baselines in-domain; the teacher transfers stably across language mod- els and datasets, while the student collapses under shift, a gap our distillation theory predicts. These results recast step-level hallucination detection as a problem of trajectory dynamics and identify the central obstacle to deployment: preserving the contrastive transport margin under distribution shift. 1Rochester Institute of Technology, Rochester, NY, USA. Cor- respondence to: Tyler Alvarez <tma9531@rit.edu>, Ali Baheri <akbeme@rit.edu>. Preprint. May 14, 2026.

## Method / Algorithms Extract

ProcessBench PRM800K HaluEval TruthfulQA Teacher (non-deployable) 91.0 98.5 94.0 96.0 Student (deployable) 75.0 99.8 88.4 96.5 TL-Entropy 57.1 54.5 50.8 64.4 TL-Perplexity 51.2 45.8 48.4 67.1 Linear Probe 67.8 91.3 78.6 90.0 LLM-Check (attention) 61.9 48.0 55.7 69.8 Table 2. First-error detection accuracy across benchmarks. Best results per column are in bold.

## Experiments / Evidence Extract

Experiments We evaluate GeoReason on two tasks: step-level hallucina- tion detection and first-error localization. Step-level detec- tion is measured with AUROC, while first-error localization is measured by the accuracy of the first step whose score crosses a validation-selected threshold. The main in-domain results are reported in Table 1 and Table 2. Benchmarks and preprocessing. We use ProcessBench, PRM800K, HaluEval, and TruthfulQA because they cover process-level mathematical reasoning, annotated solution steps, generated hallucinations, and factual truthfulness. Each example is converted to a single ordered trace. When a benchmark provides step boundaries, we preserve them; otherwise, we split generated text at newline and sentence- level delimiters and discard empty fragments. Step labels are mapped to binary correctness. For localization, the first annotated incorrect step is treated as the first-error index and all later steps are evaluated as post-error states, matching the objective in Eq. (2). Splits are prompt-level and stratified by dataset and error presence, so no reasoning trace appears in more than one split. Hidden-state extraction and model settings. The cross- model experiments use one representative instruction-tuned model from each family: Qwen, Llama, and Mistral. For every generated trace, we run a single forward pass, extract the residual-stream hidden states from the final transformer block before the language-model head, and mean-pool to- kens within each step as in Eq. (1). Unless otherwise stated, the cPCA lens uses rank k = 16, contrastive penalty α = 1, post-error weight ρ = 0.25, and smoothing window w = 3. Thresholds for first-error localization are tuned only on the validation split and then frozen for test evaluation. Training details and baselines. The teacher is a two-layer MLP over the feature block in Eq. (7). The student is a two- layer BiLSTM with a step-classification head and a training- only auxiliary head for feature distillation. Both models are trained with AdamW, early stopping on validation AUROC, and prompt-level mini-batches. Baselines are evaluated under the same splits and hidden-state extraction protocol: TL-Entropy and TL-Perplexity use token-level likelihood statistics, Linear Probe trains a linear classifier on pooled step representations, and LLM-Check uses attention-derived scores. The teacher shou...

## Conclusion / Discussion Extract

Conclusion GeoReason frames step-level hallucination detection as hidden-state trajectory geometry: a label-conditioned teacher exposes the geometric signal of a first error via trace- specific cPCA, and a deployable student distills this signal for single-pass detection from raw hidden states. We prove that cPCA is the optimal lens under a transport-separation objective (Theorem 3.1), that localization holds whenever a transport margin exists (Theorem 3.2), and that teacher- student agreement reduces to margin preservation (Proposi- tion 3.3). The teacher transfers across models and datasets while the student does not, identifying margin preservation under shift, rather than detection of the geometric signal, as the central deployment obstacle. 8 ===== PAGE 9 / 14 ===== Where Does Reasoning Break? Hidden-State Transport Geometry References [1] Abid, A., Zhang, M. J., Bagaria, V. K., and Zou, J. Exploring patterns enriched in a dataset with contrastive principal component analysis. Nature Communications, 9(1):2134, 2018. doi: 10.1038/ s41467-018-04608-8. Article number 2134. [2] Amiri Shahbazi, M. and Baheri, A. Geometry-aware uncertainty quantification via conformal prediction on manifolds. arXiv preprint arXiv:2602.16015, 2026. doi: 10.48550/arXiv.2602.16015. [3] Azaria, A. and Mitchell, T. The internal state of an LLM knows when it’s lying. In Findings of the Asso- ciation for Computational Linguistics: EMNLP 2023, pp. 967–976, Singapore, 2023. [4] Baheri, A. Logic-guided vector fields for con- strained generative modeling. arXiv preprint arXiv:2602.02009, 2026. doi: 10.48550/arXiv.2602. 02009. [5] Baheri, A. and Alm, C. O. LLMs-augmented contex- tual bandit. In NeurIPS 2023 Workshop on Foundation Models for Decision Making, 2023. FMDM@NeurIPS 2023. [6] Baheri, A. and Amiri Shahbazi, M. Conformal predic- tion across scales: Finite-sample coverage with hierar- chical efficiency. Results in Applied Mathematics, 26: 100589, 2025. doi: 10.1016/j.rinam.2025.100589. [7] Baheri, A. and Wei, P. Multi-fidelity temporal reason- ing: A stratified logic for cross-scale system spec- ifications. Logics, 3(2):5, 2025. doi: 10.3390/ logics3020005. [8] Burns, C., Ye, H., Klei...

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

TBD: add short excerpts with page markers from `../texts/alvarez和baheri-2026-where-does-reasoning-break-step-level-hallucination-detection-via-hidden-sta.txt`.
