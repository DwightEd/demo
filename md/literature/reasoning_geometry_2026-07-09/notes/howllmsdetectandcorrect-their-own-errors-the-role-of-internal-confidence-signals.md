# How LLMs Detect and Correct Their Own Errors: The Role of Internal Confidence Signals

- **Local PDF filename**: `HowLLMsDetectandCorrect Their Own Errors-The Role of Internal Confidence Signals.pdf`
- **Slug**: `howllmsdetectandcorrect-their-own-errors-the-role-of-internal-confidence-signals`
- **Pages**: 35
- **Approx Words**: 16885
- **Auto Tags**: dynamics;faithfulness;uncertainty
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.615716

## Keyword Profile

- `probe`: 52
- `causal`: 39
- `patching`: 24
- `phase`: 23
- `latent`: 9
- `dimension`: 4
- `geometric`: 2
- `transition`: 2
- `hallucination`: 1
- `hidden state`: 1
- `chain of thought`: 1

## Abstract / Opening Summary

Large language models can detect their own errors and sometimes correct them without external feedback, but the underlying mechanisms remain unknown. We investigate this through the lens of second-order models of confidence from decision neuroscience. In a first- order system, confidence derives from the generation signal itself and is therefore maximal for the chosen response, precluding error detection. Second-order models posit a partially independent evaluative signal that can disagree with the committed response, providing the basis for error detection. Kumaran et al. (2026) showed that LLMs cache a confidence representation at a token immediately following the answer (i.e. post-answer newline: PANL)—that causally drives verbal confidence and dissociates from log-probabilities. Here we test whether this PANL signal extends beyond confidence to support error detection and self-correction. Here we test whether this signal supports error detection and self- correction, deriving predictions from the second-order framework. Using a verify-then- correct paradigm, we show that: (i) verbal confidence predicts error detection far beyond token log-probabilities, ruling out a first-order account; (ii) PANL activations predict error detection beyond verbal confidence itself; and (iii) PANL predicts which errors the model can correct—where all behavioural signals fail. Causal interventions confirm that PANL signals rescue error detection behavior when answer information is corrupted. All findings replicate across models (Gemma 3 27B and Qwen 2.5 7B) and tasks (TriviaQA and MNLI). These results reveal that LLMs naturally implement a second-order confidence architecture whose internal evaluative signal encodes not only whether an answer is likely wrong but whether the model has the knowledge to fix it. 1

## Method / Algorithms Extract

A.1 Models and Datasets Models. We study two instruction-tuned language models: Gemma 3 27B (google/gemma-3-27b-it; 62 layers, 5376-dimensional residual stream) and Qwen 2.5 7B (Qwen/Qwen2.5-7B-Instruct; 28 layers, 3584-dimensional residual stream). Both models were accessed via HuggingFace Transformers and run with greedy decoding (temperature = 0) throughout. Datasets. We use TriviaQA (Joshi et al., 2017) as our primary dataset, a factual question- answering benchmark requiring retrieval of real-world knowledge. After deduplication, our TriviaQA sample comprises 7,227 questions for Gemma and 3,500 for Qwen. We additionally test on the Multi-Genre Natural Language Inference (MNLI) dataset (Williams et al., 2018), a three-way classification task (entailment, contradiction, neutral), using the development set downloaded from HuggingFace (n = 9,888) and applied to Gemma only. As entailment and contradiction trials show near-ceiling verification endorsement rates with minimal variance, all MNLI analyses are restricted to neutral trials (n = 3,395). A.2

## Experiments / Evidence Extract

We study Gemma 3 27B (Team et al., 2025) as our primary model on the TriviaQA factual knowledge dataset (Joshi et al., 2017) (n = 7,227 questions), with cross-model replication on Qwen 2.5 7B (n = 3,500) and cross-task replication on MNLI (Williams et al., 2018). All answers were generated with greedy decoding (temperature = 0), ensuring that A1 represents the model’s argmax of its token distribution. Any improvement from A1 to A2 therefore cannot arise from stochastic resampling; it requires the evaluative process at PANL to access a different weighting over alternatives in which the committed answer is no longer dominant (Figure 1), enabling the model to revise toward a response it did not initially select. Full model and dataset details are in §A.1. In the simple verify-then-self-correct paradigm we employ, the model generates an answer and reports its confidence (Phase 0), is then shown its own answer and asked to verify it (“Y”/“N”; Phase 1 (verification)), and finally produces a second answer (Phase 2 (self correction); see §A.2; Figure 1). We extract residual stream activations at PANL during the verification phase and use linear probes to predict verification behaviour and second- attempt correctness. We first characterise the predictive value of behavioural signals and then turn to the activation analyses to ask whether PANL representations explain variance that behavioural signals leave unaccounted for. 3.1

## Conclusion / Discussion Extract

Kumaran et al. (2026) showed that LLMs cache an evaluative representation at PANL that drives verbal confidence and dissociates from log-probabilities. We show that this second-order signal extends to error detection and self-correction—and, critically, predicts which errors the model can correct, where all behavioural signals fail. The second-order framework from decision neuroscience (Fleming & Daw, 2017) explains why: a first-order system cannot conclude its own output is wrong, because the signal that selected the answer is by definition peaked at that answer. We hypothesize that this automatic error signal may be precisely what reasoning models have learned to act on: redeployed at intermediate commitment points within a reasoning trace to trigger backtracking (Ward et al., 2025; 9 ===== PAGE 10 / 35 ===== Preprint. Under review. Gandhi et al., 2025; Yang et al., 2025a), and potentially offering a principled source of dense intermediate reward for reasoning training. 10 ===== PAGE 11 / 35 ===== Preprint. Under review. Acknowledgements We thank Leonidas Guibas and Andrea Banino for comments on an earlier version of this manuscript. We also acknowledge the use of Gemini for assistance with coding and improving the clarity of the writing. References Amos Azaria and Tom Mitchell. The internal state of an llm knows when it’s lying. arXiv preprint arXiv:2304.13734, 2023. Leonardo Bertolazzi, Philipp Mondorf, Barbara Plank, and Raffaella Bernardi. The validation gap: A mechanistic analysis of how language models compute arithmetic but fail to validate it. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 29375–29412, 2025. Malcolm W Brown and John P Aggleton. Recognition memory: what are the roles of the perirhinal cortex and hippocampus? Nature Reviews Neuroscience, 2(1):51–61, 2001. Lennart B¨urger, Fred A Hamprecht, and Boaz Nadler. Truth is universal: Robust detection of lies in llms. Advances in Neural Information Processing Systems, 37:138393–138431, 2024. Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. Discovering latent knowledge in language models without supervision. arXiv preprint arXiv:2212.03827,...

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

TBD: add short excerpts with page markers from `../texts/howllmsdetectandcorrect-their-own-errors-the-role-of-internal-confidence-signals.txt`.
