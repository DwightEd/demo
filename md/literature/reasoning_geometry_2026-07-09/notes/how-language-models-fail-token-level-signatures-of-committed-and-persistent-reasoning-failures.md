# How Language Models Fail: Token-Level Signatures of Committed and Persistent Reasoning Failures

- **Local PDF filename**: `How Language Models Fail-Token-Level Signatures of Committed and Persistent Reasoning Failures.pdf`
- **Slug**: `how-language-models-fail-token-level-signatures-of-committed-and-persistent-reasoning-failures`
- **Pages**: 15
- **Approx Words**: 10115
- **Auto Tags**: dynamics;faithfulness;uncertainty;emergence
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.614933

## Keyword Profile

- `chain of thought`: 12
- `entropy`: 8
- `faithful`: 7
- `trajectory`: 3
- `hallucination`: 1
- `hidden state`: 1
- `causal`: 1

## Abstract / Opening Summary

Failures in language model reasoning emerge through distinct processes that leave identifi- able signatures in the reasoning trace. We char- acterize these failures using token-level uncer- tainty signals, finding they arise through two empirically distinguishable processes. The first is committed failure, in which a model locks onto an incorrect reasoning path early in its trace. A central diagnostic signature is the com- mitment point, beyond which considering addi- tional tokens hurt rather than help failure detec- tion. In the second, persistent uncertainty, un- certainty instead accumulates throughout, and the full trace is needed to best distinguish fail- ing from successful completions. These signa- tures reproduce across 23 model-dataset config- urations, with the framework’s falsifiable pre- dictions holding in 20 of 23 cases, well above chance across both failure modes. Finally, we demonstrate our failure mode framework has di- rect implications for self-consistency, identify- ing when uncertainty signals complement it and when it can be selectively skipped. These re- sults offer a foundation for understanding when LLM reasoning failures become detectable and for adapting detection strategies accordingly. 1

## Method / Algorithms Extract

A language model’s chain-of-thought reveals how it produced its final answer and, in well-calibrated cases, should be informative of whether that answer is incorrect (Figure 1). We analyze the token-level uncertainty signals across reasoning traces to char- acterize the structure of model failures. 3.1 Failure Modes in LLM Reasoning We define model failure as any trace in which a model’s final extracted answer is incorrect. If the structure of a model’s reasoning determines even- tual failure, then the progression of token-level signals across that trace should be characteristic of how and when failure occurs. We propose that this progression takes one of two qualitatively different forms. In the first failure mode, committed failure, the model locks onto an incorrect reasoning path early in its trace. Failure becomes apparent early in the model’s reasoning, and its uncertainty signals are most informative over a prefix of the trace rather than the full se- quence. In the second, persistent uncertainty, the model never commits to a reasoning path. Uncer- tainty builds monotonically throughout the trace, and a complete reasoning path is required to dis- tinguish failed from successful traces. These two modes produce qualitatively different signatures in how uncertainty progresses across a reasoning trace, which we will formalize and test empirically. 3.2 Commitment Point If a model locks onto a reasoning path early, there likely is a token position where this is observable. We define this position as the commitment point: the point in a reasoning trace at which the uncer- tainty signals are most informative of model failure. Beyond the commitment point, the model has already selected a reasoning path, and subsequent uncertainty is downstream noise rather than signal about the eventual outcome. In the persistent un- certainty regime, no such commitment point exists as predictive power increases monotonically, and the full trace remains more informative than any prefix. 3.3 Uncertainty Features If a model has locked onto a reasoning path, its to- ken distribution should reflect its diminished uncer- tainty as the model is no longer exploring multiple paths. To reveal these failure patterns, we compute the following signals over prefixes of the reasoning trace, which we formalize below as early windows. Let p(t) = (p(t) 1 , p(t) 2 , . . .) denote the token prob- ability distribution at position t, with p(t) (1) ≥p(t) (2) ≥ · · · be the sorted probabilities. For a reasoning trace of length L, we define the early window WT = {1, . . . , min(T, L)} and compute ...

## Experiments / Evidence Extract

We test whether the two failure modes manifest empirically across a range of models, datasets and 4 ===== PAGE 5 / 15 ===== task difficulty levels. Our framework operates en- tirely on the externalized chain-of-thought trace and requires only token-level log probabilities; no access to internal model representations is needed. Our code is publicly available.1 4.1 Models and Datasets We evaluate models spanning a range of sizes, families and architectures: Qwen3.5-2B, Qwen3.5- 9B, Qwen3.5-27B, Qwen3.5-122B-A10B (Team, 2026), Llama3.1-8B-Instruct (Grattafiori et al., 2024), GPT-OSS-20B (Agarwal et al., 2025), Gemma4-31B, GPT-4o (Achiam et al., 2023), Gemini-2.5Pro (Comanici et al., 2025). We in- clude both dense models, mixture of experts, open- source and frontier models in order to capture a va- riety of patterns. These are evaluated on five bench- marks spanning mathematics, scientific, logical and coding domains: GSM8K (1319 test questions of grade-school math; Cobbe et al. 2021), MATH-500 (500 competition-level math problems represen- tative of the full MATH benchmark; Hendrycks et al. 2021, Lightman et al. 2024), GPQA Diamond (198 multiple-choice questions on graduate-level biology, chemistry and physics; Rein et al. 2024), and LiveCodeBench (451 applicable coding chal- lenges; Jain et al. 2025). We additionally evaluated AR-LSAT (230 questions from the Law School Admissions Test; Zhong et al. 2021) but every con- figuration we ran fell outside the applicability band; these results are reported in Table 3 in the appendix, for transparency and excluded from the pool as a scope decision. These datasets were selected to span across domains and a range of difficulty lev- els relative to model capability, with the intention to generalize the failure framework. Failure rates below 15% or above 60% paired with an AUROC < 0.55 render the framework inapplicable. We additionally exclude configurations whose prefinal- stripped trace contains too few failures to support reliable analysis (typically fewer than ∼10 failures in the prefinal-valid subset), where the analysis pipeline falls back to regular features. All experiments use a temperature of 0.6, which is consistent with standard LLM evaluation practice (Renze and Guven, 2024) and balances exploration and exploitation. For open-weight models, we re- trieve the top 200 log probabilities per token (cap- ture aro...

## Conclusion / Discussion Extract

We introduced a two-mode framework characteriz- ing how language-model reasoning failures mani- fest in chain-of-thought traces, requiring only log probabilities from a single completion. Across 23 configurations spanning five model families and four reasoning domains, the framework’s bidirectional prediction holds in 20 of 23 cases (sign test on committed configurations: 14/14, p = 6.1×10−5; pooled ˆ∆= +0.013, 95% CI [+0.005, +0.020]). The failure-mode classification has direct deploy- ment implications: in the committed regime we can skip self-consistency on the top 30% most- confident inputs without sacrificing failure recall, and across both regimes, combining uncertainty features with the agreement rate yields a consistent positive lift. Failure detection strategies should be adapted to failure mode rather than applied uni- formly. 8 ===== PAGE 9 / 15 ===== 7 Limitations Our framework requires failure rates within a workable applicability band; configurations at the extremes do not produce reliable PR-AUC esti- mates and are excluded from the pool. We use a single completion per question, so commitment- point identification is sensitive to the sampled trace. The commitment point T ∗is identified at the granularity of six fixed window sizes ({128, 256, 400, 512, 1024, 2048}) and represents a window range rather than an exact token posi- tion; a finer-grained sweep between adjacent win- dows could refine its localization. Closed-API constraints, where GPT-4o and Gemini-2.5Pro ex- pose only their top-20 log probabilities, limit the reliability of mean-based features that depend on tail probability mass; max-based features remain valid under truncation. Our self-consistency evalua- tion is restricted to three configurations on a single benchmark (GPQA Diamond), so consistency of the complementarity result across broader settings remains to be validated. Finally, the framework op- erates on visible chain-of-thought traces and does not address whether these traces faithfully reflect internal model computation; this question is or- thogonal to our empirical claims, which concern structure observable in the visible trace. References Josh Achiam, Steven Adler, Sandh...

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

TBD: add short excerpts with page markers from `../texts/how-language-models-fail-token-level-signatures-of-committed-and-persistent-reasoning-failures.txt`.
