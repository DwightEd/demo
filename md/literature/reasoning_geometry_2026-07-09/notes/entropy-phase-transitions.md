# When Do LLMs Reason? A Dynamical Systems View via Entropy Phase Transitions

- **Local PDF filename**: `Entropy Phase Transitions.pdf`
- **Slug**: `entropy-phase-transitions`
- **Pages**: 36
- **Approx Words**: 19982
- **Auto Tags**: geometry;dynamics;faithfulness;uncertainty;hallucination;emergence
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.610166

## Keyword Profile

- `entropy`: 133
- `manifold`: 24
- `trajectory`: 22
- `chain of thought`: 12
- `dimension`: 12
- `phase`: 9
- `transition`: 7
- `probe`: 7
- `hallucination`: 2
- `latent`: 2
- `causal`: 2
- `geometric`: 1

## Abstract / Opening Summary

Chain-of-thought (CoT) reasoning has become the default strategy for en- hancing LLM capabilities, yet its application raises a fundamental question: when is explicit reasoning actually beneficial? Empirical evidence reveals a striking paradox: CoT often provides marginal or even negative gains on factual and open-ended tasks while multiplying token consumption. In this work, we show that LLM reasoning is not a static property of tasks or mod- els, but a dynamic decoding state that emerges during generation. Through systematic analysis, we find early-stage entropy dynamics provide a re- liable signal of this state: tasks benefiting from CoT exhibit consistent entropy reduction, while others display unstable or increasing patterns. This behavior can be interpreted as a phase-transition-like shift from a high- entropy exploratory regime to a low-entropy structured reasoning regime. Based on these insights, we propose EDRM (Entropy Dynamics-based Rea- soning Manifold), a lightweight and training-free routing framework that leverages early decoding entropy to adaptively select inference strategies. EDRM embeds entropy trajectories into a compact and interpretable mani- fold representation, enabling both zero-shot deployment and fine-grained instance-level adaptation. Across 15 benchmarks and 4 LLMs of varying scales and architectures, EDRM consistently outperforms static baselines. At the dataset level, EDRM achieves 41–55% token reduction while improv- ing accuracy with as few as 50 calibration samples. At the instance level, it further improves accuracy by up to 4.7% while maintaining 27–45% token savings. These results suggest that reasoning should be invoked selectively rather than by default, and demonstrate the effectiveness of entropy-driven decoding control for efficient and adaptive LLM inference. 1

## Method / Algorithms Extract

In this section, we first present preliminary concepts and then our observations and insights about LLM decoding dynamics and their relationship to reasoning utility in our exploring investigation. Finally, we introduce EDRM, a novel framework that leverages early-stage en- tropy dynamics to adaptively route inference strategies for efficient and effective reasoning. 3.1 Preliminaries Decoding paradigms. We consider three basic decoding paradigms under identical task descriptions: 3 ===== PAGE 4 / 36 ===== AI Model TF May, 2026 Figure 3: Entropy Trajectories: average token-level entropy over the first N tokens under Standard probing. Tasks of high CoT gain show decreasing trend while low ones show oscillation or increase. • Direct: the model is instructed to output the final answer directly without explicit reasoning steps. We employ this by prompting the model to answer directly without explanation, while for thinking-oriented models we need to close the think mode to prevent over-reasoning additional. This paradigm is efficient but may fail on tasks requiring multi-step decomposition. • Standard: the model is instructed to answer the query with merely the query and the minimal prompting required to elicit its intrinsic reasoning behavior, while for thinking- oriented models we close the think mode still. This paradigm allows the model to dynamically determine its reasoning strategy based on the query, without forcing explicit CoT or suppressing reasoning entirely. And we utlize this paradigm for subsequent probing and manifold construction, as it best reflects the model’s natural decoding dynamics without heavy intervention. • CoT: the model is instructed to fullfill explicit step-by-step reasoning with CoT prompts and think mode on (if available). This paradigm encourages the model to decompose complex problems into intermediate steps, served as the heavyest reasoning intensity approach. While potentially improving accuracy on complex tasks, this approach incurs substantial token overhead and may degrade performance on some tasks. These paradigms represent a spectrum from minimal intervention (Direct) to mandatory reasoning (CoT), with Standard occupying an intermediate position that allows the model’s intrinsic behavior to manifest. For more details about the prompting templates and settings, please refer to Appendix B.3. Autoregressive generation and token-level entropy. Consider an autoregressive LLM that generates tokens sequentially. At each decoding step i, the model produces a probability distribution pi over the vocabulary V conditioned on the...

## Experiments / Evidence Extract

4.1 Experimental Setup We evaluate EDRM on 15 benchmarks spanning diverse reasoning types and difficulty levels, with 4 different LLMs to validate cross-model generalization. Implementation details are as follows. Datasets. We categorize the 15 benchmarks into 4 groups. (1) Mathematical reason- ing: gsm8k (Cobbe et al., 2021), MultiArith (Roy & Roth, 2016), and bbh (Srivastava et al., 2022). (2) Commonsense & knowledge reasoning: commonsenseqa (Speer et al., 2016), strategyqa (Geva et al., 2021), piqa (Bisk et al., 2019), siqa (Sap et al., 2019), and MuSR (Sprague et al., 2024a). (3) Scientific reasoning: arc challenge (Clark et al., 2018), arc easy (Clark et al., 2018), and gpqa (Rein et al., 2023). (4) Formal logic: FOLIO (Han et al., 2024), ContextHub abductive (Hua et al., 2025), ContextHub deductive (Hua et al., 2025), and lsat (Zhong et al., 2023). Models. We test 4 LLMs to validate generalization. Base models: (1) Llama-3.2-3B-Instruct (Grattafiori et al., 2024), (2) Llama-3.1-8B-Instruct (Grattafiori et al., 2024), (3) Qwen2.5-7B-Instruct (Hui et al., 2024), representing diverse scales (3B–8B) and families. Reasoning-enhanced model: (4) Qwen3-4B-Instruct-2507 (Yang et al., 2025), trained explic- itly for chain-of-thought generation with built-in think mode. Unlike base models, it is prone to over-reasoning, which makes it an ideal candidate for validating EDRM’s adaptive and robust routing capabilities under adversarial circumstances. Baselines. We compare 9 decoding strategies across two categories: Static regimes apply a fixed decoding mode to all instances: (1) Direct (no reasoning), (2) Standard (minimal prompting), and (3) CoT (always-on chain-of-thought). Adaptive routing dynamically selects among regimes: (4) Token-Signature, a two-way routing baseline most similar to EDRM; and EDRM variants with two granularities—(5–6) EDRM-Global-E/C for dataset- level routing and (7–8) EDRM-Inst-E/C for instance-level routing, where “E” uses empirical thresholds (SH,th=32 for base, 10 for reasoning models) and “C” uses cross-dataset calibra- tion; (9) EDRM-MLP employs a learned instance-level router. All stochastic variants are evaluated over 8 random seeds with mean and variance reported. Evaluation metrics. We report accuracy and average token consumption to capture the performance–efficiency trade-off. For multiple trials, we report mean and variance to ...

## Conclusion / Discussion Extract

This work revisits LLM reasoning from a dynamical systems perspective and establishes that the utility of explicit reasoning emerges from decoding-time entropy dynamics rather than being a fixed property of models or tasks. Our systematic analysis reveals that successful rea- soning corresponds to a phase-transition-like shift from high-entropy exploratory regimes to low-entropy structured convergence, while ineffective reasoning remains trapped in oscillatory or divergent dynamics. These distinct patterns naturally organize into separable regions in a compact three-dimensional entropy manifold (SH, Vsp, avnr), providing both theoretical insight and a practical signal for adaptive control. Based on this foundation, we introduce EDRM, a lightweight and training-free framework that embeds early-stage en- tropy dynamics into a reasoning manifold for instance-adaptive inference routing. Extensive experiments across 15 benchmarks and 4 LLMs demonstrate EDRM’s effectiveness: at the dataset level, it achieves 41–55% token reduction while improving accuracy with minimal calibration; at the instance level, it further improves accuracy by up to 4.7% while main- taining 27–45% token savings. Our results suggest that reasoning is better understood as a controllable decoding state that should be invoked selectively based on real-time generation dynamics rather than static task categories. This perspective opens new avenues for efficient LLM inference, particularly in resource-constrained and latency-sensitive applications. Limitations and Future Work. EDRM still requires a short probing stage, introducing moderate overhead compared to pure direct decoding. Our current experiments are also limited to open-source text-based models in the 3B–8B range; extending the framework to larger-scale, API-only, and multimodal systems remains important future work. In addition, we view EDRM as an initial step toward adaptive reasoning control in autonomous agents. Integrating entropy-dynamic routing into production-scale agent frameworks such as OpenCLAW may further improve efficiency and robustness in long-horizon multi-turn reasoning. References Yonatan Bisk, Rowan Zellers, Ronan Le Bra...

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

TBD: add short excerpts with page markers from `../texts/entropy-phase-transitions.txt`.
