# Reasoning emerges from constrained inference manifolds in large language models

- **Local PDF filename**: `Reasoning emerges from constrained inference.pdf`
- **Slug**: `reasoning-emerges-from-constrained-inference`
- **Pages**: 14
- **Approx Words**: 7940
- **Auto Tags**: geometry;dynamics;emergence
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.621387

## Keyword Profile

- `dimension`: 115
- `manifold`: 47
- `geometric`: 20
- `geometry`: 10
- `trajectory`: 8
- `flow`: 4
- `hidden state`: 3
- `curvature`: 2
- `faithful`: 2
- `chain of thought`: 2
- `topolog`: 1
- `causal`: 1

## Abstract / Opening Summary

Reasoning in large language models is predominantly evaluated through labeled benchmarks, conflating task performance with the quality of internal inference. Here we study reasoning as an intrinsic dynamical process by examining the evolution of internal representations during inference. We find that inference-time dynamics consistently self-organize into low-dimensional manifolds embedded within high-dimensional representation spaces. we find that such geometric compression, although pervasive, is not sufficient for stable or reliable reasoning. Instead, effective reasoning dynamics emerge within a constrained structural regime characterized by three conditions: adequate representational expressivity, spontaneous manifold compression, and preservation of non-degenerate information volume within the compressed subspace. Models outside this regime exhibit characteristic pathological inference dynamics. Based on these insights, we introduce a unified, label-free diagnostic computed solely from internal dynamics. These findings suggest that reasoning in LLMs is fundamentally governed by geometric and informational constraints, offering a complementary framework to benchmark-centric assessment. Large language models have shown remarkable im- provements in reasoning across mathematics [1, 2], science, and commonsense domains [3, 4, 5]. Yet rea- soning ability is still predominantly evaluated using labeled benchmarks and task accuracy [6, 7], implic- itly treating reasoning as an opaque input–output mapping [8, 9]. Such evaluations conflate internal inference quality with dataset alignment and prompt- ing strategies [10], offering limited insight into how reasoning is internally realized or why models with similar accuracy can differ substantially in robustness and generalization. To address this limitation, we study reasoning from an internal perspective [11, 12, 13], characterizing it as a dynamical process unfolding in representation space during inference [14]. Rather than focusing on correctness, we analyze how internal representations evolve when models are engaged by generic cognitive stimuli, independent of task-specific supervision. Across model families [15, 16, 17, 18], scales, and prompts [19], we observe a striking and consis- tent regularity: inference-time representations spon- taneously collapse onto extremely low-dimensional trajectories, despite residing in highly expressive embedding spaces. These trajectories indicate that reasoning dynamics are effectively confined to low- dimensional manifolds embedded within the ambi- ent representation ...

## Method / Algorithms Extract

The framework developed in this work applies specif- ically to inference-time reasoning dynamics in con- temporary autoregressive language models. It char- acterizes a regime in which reasoning unfolds as a low-dimensional, information-rich dynamical process embedded within a highly expressive representation space. Outside this regime, inference dynamics ex- hibit characteristic failure modes, including diffuse exploration, information starvation, or instability un- der stimulus expansion. Importantly, the admissible reasoning regime is not expected to extend indefinitely with model scale, architectural complexity, or training compute [15]. Changes that improve one structural dimension may degrade another, shifting models toward or away from the admissible regime. The framework there- fore does not predict monotonic scaling laws for rea- soning performance [5], but instead delineates a structural window within which robust reasoning dy- namics can be sustained. Whether similar regimes govern reasoning in non-autoregressive architectures, multimodal systems, or embodied agents remains an open question. Applications By grounding reasoning evaluation in internal dynam- ics rather than task outcomes, the diagnostic intro- duced here enables comparison of reasoning behavior across models, architectures, and training regimes without reliance on labeled benchmarks. This makes it applicable in settings where task-specific evaluation is limited or unavailable. More broadly, structural diagnostics provide a com- plementary perspective for monitoring and guiding model development. Instead of optimizing solely for benchmark performance, training or fine-tuning pro- cedures may be informed by constraints on inference dynamics [26], promoting regimes that support ro- bust and generalizable reasoning. Structural analysis may also help identify brittleness, characterize fail- ure modes, and assess how alignment or compression techniques affect internal computation. From this perspective, the identification of an ad- missible structural regime raises questions about how such regimes might be encouraged during model development. One possible implication, not exam- ined here, is whether training objectives that bias inference-time dynamics toward geometrically ad- missible regimes could promote healthier reasoning behavior. This does not imply a specific regulariza- tion strategy. Rather, the admissible regime provides 9 ===== PAGE 10 / 14 ===== Reasoning emerges from constrained inference manifolds in large language models a structural reference for evaluating the effects...

## Experiments / Evidence Extract

Inference-time reasoning dynamics self- organize into low-dimensional manifolds To characterize reasoning as an internal process rather than a simple input–output mapping, we analyze inference-time representation trajectories elicited by generic cognitive stimuli [19] across a range of contemporary large language models [15, 16, 17, 18]. For each stimulus, at each layer and at each inference step, we record the hidden state of the last token in the current sequence. The sequence of such states within each layer is treated as a discrete trajectory embedded in the model’s high- dimensional representation space. Across all evaluated models, we observe a striking and highly reproducible phenomenon: despite being embedded in representation spaces with thousands of dimensions, reasoning trajectories consistently and spontaneously collapse onto low-dimensional man- ifolds during inference. As shown in Figure 1, the intrinsic dimensionality [20, 21] of stimulus-induced representations decreases rapidly as inference pro- ceeds through network layers and stabilizes at values far below the ambient embedding dimension. In many cases, the intrinsic dimensionality converges to fewer than ten degrees of freedom. We further observe differences across model: weaker or earlier models tend to exhibit slower and less stable dimen- sional reduction, whereas newer-generation models converge more consistently to compact manifolds. This dimensional collapse is robust across model families and parameter scales and is not confined to specific prompts or task types. As illustrated in 2 ===== PAGE 3 / 14 ===== Reasoning emerges from constrained inference manifolds in large language models STEM Science Technology Engineering Mathematics Humanities Law Jurisprudence Formal logic Moral Disputes Social Science Econometrics Geography Macroeconomics Psychology Qwen Large Language Model Last token v Final layer DeepSeek Gemma GLM Intrinsic Dimension (ID) ID (Social Science) ID (Humanities) ID (STEM) A B C … … … Hidden Dimension vs Vocab Intrinsic Dimension Figure 2: Low-dimensional organization is robust across stimuli and decoupled from global representational capacity. A, Distribution of stimulus-induced intrinsic dimensionality (ID; Two-Nearest-Neighbor local estimator, TLE) across heterogeneous generic cognitive stimuli, demonstrating that inference-time trajectories consistently concentr...

## Conclusion / Discussion Extract

The results presented in this work motivate a shift in how reasoning in large language models is concep- tualized and evaluated. Rather than focusing exclu- sively on task-level outcomes or benchmark scores, we examine reasoning through the internal structure of inference-time dynamics. This perspective treats reasoning as a constrained dynamical process unfold- ing within a representational space [22], and invites questions not only about whether a model succeeds on a task, but about the internal regimes that make such success possible or fragile. The Discussion below situates our findings within this broader conceptual framework, clarifies what is captured by structural measures of reasoning health, delineates the regime in which the framework applies, and outlines how such structural diagnostics may be used in practice. What is being measured by reasoning health? A central contribution of this work is to clarify what is captured by internal measures of reasoning quality, as well as what is not. Rather than estimating task- specific competence or benchmark performance, the reasoning health diagnostic introduced here quanti- fies the structural organization of inference-time dy- namics. It evaluates whether internal representations evolve within a regime that supports meaningful in- termediate computation, independent of whether a particular output is correct [23, 24]. From this perspective, reasoning health character- izes how a model reasons, not what it knows or how well it performs on a given dataset. Models with sim- ilar task accuracy may operate in fundamentally dif- ferent internal regimes [25, 18], while models with 8 ===== PAGE 9 / 14 ===== Reasoning emerges from constrained inference manifolds in large language models structurally healthy inference dynamics may fail spe- cific benchmarks due to misalignment, insufficient supervision, or domain mismatch. The diagnostic therefore complements, rather than replaces, exter- nal evaluation [7] by isolating intrinsic properties of inference dynamics that are otherwise conflated with dataset effects. In this sense, reasoning health provides a measure of whether a system operates within a regime capable of su...

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

TBD: add short excerpts with page markers from `../texts/reasoning-emerges-from-constrained-inference.txt`.
