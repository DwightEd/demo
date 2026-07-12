# Literature Synthesis: Reasoning Geometry, Manifolds, and Faithful CoT

Created: 2026-07-09

Local corpus: `C:/Users/613/Desktop/papers/推理`

Record folder: `research/constrained_manifolds/demo/md/literature/reasoning_geometry_2026-07-09`

## One-Sentence Landscape

The current literature has already moved past static hidden-state probes: the active frontier frames reasoning as a dynamical trajectory on compressed/structured representation manifolds, with failure detected through entropy dynamics, geometric transport, topological signatures, causal feature graphs, and faithfulness interventions; therefore our new contribution cannot be another scalar geometry score, and must instead isolate a failure mode that prior geometry papers do not causally explain.

## What We Have Already Recorded

- 38 local PDFs were enumerated and fully text-extracted.
- Extracted text lives in `texts/`.
- Per-paper note shells live in `notes/`.
- Local inventory lives in `tables/local_paper_inventory.csv`.
- Auto-extracted sections live in `LOCAL_EXTRACTED_SECTIONS.md` and `tables/local_paper_extracted_sections.csv`.
- External arXiv API search is recorded in `search/arxiv_api_reasoning_geometry_search.md`.
- Agent notes are saved under `agent_notes/`.

## Core Theme Map

### 1. Hidden-State Geometry And Reasoning Trajectories

The strongest local and external papers in this cluster are:

- `GeoFaith`: faithful/unfaithful CoT through latent geometry plus entropy dynamics.
- `Where Does Reasoning Break?`: first error as hidden-state transport excursion from a local transition manifold.
- `LLM Reasoning as Trajectories`: step-specific subspaces and correct/wrong late divergence.
- `Truth as a Trajectory`: layer-wise displacement trajectories outperform static probes.
- `The Spectral Geometry of Thought`: spectral phase-transition signatures in hidden activations.
- `Reasoning Models Don't Just Think Longer, They Move Differently`: raw trajectory geometry is heavily confounded by length; length-corrected dynamics matter.
- `Beyond Scalars / TRACED`: geometric progress and curvature/stability instead of scalar probability.

Implication: our previous `spread`, `κ`, and transition-z scores are too close to already-covered trajectory geometry unless we add a genuinely different causal or counterfactual object.

### 2. Manifold / Riemannian / Topological Views

Important papers:

- `Reasoning emerges from constrained inference manifolds`: effective reasoning requires representational expressivity, manifold compression, and non-degenerate information volume.
- `Lines of Thought`: layer trajectories cluster along low-dimensional non-Euclidean manifolds.
- `Latent Semantic Manifolds`: Fisher-information Riemannian metric, Voronoi token projection, expressibility gap.
- `Hidden Holes` and `Persistent Topological Features`: persistent homology / zigzag persistence for LLM representation topology.
- `The Shape of Reasoning`: topological analysis of explicit reasoning traces.

Implication: “we use manifold geometry” is not enough. The gap is not the existence of manifolds, but which manifold property is mechanistically tied to reasoning failure and remains valid after length/domain controls.

### 3. Entropy Dynamics And Phase Transitions

Important papers:

- `EDIS`: entropy bursts and peak-valley rebounds diagnose reasoning failures.
- `When Do LLMs Reason?`: early entropy dynamics route Direct vs CoT; reasoning as entropy phase transition.
- `Entropy Trajectory Shape`: monotone answer entropy decrease predicts reasoning reliability.
- `Tracing Uncertainty`: uncertainty profile shape predicts final correctness.
- `Logical Phase Transitions` and `3-SAT Phase Transition`: benchmark-level phase transition under reasoning hardness.

Implication: mean entropy is obsolete as a main signal; if using uncertainty at all, it must be a time-shape or phase-transition object and must explicitly handle committed low-entropy wrong answers.

### 4. Faithful / Unfaithful Chain-of-Thought

Important papers:

- `Faithful Chain-of-Thought Reasoning`.
- `Measuring Faithfulness in Chain-of-Thought Reasoning`.
- `Language Models Don't Always Say What They Think`.
- `Dissociation of Faithful and Unfaithful Reasoning`.
- `On the Hardness of Faithful CoT`.
- `FRODO / Making Reasoning Matter`.
- `How does Chain of Thought Think?` with SAE and activation patching.
- Anthropic circuit-tracing style feature-level studies.

Implication: correctness detection and faithfulness detection are not the same. A paper claiming faithful reasoning must include intervention or counterfactual evidence that the visible/latent reasoning path actually mediates the final answer.

### 5. Error Awareness, Self-Correction, And Failure Modes

Important papers:

- `Hidden Error Awareness`: strong hidden diagnostic signal, but not necessarily causal.
- `How LLMs Detect and Correct Their Own Errors`: answer-adjacent internal evaluator can predict correctability.
- `How Language Models Fail`: committed failures vs persistent uncertainty.
- `Hallucination Basins`: separability is task-dependent; misconception-like confident wrong can overlap with correct basins.
- `Lyapunov Probes`: perturbation stability boundary.
- `Reasoning Fails Where Step Flow Breaks`: shallow lock-in and deep decay in step saliency flow.
- `CRV`: computational graph features for CoT verification.

Implication: a detector must separate at least three states: uncertain failure, committed wrong, and aware-but-not-revising. AUC alone is insufficient unless decomposed by failure mode.

## Non-Negotiable Negative Controls

Any new method must include these controls:

1. **Length and position matching**: raw trajectory geometry is length-sensitive.
2. **Domain normalization**: geometric hallucination metrics may measure domain/style rather than truth.
3. **Correctness vs confidence vs relevance vs coherence vs completeness factorization**: from `What do Geometric Hallucination Detection Metrics Actually Measure?`.
4. **Committed wrong subset**: low entropy, high self-consistency, high verbal confidence, wrong answer.
5. **Diagnostic-vs-causal test**: a probe signal must not be claimed as mechanism unless patching/intervention changes future reasoning.
6. **Wrong-question / permuted-step / random-subspace controls**: to prove geometry is constraint-specific rather than generic trajectory motion.
7. **Cross-model and cross-task transfer**: otherwise the method is likely a dataset artifact.

## Strong Existing Competitors

### GeoFaith

Covers: spatio-temporal hidden geometry, low-dimensional latent structure, entropy dynamics, step-label bootstrapping, detector training, faithfulness-aware RL.

Remaining openings:

- Does it provide online causal repair or only detection/training?
- Does it distinguish committed false belief from unstable uncertainty?
- How well does it transfer across model families and task regimes?
- Does it isolate whether hidden geometry is caused by problem constraints or by self-prefix dynamics?

### Where Does Reasoning Break?

Covers: hidden-state transport geometry, first-error localization, local transition manifold, teacher/student detector.

Remaining openings:

- Student shift robustness is a likely weak point.
- Faithfulness is not the central target.
- It treats error as transport excursion; coherent-but-wrong may stay on a wrong but smooth manifold.
- Needs stronger causal validation that transport excursion is not just a difficulty/length proxy.

### EDIS / EDRM

Covers: entropy dynamics and online routing.

Remaining openings:

- No hidden-state mechanism.
- Weak on committed low-entropy wrong answers.
- No step-level semantic cause or faithfulness mediation.

## Current Research Gap

The promising gap is not “hidden geometry detects errors.” That is already crowded.

The sharper gap is:

> Existing geometric and entropy methods detect when a reasoning trajectory becomes unstable, curved, low-progress, or transport-expensive; they do not fully explain when a trajectory remains smooth and confident but becomes controlled by its own generated prefix rather than by the original problem constraints.

This creates a possible paper thesis:

> Reasoning failure is a control-source transition: faithful reasoning remains constraint-responsive, while unfaithful or committed-wrong reasoning becomes prefix-locked. Hidden geometry is useful only after we ask what drives the geometry.

## Candidate Direction To Preserve

Call it **Constraint Responsiveness / Causal Phase Locking**.

Core object:

\[
R_q(t)=\|h_t(x_{\mathrm{constraint\ edit}})-h_t(x)\|_2
\]

\[
R_r(t)=\|h_t(x_{\mathrm{irrelevant\ edit}})-h_t(x)\|_2
\]

\[
R_p(t)=\|h_t(y_{\le t}^{\mathrm{prefix\ edit}})-h_t(y_{\le t})\|_2
\]

Correct, faithful reasoning should remain sensitive to relevant constraint edits and comparatively insensitive to irrelevant edits. Committed wrong reasoning should show reduced constraint responsiveness and increased prefix self-locking.

The response-level risk should not be a mean/max scalar over steps. It should be a survival-style hazard:

\[
P(\mathrm{error\ by\ }t)=1-\prod_{i=1}^{t}(1-p_i)
\]

where \(p_i\) is specifically tied to constraint-coupling collapse, prefix lock-in, and revision failure rather than raw spread.

## Immediate Reading Priorities

1. GeoFaith.
2. Where Does Reasoning Break?
3. What do Geometric Hallucination Detection Metrics Actually Measure?
4. Reasoning Models Don't Just Think Longer, They Move Differently.
5. Reasoning emerges from constrained inference manifolds.
6. Lines of Thought.
7. EDIS and When Do LLMs Reason?
8. Hidden Error Awareness.
9. How LLMs Detect and Correct Their Own Errors.
10. How does Chain of Thought Think? SAE + patching.

## Open Questions For The Next Experiment Design

- Can constraint responsiveness distinguish correct vs committed-wrong after matching length, step position, and calculation difficulty?
- Does hidden-state sensitivity to relevant question edits drop before the first textual error?
- Does prefix sensitivity increase after the first wrong intermediate conclusion?
- Can a model internally detect conflict but fail to revise?
- Is this effect stable across GSM8K, MATH, logical reasoning, and factual multi-hop tasks?
- Can causal patching restore constraint responsiveness or only improve detection?

