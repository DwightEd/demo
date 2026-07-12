# Agent Note: External Search - Reasoning Geometry / Manifolds

## Scope

2023-2026; arXiv / ICLR / ICML / NeurIPS / ACL / EMNLP / COLM. Focus: reasoning emergence, hidden-state geometry, manifolds, topology, phase transition, faithful CoT.

## Must-Read Papers

| Paper | Year | Method | Relation |
|---|---:|---|---|
| [GeoFaith: A Spatio-Temporal Dual View of Faithful Chain-of-Thought](https://arxiv.org/html/2605.26893v1) | 2026 | Hidden-state trajectory low-dimensional geometry and spatio-temporal view for faithful/unfaithful CoT. | Direct competitor / closest neighbor. |
| [EDIS: Diagnosing LLM Reasoning via Entropy Dynamics](https://arxiv.org/abs/2602.01288) | 2026 | Token-level entropy trajectory and instability score. | Directly relevant dynamic entropy baseline. |
| [Reasoning emerges from constrained inference manifolds in LLMs](https://arxiv.org/abs/2605.08142) | 2026 | Inference-time representations self-organize into low-dimensional constrained manifolds. | Core reasoning-emergence/manifold neighbor. |
| [Reasoning Models Don?t Just Think Longer, They Move Differently](https://arxiv.org/abs/2605.15454) | 2026 | Hidden-state CoT trajectories distinguish reasoning-trained vs non-reasoning models. | Critical for length-corrected trajectory claims. |
| [Lines of Thought in Large Language Models](https://arxiv.org/html/2410.01545v2) | 2024/2025 | Layer-wise hidden states as trajectories; low-dimensional non-Euclidean manifold; stochastic dynamics approximation. | Geometry/dynamical systems base. |
| [Latent Semantic Manifolds in Large Language Models](https://arxiv.org/html/2603.22301v1) | 2026 | Fisher information induces Riemannian metric on hidden-state manifold. | Riemannian theoretical support. |
| [Hidden Holes: topological aspects of language models](https://arxiv.org/html/2406.05798v1) | 2024 | Persistent homology / mapper on embeddings and hidden-state manifolds. | Topology + hidden states evidence. |
| [Persistent Topological Features in Large Language Models](https://arxiv.org/abs/2410.11042) | 2024/2025 | Zigzag persistence tracks cross-layer topological features. | TDA for internal representations. |
| [The Shape of Reasoning: Topological Analysis of Reasoning Traces in LLMs](https://arxiv.org/abs/2510.20665) | 2025 | TDA features of reasoning traces predict quality. | Trace topology, mostly text-level. |
| [The Topology of Ill-Posed Questions](https://arxiv.org/html/2606.23590v1) | 2026 | 0D persistent homology over prompt-token hidden states for detection/steering. | Topological anomaly descriptor for failure. |
| [The Geometry of Truth](https://arxiv.org/abs/2310.06824) | 2023/2024 | Truth/falsehood linear structure in activations with causal interventions. | Truth direction background. |
| [The Linear Representation Hypothesis and the Geometry of LLMs](https://arxiv.org/abs/2311.03658) | 2023/2024 | Counterfactual formalization and non-Euclidean inner products for probes/steering. | Theory constraints on geometric directions. |
| [Language Models Represent Space and Time](https://arxiv.org/abs/2310.02207) | 2023/2024 | Space/time world models in Llama-2 hidden states. | Evidence internal representations are structured. |
| [How to think step-by-step: A mechanistic understanding of CoT](https://arxiv.org/abs/2402.18312) | 2024 | Mechanistic probing of CoT; middle-layer functional phase shift and multi-path information flow. | CoT hidden mechanism. |
| [How does Chain of Thought Think? Sparse Autoencoding](https://arxiv.org/abs/2507.22928) | 2025 | SAE + activation patching for CoT faithfulness. | Feature-level causal study. |
| [Knowing Before Saying](https://arxiv.org/abs/2505.24362) | 2025 | Pre-generation hidden states predict CoT success/failure. | Pre-answer diagnostic neighbor. |
| [Coconut](https://arxiv.org/abs/2412.06769) | 2024 | Continuous thought via feeding last hidden state. | Latent CoT baseline. |
| [CODI](https://arxiv.org/abs/2502.21074) | 2025 | Compress explicit CoT into continuous hidden token via self-distillation. | Latent faithful reasoning baseline. |
| [Reasoning by Superposition](https://arxiv.org/abs/2505.12514) | 2025 | Continuous thoughts can encode parallel BFS frontiers. | Explains latent CoT capacity. |
| [Latent Chain-of-Thought as Planning](https://arxiv.org/html/2601.21358v1) | 2026 | Latent CoT as planning, decoupling reasoning and verbalization. | Hidden reasoning trajectory. |
| [Capabilities and Fundamental Limits of Latent CoT](https://arxiv.org/html/2602.01148v1) | 2026 | Exploration/computation tradeoff, discretization-reset explanation. | Boundary condition for latent CoT. |
| [Faithful Chain-of-Thought Reasoning](https://arxiv.org/abs/2301.13379) | 2023 | NL query ? symbolic chain ? deterministic solver. | Faithfulness starting point. |
| [Measuring Faithfulness in CoT](https://arxiv.org/abs/2307.13702) | 2023 | Intervene on CoT text, measure whether answer depends on stated reasoning. | Evaluation standard. |
| [Language Models Don?t Always Say What They Think](https://arxiv.org/abs/2305.04388) | 2023 | Biased prompts cause rationalized CoT. | Classic unfaithful CoT evidence. |
| [Dissociation of Faithful and Unfaithful Reasoning](https://arxiv.org/abs/2405.15092) | 2024 | Error recovery distinguishes faithful/unfaithful mechanisms. | Faithfulness is not a single phenomenon. |
| [On the Hardness of Faithful CoT Reasoning](https://arxiv.org/html/2406.10625v2) | 2024 | ICL/fine-tuning/activation editing have limited faithfulness gains. | Negative result. |
| [FRODO / Making Reasoning Matter](https://arxiv.org/abs/2402.13950) | 2024 | Causal mediation and counterfactual preference to make small models use intermediate reasoning. | Faithful training objective. |
| [When Do LLMs Reason? Entropy Phase Transitions](https://arxiv.org/abs/2605.22873) | 2026 | Early entropy dynamics and EDRM routing. | EDIS neighbor. |
| [The Stepwise Informativeness Assumption](https://arxiv.org/abs/2604.06192) | 2026 | Theory for why entropy dynamics correlates with correctness. | Theoretical support for entropy methods. |
| [Logical Phase Transitions](https://arxiv.org/html/2601.02902v1) | 2026 | Accuracy collapses phase-transition-like with logical complexity. | Reasoning phase transition benchmark. |
| [3-SAT Phase Transition](https://arxiv.org/abs/2504.03930) | 2025 | Random 3-SAT phase transition to test reasoning hardness. | Distinguishes shortcut vs real reasoning. |

## Theme Clusters

- Hidden-state geometry / manifold / trajectory: GeoFaith; Reasoning-emerge; Move Differently; Lines of Thought; Latent Semantic Manifolds; Geometry of Truth; Linear Representation Hypothesis; Space and Time; Tracing Representation Geometry.
- Topology / persistent homology / TDA: Hidden Holes; Persistent Topological Features; Shape of Reasoning; Topology of Ill-Posed Questions.
- Latent CoT / continuous reasoning: Coconut; CODI; Reasoning by Superposition; Latent CoT as Planning; Limits of Latent CoT.
- Entropy dynamics / phase transition / emergence: EDIS; EDRM; Stepwise Informativeness; Entropy Trajectory Shape; Logical Phase Transitions; 3-SAT Phase Transition.
- Faithful / unfaithful CoT: Faithful CoT; Measuring Faithfulness; Turpin et al.; Dissociation; Hardness; FRODO; Unlearning Reasoning Steps.

## Key Takeaway

The true nearby intersection is **GeoFaith + Reasoning-emerge + Lines of Thought + EDIS/EDRM + faithful-CoT evaluation**, not any single scalar detector.
