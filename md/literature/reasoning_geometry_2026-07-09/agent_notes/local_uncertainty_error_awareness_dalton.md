# Agent Note: Local Uncertainty / Error Awareness / Step Flow Papers

## Scope

Local PDFs in `C:/Users/613/Desktop/papers/??` with titles related to uncertainty, hallucination, error awareness, self-correction, step flow, verifier, entropy dynamics, token-level signatures. `Alvarez?Baheri...Where Does Reasoning Break...pdf` and `WhereDoesReasoningBreak.pdf` are duplicates.

| Paper | Core Signal | Proxy Type | Coherent-but-Wrong? | Mechanism / Negative Control For Us |
|---|---|---|---|---|
| TokUR | Low-rank random weight perturbations produce token-level epistemic/aleatoric uncertainty; error paths show higher uncertainty near key tokens. | Uncertainty proxy; response aggregation. | Partial; misses committed low-uncertainty wrong. | EU/AU decomposition; compare against log-likelihood, entropy, response scores. |
| Isotropic Gradient-Norm Uncertainty | Single forward-backward gradient norm approximates epistemic uncertainty. | Parameter-sensitivity / uncertainty proxy. | Partial. | Negative control: not all uncertainty is good for step localization. |
| EDIS | Token entropy dynamics: bursts and peak-valley rebounds. | Trajectory-level uncertainty proxy. | Partial; smooth low-entropy committed wrong can escape. | Shape beats mean; compare against mean entropy/static confidence. |
| Entropy Trajectory Shape | Step answer-distribution entropy monotonic decrease violations correlate with correctness. | Step-position uncertainty-shape proxy. | Partial. | Shape-over-magnitude; control final entropy/total entropy reduction. |
| Tracing Uncertainty | Trace profiles: slope, linearity, early/middle/late stats. | Trace-level uncertainty + position/length profile. | Partial. | Dynamic features matter more than static uncertainty. |
| Hidden Error Awareness | Hidden-state linear probe predicts trace correctness; text confidence similar for correct/wrong; interventions fail. | Hidden error probe, diagnostic not causal. | Detects high verbal confidence wrong. | Red line: diagnostic signal is not necessarily causal lever. |
| Internal Confidence / Self-Correction | PANL position caches second-order evaluative confidence; predicts error detection/correctability with causal evidence. | Answer-adjacent evaluator. | Can handle coherent wrong post-commit. | Second-order evaluator decoupled from generation signal. |
| Token-Level Signatures of Committed and Persistent Failures | Failure modes: committed early lock-in vs persistent uncertainty. | Token-level uncertainty + position/length. | Can distinguish modes; self-consistency reinforces committed wrong. | First classify failure mode before choosing detector. |
| Hallucination Basins | Hallucination as latent basin/attractor; separability is task-dependent. | Hidden trajectory/basin geometry. | Misconception/confident wrong may be inseparable. | Negative control: task-specific basin separability is not universal truth. |
| Lyapunov Probes | Stability boundary under perturbations and confidence decay. | Stability/confidence proxy. | Partial. | Control random perturbation sensitivity, logit entropy, ordinary probes. |
| Step Flow Breaks | Step-saliency maps step-to-step flow; shallow lock-in/deep decay; intervention improves some performance. | Information-flow / position proxy. | Partial; explains lock-in/forgetting. | Step-level flow map; token attention too dense. |
| CRV Computational Graph | Attribution graph structural features verify CoT step correctness; targeted interventions. | White-box computational graph verifier. | Can handle coherent wrong if graph structure differs. | Upgrade from activation probe to causal graph features. |
| What Do Geometric Metrics Measure? | Geometric metrics measure correctness/confidence/relevance/coherence/completeness differently; domain shift can dominate. | Proxy audit. | Essential. | Required control: separate correctness, confidence, relevance, coherence, completeness. |
| Where Does Reasoning Break? | First error as transport-cost excursion from locally coherent manifold; teacher cPCA + student BiLSTM. | Step-level hidden transport geometry. | Target is fluent but incorrect first-error localization. | Strong direct competitor; note shift collapse risk. |

## Overall Takeaway

Positive mechanism candidates: `Where Does Reasoning Break`, `Token-Level Signatures`, `EDIS`, `PANL self-correction`, `CRV`.

Critical negative controls: `What do Geometric Metrics Actually Measure`, `Hidden Error Awareness`, `Hallucination Basins`.

These prevent us from confusing coherence with correctness, diagnostic separability with causality, and task-specific basin separability with universal truth.
