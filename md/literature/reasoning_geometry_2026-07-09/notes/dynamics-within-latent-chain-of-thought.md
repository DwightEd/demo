# ===== PAGE 1 / 22 ===== Published at Latent & Implicit Thinking Workshop @ ICLR 2026

- **Local PDF filename**: `DYNAMICS WITHIN LATENT CHAIN-OF-THOUGHT.pdf`
- **Slug**: `dynamics-within-latent-chain-of-thought`
- **Pages**: 22
- **Approx Words**: 11140
- **Auto Tags**: dynamics;faithfulness;emergence
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.606335

## Keyword Profile

- `latent`: 152
- `causal`: 48
- `probe`: 23
- `trajectory`: 20
- `chain of thought`: 17
- `faithful`: 8
- `hidden state`: 7
- `transition`: 4
- `flow`: 2
- `manifold`: 1
- `topolog`: 1
- `entropy`: 1

## Abstract / Opening Summary

Latent or continuous chain-of-thought methods replace explicit textual rationales with a number of internal latent steps, but these intermediate computations are difficult to evaluate beyond correlation-based probes. In this paper, we view la- tent chain-of-thought as a manipulable causal process in representation space by modeling latent steps as variables in a structural causal model (SCM) and an- alyzing their effects through step-wise do-interventions. We study two repre- sentative paradigms (i.e., Coconut and CODI) on both mathematical and gen- eral reasoning tasks to investigate three key questions: (1) which steps are causally necessary for correctness and when answers become decidable early; (2) how does influence propagate across steps, and how does this structure com- pare to explicit CoT; and (3) do intermediate trajectories retain competing an- swer modes, and how does output-level commitment differ from representational commitment across steps. We find that latent-step budgets behave less like ho- mogeneous extra depth and more like staged functionality with non-local rout- ing, and we identify a persistent gap between early output bias and late repre- sentational commitment. These results motivate mode-conditional and stability- aware analyses—and corresponding training/decoding objectives—as more reli- able tools for interpreting and improving latent reasoning systems. Code is avail- able at https://github.com/J1mL1/causal-latent-cot. 1

## Method / Algorithms Extract

methods replace explicit textual rationales with a number of internal latent steps, but these intermediate computations are difficult to evaluate beyond correlation-based probes. In this paper, we view la- tent chain-of-thought as a manipulable causal process in representation space by modeling latent steps as variables in a structural causal model (SCM) and an- alyzing their effects through step-wise do-interventions. We study two repre- sentative paradigms (i.e., Coconut and CODI) on both mathematical and gen- eral reasoning tasks to investigate three key questions: (1) which steps are causally necessary for correctness and when answers become decidable early; (2) how does influence propagate across steps, and how does this structure com- pare to explicit CoT; and (3) do intermediate trajectories retain competing an- swer modes, and how does output-level commitment differ from representational commitment across steps. We find that latent-step budgets behave less like ho- mogeneous extra depth and more like staged functionality with non-local rout- ing, and we identify a persistent gap between early output bias and late repre- sentational commitment. These results motivate mode-conditional and stability- aware analyses—and corresponding training/decoding objectives—as more reli- able tools for interpreting and improving latent reasoning systems. Code is avail- able at https://github.com/J1mL1/causal-latent-cot. 1 INTRODUCTION Large language models (LLMs) have achieved strong performance on mathematical problem solving and logical question answering (Cobbe et al., 2021; Geva et al., 2021). A widely adopted technique is Chain-of-Thought (CoT) prompting, which improves accuracy by eliciting intermediate reason- ing steps in natural language (Wei et al., 2022). Despite its empirical effectiveness, explicit CoT incurs substantial decoding cost, often produces verbose outputs, and may contain post-hoc ratio- nalizations that do not faithfully reflect the computations driving model predictions (Pruthi et al., 2020; Turpin et al., 2023). These limitations motivate a shift from reasoning in tokens to reasoning in representations. Recent work explores latent or continuous CoT, where multi-step inference is carried out in contin- uous hidden representations rather than long textual explanations (Hao et al., 2024; Shen et al., 2025; Zhang et al., 2025; Xu et al., 2025; Gozeten et al., 2025). This paradigm promises a higher-bandwidth internal workspace and reduced decoding overhead, but it faces two fundamen- tal interpretability challenges: 1) intermediate computat...

## Experiments / Evidence Extract

experiments, we intervene on the intermediate states produced by a latent thinking rollout. We denote realized latent states by lowercase ht and write h1:T ∼pθ(H1:T | x), (3) where ht ∈Rd instantiates the random variable Ht at step t under the model dynamics. Latent- reasoning models expose these variables through a fixed-length sequence of hidden states h1:T = (h1, . . . , hT ), where ht is the last-layer hidden representation associated with the t-th latent step (e.g., a continuous “thought token” in COCONUT or the designated reasoning position in CODI) and is used as the step-t reasoning input embedding. Given input x and a realized trajectory h1:T , the intervention do(ht ←˜ht) replaces the latent state at step t by ˜ht and then propagates the resulting change through all later steps using the same transition mechanism, yielding a counterfactual trajectory ˜h1:T . Formally, let ˜h<t = h<t and let ˜ht 3 ===== PAGE 4 / 22 ===== Published at Latent & Implicit Thinking Workshop @ ICLR 2026 be the overwritten state; for t′ > t we set ˜ht′ := ft′(˜h<t′, x, ˜ϵt′; θ), (4) where ˜ϵt′ matches the baseline randomness when applicable. The corresponding counterfactual output is obtained by the same readout mechanism, ˜y = g(˜h1:T , x, ˜ϵy; θ). (5) Unless otherwise stated, we use deterministic rollouts whenever possible; otherwise we control ran- domness (e.g., fixed seeds) and isolate propagation effects via teacher-forced readouts (Figure 2(V)) to reduce sampling noise. 2.3 PARADIGMS OF LATENT-REASONING MODELS We instantiate it on two latent-reasoning paradigms diverging in their realization of latent steps. Coconut (Hao et al., 2024) uses an explicit latent mode: it treats the final hidden state as a continuous reasoning token and feeds it back as input to the next step, rather than decoding a discrete token. CODI (Shen et al., 2025) compresses discrete CoT into continuous space via self-distillation: a continuous-CoT student is trained to both produce the correct answer and align its hidden states, at specific reasoning steps, with those of a discrete-CoT teacher, encouraging the latent trajectory to inherit stepwise structure. 2.4 MODELS AND DATA Models. We experiment with CODI and Coconut on multiple backbones. We use official CODI checkpoints for GPT-2 (Radford et al., 2019) and Llama3-1B (Grattafiori et al., 2024), and reproduce it on Qwen3-4B-Instruct (Yang e...

## Conclusion / Discussion Extract

Our experiments offer a unified, step-centric perspective on latent-token reasoning by connecting three complementary views: step-wise causal necessity and early decodability (RQ1), directed prop- agation summarized at the step level (RQ2), and trajectory-level mode dynamics under stochastic rollouts (RQ3). Taken together across COCONUT and CODI, these analyses suggest that a fixed latent budget functions less like homogeneous extra depth and more like a structured interface: steps have unequal causal leverage, influences can route non-locally across the trajectory, and apparent output-level commitment need not coincide with the underlying representational state. Latent steps are causally functional, with heterogeneous leverage. RQ1 indicates that latent computation is broadly engaged: intervening on a single step can change the decoded decision, but the effect is not evenly distributed across the budget. A useful lens is to treat the step index as an implicit interface for division of labor: certain steps act as high-leverage intervention sites whose removal disrupts downstream computation, while others appear to contribute more conditionally, surfacing as sensitivity only on specific inputs or reasoning modes. This helps interpret the non- monotonic profiles without assuming that the model simply “refines the same state” at every step; instead, latent reasoning may introduce step-specific updates whose downstream effect is later am- plified, transformed, or gated. The fact that COCONUT and CODI allocate leverage differently on matched backbones further suggests that training paradigm shapes where decision-relevant depen- dence concentrates along the trajectory. Minimal computation towards the correct answer is distinct from the corresponding commit- ment. Early-stop decoding offers a complementary notion of “how much reasoning is needed”: it measures when the correct answer first becomes readable from the latent state, rather than whether a step remains behaviorally necessary when removed. Seen together with intervention sensitivity, this separates availability from stability: a solution can become decodable at an intermediate step while later computation stil...

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

TBD: add short excerpts with page markers from `../texts/dynamics-within-latent-chain-of-thought.txt`.
