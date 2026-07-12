# Reasoning Models Don't Just Think Longer, They Move Differently

- **Local PDF filename**: `Reasoning Models Don’t Just Think Longer,.pdf`
- **Slug**: `reasoning-models-dont-just-think-longer`
- **Pages**: 25
- **Approx Words**: 13238
- **Auto Tags**: geometry;dynamics;faithfulness;uncertainty;emergence
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.623869

## Keyword Profile

- `trajectory`: 69
- `geometry`: 66
- `curvature`: 26
- `probe`: 26
- `geometric`: 20
- `dimension`: 16
- `hidden state`: 12
- `causal`: 10
- `chain of thought`: 5
- `manifold`: 4
- `latent`: 3
- `flow`: 2

## Abstract / Opening Summary

Reasoning-trained language models often spend more tokens on harder problems, but longer chains of thought do not show whether a model is merely computing for more steps or following a different internal trajectory. We study this distinc- tion through hidden-state trajectories during chain-of-thought generation across competitive programming, mathematics, and Boolean satisfiability. Raw trajectory geometry is strongly shaped by generation length: longer generations mechanically alter path statistics, so difficulty-dependent comparisons are misleading without adjustment. After residualizing trajectory statistics on length, difficulty remains systematically coupled to corrected trajectory geometry across all domains stud- ied. The clearest reasoning-specific separation appears in the code domain, where harder problems show more direct corrected trajectories and less heterogeneous local curvature in reasoning-trained models than in matched instruction-tuned baselines. Corrected difficulty-geometry coupling is weaker, but still present, in mathematics and Boolean satisfiability. Prompt-stage linear probes do not mirror the code-domain separation, and behavioral annotations show that stronger cor- rected coupling co-occurs with strategy shifts and uncertainty monitoring. Together, these findings establish length correction as a prerequisite for generation-time tra- jectory analysis and show that reasoning training can be associated with distinct corrected trajectory geometry, with the strength of the effect depending on the domain. 1

## Method / Algorithms Extract

methodological concern, since geometric metrics can change mechanically with trajectory length. Sun et al. [2026] characterize reasoning as trajectories through step-specific representation subspaces, showing that correct and incorrect solutions diverge at late steps and that trajectory-based steering can redirect reasoning. Our question is complementary: we study token-time trajectories at a fixed layer rather than layer-indexed step representations, and ask whether problem difficulty modulates trajectory geometry after removing the mechanical effects of generation length, a confound not addressed in step-indexed analyses. Difficulty in LLMs. A separate line of work studies how LLMs encode or measure problem difficulty. Linear probes can decode difficulty from hidden states with high accuracy [Lugoloobi and Russell, 2025]. IRT has also been adopted for LLM benchmarking and evaluation [Polo et al., 2024, Zhou et al., 2025, Xu et al., 2025]. Zhu et al. [2025] estimated model-perceived difficulty from hidden representations via a value-function framework, while Lee et al. [2025] identified attention heads with distinct activation patterns for easy versus hard problems. These works show that difficulty is represented in model internals and can be measured continuously. Our goal, however, is not to show that difficulty is encoded, but to use a continuous difficulty variable to study how internal computation changes across problems. Difficulty-dependent reasoning behavior. Work on overthinking, underthinking, and inference- time compute has shown that reasoning models allocate computation differently across easy and hard problems. Snell et al. [2024] showed that optimal compute allocation depends on difficulty. Chen et al. [2025] documented overthinking on easy problems, while Wang et al. [2025] identified underthinking on hard problems; Su et al. [2025] showed that both behaviors can coexist. Huang et al. [2025] linked overthinking to a low-dimensional activation manifold and proposed steering-based mitigation. These works primarily characterize difficulty-dependent adaptation through outputs or pathological regimes. Our paper asks the complementary internal question: whether reasoning training changes the geometry of the generation-time trajectory itself, across the full difficulty continuum and after controlling for response length. Taken together, these literatures motivate geometry, difficulty, and inference-time adaptation as relevant lenses, but leave open whether reasoning training changes generation-time internal dynamics as a function of problem dif...

## Experiments / Evidence Extract

We use a matched design to separate four quantities that are otherwise entangled: problem difficulty, generation length, model class, and trajectory geometry. We define comparable item sets across three domains, calibrate a continuous difficulty scale within each domain, compare matched reasoning and 3 ===== PAGE 4 / 25 ===== Table 1: Matched model pairs used in the main comparison. Reasoning Model Baseline Family Training R1-Distill-Qwen-7B Qwen2.5-7B-Instruct Qwen SFT distillation (R1) R1-Distill-Qwen-14B Qwen2.5-14B-Instruct Qwen SFT distillation (R1) R1-Distill-Qwen-32B Qwen2.5-32B-Instruct Qwen SFT distillation (R1) R1-Distill-Llama-8B Llama-3.1-8B-Instruct Llama SFT distillation (R1) QwQ-32B Qwen2.5-32B-Instruct Qwen SFT + RL Phi-4-Reasoning Phi-4 Phi SFT distillation (o3-mini) instruction-tuned model pairs on the same items, and extract hidden-state trajectories from generated solution segments. Datasets. We evaluate on 500 Easy2Hard-Bench competitive-programming problems [Ding et al., 2024], 500 MATH problems [Hendrycks et al., 2021], and 500 SATBench problems [Wei et al., 2025]. SATBench items are stratified into five clause-count bins spanning 4–45 clauses and are approximately balanced between satisfiable and unsatisfiable instances within each bin. This yields 1,500 items across competitive programming, mathematics, and Boolean satisfiability. Difficulty calibration. Native difficulty labels are platform-specific (Codeforces Glicko-2 ratings), coarsely ordinal (MATH levels 1–5), or structural (SAT clause counts; SATBench clause count is the dominant proxy for instance hardness in the synthetic regime studied here). To obtain a continuous latent difficulty scale within each domain, we fit a Rasch model [Rasch, 1960] with a binomial likelihood over repeated runs: kij ∼Binomial nij, σ(θj −bi)  , (1) where kij is the number of correct completions by model j on item i, and bi is item difficulty. IRT is calibrated separately per domain from 32 models and validated against external labels: Spearman ρ = 0.55 with Codeforces ratings, ρ = 0.43 with MATH levels, and ρ = 0.56 (r = 0.58) with SAT clause counts. We use bi as the continuous independent variable throughout. Appendix A.6 reports calibration diagnostics, external-label agreement, 1PL–2PL comparisons, and leave-one-out recalibration checks. Matched model pairs. The core analysis uses six matched ...

## Conclusion / Discussion Extract

Generation length is a structural variable in generation-time trajectory geometry. Straightness-style path statistics depend on path structure and length, and prior language-model work has shown that trajectory geometry can be informative when the trajectory regime is well specified [Benhamou, 2004, Hosseini and Fedorenko, 2023]. In token-time generation, response length varies with problem difficulty, correctness, and model class, so raw geometric statistics mix trajectory organization with path-length mechanics. Length correction therefore changes the object of analysis: it separates geometry associated with how generation unfolds from geometry induced by how long generation continues. This length-aware view reveals difficulty-dependent trajectory structure across the domains we study. Corrected geometry retains systematic coupling with item difficulty after the dominant length component is removed, showing that harder problems are not characterized only by longer traces. This is especially relevant for reasoning models, where test-time compute, problem difficulty, response 8 ===== PAGE 9 / 25 ===== length, and correctness interact in nontrivial ways [Snell et al., 2024, Chen et al., 2025, Wang et al., 2025, Su et al., 2025]. The corrected statistics are not a direct measure of reasoning quality; they are a controlled description of how hidden-state trajectories vary with difficulty during generation. The reasoning-specific pattern is strongest in competitive programming. In that domain, matched reasoning models and instruction-tuned baselines differ most clearly after length correction, suggest- ing that reasoning training changes how trajectories adapt as problems become harder. A plausible explanation is that hard code problems more visibly elicit strategy selection, revision, and verification over extended traces. Mathematics and Boolean satisfiability still show corrected difficulty–geometry coupling, though the separation between model classes is weaker. This domain dependence is infor- mative: corrected geometry captures both general difficulty-conditioned generation and, in the code setting, a sharper reasoning-training contrast. The probe and behavior...

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

TBD: add short excerpts with page markers from `../texts/reasoning-models-dont-just-think-longer.txt`.
