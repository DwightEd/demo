# LaST$_{0}$: Latent Spatio-Temporal Chain-of-Thought for Robotic Vision-Language-Action Model

- **Local PDF filename**: `Latent Spatio-Temporal Chain-of-Thought for Robotic.pdf`
- **Slug**: `latent-spatio-temporal-chain-of-thought-for-robotic`
- **Pages**: 18
- **Approx Words**: 12157
- **Auto Tags**: geometry;dynamics;faithfulness
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.617801

## Keyword Profile

- `latent`: 152
- `chain of thought`: 29
- `dimension`: 14
- `geometric`: 10
- `flow`: 9
- `geometry`: 2
- `transition`: 2
- `faithful`: 2
- `causal`: 2
- `trajectory`: 1
- `hidden state`: 1

## Abstract / Opening Summary

Vision-Language-Action (VLA) models have re- cently shown strong generalization, with some approaches seeking to explicitly generate linguis- tic reasoning traces or predict future observa- tions prior to execution. However, explicit rea- soning typically incurs non-negligible inference latency, which constrains the temporal resolu- tion required for robotic manipulation. More- over, such reasoning is confined to the linguis- tic space, imposing a representational bottleneck that struggles to faithfully capture ineffable phys- ical attributes. To mitigate these limitations, we propose LaST0, a framework that enables effi- cient reasoning before acting through a Latent Spatio-Temporal Chain-of-Thought (CoT), cap- turing fine-grained physical and robotic dynamics that are often difficult to verbalize. Specifically, we introduce a token-efficient latent CoT space that models future visual dynamics, 3D structural information, and robot proprioceptive states, and further extends these representations across time to enable temporally consistent implicit reasoning trajectories. Furthermore, LaST0 adopts a dual- system architecture implemented via a Mixture- of-Transformers design, where a reasoning ex- pert conducts low-frequency latent inference and an acting expert generates high-frequency ac- tions conditioned on robotics-oriented latent rep- resentations. To facilitate coordination, LaST0 is trained with heterogeneous operation frequencies, enabling adaptive switching during deployment. Across 10 real-world tasks spanning tabletop, mo- bile, and dexterous hand manipulation, LaST0 improves mean success rates by 13%, 14% and 14% over prior SOTA VLA methods, respectively.

## Method / Algorithms Extract

methods that explicitly generate linguistic reasoning traces or future visual observations, (b) we propose LaST0, a framework that enables efficient reasoning before acting through a Latent Spatio-Temporal CoT. This latent CoT captures multimodal physical and robotic dynamics that are difficult to verbalize and propagates them over time to form temporally consistent reasoning. LaST0 achieves SOTA performance across a wide range of tasks while enabling more efficient model inference. Abstract Vision-Language-Action (VLA) models have re- cently shown strong generalization, with some approaches seeking to explicitly generate linguis- tic reasoning traces or predict future observa- tions prior to execution. However, explicit rea- soning typically incurs non-negligible inference latency, which constrains the temporal resolu- tion required for robotic manipulation. More- over, such reasoning is confined to the linguis- tic space, imposing a representational bottleneck that struggles to faithfully capture ineffable phys- ical attributes. To mitigate these limitations, we propose LaST0, a framework that enables effi- cient reasoning before acting through a Latent Spatio-Temporal Chain-of-Thought (CoT), cap- turing fine-grained physical and robotic dynamics that are often difficult to verbalize. Specifically, we introduce a token-efficient latent CoT space that models future visual dynamics, 3D structural information, and robot proprioceptive states, and further extends these representations across time to enable temporally consistent implicit reasoning trajectories. Furthermore, LaST0 adopts a dual- system architecture implemented via a Mixture- of-Transformers design, where a reasoning ex- pert conducts low-frequency latent inference and an acting expert generates high-frequency ac- tions conditioned on robotics-oriented latent rep- resentations. To facilitate coordination, LaST0 is trained with heterogeneous operation frequencies, enabling adaptive switching during deployment. Across 10 real-world tasks spanning tabletop, mo- bile, and dexterous hand manipulation, LaST0 improves mean success rates by 13%, 14% and 14% over prior SOTA VLA methods, respectively. 1. Introduction By inheriting the semantic understanding and common- sense reasoning capabilities of Vision-Language Models (VLMs) (Alayrac et al., 2022; Karamcheti et al., 2024; Deng 1 arXiv:2601.05248v3 [cs.RO] 30 Mar 2026 ===== PAGE 2 / 18 ===== LaST0: Latent Spatio-Temporal Chain-of-Thought for Robotic Vision-Language-Action Model et al., 2025a), Vision-Language-Action (VLA) models inte- grate rich pr...

## Experiments / Evidence Extract

experiments on 10 RLBench tasks. Importance of latent CoT modalities. As shown in Fig. 5(a), we assess each latent modality by ablating indi- vidual components while keeping all other parameters fixed at their optimal settings (i.e., latent tokens = 1, temporal coverage = 4). When using only the image, point cloud, or robot state latent, the model achieves success rates of 74%, 76%, and 75%, respectively, indicating that each modality- specific latent provides a strong basis for action generation. The combination of multiple modality latents continues to provide additional performance improvements, even when the manipulation accuracy is already high. These results val- idate the importance of modeling comprehensive physical dynamics in the latent space, and further demonstrate that enabling the model to autonomously reason about the rela- tionship between the robot and its interactive environment is effective for robotic manipulation. Number of tokens per latent modality. As shown in Fig. 5 6 ===== PAGE 7 / 18 ===== LaST0: Latent Spatio-Temporal Chain-of-Thought for Robotic Vision-Language-Action Model Table 1. Comparison of LaST0 and baselines on RLBench. All methods are trained in the multi-task setting (Shridhar et al., 2022), and we report mean success rates (S.R.). Inference speed is evaluated on an NVIDIA 4090 GPU. Models Close Close Toilet Sweep Close Phone Umbrella Frame Wine at Water Mean S.R. ↑ Infer. box laptop lid seat down to dustpan fridge on base out off hanger rack plants & Var ↓ speed ↑ OpenVLA 0.60 0.35 0.75 0.55 0.85 0.20 0.30 0.15 0.20 0.05 0.40 ±0.02 6.3 Hz SpatialVLA 0.80 0.70 0.85 0.20 0.80 0.15 0.25 0.40 0.15 0.30 0.46 ±0.03 7.9 Hz CogACT 0.90 0.80 0.95 0.50 0.85 0.50 0.55 0.45 0.30 0.25 0.61 ±0.04 9.8 Hz CoT-VLA 0.95 0.75 1.00 0.80 0.65 0.50 0.40 0.50 0.55 0.50 0.66 ±0.03 1.1 Hz π0.5 0.90 0.95 0.85 0.75 1.00 0.05 0.10 0.80 0.75 0.35 0.65 ±0.04 13.8 Hz HybridVLA 0.85 0.95 1.00 0.90 1.00 0.50 0.50 0.70 0.50 0.50 0.74 ±0.04 6.1 Hz LaST0 0.95 0.95 1.00 0.80 0.85 0.75 0.75 0.70 0.85 0.60 0.82 ±0.03 15.4 Hz 2D 3D state 0 token 1 token 2 tokens 4 tokens 0 step 1 step 2 steps 4 steps 1:1 1:2 1:4 1:8 Mix a) Importance of modalities d) Collaboration frequency c) Temporal coverage b) Number of latent tokens 2D+3D 2D+state all 5 steps 6 steps Figure 5. Ablation study on key design choices of LaST0. We analyze (a) the importance of different late...

## Conclusion / Discussion Extract

Conclusion We introduced LaST0, a dual-system VLA model that en- ables efficient reason-before-act behavior for robotic manip- ulation through a Latent Spatio-Temporal Chain-of-Thought (LaST CoT). By shifting reasoning from explicit traces to a compact latent space, LaST0 overcomes the latency and representational bottlenecks inherent in prior CoT VLA ap- proaches, while preserving the ability to model fine-grained physical dynamics essential for closed-loop control. Central to our framework is a token-efficient spatio-temporal latent representation that autoregressively captures future seman- tic, geometric, and proprioceptive dynamics. Building upon this LaST CoT, we further proposed a fast-slow dual-system implemented via a MoT, which decouples low-frequency de- 8 ===== PAGE 9 / 18 ===== LaST0: Latent Spatio-Temporal Chain-of-Thought for Robotic Vision-Language-Action Model liberative reasoning from high-frequency action generation. We believe LaST0 represents a step toward more physically grounded reasoning in robotic foundation models. Impact Statement This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. References Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736, 2022. Belkhale, S., Cui, Y., and Sadigh, D. Hydra: Hybrid robot actions for imitation learning. arxiv, 2023. Belkhale, S., Ding, T., Xiao, T., Sermanet, P., Vuong, Q., Tompson, J., Chebotar, Y., Dwibedi, D., and Sadigh, D. Rt-h: Action hierarchies using language, 2024. URL https://arxiv.org/abs/2403.01823. Bjorck, J., Casta˜neda, F., Cherniadev, N., Da, X., Ding, R., Fan, L., Fang, Y., Fox, D., Hu, F., Huang, S., et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025. Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., et al. pi0: A vision-...

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

TBD: add short excerpts with page markers from `../texts/latent-spatio-temporal-chain-of-thought-for-robotic.txt`.
