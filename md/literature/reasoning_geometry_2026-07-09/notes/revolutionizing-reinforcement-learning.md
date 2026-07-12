# ===== PAGE 1 / 26 ===== Published as a conference paper at ICLR 2026

- **Local PDF filename**: `REVOLUTIONIZING REINFORCEMENT LEARNING.pdf`
- **Slug**: `revolutionizing-reinforcement-learning`
- **Pages**: 26
- **Approx Words**: 14148
- **Auto Tags**: dynamics;faithfulness;uncertainty;emergence
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.624950

## Keyword Profile

- `trajectory`: 30
- `transition`: 4
- `latent`: 3
- `flow`: 3
- `chain of thought`: 2

## Abstract / Opening Summary

The extension of diffusion models to language tasks has shown promising re- sults, but their post-training methods remain largely unexplored. We highlight the importance of aligning a diffusion language model’s preference-inference tra- jectory with its post-training objective. To this end, we propose TraceRL, a trajectory-aware reinforcement learning framework for DLMs that incorporates information from inference trajectories into post-training and is applicable to both full-attention and block-attention diffusion models. We also introduce a diffusion- based value model that enhances training stability and naturally accommodates process rewards. We demonstrate TraceRL’s superiority in enhancing a model’s reasoning ability on complex math and coding tasks, as well as its applicabil- ity in scaling block diffusion models to larger block sizes. Employing TraceRL, we derive a series of state-of-the-art diffusion language models, namely TraDo. Although smaller than Qwen2.5-7B-Instruct, TraDo-4B-Instruct consistently out- performs it on complex math reasoning tasks. TraDo-8B-Instruct achieves 4.5% higher accuracy on MATH500 than Qwen2.5-7B-Instruct and 6.6% higher ac- curacy on LiveCodeBench-V2 than Llama3.1-8B-Instruct. Through curriculum learning, we also develop the first 8B-scale long-CoT diffusion language model. 1

## Method / Algorithms Extract

FOR DIFFUSION LARGE LANGUAGE MODELS Yinjie Wang1,2∗, Ling Yang1∗†, Bowen Li3, Ye Tian3, Ke Shen, Mengdi Wang1 1Princeton University, 2University of Chicago, 3Peking University Project: https://github.com/Gen-Verse/dLLM-RL

## Experiments / Evidence Extract

In this section, we demonstrate the superiority of TraceRL across diverse tasks and models, as well as the advantages of incorporating a value model, including enhanced training stability and the natural 5 ===== PAGE 6 / 26 ===== Published as a conference paper at ICLR 2026 Table 2: The main benchmark results across different math and coding tasks. “Static” denotes static sampling, and “Dynamic” denotes dynamic sampling. The long-CoT model TraDo-8B-Instruct is evaluated using dynamic sampling with threshold 0.9. Model MATH500 AIME2024 GSM8K LiveCodeBench-v2 LiveBench Autoregressive Models Llama3.1-8B-Instruct 51.9 6.7 84.5 20.0 19.7 Qwen2.5-7B-Instruct 74.0 8.2 89.9 26.9 31.1 Diffusion Language Models Static Dynamic Static Dynamic Static Dynamic Static Dynamic Static Dynamic LLaDA-8B-Instruct 37.3 38.3 0.5 1.7 82.5 82.5 5.9 5.5 4.9 6.0 Dream-7B-Instruct 38.7 32.3 / / 72.7 57.8 10.7 4.7 10.7 4.9 SDAR-4B-Chat 70.2 67.4 5.0 8.2 90.2 88.9 15.6 11.2 14.0 7.6 TraDo-4B-Instruct 75.6 +5.4 71.8 +4.4 8.3 +3.3 10.3 +2.1 91.2 +1.0 90.3 +1.2 18.7 +3.1 15.1 +3.9 12.9 10.4 +2.8 SDAR-8B-Chat 74.3 70.7 11.8 8.3 91.1 90.4 18.5 15.3 11.5 11.2 TraDo-8B-Instruct 78.5 +4.2 75.5 +4.8 13.3 +1.5 11.0 +2.7 92.3 +1.2 91.2 +0.8 25.9 +7.4 22.4 +7.1 22.7+11.2 20.6 +9.4 TraDo-8B-Thinking 87.4+13.1 35.5+23.7 94.2 +3.1 34.6+16.1 36.0+23.8 ability to accommodate process rewards for trajectory-wise supervision. We present comprehensive evaluation results for our state-of-the-art TraDo models and highlight interesting applications such as block size enlargement. More ablation studies are included in Appendix C. 5.1

## Conclusion / Discussion Extract

We present a new reinforcement learning method for diffusion language models with diverse archi- tectures. Extensive experiments demonstrate the effectiveness of this method across multiple RL tasks, resulting in three state-of-the-art diffusion language models. We also highlight its benefits for accelerating inference and scaling block size, pointing to promising directions for future research. 10 ===== PAGE 11 / 26 ===== Published as a conference paper at ICLR 2026 7 ETHICS STATEMENT Our research relies solely on publicly available benchmarks for math and coding tasks, and does not involve human subjects, private data, or applications with direct ethical risks. 8 REPRODUCIBILITY STATEMENT The code is included in the supplementary material. We also describe the experimental details in Section 5.1 and Appendix D. REFERENCES Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Sub- ham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autore- gressive and diffusion language models. arXiv preprint arXiv:2503.09573, 2025. Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021. Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11315–11325, 2022. Shuang Cheng, Yihan Bian, Dawei Liu, Yuhua Jiang, Yihao Liu, Linfeng Zhang, Wenghai Wang, Qipeng Guo, Kai Chen, Biqing Qi*, and Bowen Zhou. Sdar: A synergistic diffu- sion–autoregression paradigm for scalable sequence generation, 2025. URL https://gith ub.com/JetAstra/SDAR. Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H Richemond, Arnaud Doucet, Robin Strudel, Chris Dye...

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

TBD: add short excerpts with page markers from `../texts/revolutionizing-reinforcement-learning.txt`.
