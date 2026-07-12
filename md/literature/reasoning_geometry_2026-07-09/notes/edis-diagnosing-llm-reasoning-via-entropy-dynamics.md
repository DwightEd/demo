# EDIS: Diagnosing LLM Reasoning via Entropy Dynamics

- **Local PDF filename**: `EDIS-Diagnosing LLM Reasoning via Entropy Dynamics.pdf`
- **Slug**: `edis-diagnosing-llm-reasoning-via-entropy-dynamics`
- **Pages**: 16
- **Approx Words**: 8735
- **Auto Tags**: dynamics;faithfulness;uncertainty;hallucination;emergence
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.607978

## Keyword Profile

- `entropy`: 130
- `trajectory`: 15
- `hallucination`: 2
- `chain of thought`: 2
- `transition`: 1

## Abstract / Opening Summary

Entropy-based confidence signals are increasingly leveraged to improve reasoning in large language models (LLMs), yet existing approaches treat con- fidence as a static quantity—typically aggregated over tokens. We show that the temporal evolution of confidence during generation carries richer in- formation than aggregate statistics alone. Analyz- ing token-level entropy trajectories, we identify characteristic patterns distinguishing correct from incorrect reasoning: erroneous solutions exhibit unstable dynamics, including burst spikes (sus- tained uncertainty growth) and peak-valley spikes (sharp rebounds following transient confidence). These patterns persist across models and train- ing stages, suggesting they reflect intrinsic prop- erties of reasoning failure rather than superficial noise. To formalize this observation, we introduce the Entropy Dynamics Instability Score (EDIS), a trajectory-level metric quantifying instability in entropy evolution. EDIS serves as an effec- tive diagnostic signal for inference-time selection, substantially improving reasoning accuracy, and offers a promising direction for training-time sam- ple curation. Our findings establish entropy dy- namics as an underexplored yet informative lens for understanding and improving LLM reasoning.

## Method / Algorithms Extract

methods treat confidence as a static quantity, ag- gregating token-level uncertainty into summary statistics or examining only the final output. Recent evidence suggests that entropy calibration degrades during autoregressive gen- eration (Cao et al., 2025), indicating that this static view may miss important structure. More fundamentally, it over- looks a key aspect of autoregressive generation: reasoning unfolds sequentially, and confidence evolves throughout the process. In this work, we demonstrate that how confidence evolves during generation is more informative than its aggregate value. Through systematic analysis of token-level entropy trajectories, we uncover a striking pattern: incorrect reason- ing is not merely associated with higher uncertainty, but with instability in how uncertainty evolves. As illustrated in Figure 1, correct reasoning produces relatively smooth entropy curves where most tokens exhibit low entropy with few spikes or oscillations. In contrast, incorrect reasoning shows frequent high-entropy tokens and characteristic in- stability patterns. We identify two typical failure modes: burst spikes, where entropy rises steadily over consecutive tokens as the model becomes progressively confused, and peak-valley (rebound) spikes, where entropy drops to a lo- cal minimum before sharply rebounding—indicating false confidence followed by renewed uncertainty. These insta- bility patterns are remarkably consistent: across models, temperatures, and training stages, incorrect responses ex- hibit 1.7–3.6× more entropy fluctuations than correct ones (Cohen’s d ≈1.0), suggesting they reflect fundamental properties of reasoning failure rather than incidental noise. To operationalize this observation, we introduce the Entropy Dynamics Instability Score (EDIS), a simple trajectory-level metric that captures two complementary forms of instabil- ity: burst spikes (cumulative entropy growth within a sliding window) and peak-valley spikes (sharp increases from histor- ical minima). As shown in Figure 2, EDIS distributions for correct and incorrect responses concentrate around distinct central values, enabling clear separation. In contrast, mean entropy—a common baseline—fails to distinguish response quality, highlighting the value of trajectory-level analysis. We validate EDIS through extensive experiments on math- ematical reasoning. For inference-time selection, EDIS- 1 arXiv:2602.01288v2 [cs.LG] 6 Mar 2026 ===== PAGE 2 / 16 ===== EDIS: Diagnosing LLM Reasoning via Entropy Dynamics Figure 1. Token entropy trajectories for correct (top) and incorre...

## Experiments / Evidence Extract

span four mathematical rea- soning benchmarks—GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021), AMC23 (knoveleng, 2025), and AIME24 (Hugging Face H4, 2025)—using three mod- els: Qwen2.5-Math-1.5B (Qwen Team, 2024a; Yang et al., 2024), Qwen3-4B-Instruct (Qwen Team, 2025; Yang et al., 2025), and Qwen2.5-Math-7B (Qwen Team, 2024b; Yang et al., 2024). For GSM8K and MATH, we randomly sample 100 problems; for AMC23 and AIME24, we use the full test sets. All experiments use three sampling temperatures (0.2, 0.6, 1.0), with results averaged across temperatures. For each problem, we generate N = m · k candidates (k = 8, m ∈{1, 2, 4, 8, 16}), rank by EDIS, and retain the k most stable responses (lowest EDIS). Results. Figure 4 shows remarkably consistent improve- ments: across all three models, four benchmarks, and three metrics (average accuracy, EDIS-best, and majority voting), accuracy increases monotonically with the oversampling multiplier. The gains are substantial, particularly for mod- els with lower baseline performance. Aggregating across benchmarks, Qwen2.5-Math-1.5B improves from 29.9% to 54.5% in average accuracy as m increases from 1 to 16—a gain of 24.6 percentage points that nearly doubles the base- line. Similarly, Qwen2.5-Math-7B improves from 40.9% to 61.9% (+21.0 pp). Even the stronger Qwen3-4B-Instruct, starting at 58.8%, achieves consistent gains to 62.2% (+3.4 pp). The pattern holds for majority voting, where Qwen2.5- Math-1.5B improves from 39.4% to 62.6% (+23.2 pp) and Qwen2.5-Math-7B from 55.9% to 69.8% (+13.9 pp). No- tably, even EDIS-best accuracy improves (e.g., 46.3% to 54.1% for Qwen2.5-Math-1.5B), indicating that EDIS fil- tering enriches the candidate pool with higher-quality re- sponses rather than merely improving answer aggregation. These results confirm that EDIS reliably indicates reasoning quality, enabling substantial improvements without exter- nal supervision. Detailed per-model breakdowns across all temperatures are provided in Appendix C. 5.2. Comparison with Other Selection Methods To evaluate the effectiveness of EDIS relative to other se- lection methods, we compare against several baselines: Mean (unweighted average), Majority Voting (most fre- quent answer), Sequence Entropy (mean token-level en- 5 ===== PAGE 6 / 16 ===== EDIS: Diagnosing LLM Reasoning via Entropy Dynamics Figure 4. EDIS-based best-k-of-N selec...

## Conclusion / Discussion Extract

Conclusion We introduced EDIS, a trajectory-level metric that captures instability patterns in entropy evolution during LLM rea- soning. The central insight is that reasoning quality can be diagnosed from how confidence evolves during generation, not just its average value. By shifting from static to dynamic analysis, EDIS extracts richer signal from token-level data that prior methods reduce to summary statistics. The char- acteristic instability patterns—burst spikes and peak-valley spikes—persist across models, temperatures, and training stages, suggesting they reflect fundamental properties of reasoning failure. EDIS achieves an 82% relative accuracy improvement for inference-time selection and up to +7.7 percentage points gains for RL training, consistently outper- forming alternative confidence measures. 8 ===== PAGE 9 / 16 ===== EDIS: Diagnosing LLM Reasoning via Entropy Dynamics References Brown, B., Juravsky, J., Ehrlich, R., Clark, R., Le, Q. V., R´e, C., and Mirhoseini, A. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787, 2024. Cao, S., Valiant, G., and Liang, P. On the entropy calibration of language models. arXiv preprint arXiv:2511.11966, 2025. Chen, J. and Mueller, J. Quantifying uncertainty in answers from any language model and enhancing their trustwor- thiness. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5186–5200, 2024. Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. Cui, G., Zhang, Y., Chen, J., Yuan, L., Wang, Z., Zuo, Y., Li, H., Fan, Y., Chen, H., Chen, W., et al. The entropy mech- anism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025. Desai, S. and Durrett, G. Calibration of pre-trained trans- formers. arXiv preprint arXiv:2003.07892, 2020. Farquhar, S., Kossen, J., Kuhn, L., and Gal, Y. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630, 2024. Guo, C...

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

TBD: add short excerpts with page markers from `../texts/edis-diagnosing-llm-reasoning-via-entropy-dynamics.txt`.
