# Same-Problem Multisample Notes

## Result Analysis

The same-problem multisample audit is a stronger gate than the earlier cross-problem ProcessBench setting.  It asks whether a score can rank an incorrect sample above a correct sample for the same question, so problem difficulty is largely controlled.

Current results show that most trajectory summaries collapse under this gate.  On `gsm8k_v2_5shot.npz` with `answer_format_ok`, the run has 2646 samples, 291 problems, 94 contrastive problems, 701 contrastive samples, 279 answer errors, and 1756 correct samples.  Most within-problem AUROCs are near 0.50-0.60.  On `gsm8k_v2_custom.npz`, the run has 3600 samples, 300 problems, 147 contrastive problems, 1658 contrastive samples, 532 answer errors, and 2920 correct samples; the same collapse largely holds.

This means the earlier strong detection numbers cannot be interpreted as proof that the current features learned a reasoning-trajectory failure pattern.  A large part of the signal is likely question difficulty, response length, format behavior, or global uncertainty.

The label policy matters.  Local comprehensive stats show that `gsm8k_v2_5shot.npz` has 2646 total samples, 2118 lenient-correct samples, but only 1756 strict-correct samples.  Under `answer_format_ok`, 2035 samples remain, so roughly 611 samples are format failures; 362 of these were counted correct by lenient last-number fallback.  This can create artificial separation if format failures are mixed with reasoning failures.  `gsm8k_v2_custom.npz` is cleaner: 3600 total samples and 3452 format-ok samples, so roughly 148 format failures.

The saved multisample npz is useful but uneven:

- Always saved: `problem_ids`, `sample_idx`, `is_correct`, `is_correct_strict`, `format_ok`, `pred_source`, `n_steps`, `pred_answers`, `gold_answers`, `layers_used`, `sv_modes`, `sv_pr_<mode>`, `sv_ae_<mode>`, `sv_out_entropy`, `responses`, `steps_text`, `step_split`, `prompt_style`, `model_name`, and dataset metadata.
- Optionally saved: raw step vectors `sv_vec_<mode>` when `--store_vectors` is used.
- Optionally saved: raw per-step token clouds `sv_clouds`, `sv_cloud_sizes`, and `cloud_layers` when `--store_clouds` is used.
- Newly fixed for future extraction: `sv_out_committal`, and optional response-token traces `sv_tok_entropy` / `sv_tok_committal` are now saved when computed.

## Follow-Up Research Direction

The main question should shift from "do error chains have larger spread?" to "holding the problem fixed, where does an incorrect sample leave the constrained transition tube that correct samples still respect?"

Because the target module must work during normal generation, methods should not depend on an extra verification prompt.  The useful part of the internal-confidence literature is therefore not the extra verifier forward, but the idea that boundary positions can carry a second-order self-evaluation state.  In our setting this should be approximated from the original forward pass:

- Step-boundary hidden state: the last token or newline at the end of each generated step.
- Boundary logits: entropy and committal for the next realized token at the step boundary.
- Boundary geometry: local spread, resultant, and prompt/prefix-conditioned residual at the boundary.
- Optional token trace: within-step entropy/committal and hidden reconvergence when token traces are saved.

The next detector should be local and non-cumulative.  A candidate score is a one-step transition residual:

```text
r_t = distance(z_t, T(z_{t-1}, prompt_state, position, step_length))
```

where `z_t` is a boundary/step representation and `T` is a learned or robustly fitted healthy transition tube.  The score must be time-debiased and tested on correct-chain false alarms.  CUSUM-style accumulation should not be promoted as an onset detector.

Same-problem data should also be used to split failure modes:

- Committed failure: confidence/committal rises or stays high while geometry leaves the prompt-conditioned tube.
- Persistent uncertainty: entropy and spread stay high without a clean commitment.
- Format/truncation failure: marker missing or no final answer; analyze separately from reasoning failure.
- Wrong-but-coherent failure: low spread but off-anchor or wrong attractor; this is where simple spread cannot work.

## Optimization Suggestions

Every new detector should report:

- same-problem paired AUROC;
- contrastive-problem count and same-problem pair count;
- cross-problem AUROC only as secondary context;
- length and format baselines;
- correct-chain false localization / false alarm position;
- `answer_format_ok` as the main reasoning-failure policy, with strict/format failures reported separately.

Before building heavier models, run the new data audit:

```bash
python multisample_data_audit.py \
  --input /gz-data/research/demo/data/gsm8k_v2_5shot.npz \
  --output_dir outputs/multisample_data_audit

python multisample_data_audit.py \
  --input /gz-data/research/demo/data/gsm8k_v2_custom.npz \
  --output_dir outputs/multisample_data_audit
```

Then inspect the distributions of the feature families that previously looked useful:

```bash
python multisample_feature_distribution.py \
  --input /gz-data/research/demo/data/gsm8k_v2_5shot.npz \
  --output_dir outputs/multisample_feature_distribution

python multisample_feature_distribution.py \
  --input /gz-data/research/demo/data/gsm8k_v2_custom.npz \
  --output_dir outputs/multisample_feature_distribution
```

For the actual "is there a local rupture signal?" question, run the temporal audit.  This keeps each sampled answer as a full per-step trajectory and compares local jumps/window contrasts against static level summaries:

```bash
python multisample_temporal_rupture_audit.py \
  --input /gz-data/research/demo/data/gsm8k_v2_5shot.npz \
  --policies answer_format_ok \
  --output_dir outputs/multisample_temporal_rupture

python multisample_temporal_rupture_audit.py \
  --input /gz-data/research/demo/data/gsm8k_v2_custom.npz \
  --policies answer_format_ok \
  --output_dir outputs/multisample_temporal_rupture
```

Read this audit as follows:

- If `*.jump_max`, `*.zjump_max`, `*.contrast_max`, or `multi.zjump_*` do not beat the corresponding `*.level_*` summaries under same-problem AUROC, the current data/signals do not yet support a genuine abrupt-transition detector.
- If a local score does survive, inspect its `argpos_error` and `argpos_correct`.  A useful intervention trigger should not simply fire late in every chain.
- Because the same-problem data only has final-answer labels, this audit can establish whether a rupture signature exists, but cannot prove that the rupture aligns with the true first wrong step.

Current temporal audit finding:

- On `gsm8k_v2_5shot.npz` under `answer_format_ok`, the strongest local-looking score is `vec_norm_mid.contrast_max` with within-problem AUROC 0.646, but its median argpos is late for both error and correct chains (`0.80` vs `0.75`).  This is not a clean online rupture signal.
- On `gsm8k_v2_custom.npz` under `answer_format_ok`, the strongest scores are still static cloud spread/resultant levels: `cloud_spread.level_late` within 0.659 and `cloud_spread.level_max` within 0.639.  The local contrast version `cloud_spread.contrast_max` drops to 0.579, and multi-channel rupture scores do not help.
- Several apparent dynamic scores point in the wrong direction or fire at the end of both correct and error chains, e.g. `abs_zjump_max` / `multi.zcontrast_l2_max` with argpos near `1.00`.  Treat these as endpoint/length artifacts, not first-failure detectors.
- Conclusion: with the currently saved same-problem signals, there is weak same-problem static separation but no robust evidence of an abrupt local transition.  The current signal family should not be presented as a dynamic rupture detector.

To test a stronger constrained-manifold reading, fit a low-rank transition tube from correct step-vector transitions:

```bash
python multisample_transition_tube_audit.py \
  --input /gz-data/research/demo/data/gsm8k_v2_5shot.npz \
  --policies answer_format_ok \
  --band mid \
  --normalize l2 \
  --output_dir outputs/multisample_transition_tube

python multisample_transition_tube_audit.py \
  --input /gz-data/research/demo/data/gsm8k_v2_custom.npz \
  --policies answer_format_ok \
  --band mid \
  --normalize l2 \
  --output_dir outputs/multisample_transition_tube
```

Read this audit carefully:

- `global.*` is the deployable-style setting: fit the transition tube on correct chains from training problems and score held-out problems.
- `oracle.*` is a diagnostic setting: fit the tube from same-problem correct samples.  It tests whether a question-specific correct transition manifold exists, but it is not an online detector.
- Important leakage guard: correct target samples must be excluded from their own same-problem tube.  Otherwise the correct class is evaluated on training residuals and `oracle.off_energy_ratio` can become artificially near-perfect.
- If `oracle.*` works but `global.*` fails, the correct manifold is problem-conditioned and needs prompt/anchor conditioning.
- If `rank_energy` or `transition_eff_rank` works, error chains need more transition directions.  If only `off_*` works, error chains leave the tube by residual magnitude rather than intrinsic rank.

For future data extraction, preserve the single-forward confidence traces:

```bash
python 10_sample_and_extract.py ... \
  --store_vectors \
  --store_token_uncertainty
```

Refined tube validation now lives in `multisample_tube_refinement_audit.py`.
It is the preferred next audit after the basic transition-tube script because
it separates three claims:

- `support_oracle.*`: same-problem correct support tube with heldout correct
  samples and error samples scored against the same support folds.
- `global_tail.*`: deployable-style cross-problem tube plus spectral-tail
  features such as `tail_k90`, `tail_auc`, and residual-at-k.
- `conditioned.*`: local tube from nearest training problems under `qvec` /
  prompt vectors if present, or a clearly labeled first-step hidden proxy if
  the multisample file lacks prompt vectors.
- `layer.*`: adjacent-layer transition desynchronization and layer effective
  rank, only when `sv_vec_step_exp` stores multiple layers.

Run:

```bash
python multisample_tube_refinement_audit.py \
  --input /gz-data/research/demo/data/gsm8k_v2_5shot.npz \
  --policies answer_format_ok \
  --band mid \
  --normalize l2 \
  --output_dir outputs/multisample_tube_refinement

python multisample_tube_refinement_audit.py \
  --input /gz-data/research/demo/data/gsm8k_v2_custom.npz \
  --policies answer_format_ok \
  --band mid \
  --normalize l2 \
  --output_dir outputs/multisample_tube_refinement
```

If storage allows, also use `--store_clouds --cloud_layers 10,14,18,22` for within-step reconvergence and boundary-free step discovery experiments.

Latent-state validation now lives in `latent_constraint_em_audit.py`.  This
script does not reconstruct a concrete tube.  It treats the constrained
reasoning manifold as a hidden state learned by EM:

- observations: `cloud_spread`, `sv_out_entropy`, `sv_out_committal`,
  `sv_pr_step_exp`, `sv_ae_step_exp`, and step-vector jump;
- no layer desynchronization or cross-layer tensor signal;
- EM is unsupervised; chain labels only orient learned states into a risk score
  on the training fold;
- scoring uses online filtered posteriors `p(z_t | x_1...x_t)`.

Recommended first run:

```bash
python latent_constraint_em_audit.py \
  --input /gz-data/research/demo/data/gsm8k_v2_5shot.npz \
  --policies answer_format_ok \
  --band mid \
  --states 4 \
  --feature_groups all_effective=cloud_spread,out_entropy,out_committal,pr_mid,ae_mid,step_jump,direction_jump \
  --output_dir outputs/latent_constraint_em

python latent_constraint_em_audit.py \
  --input /gz-data/research/demo/data/gsm8k_v2_custom.npz \
  --policies answer_format_ok \
  --band mid \
  --states 4 \
  --feature_groups all_effective=cloud_spread,out_entropy,out_committal,pr_mid,ae_mid,step_jump,direction_jump \
  --output_dir outputs/latent_constraint_em
```

Then run the default feature-group ablation.  The key comparison is whether
`hmm.spread_entropy_moment.*` or `hmm.all_effective.*` beats
`hmm.spread_entropy.*` and the corresponding single-feature baselines under
same-problem paired AUROC.

Second-moment and matrix-valued dynamics are now separated into
`SECOND_MOMENT_DYNAMICS_RESEARCH_PLAN.md`.  That note records why the scalar
EM/HMM audit is not enough, how to reinterpret second moments as Gram-matrix
trajectory structure, and what gates the next audit must pass before we treat
this branch as a serious paper claim.
