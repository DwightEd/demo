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

For future data extraction, preserve the single-forward confidence traces:

```bash
python 10_sample_and_extract.py ... \
  --store_vectors \
  --store_token_uncertainty
```

If storage allows, also use `--store_clouds --cloud_layers 10,14,18,22` for within-step reconvergence and boundary-free step discovery experiments.
