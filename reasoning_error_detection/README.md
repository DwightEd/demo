# ProcessBench Reasoning Error Detection

This project tests whether a reasoning step is the first incorrect step using
features already extracted from ProcessBench. It does not generate synthetic
alias tasks, reload the language model, or claim that a probe explains the
causal mechanism of an error.

## Execution path

The entry point is `main.py`. It performs one direct workflow:

```text
ProcessBench geometry/trace.npz
  -> ProcessBenchFeatureLoader.load()
  -> ProcessBenchErrorDetector.run()
  -> summary.json + oof_predictions.npz
```

Run it as a module from the repository root:

```bash
python -m reasoning_error_detection.main \
  --input /path/to/processbench/gsm8k/geometry/trace.npz \
  --output_dir /path/to/results/gsm8k \
  --layers 8,10,12,14,16,18,20,22 \
  --device cuda:0
```

`run_processbench.sh` runs GSM8K, MATH, OlympiadBench, and OmniMath. Its paths
and experiment parameters are grouped at the top of the file so a remote run
does not depend on hidden configuration.

## Target and features

The positive row is the annotated `gold_error_step`. Negatives are steps before
the first error and all steps from process-correct chains. Steps after the first
error are excluded from onset training and onset metrics. Cross-validation is
grouped by `problem_group_id`, preventing a problem from appearing in both the
training and test folds.

The reported feature sets separate the main hypotheses:

- `controls`: step index and current, previous, and cumulative token counts;
- `output`: output uncertainty features such as entropy and NLL, when present;
- `routing`: attention-derived prompt-flow or ICR features, when present;
- `residual`: fixed random projections and norms of the per-layer update
  `step_end_state - step_pre_state`, always conditioned on controls;
- `joint`: all available modalities plus controls.

This control baseline is essential: improvement by residual or routing features
must be measured over length and step-position information, not inferred from a
raw correlation. The random projection is fixed by layer and seed and is never
fit to labels.

## Interpretation

The current experiment is an offline, retrospective detector. It observes the
end of a step, so it can test whether the just-completed step was the first
error. It is not yet an online pre-step warning system and is not a drop-in
module inside model generation. A deployable monitor can be built later from
the same public interface, but it must use only causal pre-step features and be
evaluated separately.

Attention tensors are not required at analysis time. If the extraction stored
attention-derived step scores such as `prompt_frac`, `off_prompt`, or `icr_*`,
the loader includes them as routing features. Hidden-state pre/end views are
required because they define the residual update being tested.

## Outputs

`summary.json` contains problem-balanced onset AUROC/AUPRC/NLL, first-error
localization rank/top-1, full-chain retrospective detection, split diagnostics,
and the exact feature names. `oof_predictions.npz` stores every out-of-fold
score with chain, problem, step, and label alignment for later statistics and
plots.

The strongest supported claim from this experiment is predictive: whether
routing/output/residual features add held-out detection value beyond explicit
length controls. A causal routing or Bayesian-update claim requires additional
interventions, ablations, and cross-model replication.
