# All-ProcessBench Two-GPU Response Pipeline

The audited all-dataset entry point runs in the foreground and streams progress:

```bash
mkdir -p outputs/job_logs
PYTHONUNBUFFERED=1 bash hypergraph/attention/scripts/run_all_processbench_response_pipeline.sh \
  --layer 14 \
  --folds 5 \
  --seeds 17 \
  --generator-model Llama-3.1-8B-Instruct \
  2>&1 | tee outputs/job_logs/all_processbench_layer14.log
```

This is the generator-tag-matched reconstructed-observer main experiment. The wrapper
accepts only the explicit `Meta-Llama-*`/`Llama-*` naming alias, but a local path does
not prove an exact weight revision. Remove
`--generator-model` only for the separately reported all-generator observer
experiment. Existing complete all-generator traces are filtered and reused; when no
complete cache exists, the wrapper first materializes an audited generator cohort and
forwards only the matching rows.

Do not append `&` or wrap this command in `nohup` when interactive progress is
required. Extraction reports sample progress for the active extraction worker, and
training streams each epoch record while preserving the same output in log files.

It runs these ProcessBench subsets independently:

```text
gsm8k
math
olympiadbench
omnimath
```

Each name resolves to:

```text
data/hf_datasets/ProcessBench/<dataset>.json
```

## GPU Scheduling

For each dataset, extraction defaults to `MODE=model_parallel` with
`QUERY_CHUNK_SIZE=0`:

- the observer model is balanced over physical GPUs 0 and 1;
- attention comes from an exact full-sequence teacher-forcing forward;
- only the requested decoder layer is retained through a temporary self-attention hook;
- `hypergraph.attention.shards` verifies the resulting trace scope and writes
  `shard_audit.json`.

`MAX_SEQ_LEN` defaults to `0`, meaning there is no user-imposed token cap and
no truncation. The extractor still rejects a sequence that exceeds the model's
declared context window. Dense trace storage remains quadratic, so
`MAX_ATTENTION_GIB` is an independent allocation guard rather than a sequence
length limit. Interactive extraction is rendered by `tqdm` with elapsed time,
rate, and ETA.

Cached query chunks are not part of the strict pipeline. On the real
Llama-3.1-8B checkpoint they changed edges selected at the `0.01` topology
threshold, despite satisfying the synthetic tensor contract.

Training then schedules at most two fold jobs concurrently:

- one fold on physical GPU 0;
- one fold on physical GPU 1;
- the next pair starts only after the current wave finishes.

Datasets run sequentially. Override physical device identifiers with:

```bash
GPU0=0 GPU1=1 TRAIN_GPUS=0,1 \
bash hypergraph/attention/scripts/run_all_processbench_response_pipeline.sh \
  --layer 14
```

## Outputs

Per-dataset traces:

```text
outputs/attention_traces/<dataset>_llama31_layer14/
```

Per-dataset held-out results:

```text
outputs/attention_hypergraph/<dataset>_response_layer14_matched_Llama-3.1-8B-Instruct/aggregate_results.json
outputs/attention_hypergraph/<dataset>_response_layer14_matched_Llama-3.1-8B-Instruct/pooled_oof_results.json
outputs/attention_hypergraph/<dataset>_response_layer14_matched_Llama-3.1-8B-Instruct/predictions_pooled_oof_seed_ensemble.csv
outputs/attention_hypergraph/<dataset>_response_layer14_observer_all/aggregate_results.json
```

The old unsuffixed path is legacy. New all-generator results use `_observer_all`;
matched-generator results use `_matched_<generator>` so the two
experimental populations cannot overwrite each other.

Cross-dataset report:

```text
outputs/attention_hypergraph/all_processbench_response_layer14_matched_Llama-3.1-8B-Instruct/aggregate_results.json
outputs/attention_hypergraph/all_processbench_response_layer14_matched_Llama-3.1-8B-Instruct/summary.md
outputs/attention_hypergraph/all_processbench_response_layer14_observer_all/aggregate_results.json
outputs/attention_hypergraph/all_processbench_response_layer14_observer_all/summary.md
```

The primary per-dataset result is `pooled_oof_test.seed_ensemble`: each trace is
predicted exactly once by its held-out fold for each seed, probabilities are averaged
per trace across seeds, and one final AUROC/AUPRC is computed. `test_aggregate` remains
the mean/std/min/max of individual fold runs and is a variability diagnostic, not the
final test AUROC. The primary cross-dataset result is the unweighted macro average of
the four dataset-level pooled OOF metrics. Each per-dataset JSON also contains
`generator_test_aggregate`; undefined single-class generator/fold metrics remain
explicit rather than being silently pooled into the overall score.

The default five-fold protocol is problem-disjoint. For fold `k`, fold `k` is test,
fold `(k+1) mod 5` is validation, and the remaining three folds are training data.
After all five runs, every problem has appeared in test exactly once per seed. This is
used because the ProcessBench traces in this pipeline do not provide a reusable
official train/validation/test partition. If an official split is present, training
refuses to repartition it by default.

To aggregate already completed folds without extracting or training again:

```bash
python hypergraph/attention/aggregate_oof.py \
  --run-root outputs/attention_hypergraph/gsm8k_response_layer14_matched_Llama-3.1-8B-Instruct \
  --folds 5 \
  --seeds 17
```

Completed audited traces and completed fold runs are reused. Before reuse, manifests
are freshly audited and preflight runs the same cohort gate as training. The legacy
monolithic code hash remains untouched as historical provenance; extraction and
training hashes are separate for new requests. A partial directory is rejected rather
than overwritten. The cross-dataset aggregator verifies that every `preflight.json`
matches the SHA256 bound by its per-dataset `pipeline_request.json`, and then requires
the observer/template/axis/graph plus current validation/training-code signatures to
agree across datasets. Legacy/v2 request schemas remain recorded per dataset but do
not create a false incompatibility when the trace-embedded representation provenance
itself agrees.

Preserve each trace directory together with its `pipeline_request.json`,
`shard_audit.json`, and any matched-cohort `.report.json`. Bare NPZ files are not a
self-contained audit package. The preflight is a cohort/provenance/graph gate only; it
does not prove fold class coverage, successful optimization, causal recovery of the
original generation, or control of response-length confounding.

The failed legacy run `gsm8k_response_layer14` is not reused by either new suffix. If
you want it out of the way, preserve it with a timestamped backup rather than deleting
the completed trace cache:

```bash
stamp="$(date +%Y%m%d_%H%M%S)"
mv -- outputs/attention_hypergraph/gsm8k_response_layer14 \
  "outputs/attention_hypergraph/gsm8k_response_layer14_failed_${stamp}"
```
