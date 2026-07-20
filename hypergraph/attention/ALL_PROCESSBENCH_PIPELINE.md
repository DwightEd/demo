# All-ProcessBench Two-GPU Response Pipeline

The audited all-dataset entry point runs in the foreground and streams progress:

```bash
mkdir -p outputs/job_logs
PYTHONUNBUFFERED=1 bash hypergraph/attention/scripts/run_all_processbench_response_pipeline.sh \
  --layer 14 \
  --folds 5 \
  --seeds 17 \
  2>&1 | tee outputs/job_logs/all_processbench_layer14.log
```

Do not append `&` or wrap this command in `nohup` when interactive progress is
required. Extraction reports a per-shard sample progress bar, and training streams
each epoch record while preserving the same output in log files.

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

For each dataset, extraction uses `MODE=data_parallel`:

- physical GPU 0 loads the observer model and extracts complementary `shard0`;
- physical GPU 1 loads the observer model and extracts complementary `shard1`;
- `hypergraph.attention.shards` verifies coverage and writes `shard_audit.json`.

Training then schedules at most two fold jobs concurrently:

- one fold on physical GPU 0;
- one fold on physical GPU 1;
- the next pair starts only after the current wave finishes.

Datasets run sequentially so that extraction never creates more than two simultaneous
8B model replicas. Override physical device identifiers with:

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
outputs/attention_hypergraph/<dataset>_response_layer14/aggregate_results.json
```

Cross-dataset report:

```text
outputs/attention_hypergraph/all_processbench_response_layer14/aggregate_results.json
outputs/attention_hypergraph/all_processbench_response_layer14/summary.md
```

The cross-dataset values are unweighted macro averages of the four dataset-level
held-out means. They are not pooled predictions and must not be reported as a pooled
ProcessBench AUROC.

Completed audited traces and completed fold runs are reused. A partial directory is
rejected rather than overwritten; inspect its logs before removing it.

To remove only failed or interrupted layer-14 trace directories, while retaining
every completed audited extraction:

```bash
for dataset in gsm8k math olympiadbench omnimath; do
  dir="$PWD/outputs/attention_traces/${dataset}_llama31_layer14"
  if [ -d "$dir" ] && [ ! -f "$dir/shard_audit.json" ]; then
    echo "removing incomplete trace: $dir"
    rm -rf -- "$dir"
  fi
done
```
