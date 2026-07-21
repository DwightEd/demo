# All-ProcessBench Two-GPU Response Pipeline

The audited entry point runs GSM8K, MATH, OlympiadBench, and OmniMath in the
foreground and streams extraction and training progress.

## Attention-Only Baseline

```bash
mkdir -p results/job_logs

MODEL=/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct \
GPU0=0 GPU1=1 TRAIN_GPUS=0 PYTHONUNBUFFERED=1 \
bash hypergraph/attention/scripts/run_all_processbench_response_pipeline.sh \
  --layer 14 \
  --seed 17 \
  --generator-model Llama-3.1-8B-Instruct \
  2>&1 | tee results/job_logs/all_processbench_layer14_attention.log
```

This is the closest ProcessBench adaptation of the local original method:

- every token is a node;
- node content is the 32-dimensional self-attention diagonal from the selected
  Llama layer;
- response attention rows from topology head 0 create hyperedges;
- source selection uses `tau=0.05`, positive top-16 fallback only when no source
  crosses the threshold, and at least two sources before adding the receiver;
- every hyperedge has exactly three attributes: attention mean, attention max,
  and normalized flattened stored-head index.

## Hidden-State Node Variants

To concatenate hidden state with the attention-diagonal node feature:

```bash
NODE_FEATURE_MODE=diagonal_plus_activation MAX_SEQ_LEN=0 \
MODEL=/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct \
GPU0=0 GPU1=1 TRAIN_GPUS=0 PYTHONUNBUFFERED=1 \
bash hypergraph/attention/scripts/run_all_processbench_response_pipeline.sh \
  --datasets gsm8k,math,olympiadbench,omnimath \
  --layer 14 \
  --seed 17 \
  --generator-model Llama-3.1-8B-Instruct \
  2>&1 | tee results/job_logs/all_processbench_layer14_attention_hidden.log
```

`ACTIVATION_LAYER` is a Hugging Face `hidden_states` index. In hidden-node modes
it defaults to `layer + 1`; therefore `--layer 14` stores `hidden_states[15]`, the
output after zero-based decoder block 14. Override it explicitly only for a
pre-registered input-state versus output-state ablation.

Use `NODE_FEATURE_MODE=activation_only` to set each node to hidden state alone.
In both hidden variants, attention still defines topology and the three
hyperedge attributes remain unchanged. Hidden modes are innovations rather than
faithful reproductions. They also increase input dimension and parameter count,
so they must be compared with feature-only, parameter-matched, and length-control
baselines before attributing gains to hypergraph reasoning.

## Sequence And Cache Policy

`MAX_SEQ_LEN=0` disables the user-imposed token cap. It does not disable the
model's context window or the independent `MAX_ATTENTION_GIB` allocation guard.
Truncation is forbidden.

Trace paths encode the sequence and node-content policy:

```text
attention-only, no cap:
  data/attention_traces/<dataset>_llama31_layer14_nocap/

hidden_states[15], no cap:
  data/attention_traces/<dataset>_llama31_layer14_nocap_hidden_hs15/
```

Consequently, an old cache whose request says `max_seq_len=2048` is never reused
as an uncapped cache. The old directory is preserved; the new run writes a
separate `_nocap` directory without requiring deletion or manual renaming.

Extraction uses both GPUs in `model_parallel` mode. The response-level graph
model is small and a single fixed run trains on `TRAIN_GPUS`' first device. Do
not append `&` or use `nohup` when interactive `tqdm` progress is required.

## Evaluation Protocol

ProcessBench does not provide the original project's external RAGTruth train and
test directories. Each subset therefore receives one deterministic,
problem-disjoint 70/10/20 train/validation/test split:

- split seed 17 fixes the data partition;
- validation AUPRC selects the best epoch;
- the held-out test is evaluated exactly once;
- no five-fold OOF aggregation is performed.

The split manifest stores all trace and problem IDs. A single-class validation
or test partition fails closed because AUROC would be undefined.

## Outputs

Attention-only, no-cap, matched-generator output:

```text
results/attention_hypergraph/
  <dataset>_response_layer14_matched_Llama-3.1-8B-Instruct_node_attention_nocap_fixed_original/
    fixed_seed17/results.json
    aggregate_results.json
    predictions_test.csv
    split_manifest.json
```

Combined hidden output replaces `_node_attention_` with
`_node_attention_hidden_hs15_`; hidden-only output uses `_node_hidden_hs15_`.

The four-dataset report follows the same suffix and contains:

```text
aggregate_results.json
summary.md
pipeline_request.json
```

It reports each dataset's final test AUROC/AUPRC and their unweighted macro
average. Old `fold*_seed*`, `pooled_oof_results.json`, and
`predictions_pooled_oof_seed_ensemble.csv` files belong to the retired protocol
and are not read by this entry point.

## One-Time Migration From The Old Layout

Existing traces and completed runs do not need to be recomputed. After pulling
the layout change, run once:

```bash
bash hypergraph/attention/scripts/migrate_artifacts_layout.sh
```

The migration is fail-closed: it moves only the three hypergraph-owned
directories and refuses to merge when a destination already exists. It does
not touch residual-flow data, unrelated `outputs/` content, or source datasets.

Every run remains bound to its source hashes, trace request, extraction manifest,
graph configuration, node mode, hidden-state index, sequence policy, and split
seed. A partial or semantically different cache fails closed rather than being
silently mixed into the experiment.
