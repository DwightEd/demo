# Reasoning Trace Data Contract

This document is the source of truth for data used by hidden-state geometry,
output uncertainty, and causal intervention experiments. The contract separates
three different estimands that cannot be recovered from one another after the
fact.

## 1. Three estimands, three evidence tiers

| tier | question | source | valid claim |
|---|---|---|---|
| Benchmark observer | Can one frozen observer diagnose a published candidate solution? | ProcessBench problem, reformatted steps, human first-error label | Observer-state detection and first-error localization |
| Same-problem self generation | Holding problem, checkpoint, and prompt fixed, how do sampled correct and incorrect trajectories differ? | (K) generations per problem from one checkpoint | Difficulty-controlled response geometry and consensus |
| Generation-matched trace | Did the generating model already exhibit a signal online, and is it causally useful? | Exact rendered generation prompt and exact generated token IDs | Online self-state analysis and residual intervention |

An observer trace is not an original-generator trace. A same-problem legacy
artifact without exact token IDs is not a generation-matched trace.

## 2. What ProcessBench does and does not contain

ProcessBench contains 3,400 expert-annotated solutions across GSM8K, MATH,
OlympiadBench, and Omni-MATH. Its central label is the earliest erroneous step.
The published solutions were normalized and re-segmented before annotation:
line breaks were removed and a separate model inserted double line breaks.
Consequently, the public JSONL does not preserve the original generator's
rendered prompt, token IDs, token boundaries, KV-cache trajectory, or hidden
states.

Primary sources:

- [ACL 2025 paper](https://aclanthology.org/2025.acl-long.50/)
- [Official repository](https://github.com/QwenLM/ProcessBench)
- [Official dataset](https://huggingface.co/datasets/Qwen/ProcessBench)

Therefore ProcessBench supports a strong **benchmark-observer** experiment:
teacher-force every published candidate through one frozen observer under one
declared prompt. It cannot by itself support the claim that the original
solution generator knew it was wrong while generating.

The current local source audit reports:

| subset | rows | unique problems | process errors | correct-final but process-error |
|---|---:|---:|---:|---:|
| GSM8K | 400 | 375 | 207 | 7 |
| MATH | 1,000 | 955 | 594 | 94 |
| OlympiadBench | 1,000 | 561 | 661 | 161 |
| Omni-MATH | 1,000 | 951 | 759 | 259 |

The last column is why `final_answer_correct` must never substitute for
`gold_error_step`. The source audit is executable:

```bash
python audit_reasoning_data.py \
  data/hf_datasets/ProcessBench/gsm8k.jsonl \
  data/hf_datasets/ProcessBench/math.jsonl \
  data/hf_datasets/ProcessBench/olympiadbench.jsonl \
  data/hf_datasets/ProcessBench/omnimath.jsonl \
  --strict_source \
  --output outputs/data_audit/processbench_sources.json
```

Paths can point to the canonical raw JSONL directory on the machine; the local
checkout currently uses `../data/processbench`.

## 3. Teacher forcing and token semantics

Teacher forcing reproduces a generation trace only when all of the following
are fixed:

1. identical model checkpoint and revision;
2. identical tokenizer and chat template revision;
3. exact rendered prompt token IDs;
4. exact generated response token IDs, including terminal-token handling;
5. identical attention mask and non-truncated prefix;
6. evaluation mode with no stochastic model layers.

A full causal forward over these IDs is mathematically equivalent to replaying
the same prefix incrementally, apart from small kernel/numerical differences.
Retokenizing decoded text is not equivalent: boundary merges, BOS insertion,
chat-template changes, and whitespace normalization can all change the state.

The shared trace schema fixes the causal indexing convention:

\[
h_i = \text{state after reading token } x_i,
\qquad
\ell_i = \text{logits predicting } x_{i+1}.
\]

For a target token at index (j), the state and logits available before its
generation are (h_{j-1}) and \(\ell_{j-1}\). The state (h_j) is a post-token
diagnostic and must not be called a predictor of token (j). The NPZ records:

```text
hidden_state_token_semantics = h_i_after_reading_token_i
logit_prediction_semantics = logits_i_predict_token_i_plus_1
step_prediction_position_shift = -1
```

## 4. Label contract

The following fields are distinct:

```text
gold_error_step       ProcessBench index; -1 means every annotated step is valid
process_correct       1/0; -1 means unavailable
final_answer_correct  1/0; correctness of the final answer only
format_ok             1/0; required output format was observed
is_correct            compatibility alias only; read is_correct_semantics first
```

New ProcessBench artifacts store both process and final-answer labels. New
self-sampled artifacts set `process_correct=-1` until a process annotation is
available. They must not fabricate `gold_error_step=-1`, because that value
means the entire process was inspected and found correct.

For same-problem training, automatic final-answer checks are acceptable for
GSM8K response-level labels. Confirmatory first-error localization requires
human/expert process labels. A symbolic verifier or strong critic can provide
weak training labels, but the held-out test labels must remain independent.

## 5. Canonical trace schema

### Provenance and text

```text
data_contract_version
trace_schema_version
dataset, subset, split, problem_id, sample_idx
source_generators
model_name, model_revision, tokenizer_name, tokenizer_revision
prompt_style, prompt_provenance, response_provenance
prompts, questions, responses, steps_text
temperature, top_p, seed, max_new_tokens
trace_semantics, label_semantics, is_correct_semantics
```

### Exact token axis

```text
prompt_token_ids, prompt_attention_mask
generated_token_ids, generation_terminal_token_ids
input_ids, attention_mask, token_offsets
full_input_ids, full_attention_mask, full_token_offsets
prompt_token_counts, response_token_ranges
step_token_ranges, time_axis_token_ranges
model_input_truncated
```

### Output-side evidence

At minimum retain compact per-token output summaries aligned to the prediction
position:

```text
chosen_token_logprob
token_entropy
top1_top2_margin
topk_mass
optional top-k token IDs and logits
```

Do not persist the full vocabulary tensor unless a specific experiment needs
it. Compact summaries are enough for output-only baselines and conditional
geometry tests.

### Hidden-state evidence

Raw response-token hidden states at selected declared layers are the source of
truth. Derived step means, spectra, curvature, tangent coordinates, and scalar
scores should be recomputable offline. Prompt-token hidden states should be
retained when prompt anchors or question-conditioned spaces are planned.

Full attention is quadratic and remains opt-in. Save compact ICR or routing
summaries unless a small diagnostic subset explicitly needs full attention.

## 6. Storage layout

A single object-array NPZ is acceptable for a 10-sample smoke test, but it is
not the long-term full-data format. The canonical full extraction should use:

```text
manifest.parquet or manifest.jsonl     row metadata, text, labels, provenance
tokens.npz or packed .npy              ragged IDs/ranges plus offsets
hidden/layer_<L>/shard_<S>.safetensors raw hidden states
outputs/shard_<S>.npz                  compact output-side summaries
derived/<experiment>.npz               recomputable geometry/features
skip_report.jsonl                      every rejected row and exact reason
```

This keeps raw evidence immutable, supports GPU-batched offline derivation,
and avoids loading multi-gigabyte object arrays to inspect metadata.

## 7. Current artifacts and their valid uses

| artifact | current status | valid now | not valid now |
|---|---|---|---|
| `data/features/full_*.npz` | Legacy ProcessBench observer features; sparse step vectors and selected token-hidden shards; no original prompt/token trace | Cross-problem and first-error exploratory geometry | Original-generator state, exact online replay |
| `data/gsm8k_v2_custom.npz` | Legacy same-problem samples; raw layer-16 token clouds; no exact generation token axis | Same-problem response geometry and donor exploration | Confirmatory causal intervention or token-ID controls |
| `data/gsm8k_v2_5shot.npz` | Same limitation under a different prompt style | Prompt-style replication at response level | Generation-matched causal claims |
| New `10_sample_and_extract.py` output | Exact chat-rendered prompt and generated token IDs, then exact teacher-forcing replay | Generation-matched self-state and same-problem analysis | First-error claims without new process labels |
| New `01_extract_spectral_field.py` output | Exact token axis for a declared fixed observer prompt over ProcessBench text | Benchmark-observer first-error analysis | Original generator's internal trajectory |

Run the same audit on NPZ artifacts:

```bash
python audit_reasoning_data.py \
  data/features/full_gsm8k.npz \
  data/gsm8k_v2_custom.npz \
  data/gsm8k_v2_5shot.npz \
  --output outputs/data_audit/current_artifacts.json
```

## 8. Recommended experiment order

1. **Source audit first.** Fail on malformed or missing ProcessBench rows.
2. **Benchmark observer baseline.** Extract all four subsets with one frozen
   observer and one declared prompt; split/group by problem and report by source
   generator. This establishes what is detectable in the benchmark.
3. **Same-problem pilot.** Use one checkpoint and exact prompt with at least
   (K=8) samples per problem. Promote to (K=16\text{-}32) for stable
   conditional-neighborhood or consensus estimation.
4. **Contrastive coverage audit.** Report problems with both correct and error
   samples. Never keep sampling only until a desired label appears without
   recording the fixed budget.
5. **Process annotation.** Add human first-error labels to a held-out subset of
   self-generations before making token/step localization claims.
6. **Conditional tests.** Compare output-only, geometry-only, and joint models
   with problem-grouped out-of-fold predictions. Report incremental log loss,
   usable bits, partial (R^2), and matched nulls.
7. **Causal tests last.** Intervene only on generation-matched raw residual
   states after replay identity and coverage gates pass.

ProcessBench is the right starting benchmark for diagnosis. Same-problem data
is a necessary complementary dataset for difficulty-controlled geometry, not a
replacement for the benchmark.

## 9. Current extraction entry points

Generation-matched same-problem data (use (K=8) only as a pilot; increase the
frozen budget after the pipeline passes):

```bash
python 10_sample_and_extract.py \
  --model /gz-data/models/Meta-Llama-3.1-8B-Instruct \
  --dataset_format processbench \
  --dataset data/hf_datasets/ProcessBench \
  --subset gsm8k \
  --n_problems 300 \
  --k_samples 8 \
  --prompt_style custom_zeroshot \
  --layers 8,10,12,14,16,18,20,22 \
  --store_vectors \
  --sv_modes step_exp \
  --store_clouds \
  --cloud_layers 16 \
  --store_prompt_hidden \
  --prompt_hidden_layers 16 \
  --store_token_uncertainty \
  --output data/exact/gsm8k_same_problem_k8_l16.npz
```

Exact benchmark-observer pilot over ProcessBench (the fixed plain prompt is
declared observer context, not the unknown original generation prompt):

```bash
python 01_extract_spectral_field.py \
  --model /gz-data/models/Meta-Llama-3.1-8B-Instruct \
  --dataset data/hf_datasets/ProcessBench \
  --subset gsm8k \
  --n_correct 50 \
  --n_error 50 \
  --layers 8,10,12,14,16,18,20,22 \
  --no_reasoning_subspace \
  --step_vectors \
  --sv_modes step_exp \
  --store_vectors \
  --store_clouds \
  --cloud_layers 14,22 \
  --store_prompt_hidden \
  --prompt_hidden_layers 14,22 \
  --store_token_outputs \
  --output data/exact/processbench_gsm8k_observer_pilot.npz
```

The current NPZ writer is appropriate for pilots. Do not scale the raw-cloud
configuration to all 3,400 rows and many layers in one object-array file. Use
the sharded layout in Section 6 for the confirmatory extraction.
