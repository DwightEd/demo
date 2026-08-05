# Component-Resolved Hazard Method

This method uses stored teacher-forcing replay traces to build
`component_step_v1` artifacts for the already implemented
`component_resolved_hazard` detector.

## Required Data

For every selected ProcessBench domain under
`<data_root>/<domain>/selected/`, extraction requires:

- `trace.raw_residual_stream.npz` with response-token hidden-state shard paths,
  layer IDs, prompt token counts, step token ranges, gold first-error labels,
  chain IDs, dataset names, and model provenance;
- `trace.npz` with `full_input_ids`, `full_attention_mask`, `chain_idx`,
  `prompt_token_counts`, `n_steps`, `step_token_ranges`, `gold_error_step`, dataset,
  model/tokenizer provenance, and output summaries;
- response-token hidden shards referenced by the raw residual manifest.

The extractor never re-tokenizes. It loads each stored full token sequence by
`chain_idx`, trims token padding proven by `full_attention_mask` and range-row
padding declared by `n_steps`, and
rejects interior padding or any disagreement with `ChainSample` prompt counts,
step ranges, gold label, dataset, or response-token count.

When an explicit model or tokenizer revision is requested, the stored revision
must match. With the default `auto` setting, a revision omitted by the source
trace is recorded as `unavailable_in_source_trace`, never inferred. Checkpoint
compatibility is then established by the mandatory selected-layer, step-end
activation replay-fidelity check. Tokenizer compatibility relies on replaying
the exact stored input IDs without re-tokenization.

The older `outputs/attention_traces` arrays are not sufficient for the full
four-domain experiment: they mostly cover a matched subset and mainly layer 14,
and they store large raw attention arrays rather than full replay token IDs,
response-token residual shards, and per-layer source-message residual writes.

## Artifact

Extraction writes one file per chain:

```text
<data_root>/<domain>/selected/component_step_v1/chain_<chain_id>.component_step_v1.npz
```

Each artifact contains:

- right-open absolute step starts/ends and
  `boundary_token_end = [prompt_token_count, step_0_end, ...]`;
- step labels with `0` before first error, `1` at first error, and `-1` after;
- source IDs where prompt is `-1`, completed steps are `0..current`, and padding
  is `-32768`;
- residual boundary states, attention source-message residual writes, attention
  outputs, and MLP outputs on the same selected one-based decoder depth grid;
- metadata with chain, dataset, model/tokenizer names and revisions, extractor
  commit, source trace SHA-256, and replay config SHA-256.

Existing aligned artifacts are validated and reused. Use `--overwrite` only to
force replacement.

## Message Interpretation

For each current step boundary, source blocks partition every token from the
beginning of the prompt through the current target token exactly once:

```text
prompt:  [0, prompt_end)
step 0:  [prompt_end, step_0_end)
step k:  [step_{k-1}_end, step_k_end)
```

At each selected decoder depth, the extractor captures `v_proj` values and the
current attention weights, expands grouped-query values from KV heads to query
heads, aggregates attended value context by source block, applies that layer's
`o_proj`, and stores residual-space source writes. The sum of source writes must
reconstruct the attention-module output at the same boundary within tolerance.

MLP outputs and decoder block output residual states are captured at matching
step-end boundaries. Replayed block outputs are compared against the existing
response-token hidden shard at the same stored depths; a fidelity mismatch is a
hard failure.

## Timing And Claims

The detector is retrospective at each completed step: it asks whether component
innovations after observing the current step diagnose that this step is the
first erroneous step. It does not predict before the step is generated, and it
does not establish causality. The claim boundary is observational held-domain
diagnostic association; causal attribution would require intervention or
activation patching.

## Commands

Smoke extraction and analysis:

```bash
./run_hidden_geometry_remote.sh component-smoke
```

This runs focused component tests when `pytest` is available, extracts or reuses
exactly 32 selected records per domain with `seed=17`, then runs the post-step
`component_resolved_hazard` analysis with bootstrap 200.

Full extraction and analysis:

```bash
./run_hidden_geometry_remote.sh component-full
```

This resumes extraction for all eligible records and runs bootstrap 2000.

Useful editable runner variables:

```bash
MODEL_DIR=/path/to/Meta-Llama-3.1-8B-Instruct
GPU_ID=0
COMPONENT_LAYERS=8,12,16,20,24,28
DATA_ROOT=/path/to/processbench_observer_llama31_full
```

Depth 32 is intentionally excluded for this 32-block checkpoint: the stored
hidden-state depth 32 is the final normalized state, not decoder block 32's
pre-normalization output, so mixing it with block-level attention/MLP writes
would fail the replay-fidelity semantics.

Outputs are written under
`reasoning_activation_divergence/outputs/hidden_state_geometry/component_*_<run_tag>/`
for analysis results, while reusable component artifacts remain beside the
selected trace files under each domain.
