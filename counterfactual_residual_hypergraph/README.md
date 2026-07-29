# Counterfactual Residual-Write Hypergraph (CRWH)

CRWH is a research prototype for semi-supervised and normal-reference
hallucination detection with attention-selected, residual-write-attributed
hypergraphs. Its central question is:

> Can a response state look locally normal while the residual-write provenance
> that produced it is unsupported by the supplied evidence?

The project consumes pre-extracted, teacher-forced tensors. It does **not**
download an LLM, generate responses, implement a real-model OV extractor, or
claim that attention alone is causal.

## What was integrated

This project keeps stable ideas from the two local hypergraph repositories
without importing their script-style code:

- `research/hypergraph-hallucination`
  - compatible incidence schema (`[node_id, hyperedge_id]`);
  - one head/query group per hyperedge;
  - HyperCHARM-style node-to-edge-to-node aggregation.
- `research/multiview-hypergraph-hallucination`
  - shared multi-view encoder;
  - cross-view gated fusion;
  - response-only prediction.
- `demo/hypergraph/attention/cct`
  - directed receiver semantics;
  - fail-closed tensor contracts;
  - separation between attention routing and output-effective writes.

The old formats are exposed only as `adapted_attention_ablation` adapters. They
are not silently upgraded into residual-write graphs, and these adapters are
not faithful reproductions of the original encoders.

## Method

For a fixed response, extract the same response token IDs under:

1. `factual`: original evidence;
2. `paraphrase`: meaning-preserving evidence perturbation;
3. `counterfactual`: evidence removal, replacement, contradiction, or a
   validated random-context control.

For layer/head/query/source tuple `(l, h, t, k)`, a future validated extractor
should supply a source-resolved OV residual write

```text
w[l,h,t,k] = attention[l,h,t,k] * W_O[l,h] W_V[l,h] LN(r[l,k])
```

and an independently captured expected attention update. Given those inputs,
the builder checks

```text
sum_k w[l,h,t,k] == expected_update[l,h,t]
```

within a configured absolute-plus-relative tolerance. A failed reconstruction
aborts graph construction. This gate proves only that the supplied decomposition
is internally consistent; it does not prove that the extractor recovered the
real model computation. A real experiment must validate the extractor against
an independent model hook.

Attention selects each retained head/query topology. Supplied write tensors then
attribute that hyperedge, whose non-receiver sources write to an explicit
response receiver. Edge features include attention mass, source-role
composition, write mass, resultant norm, path cancellation, effective source
count, dominance, signed support in a fixed output direction, and separately
recorded receiver/self-write diagnostics.

The implementation provides two learning routes:

- **Normal-reference anomaly scoring:** separate shrunk-Mahalanobis scorers for
  factual response state, factual hypergraph relation, aligned
  factual-counterfactual response-state discrepancy, and the corresponding
  relation discrepancy. The response-state discrepancy is the contextualized
  embedding-influence signal; keeping it separate from relation change supports
  direct ablation. Wide vectors are mapped through a fixed seeded random
  projection before covariance fitting. Every fit example must be explicitly marked
  `normal_reference`; a positive token label in any paired view is rejected.
  Knowing that these samples are correct is sample-level supervision even when
  no hallucination token labels are used. The caller remains responsible for a
  train-only, group-safe reference split. Outputs are raw anomaly scores and
  empirical reference percentiles, not calibrated hallucination probabilities.
- **Semi-supervised learning:** masked token BCE for available labels,
  paraphrase embedding consistency for all examples, and a
  factual/paraphrase/counterfactual contrastive objective only where the caller
  supplies a boolean `contrastive_eligible` token mask. Correctness labels and
  `normal_reference` never implicitly activate this term, because a correct
  answer need not depend on the perturbed evidence. Unlabeled or withheld tokens
  use label `-1` and never enter BCE.

Detector disagreements can be exported as a bounded review queue for an
external LLM verifier. Selection compares decisions made with separate
validation-selected thresholds; it never subtracts a normal-reference
percentile from a supervised posterior as if they shared a scale. LLM judgments
are intentionally not treated as ground truth by this package.

## Project layout

```text
counterfactual_residual_hypergraph/
|-- configs/pilot.json
|-- scripts/run_remote_4090.sh
|-- src/crwh/
|   |-- contracts.py       # strict trace, graph, and paired-view contracts
|   |-- builder.py         # reconstruction gate and directed HG builder
|   |-- geometry.py        # cancellation, N_eff, dominance
|   |-- features.py        # deterministic token-level graph signatures
|   |-- one_class.py       # shrunk Mahalanobis
|   |-- detector.py        # four-component normal-reference detector
|   |-- model.py           # directed HG encoder and gated multi-view detector
|   |-- objectives.py      # supervised + consistency + contrastive losses
|   |-- training.py        # small transparent training loop
|   |-- adapters.py        # adapted attention-only graph ablations
|   |-- selection.py       # selective LLM-review queue policy
|   |-- cohort.py          # balanced, problem-unique real-data selection
|   |-- real_cct_audit.py  # split, graph, checkpoint, and prediction audit
|   |-- synthetic.py       # deterministic CPU smoke data
|   `-- cli.py
`-- tests/
```

## Installation and smoke run

From this directory:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m crwh synthetic \
  --output-dir outputs/smoke \
  --examples 16 \
  --epochs 3 \
  --seed 42
```

The smoke command writes `config.json`, `metrics.json`, `scores.json`, and
`review_queue.json`. Synthetic metrics are contract checks, not scientific
results.

## RTX 4090 real-data baseline

The repository includes one audited shell entry point for the declared server:

```bash
bash scripts/run_remote_4090.sh
```

Its defaults are:

```text
model:
/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct

ProcessBench:
/share/home/tm902089733300000/a903202310/lys/research/demo/data/hf_datasets/ProcessBench

GPU=0, subset=gsm8k, layer=14, BF16, SDPA
```

The safe first run reuses the active CUDA-enabled PyTorch environment:

```bash
INSTALL_DEPS=0 PROFILE=smoke bash scripts/run_remote_4090.sh
```

If NumPy, pytest, Transformers, Accelerate, Safetensors, or tqdm are missing,
install only those non-Torch dependencies:

```bash
INSTALL_DEPS=1 PROFILE=smoke bash scripts/run_remote_4090.sh
```

Do not replace a working CUDA PyTorch build through this project. The remote
requirements intentionally omit `torch`; the script validates CUDA/BF16 first,
pins the detected Torch version as a pip constraint, verifies that it did not
change, and runs `pip check`.

The default `smoke` profile scans the whole GSM8K source instead of taking its
ordered prefix. It deterministically selects 24 traces with at most 768 tokens:
12 normal (`label=-1`) and 12 containing a first error (`label>=0`), with one
trace per problem ID. It uses `top_sources=2`, `node_dim=32`, then performs a
60/20/20 problem-disjoint split, trains a one-layer CCT-HG detector, and writes
held-out metrics and a checkpoint. This is an evaluable engineering smoke, but
24 traces are too few for a paper result. At the pre-specified
`min_effect=min_synergy=0.01`, the smoke gate requires at least four
intervention-supported higher-order edges overall and hyperedges in at least
25% of training traces; otherwise it stops instead of silently reporting a
sparse graph as a hypergraph result. The pilot raises the total-edge threshold
to 12 while retaining 25% training-trace coverage. Setting
`REQUIRE_HYPEREDGES=0` explicitly permits an engineering witness, but
`summary.json` then downgrades the result to
`real_processbench_cct_sparse_hyperedge_witness`,
`real_processbench_cct_pair_graph_witness`, or
`real_processbench_cct_no_edge_witness` when warranted.

After that succeeds with safe GPU headroom, the larger engineering pilot is:

```bash
PROFILE=pilot INSTALL_DEPS=0 bash scripts/run_remote_4090.sh
```

The `pilot` profile uses 60 balanced, problem-unique traces, `top_sources=2`,
`node_dim=64`, a two-layer detector, and 1,000 problem-bootstrap replicates.
The auditable entry point intentionally does not run the optional CCT control
suite yet: those transformed-cohort checkpoints still need independent
provenance and replay validation before control comparisons can be reported.
Environment variables can override every path and resource setting, for
example:

```bash
GPU=0 MAX_TOKENS=768 OUTPUT_ROOT=/path/to/output \
  bash scripts/run_remote_4090.sh
```

All path overrides must be absolute. If a scheduler already supplied one
`CUDA_VISIBLE_DEVICES` entry, the script preserves that allocation and treats
it as the selected physical GPU; otherwise it exports `GPU` (default `0`).

If the initial gradient-bearing extraction still runs out of memory, retry
with `MAX_TOKENS=512`. A successful 768-token smoke is required before raising
the limit toward 1024.

The shell performs:

1. absolute-path, minimum-free-disk, Python, and Llama-config checks;
2. CUDA/BF16, total/free-memory, and seeded matrix-multiplication witnesses;
3. dependency and source-tree fingerprints;
4. the full CRWH test suite;
5. the deterministic CRWH synthetic CPU pipeline;
6. tokenizer-based, response-label-balanced, problem-unique ProcessBench
   selection using `label`, not `final_answer_correct`;
7. real Llama-3.1-8B CCT extraction and trace inspection;
8. pair/hyperedge counts and a fail-closed audit that every train,
   validation, and test partition contains both response classes;
9. CCT-HG training with validation checkpoint selection and held-out test
   evaluation;
10. reload-and-reproduction validation of `model.safetensors`,
    `checkpoint.json`, normalization, split, prediction, history, and metric
    artifacts plus `_SUCCESS`/`_FAILED` markers.

The main real artifacts are:

```text
outputs/remote_4090/<run-id>/
|-- cohort-manifest.json
|-- cct-graph-audit.json
|-- cct-hypergraph-gate.json
|-- cct-split-preflight.json
|-- cct-real-result.json
|-- source-tree.sha256
|-- summary.json
`-- cct-real-training/
    |-- metrics.json
    |-- model.safetensors
    |-- checkpoint.json
    |-- model.pt              # trusted-local compatibility only
    |-- normalizer.npz
    |-- split.json
    |-- history.csv
    |-- predictions_validation.csv
    `-- predictions_test.csv
```

Inspect the newest completed run with:

```bash
RUN_DIR="$(find outputs/remote_4090 -mindepth 2 -maxdepth 2 \
  -type f -name _SUCCESS -printf '%T@ %h\n' \
  | sort -nr | head -n1 | cut -d' ' -f2-)"
cat "${RUN_DIR}/summary.json"
cat "${RUN_DIR}/cct-real-training/metrics.json"
```

This is deliberately **not** described as a real CRWH ProcessBench result. The
real output is a supervised, single-view CCT-HG baseline using constraint
geometry and intervention-tested pair/hyperedges. It proves that real
extraction, hypergraph training, held-out evaluation, and checkpointing work.
It does not prove that the counterfactual multi-view semi/unsupervised CRWH
method works. A real paired-view CRWH run still requires:

- per-head writes shaped `[H,Q,N,R]` rather than CCT's head-aggregated
  `[Q,N,R]`;
- per-head expected updates `[H,Q,R]`;
- fixed-response factual/counterfactual construction;
- one factual output-direction/projection anchor reused across paired views;
- a safe graph/trace serialization and ProcessBench experiment driver.

The script does not use low-bit quantization because exact access to ordinary
`v_proj` and `o_proj` weights is part of the current extraction contract.

## Residual-write input contract

`ResidualWriteTrace` expects:

| Field | Shape | Meaning |
|---|---:|---|
| `provenance` | metadata | model/tokenizer revisions, prompt format, layer, extractor, projection, output anchor, and teacher-forcing offset |
| `perturbation_*` | metadata | perturbation kind, ID, seed, and validation status |
| `node_features` | `[N,D]` | hidden/node features |
| `attention` | `[H,Q,N]` | causal attention mass |
| `source_writes` | `[H,Q,N,R]` | supplied source-resolved residual writes |
| `expected_updates` | `[H,Q,R]` | independently captured attention updates |
| `output_directions` | `[Q,R]` | fixed output-sensitive directions |
| `receiver_positions` | `[Q]` | response receiver node per query |
| `response_token_ids` | `[Q]` | fixed response IDs used for view alignment |
| `source_roles` | `[N]` | role IDs; default evidence role is `0` |
| `token_labels` | `[Q]` | `0`, `1`, or `-1` for unlabeled |

Prompt lengths may differ across context views. Alignment is performed on the
fixed response token IDs and response-query order, not prompt token positions.
All views must also share extraction provenance and fixed output directions, and
their perturbations must be explicitly validated. Set
`CounterfactualExample.normal_reference=True` only for a corpus externally
verified as normal; the contract rejects a positive hallucination label in any
view. Supply `contrastive_eligible` independently as a boolean `[Q]` mask.
When simulating a label budget, replace withheld labels with `-1` in **every**
view; the trainer merges any still-visible aligned labels by design.

## Using the existing attention hypergraphs

The adapters accept already-loaded dictionaries; file loading remains with the
caller so unsafe pickle policy is explicit.

```python
from crwh.adapters import from_attnhyper, from_mvht_view

attn_graph = from_attnhyper(saved_dict, num_heads=32)
semantic_graph = from_mvht_view(
    mvht_data,
    sample_id="question-17/response-3",
    view="semantic",
    num_heads=32,
    response_token_ids=response_ids,
)
reasoning_graph = from_mvht_view(
    mvht_data,
    sample_id="question-17/response-3",
    view="reasoning",
    num_heads=32,
    response_token_ids=response_ids,
)
```

These graphs are useful adapter tests and same-encoder attention-only
ablations. They cannot be passed off as residual-write graphs because their edge
attributes contain attention statistics rather than OV write vectors. They are
marked `perturbation_validated=False`, so they cannot enter the strict paired
counterfactual contract.

For publication experiments, independently run the original AttnHyper and MVHTR
training/inference code as reproduced baselines. The adapters do not preserve
their original message passing, auxiliary metadata, or semantic/reasoning view
semantics.

## Experimental controls

At minimum compare:

- nuisance-only features (length, position, token rarity, entropy);
- hidden-state Mahalanobis;
- contextual influence only;
- matched same-encoder attention-only graph ablation;
- reproduced original AttnHyper and MVHTR baselines;
- residual-write feature MLP;
- signed pair/factor graph;
- full residual-write hypergraph;
- receiver/degree/cardinality/role-preserving rewiring.

Also compare attention-threshold, write-norm-threshold, and write-mass-coverage
topology rules before attributing gains to the residual-write construction.

Keep paired contexts in the same data split. Select thresholds on training or
validation data only. Report token AUPRC, calibration/risk-coverage, worst-group
performance, and first-error localization when step labels exist.

## Scientific guardrails

- `cancellation` means path cancellation/inefficiency, not semantic conflict.
- Signed support means support for the model's current decision direction, not
  factual truth.
- The reconstruction gate is conditional self-consistency, not extractor
  correctness. Real-model claims require an independently hooked update and a
  validated source-write extractor with locked provenance.
- A `normal_reference` partition uses sample-level correctness supervision.
  Describe it as normal-reference or one-class detection, not label-free in the
  absolute sense.
- Training fails closed if no example has an active BCE, paraphrase-consistency,
  or explicitly eligible contrastive objective, and skips no-signal examples in
  a mixed set. With a zero-label budget, consistency alone still admits
  representation collapse; use the one-class route as the primary zero-label
  baseline or add a separately validated stop-gradient/variance regularizer.
- Better performance than clique expansion does not prove irreducible
  higher-order computation. Add a matched factor-GNN and explicit coalition
  interventions before making that claim.
- The current package constructs grouped write hyperedges. Pure third-order
  Möbius interaction and suffix-replay intervention are a future experimental
  gate, not implemented evidence.
- Human-written distractors and model confabulations should be evaluated as
  distinct error regimes.
