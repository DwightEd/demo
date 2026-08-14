# Source-message Fisher experiment

This experiment measures how attention-source messages and the local FFN update
change the model's actual next-token distribution immediately before a first
reasoning error. It does not train a probe and does not assume a thermodynamic
phase transition.

## Computation

For each selected layer and decision boundary, attention is decomposed into
source-step residual writes:

```text
M[source] = sum(head, source tokens) o_proj(attention * value)
```

The extraction must satisfy both reconstruction identities:

```text
sum(source) M[source] ~= AttentionOut
ResidualPre + AttentionOut + FFNOut ~= ResidualPost
```

Each source message and the FFN output define an intervention coordinate. The
code applies central coefficient perturbations, measures the resulting full
vocabulary logit derivatives, and computes the categorical Fisher Gram matrix:

```text
G = J_logits (diag(p) - p p^T) J_logits^T
```

It also stores the Euclidean message Gram as the required control. The
quadratic prediction `0.5 * epsilon^2 * diag(G)` is compared with the observed
symmetric KL under the actual interventions. A poor match means epsilon is not
local enough or numerical precision is insufficient; such a run must not be
interpreted geometrically.

## Cohort

For every ProcessBench error chain whose first error occurs after step zero,
the experiment compares two future-free boundaries from the same chain:

1. before the immediately preceding correct step;
2. before the first-error step.

No token from the target step is supplied to either replay. Step-zero errors
are excluded because they have no within-chain preceding-step control.

Source IDs are:

```text
-2      unassigned separators/control tokens
-1      prompt tokens
0..N    completed reasoning steps
-32768  typed local FFN direction (not an attention source)
```

## Remote commands

Run from the existing remote terminal; no SSH command is needed:

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/demo/reasoning_activation_divergence
git pull --ff-only

PYTHON_BIN=/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
  bash run_hidden_geometry_remote.sh causal-fisher-smoke
```

Smoke uses one paired case per domain. A bounded formal pilot with 20 cases per
domain is:

```bash
FISHER_CASES_PER_DOMAIN=20 \
PYTHON_BIN=/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
  bash run_hidden_geometry_remote.sh causal-fisher-full
```

Set `FISHER_CASES_PER_DOMAIN=0` only when intentionally running every eligible
case. This is expensive because every source direction is intervened on at
every selected layer. `FISHER_BATCH_SIZE` controls the number of simultaneous
directions; lower it after CUDA OOM. `FISHER_EPSILON` defaults to `0.05`, which
is safer than very small coefficients under bfloat16 inference.

## Outputs

Each run writes under:

```text
outputs/hidden_state_geometry/causal_fisher_<smoke|full>_<timestamp>/
```

The directory contains:

```text
events.jsonl
summary.json
artifacts/<domain>/*.source_message_fisher_v1.npz
```

Each NPZ retains exact prefix token IDs, token-to-source assignments,
head-wise source attention mass, source residual messages, Attention/FFN and
pre/post residual vectors, Fisher and Euclidean Gram matrices, intervention KL
audits, and reconstruction errors. It does not overwrite or duplicate the
central raw hidden-state shards.

The primary comparison in `summary.json` is paired
`first_error - previous_correct`. Evidence is credible only when component
reconstruction passes, the quadratic KL audit is acceptably local, and Fisher
changes cannot be explained by the Euclidean Gram control or source count.
