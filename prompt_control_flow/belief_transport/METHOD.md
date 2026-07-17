# Constraint-Supported Belief Transport

## 1. Purpose

This subproject tests whether a belief-transport interpretation exists before
using it to diagnose ProcessBench errors. It is not another collection of
trajectory scalars and it does not assume that low dimension implies
correctness.

The first stage asks two falsifiable questions:

1. Can a frozen model's residual state decode the exact feasible hypothesis
   set on held-out problems, beyond position, prompt length, template family,
   and compact next-token uncertainty?
2. Does the condition-specific update operator explain the next residual belief
   better than a minimally direction-scrambled operator with exactly the same
   information gain?

Only if both tests pass is it justified to construct a ProcessBench detector or
to localize attention/MLP causes.

## 2. Exact Reference System

Each synthetic problem defines two integer variables over a finite domain. The
hypothesis universe is

\[
\mathcal H = \{(x,y): x,y \in \{0,\ldots,d-1\}\}.
\]

Every generated condition is true for one sampled target and strictly reduces
the feasible set. The sequence ends at a unique assignment. At prefix \(t\),
the exact reference belief is uniform on the feasible set \(F_t\):

\[
b_t(h)=\frac{\mathbf 1[h\in F_t]}{|F_t|}.
\]

For a new condition with mask \(m_t\), the exact transition is

\[
T_{m_t}(b_{t-1})
=
\frac{b_{t-1}\odot m_t}
{\langle b_{t-1},m_t\rangle}.
\]

This gives actual belief targets and legal transitions. ProcessBench does not
provide either object directly, which is why mechanism existence is tested here
first.

## 3. Frozen-Model Observation

For every problem prefix, the extractor renders the observer model's own chat
template with `add_generation_prompt=True`. It saves the exact rendered prompt
and token IDs. The observed state is the normalized residual state at the last
non-padding token, immediately before the assistant generates an answer.

Extraction uses selective forward hooks:

- depth 0 is captured at the embedding output;
- intermediate depths are captured after the corresponding transformer block;
- the final depth is captured after final residual normalization;
- `output_hidden_states=True` is not used, so full sequence histories are not
  retained;
- only one boundary vector per selected depth is stored;
- the LM head is applied only to the final boundary state, avoiding a full
  sequence-by-vocabulary logits tensor.

The compact output control stores next-token entropy, top-1/top-2 margin, top-k
probability mass, and a fixed random projection of centered, L2-normalized
full-vocabulary logits. The sketch preserves much more output identity than
three uncertainty scalars while avoiding full-logit persistence. Attention
matrices are not persisted.

## 4. Learned Belief Chart

At every selected depth \(\ell\), a fold-local decoder maps residual state to a
categorical belief:

\[
\hat b_{t,\ell}=\operatorname{softmax}(D_\ell r_{t,\ell}).
\]

The primary implementation is a linear soft-belief probe. Linear accessibility
is a stronger, easier-to-falsify existence claim than an unconstrained VAE or a
high-capacity sequence classifier. An MLP is available only as an ablation.

Training uses grouped cross-fitting. All prefixes of one problem stay in one
outer fold, preprocessing is fit on training rows only, validation groups drive
early stopping, and every problem receives equal total loss weight.

Four prediction sources use identical folds:

1. nuisance only: prefix index, relative prefix position, prompt token count,
   and template family;
2. output control: nuisance features plus compact next-token uncertainty;
3. residual state: the selected hidden vector.
4. joint model: nuisance and compact output controls plus the hidden vector.

The central conditional-information estimate is

\[
I_{\mathrm{usable}}(B;R\mid Z,C)
\approx
\frac{\mathcal L_{Z,C}-\mathcal L_{Z,C,R}}{\log 2},
\]

where \(\mathcal L_{Z,C}\) is the output-control model and
\(\mathcal L_{Z,C,R}\) is the joint model's held-out soft cross-entropy;
\(Z\) denotes compact output statistics and \(C\) denotes nuisance controls.
Confidence intervals use problem-cluster bootstrap. Hidden-only performance is
reported separately and is never substituted for this conditional comparison.

## 5. Geometric Direction Test

The categorical Fisher-Rao distance is

\[
d_{\mathrm{FR}}(p,q)
=2\arccos\left(\sum_h\sqrt{p_hq_h}\right).
\]

For each held-out transition, the true transport residual is

\[
R_t^{\mathrm{true}}
=d_{\mathrm{FR}}\!\left(
\hat b_t,
T_{m_t}(\hat b_{t-1})
\right).
\]

The primary null is a local one-swap perturbation of the true posterior
support. It removes one retained hypothesis and inserts one hypothesis rejected
by the true condition but still feasible under the previous prefix. Therefore
the null differs in direction by the smallest non-zero support edit while
preserving the exact reduction

\[
\log\frac{|F_{t-1}|}{|F_t|},
\]

The information-gain mismatch is reported and must be numerically zero. This
prevents update strength or a more destructive random condition from
masquerading as direction. The primary directional statistic is

\[
\Delta R_t=R_t^{\mathrm{wrong}}-R_t^{\mathrm{true}}.
\]

The audit also reports contraction, support-margin gain, and unsupported
contraction

\[
U_t=[H(\hat b_{t-1})-H(\hat b_t)]_+
[-\Delta M_t]_+,
\]

but these are secondary diagnostics, not independent features combined into a
classifier.

## 6. Predeclared Gates

The primary layer is supplied before the audit. Other layers describe depth
evolution and do not replace the primary result.

The belief-state gate requires:

- primary support AUROC at least `0.70`;
- exact-vs-decoded entropy Spearman correlation at least `0.50`;
- the lower 95% confidence bound of joint-over-output usable bits above zero.

The direction gate requires:

- matched operator AUROC at least `0.60`;
- the lower 95% confidence bound of \(\Delta R_t\) above zero.
- the 95th percentile of true/null information-gain mismatch at most
  \(10^{-8}\) nats.

Failure of either gate blocks ProcessBench transfer. Passing is evidence that
the objects exist, not evidence that they detect reasoning errors.

## 7. Artifact Contract

The trace NPZ contains:

```text
schema_version
state_semantics
problem_ids
template_families
prefix_index
previous_prefix_index
target_hypothesis
feasible_mask
condition_mask
hypotheses
layers
states
prompts
prompt_sha256
input_ids
prompt_token_count
output_entropy
output_margin
output_topk_mass
output_logit_sketch
metadata_json
```

Validation checks every transition exactly:

```text
feasible_t == feasible_(t-1) AND condition_t
```

Shards may contain disjoint problems only. Merge fails if model identity,
tokenizer identity, hypothesis universe, layer set, or state semantics differ.

## 8. Run Order

### Build the reference set

```bash
python build_belief_wind_tunnel.py \
  --output data/belief_transport/wind_tunnel_v1.jsonl \
  --num_problems 2000 \
  --domain_size 8 \
  --min_steps 3 \
  --max_steps 6 \
  --seed 17
```

### Two-GPU extraction

Run from:

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/demo
```

Terminal 1:

```bash
CUDA_VISIBLE_DEVICES=0 python extract_belief_transport.py \
  --input data/belief_transport/wind_tunnel_v1.jsonl \
  --model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct \
  --output data/belief_transport/trace_shard0.npz \
  --layers 8,12,16,20,24,28,32 \
  --device cuda \
  --dtype bfloat16 \
  --batch_size 16 \
  --max_batch_tokens 3072 \
  --num_shards 2 \
  --shard_index 0
```

Terminal 2 uses the same command with `CUDA_VISIBLE_DEVICES=1`,
`trace_shard1.npz`, and `--shard_index 1`.

Merge:

```bash
python merge_belief_transport.py \
  --inputs data/belief_transport/trace_shard0.npz \
           data/belief_transport/trace_shard1.npz \
  --output data/belief_transport/trace_full.npz
```

### Cross-fitted audit

```bash
CUDA_VISIBLE_DEVICES=0 python audit_belief_transport.py \
  --input data/belief_transport/trace_full.npz \
  --output_dir outputs/belief_transport/llama31_8b_v1 \
  --primary_layer 16 \
  --device cuda \
  --folds 5 \
  --bootstrap 2000
```

For a pilot, add `--max_problems 200` to both extraction commands and use
`--bootstrap 200` in the audit.

## 9. Stage 2, Only After Passing

If both gates pass, the next experiment freezes the learned chart and tests
ProcessBench step boundaries. An error hypothesis can then be stated precisely:
the model contracts while losing feasible-condition support, or follows a
transition that the learned legal operator cannot explain. Attention routing,
MLP updates, and activation patching enter only at that causal-localization
stage.
