1. **Dominant Contribution**

A single `hidden_state_geometry` method plugin that tests whether source-binned attention message contributions, MLP residual updates, and step-to-step residual propagation add LODO first-error signal beyond capacity-matched hidden-state baselines, with post-step diagnosis as the primary task and pre-step risk kept separate.

2. **Artifact Contract**

Use one component artifact per `ChainSample`, stored as `.npz` plus JSON metadata.

Required metadata:

```text
contract_version = "component_step_v1"
sample_id
dataset_id
domain_id
model_name
model_revision
tokenizer_name
tokenizer_revision
extractor_commit
source_trace_sha256
generation_config_sha256
dtype
selected_attention_layers: int[La]
selected_mlp_layers: int[Lm]
selected_residual_layers: int[Lr]
hidden_size: D
max_source_blocks: Bmax
```

Required index arrays:

```text
input_ids: int32[T]
step_token_start: int32[S]        # inclusive, 0-based token coordinates
step_token_end: int32[S]          # exclusive
boundary_token_end: int32[S + 1]  # boundary 0 = prompt end, boundary s+1 = after step s
first_error_step: int32           # -1 if fully correct
step_label: int8[S]               # 1 first error, 0 pre-error/correct, -1 post-error/unknown
source_step_id: int16[S, Bmax]    # -1 prompt, 0..S-1 step source, padded with -32768
source_mask: bool[S, Bmax]
```

Required residual arrays:

```text
resid_boundary: float16[S + 1, Lr, D]
```

Timing: `resid_boundary[s]` is the post-layer residual stream at the last token before step `s`; `resid_boundary[s+1]` is after step `s`.

Required attention arrays:

```text
attn_msg_resid_by_source: float16[S, La, Bmax, D]
attn_out_step: float16[S, La, D]
```

Definition:

```text
attn_msg_resid_by_source[s,l,b] =
mean over target tokens t in step s of
W_o,l applied to the concatenated head messages
sum_{u in source block b} A[l,h,t,u] V[l,h,u]
```

This is the **message contribution** in residual-stream space. It is not raw attention weight. `attn_out_step[s,l]` must equal the sum over valid source blocks up to tolerance.

Optional attention controls:

```text
attn_weight_mass_by_source: float16[S, La, H, Bmax]
attn_msg_head_by_source: float16[S, La, H, Bmax, Dh]
```

Raw attention weights are for control arms only. They cannot support a routing claim without message contributions.

Required MLP array:

```text
mlp_out_step: float16[S, Lm, D]
```

Definition: mean over tokens in step `s` of the MLP `down_proj` output added to the residual stream.

Optional logits:

```text
logit_topk_ids_boundary: int32[S + 1, K]
logit_topk_values_boundary: float16[S + 1, K]
```

Logits are separate arms only, never silently mixed into mechanism arms.

3. **Task Semantics**

Primary task: **post-step first-error diagnosis**.

For step `s`, predict:

```text
y_post[s] = 1 iff first_error_step == s
```

Allowed inputs:

```text
resid_boundary[s]
resid_boundary[s+1]
attn_msg_resid_by_source[s]
attn_out_step[s]
mlp_out_step[s]
```

Drop all `step_label == -1` examples. No future steps, future component outputs, textual judge summaries, or answer correctness fields.

Secondary task: **pre-step first-error risk**.

For boundary before step `s`, predict:

```text
y_pre[s] = 1 iff first_error_step == s
```

Allowed inputs:

```text
resid_boundary[s]
component summaries for target steps < s only
optional logits at boundary s only
```

Forbidden: current-step tokens, current-step attention, current-step MLP, `resid_boundary[s+1]`, post-step logits, and any post-error trace.

All scalers, projections if fitted, ridge maps, logistic heads, hyperparameters, and calibration must be fit inside the training side of each LODO fold only.

4. **Smallest Model**

No typed sparse GNN for v1. The source-step attention structure is scientifically necessary in the artifact, but a graph encoder is not. A GNN would add capacity and attribution ambiguity before proving the channels matter.

Use fixed structural buckets from `source_step_id`:

```text
prompt
earlier_steps
previous_step
current_step
```

For each valid post-step example:

```text
x_s      = RP(resid_boundary[s])
zA_s     = RA(bucket(attn_msg_resid_by_source[s]))
zF_s     = RF(mlp_out_step[s])
zP_s     = RQ(resid_boundary[s+1])
```

`RP, RA, RF, RQ` are frozen random projections with stored seed, not trainable.

Trainable component 1: component normal maps, trained on correct/pre-error training steps only:

```text
W_c, b_c = argmin ||zC_s - W_c x_s - b_c||^2 + lambda ||W_c||^2
eC_s     = zC_s - W_c x_s - b_c
```

for `C in {A,F,P}`.

Trainable component 2: logistic hazard head:

```text
p_post_s = sigmoid(beta0 + beta_x^T x_s + beta_A^T eA_s + beta_F^T eF_s + beta_P^T eP_s)
```

Pre-step uses the same form, but replaces current-step residuals with previous-step residual summaries:

```text
p_pre_s = sigmoid(beta0 + beta_x^T x_s + beta_A^T eA_{s-1} + beta_F^T eF_{s-1} + beta_P^T eP_{s-1})
```

For `s=0`, previous component residuals are zero with an explicit mask bit.

Every arm gets the same final detector dimension, e.g. `256`, and the same logistic regularization grid.

5. **Arms, Ablations, Gates**

Core LODO arms:

```text
metadata_control: step index, token count only
hidden_pre: resid_boundary[s]
hidden_post: resid_boundary[s], resid_boundary[s+1]
innovation_hazard_repro: current layer14->16 style baseline
attn_weights_only: raw attention mass control
attn_out_only: attn_out_step
attn_msg_only: source-binned attn_msg_resid_by_source
mlp_only: mlp_out_step
prop_only: resid_boundary[s] -> resid_boundary[s+1]
components_all: attn_msg + mlp + prop
components_all_plus_hidden
```

Attribution ablations:

```text
all_minus_attn
all_minus_mlp
all_minus_prop
attn_msg_source_bucket_shuffle
mlp_step_shuffle
prop_step_shuffle
label_shuffle_negative_control
```

Predictive-gain gate:

```text
components_all_plus_hidden beats hidden_post on weighted LODO AUPRC
95% paired bootstrap CI lower bound > 0
median held-out dataset delta > 0
ECE not worse by more than 0.02
shuffle controls show no gain
```

Attention-routing attribution gate:

```text
attn_msg_only > attn_weights_only
all_minus_attn drops vs components_all
source-bucket shuffle kills most attention gain
stable source/layer localization across held-out datasets
```

FFN attribution gate:

```text
mlp_only beats matched hidden control
all_minus_mlp drops vs components_all
mlp step-shuffle removes the gain
```

Propagation attribution gate:

```text
prop_only beats hidden_pre
all_minus_prop drops vs components_all
gain is strongest on post-step diagnosis, not leaked pre-step risk
```

Failure of any attribution gate means the prior remains unsupported, even if the full detector predicts well.

6. **File-by-File Plan**

Inside:

```text
reasoning_activation_divergence/src/functional_divergence/hidden_state_geometry
```

Add:

```text
component_contract.py
```

Responsibilities:

```text
ComponentStepArtifact dataclass
load_component_artifact(path)
validate_component_artifact(...)
shape, timing, provenance, and contribution-vs-weight checks
```

Add:

```text
component_features.py
```

Responsibilities:

```text
align ChainSample to component artifacts by sample_id
construct post-step and pre-step examples
apply strict_prefix leakage rules
bucket source-step attention messages
build arm-specific raw vectors
return X/y/group/domain/sample-step indices
```

Add one method plugin:

```text
methods/component_resolved_hazard.py
```

Responsibilities:

```text
ComponentResolvedHazardMethod.run(fold: FoldInput) -> MethodFoldResult
frozen random projections
ridge normal maps
logistic hazard heads
LODO arm execution
ablation metrics
prediction/artifact emission
```

Modify only registry glue:

```text
methods/__init__.py or existing method_registry.py
```

Register:

```text
component_resolved_hazard
```

Add extraction adapter, not a new project:

```text
scripts/extract_processbench_components.py
```

Responsibilities:

```text
wrap prompt_control_flow/.../routing_extraction.py hooks
emit component_step_v1 artifacts
reuse routing_schema.py provenance conventions
validate artifacts immediately after write
```

Tests:

```text
tests/test_component_contract.py
tests/test_component_features.py
tests/test_component_resolved_hazard_synthetic.py
```

Reuse, do not copy:

```text
ChainSample, FoldInput, MethodFoldResult
strict_prefix task logic
LODO splitting and group/domain weights
existing ridge/logistic helpers
innovation_hazard fold/output pattern
prompt_control_flow routing_extraction hooks
routing_schema metadata/provenance ideas
```

Do not reuse the hypergraph/CCT model stack. At most borrow contract sanity ideas.

7. **First RED Tests**

Synthetic dimensions:

```text
N=64 samples
S=4 steps
La=Lm=Lr=2
D=8
Bmax=5
K=5
```

First failing tests:

```text
contract rejects artifacts with raw attention weights but no attn_msg_resid_by_source
contract rejects attn_out_step != sum(attn_msg_resid_by_source)
post-step builder drops step_label == -1
pre-step builder never reads current-step component arrays
LODO split has no dataset overlap between train/test
capacity-matched arms all expose 256 detector dimensions
attn synthetic: label depends on previous-step message vector; attn_msg passes, attn_weights fails
mlp synthetic: label depends on planted mlp_out direction; all_minus_mlp drops
prop synthetic: label depends on residual transition; hidden_pre fails, prop_only passes
shuffle controls destroy planted channel signal
```

8. **Local Work Before Remote Artifacts**

Implement contract loader, validator, feature builder, method plugin, registry entry, synthetic tests, LODO metric plumbing, capacity-matched arm runner, and output schemas.

Run locally on synthetic artifacts and existing residual-only ProcessBench artifacts for:

```text
metadata_control
hidden_pre
hidden_post
innovation_hazard_repro
```

Component arms should fail explicitly with `missing component_step_v1 artifacts`, not silently degrade.

9. **Identifiability Limits**

This design can show that component-resolved channels predict first errors and that removing a channel hurts prediction. It cannot prove the model failed because attention routed the wrong fact, the FFN failed a knowledge update, or recursion causally broke.

Minimum later intervention:

```text
attention: patch source-binned message contribution from matched correct traces
MLP: patch mlp_out_step from matched correct traces
propagation: patch resid_boundary transition states
```

Then measure whether next-token distribution, generated step correctness, or post-step hazard changes in the predicted direction, with random-layer/source and unrelated-sample patch controls.

10. **GO / NO-GO**

GO for implementation as one `reasoning_activation_divergence` method plugin plus one extraction contract.

NO-GO for causal mechanistic claims until intervention results exist.

Highest-risk assumption: source-binned attention message contributions and MLP outputs can be re-extracted with exact ProcessBench step alignment at scale, and still add LODO signal beyond a capacity-matched post-step hidden-state baseline.
