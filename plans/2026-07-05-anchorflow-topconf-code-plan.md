# AnchorFlow 顶会级实验代码规划（2026-07-05）

目标不是继续堆标量，而是把当前零散发现升级成一个可投稿叙事：

```text
Reasoning failures are online breaks in constraint-anchored representational flow.
```

当前负结果很关键：

```text
1. hypergraph HGN 能定位，但重、慢、增量有限；
2. manifold_health v1 独立 detector 变差；
3. cloud_V 几乎等于 length/difficulty proxy；
4. q_align/anchor_loss 只是 single-vector cosine，机制解释太薄；
5. pos/logN 偏置很强，所有 localization 必须 residualize。
```

因此下一步代码必须围绕一个统一对象：**multi-anchor transport field**。

---

## 0. 论文级主张与代码判据

### Claim A：错误不是泛化发散，而是题设约束流断裂

代码判据：

```text
AnchorFlow features > anchor_uncertainty / EDIS / spread
在 residualized AUROC、high-spread subset、confident-wrong subset 上稳定有增量。
```

### Claim B：无需强制分步骤也能发现潜在相变边界

代码判据：

```text
boundary-free token/window phase detector
能恢复人工 step boundary，并能在 first-error 前后产生局部 phase break。
```

### Claim C：检测信号能转化成 test-time intervention

代码判据：

```text
AnchorFlow-triggered repair > random / high-entropy / max-spread trigger
报告 wrong-to-right、correct-to-wrong damage、token cost、delay。
```

若只提高 AUROC 但不能干预，论文故事仍然弱；若检测增量一般但干预收益强，可以转成：

```text
not a better classifier, but a better repair trigger.
```

---

## 1. 新代码结构

新增包：

```text
anchorflow/
  __init__.py
  data.py
  anchors.py
  anchor_repr.py
  transport.py
  volume.py
  phase.py
  residualize.py
  detectors.py
  online.py
  intervention.py
  eval.py
  report.py

anchorflow_anchor_audit.py
anchorflow_phase_audit.py
anchorflow_validation_suite.py
anchorflow_intervention_suite.py

tests/
  test_anchorflow_anchors.py
  test_anchorflow_transport.py
  test_anchorflow_residualize.py
  test_anchorflow_phase.py
```

保留旧脚本为 baseline，不再在旧脚本里继续加新逻辑：

```text
chain_dynamics_audit.py          # baseline: spread/anchor/uncertainty/transition
manifold_health_audit.py         # negative-result baseline
hypergraph_token_hgn.py          # heavy upper-bound baseline
mainline_validation_suite.py     # old summary suite
```

---

## 2. 数据层：`anchorflow/data.py`

输入：

```text
data/features/full_{dataset}.npz
data/hidden/{dataset}/{chain_id}.npy
optional: attention dumps / logits entropy if present
```

输出统一对象：

```python
Trace:
  chain_id: str
  problem_id: int
  dataset: str
  correct: bool
  gold_error_step: int
  step_token_ranges: (T,2)
  prompt_token_range: (2,)
  steps_text: list[str]
  features: dict[str, np.ndarray]
  hidden_path: str | None

TokenWindow:
  chain_id
  t0, t1
  step_id | None
  hidden: optional lazy view
```

关键要求：

```text
1. 支持 step mode 和 boundary-free token/window mode；
2. 全部 token/window 指标必须 causal，不能读未来 token；
3. gold_error_step 之后默认 mask，另存 no-mask appendix；
4. 所有 split 按 problem_id GroupKFold；
5. 输出 schema manifest，避免后续脚本各自猜字段。
```

第一版代码任务：

```text
anchorflow/data.py
  load_traces(npz, hidden_dir=None, layer=14, max_chains=0)
  iter_step_windows(trace)
  iter_token_windows(trace, win=32, stride=8, causal=True)
  make_labels(trace, mode="first_error", mask_post_error=True)
```

---

## 3. Anchor 层：`anchorflow/anchors.py`

不要再用 single `qvec`。prompt anchors 至少分四类：

```text
number/value anchors: 题目中的数字、单位、数量
entity anchors: 人名、物体、变量名
constraint anchors: 比较、条件、等式、不等式、限制
goal anchors: 问题最终要求的目标量
```

第一版用规则解析，避免依赖外部 LLM：

```text
regex numbers / fractions / percentages
simple noun spans around numbers
keywords: total, each, more than, less than, left, remaining, cost, ratio
last question sentence as goal anchor
```

输出：

```python
Anchor:
  anchor_id: int
  kind: str              # number/entity/constraint/goal
  text: str
  char_span: tuple[int,int] | None
  token_span: tuple[int,int] | None
  value: float | None
```

代码任务：

```text
anchorflow/anchors.py
  parse_anchors(prompt_text, tokenizer_offsets=None)
  anchors_to_jsonl(...)
  anchor_coverage_stats(...)
```

验证：

```text
每个问题至少 1 个 goal anchor；
GSM8K/MATH/OmniMath 统计 anchor 数量分布；
随机抽样 30 个 prompt 人工检查 anchor 是否合理。
```

---

## 4. Anchor 表征层：`anchorflow/anchor_repr.py`

把 anchor 从文本 span 变成 hidden anchor vectors。

第一版表征：

```text
anchor vector = prompt span hidden mean at selected layer
goal vector = final question span hidden mean
prompt global = original qvec fallback
```

若没有 tokenizer span：

```text
v0 fallback: 用 qvec + prompt/global anchor，只跑流程；
v1: 接入 tokenizer offsets 重新抽取 prompt-span anchors。
```

代码任务：

```text
build_anchor_bank(trace, anchors, hidden, layer)
  returns A: (K,d), anchor_meta

normalize_anchor_bank(A, mode="unit|whitened")
```

验证：

```text
anchor vectors 不能全退化到 qvec；
不同 anchor 之间 cosine 分布要有区分度；
random anchors / shuffled anchors 作为负对照。
```

---

## 5. Transport 层：`anchorflow/transport.py`

核心对象：response window 到 prompt anchors 的软运输。

```text
H_t: 当前 response token/window hidden cloud, shape (n,d)
A: prompt anchor vectors, shape (K,d)
S_t = cosine(H_t, A) or whitened dot
P_t = softmax(S_t / tau) or Sinkhorn(S_t)
z_t = mean_token_mass(P_t) in simplex Δ^K
```

第一版不用复杂 Sinkhorn，先做 causal softmax transport：

```text
tau calibrated on correct chains
window hidden unit-normalized
optional qvec/global anchor as extra anchor
```

输出特征：

```text
anchor_entropy_t       # 运输质量是否过分扩散
anchor_coverage_t      # 有多少 anchor 被当前窗口触达
goal_mass_t            # 是否仍对目标 anchor 有质量
constraint_mass_t      # 是否仍连到条件/数字 anchors
anchor_detach_t        # max/mean similarity drop
transport_jump_t       # ||z_t - z_{t-1}||
transport_cusum_t      # online cumulative break
```

代码任务：

```text
compute_transport(H, A, tau=0.07, mode="softmax")
transport_features(P, anchor_meta)
causal_transport_trace(trace, anchors, hidden, win, stride)
```

必要消融：

```text
single qvec anchor
multi-anchor but shuffled kind
random anchor vectors
no goal anchor
numbers-only
constraints-only
no hidden, only text anchor counts
```

---

## 6. Anchored Information Volume：`anchorflow/volume.py`

`cloud_V` 已经被证明接近 `logN`，不能再用 raw volume。

新定义必须 length-controlled and anchor-conditioned：

```text
1. local residual cloud:
   R_t = H_t - projection_to_anchor_subspace(H_t, A_active)

2. anchored logdet:
   logdet(cov(R_t) + eps I), after shrinkage and fixed rank

3. anchored effective rank:
   entropy(eigenvalues(cov(R_t)))

4. length residual:
   fit volume ~ logN + pos on train folds / correct chains,
   score = residual volume.
```

解释：

```text
健康复杂推理可以高 volume；
错误风险是 anchor mass 断裂后 volume 异常升高，或过早坍缩到低 volume。
```

代码任务：

```text
anchor_residual_cloud(H, A, active_mass=None)
anchored_logdet(R, rank=16, shrink=0.1)
anchored_eff_rank(R)
length_residualize_volume(feature, controls=[logN,pos], groups)
```

验证：

```text
anchored volume 与 logN 的相关性必须显著低于 cloud_V；
若 residual volume 仍无增量，volume 叙事降级，只保留 transport flow。
```

---

## 7. Boundary-Free Phase：`anchorflow/phase.py`

目标：不依赖强制分步骤，也能找潜在推理相变点。

输入序列：

```text
z_t                    # anchor transport simplex
anchor_detach_t
transport_jump_t
anchored_volume_resid_t
uncertainty_t
hidden_jump_t
```

第一版方法：

```text
causal CUSUM on transport_jump / detach
Bayesian online change point detection lite
PELT offline upper bound for boundary recovery
```

输出：

```text
phase_score_t
phase_alarm_t
predicted_boundaries
```

评估：

```text
1. boundary recovery: predicted boundaries vs step_token_ranges
2. first-error localization: phase peak around gold_error_step
3. early warning: alarm before or at gold step
4. residualized localization: remove pos/logN first
```

代码任务：

```text
cusum_phase(scores, lam, kref)
bocpd_lite(X, hazard)
pelt_boundaries(X, penalty)
evaluate_boundary_recovery(pred, gold_step_ranges)
```

---

## 8. Detector 与评估层

`anchorflow/detectors.py`：

```text
Baseline groups:
  length_pos
  EDIS_uncertainty
  spread_resultant
  anchor_uncertainty
  transition_surprise
  manifold_health_v1
  hypergraph_HGN_oof if available

AnchorFlow groups:
  transport_only
  anchored_volume_only
  phase_only
  transport_phase
  full_anchorflow
```

`anchorflow/residualize.py`：

```text
crossfit_residualize(feature, controls=[logN,pos], groups)
residualized_auroc(score, y, controls, groups)
residualized_localization(score, gold_step, controls, groups)
cluster_boot_increment(score_new, score_base, y, groups)
```

`anchorflow/online.py`：

```text
calibrate_threshold_on_correct_chains(score, eps_fpr)
online_alarm_curve(score_t)
evaluate_alarm(fpr, recall, delay, early_rate)
```

报告必须包含：

```text
step AUROC / AUPR
chain AUROC / AUPR
within-chain top1 and expected top1
residualized localization gain
high-spread subset
confident-wrong subset
online FPR/recall/delay
bootstrap CI by problem
```

---

## 9. Intervention 层

`anchorflow/intervention.py` 不做 hidden steering，先做最低风险 prompt repair / micro-replay。

触发器：

```text
random step/window
highest entropy
highest spread
anchor_uncertainty
AnchorFlow phase break
AnchorFlow constraint detachment
```

干预策略：

```text
micro-replay:
  "Re-check the constraints involving {anchors}. Continue from the last reliable step."

prompt repair:
  insert a short reminder of detached anchors before continuing.

best-of-n local:
  sample N continuations only after flagged point, choose by answer verifier.
```

评估：

```text
wrong-to-right flip rate
correct-to-wrong damage
final accuracy gain
extra tokens
repair locality
trigger delay
```

代码任务：

```text
build_repair_prompt(trace, alarm, detached_anchors)
run_repair_batch(...)
score_repair_results(...)
```

顶会红线：

```text
如果 detection 小增量但 intervention 无收益，故事不够；
如果 intervention 显著高于 high-entropy/random trigger，故事成立。
```

---

## 10. Validation Suite

新增：

```text
anchorflow_validation_suite.py
```

命令：

```bash
python anchorflow_validation_suite.py \
  --datasets gsm8k,math,omnimath \
  --layers 10,14,18,22 \
  --data_dir /gz-data/research/demo/data \
  --hidden_dir /gz-data/research/demo/data/hidden \
  --folds 5 \
  --n_boot 500 \
  --output_dir outputs/anchorflow_validation
```

输出：

```text
outputs/anchorflow_validation/
  summary.json
  summary.md
  per_dataset_layer/*.json
  ablations/*.json
  figures/
```

每个结果 JSON 必须保存：

```text
config
git commit
dataset/layer/schema
feature names
fold assignments
raw metrics
bootstrap CIs
failure notes
```

---

## 11. 分阶段执行顺序

### Phase 1：无模型、无干预的 AnchorFlow 诊断

写：

```text
anchorflow/data.py
anchorflow/anchors.py
anchorflow/transport.py
anchorflow_anchor_audit.py
tests/test_anchorflow_transport.py
```

跑：

```bash
python anchorflow_anchor_audit.py --dataset gsm8k --data_dir /gz-data/research/demo/data --layer 14 --max_chains 120 --output_dir outputs/anchorflow_smoke
```

成功：

```text
transport features 在 high-spread / confident-wrong subset 有增量；
random/shuffled anchors 显著下降。
```

### Phase 2：anchored volume 与 residualization

写：

```text
anchorflow/volume.py
anchorflow/residualize.py
anchorflow_validation_suite.py
```

成功：

```text
anchored volume 与 logN 解耦；
full_anchorflow > anchor_uncertainty +0.02，且 bootstrap 稳定。
```

失败处理：

```text
删除 volume 主张，保留 transport/phase。
```

### Phase 3：boundary-free phase

写：

```text
anchorflow/phase.py
anchorflow_phase_audit.py
```

成功：

```text
token/window phase boundary 能恢复 step boundary；
first-error localization 在 residualized setting 仍高于随机。
```

### Phase 4：在线报警

写：

```text
anchorflow/online.py
```

成功：

```text
FPR 0.10 下 recall >= 0.40，median delay <= 0。
```

### Phase 5：干预闭环

写：

```text
anchorflow/intervention.py
anchorflow_intervention_suite.py
```

成功：

```text
AnchorFlow trigger 的 wrong-to-right flip rate 明显高于 entropy/random；
correct-to-wrong damage 可控。
```

---

## 12. 第一批代码任务清单

先写最小可行闭环，不碰生成模型：

```text
Task A. scaffold anchorflow package + data loader
Task B. rule-based prompt anchors + JSON inspection output
Task C. causal softmax transport + transport features
Task D. anchorflow_anchor_audit.py:
        compare transport_only vs anchor_uncertainty
        include random/shuffled anchor ablation
Task E. residualized evaluation helper:
        remove logN/pos and report honest increments
```

本阶段不写：

```text
Sinkhorn transport
HMM/EM
hypergraph model
hidden steering
LLM-based anchor parser
full repair loop
```

原因：

```text
先证明 multi-anchor transport 不是 qvec cosine 的包装；
否则后面的复杂模型都会变成昂贵的分类器。
```

---

## 13. 远端第一轮命令草案

代码写完后先跑 smoke：

```bash
cd /gz-data/research/demo
git pull

python anchorflow_anchor_audit.py \
  --dataset gsm8k \
  --data_dir /gz-data/research/demo/data \
  --hidden_dir /gz-data/research/demo/data/hidden \
  --layer 14 \
  --max_chains 120 \
  --folds 3 \
  --n_boot 50 \
  --output_dir outputs/anchorflow_smoke
```

再跑正式 L14：

```bash
for d in gsm8k math omnimath; do
  python anchorflow_anchor_audit.py \
    --dataset $d \
    --data_dir /gz-data/research/demo/data \
    --hidden_dir /gz-data/research/demo/data/hidden \
    --layer 14 \
    --folds 5 \
    --n_boot 200 \
    --output_dir outputs/anchorflow_l14
done
```

最后多层：

```bash
for d in gsm8k math omnimath; do
  for l in 10 14 18 22; do
    python anchorflow_anchor_audit.py \
      --dataset $d \
      --data_dir /gz-data/research/demo/data \
      --hidden_dir /gz-data/research/demo/data/hidden \
      --layer $l \
      --folds 5 \
      --n_boot 200 \
      --output_dir outputs/anchorflow_layers
  done
done
```

---

## 14. 结果记录模板

每次运行后必须写：

```text
Result analysis:
  哪些 claim 被支持/反驳；
  是否超过 anchor_uncertainty；
  是否摆脱 logN/pos；
  是否能解释 high-spread/confident-wrong。

Follow-up research:
  继续 transport / volume / phase / intervention 哪一支；
  是否需要放弃某个叙事。

Optimization suggestions:
  哪个 proxy 太粗；
  哪个模块需要消融；
  下一轮最小代码改动是什么。
```

顶会写作底线：

```text
每一个漂亮故事都必须配一个杀伤性消融。
没有 random/shuffled anchors 的 AnchorFlow，不可信；
没有 residualized localization 的 phase break，不可信；
没有 intervention 的 online detector，不够动人。
```
