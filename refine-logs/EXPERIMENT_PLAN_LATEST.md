# ECGH claim-driven 实验路线图

生成时间：2026-07-13 01:02（Asia/Shanghai）
上游：`research-reports/idea-discovery-20260713-010210.md`

## 1. 研究问题与判定对象

目标不是最大化某一个 chain AUROC，而是依次回答：

1. 模型生成过程中是否仍与题设中的具体证据锚点保持连接？
2. 当连接减弱时，anchor 不能解释的二阶表示几何是否出现局部变化？
3. 这些变化是否在不使用未来信息和人工 step 边界时定位首次错误？
4. 以该事件触发局部重放，是否在等计算量下改善最终正确率且不伤害正确链？

四个问题分别对应 Trace/Anchor、Gram、Hazard、Intervention 四级 gate。后一级不能掩盖前一级失败。

## 2. 数据与切分

### 2.1 主数据

- ProcessBench：GSM8K、MATH、OlympiadBench 子集，提供首次错误 step。
- PRM800K：用作跨标注风格复现。
- 同题多样本：每题至少 \(K=8\) 条生成，覆盖正确/错误配对；主检验必须在同题内完成。
- HaluEval / TruthfulQA：只做 response-level 外部迁移，不作为首错定位主证据。

### 2.2 模型

- 开发模型：单个 7B/8B instruct 模型。
- 迁移：至少一个不同模型家族和一个不同规模。
- tokenizer、model revision、chat template 与 generation config 全部写入 artifact。

### 2.3 切分协议

- 外层按 `problem_id` GroupKFold，推荐 5 folds。
- 同一题的所有 sample 永不跨 fold。
- 插补、标准化、残差化、阈值、温度、正则和特征选择仅在训练 fold 拟合。
- 若调参，使用训练 fold 内层 GroupKFold；test fold 每个配置只评价一次。
- 正确链作为 survival 右删失；错误首次发生后的 token/window 不参与 hazard loss。

## 3. Stage 0：数据忠实性与合成集成测试

### 实现

建立统一 trace schema，保存：

```text
chain_id, problem_id, correct, gold_error_step
rendered_prompt, prompt_token_ids, response_token_ids
attention_mask, offset_mapping, question_char_span
all_step_text, kept_step_indices, step_token_ranges
prompt_hidden[selected_layers], response_hidden/windows
time_axis metadata
model/tokenizer revisions, generation config, seed, package versions
```

### 必测断言

1. `rendered_prompt` 重新 tokenize 后与生成 `input_ids` 完全一致。
2. response token IDs 与 model.generate 返回后缀一致。
3. 每个 step token range 落在 response 轴内，且与 `kept_step_indices` 一一对应。
4. token feature 只有经 token-to-step mapping 聚合后才能进入 step multichannel。
5. prompt anchor char span 经 offsets 映射后，解码文本覆盖原 anchor。

### 通过标准

- 合成 tokenizer integration test 全过。
- 真实 GPU 抽样 100 条，ID/offset/range 零错位；发现一条错位即 fail fast。

## 4. Stage 1：非学习型信号 gate

### 4.1 Baselines

- length / relative position / step size；
- token entropy、committal score；
- raw spread、effective rank、cloud resultant；
- qvec cosine / EDIS；
- Lookback-style prompt mass；
- hidden transition norm、旧 transition tube。

### 4.2 ECGH primitives

- `anchor_mass`, `goal_mass`, `constraint_mass`；
- anchor entropy、coverage、transport jump/churn；
- anchor-residual energy；
- Gram effective rank、tail mass、logdet、subspace drift；
- readout visibility（可用 logit lens / low-rank Jacobian 近似）；
- strictly causal robust change score。

### 4.3 Null controls

- random anchor vectors；
- prompt spans permutation；
- anchor kind permutation；
- degree / window-size preserving evidence edge shuffle；
- time shuffle；
- future-reading noncausal upper bound；
- same feature with absolute position and length residualization。

### 4.4 Gate

先用纯非参数统计检验，不训练 Matrix-HMM：

- same-problem pair-micro AUROC；
- problem-macro AUROC；
- problem-cluster bootstrap 95% CI；
- 对 entropy/spread/length baseline 的 OOF incremental AUROC；
- high-spread / low-entropy 子集；
- 非链尾定位率。

只有真实 anchor 与 conditional Gram 同时优于相应 null control，才进入学习型 reader。

## 5. Stage 2：Boundary-free event discovery

### 设计

- token window \(w\in\{16,32,64\}\)，stride \(s\in\{4,8,16\}\)。
- 每个时刻的 median/MAD 只由过去窗口估计。
- 峰值需要最小 separation，避免一个 rupture 被重复计数。
- 禁止用 gold step boundary 构造 feature 或 segment。

### 评价

- 固定正确链 FPR：1%、5%、10%。
- event recall within \(\pm 1\) step / 对应 token 区间。
- signed delay：\(\hat T-T\)，同时报告 median、IQR 和 early/late 比率。
- false-alarm position histogram，专门检查链尾偏置。
- 同题 paired event score。

### 消融

`lookback only`、`transport only`、`Gram only`、`visibility only`、全部联合；再与随机位置、entropy peak、max-spread、CUSUM 和 supervised step boundary upper bound 比较。

## 6. Stage 3：First-error hazard reader

### 轻量 reader

首选可解释模型：

\[
\operatorname{logit}q_t=\beta^\top x_t+b.
\]

在风险集上拟合 regularized logistic hazard；所有预处理放入 fold 内 pipeline。若轻量 reader 已成立，再比较 TCN/GRU/Transformer 与 HyperCHARM，不允许直接用重模型掩盖原始信号失败。

### 上界模型

- causal compact evidence graph；
- causal evidence-geometry hypergraph；
- noncausal hidden/attention graph，仅作为 upper bound；
- 等参数普通 GNN、DeepSets、TCN。

### 指标

- token/window hazard AUROC 与 AUPRC；
- first-error top-1 / top-3 localization；
- chain cumulative-risk AUROC；
- integrated Brier score、ECE；
- 固定 FPR recall 与 delay；
- problem-macro 和跨模型零样本迁移。

## 7. Stage 4：Detection–Expression gap

### 假设

若模型已出现 anchor-residual rupture，但输出 entropy 很低且 rupture direction 对 logits 的可见性低，则更可能发生 confident hallucination。

### 实验

1. 用 held-out train fold 估计低秩 readout/Jacobian subspace。
2. 比较 correct、uncertain-wrong、confident-wrong 的 residual visibility。
3. 在匹配问题、长度、位置、entropy 后做条件检验。
4. 对 rupture direction 做小幅 steering，比较 logit KL、拒答概率与正确答案概率。
5. 使用随机正交方向、范数匹配方向和 anchor direction 作因果对照。

### Gate

只在 held-out paired data 上报告；若 visibility 只区分数据集或只在大幅 steering 下有效，则降级为解释性附录。

## 8. Stage 5：干预

### 策略

- `micro-replay`：退回最后安全 event，附加最近脱离的 1–3 个 anchors，局部重生成。
- `repath`：保持答案前缀，要求从证据约束重新给出下一步。
- `abstain`：高风险且多次 replay 未恢复 evidence coupling 时拒答。

### 对照

- no intervention；
- random trigger；
- entropy trigger；
- max-spread trigger；
- self-consistency / best-of-\(N\)；
- 相同额外 token budget 的无锚点 replay；
- oracle first-error upper bound。

### 指标

```text
wrong-to-right
correct-to-wrong damage
net accuracy gain
trigger precision / coverage
extra tokens and wall time
accuracy-cost frontier
abstention coverage-accuracy
```

阈值只在 calibration fold 选择，并在所有触发链上评价，而不是只对已知错误链运行。

## 9. 核心消融矩阵

| ID | Exact trace | Semantic anchors | Lookback | Conditional Gram | Causal events | Hazard | Intervention |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | ✓ | – | – | – | – | – | – |
| B1 | ✓ | qvec | – | – | – | logistic | – |
| B2 | ✓ | – | prompt mass | – | ✓ | logistic | – |
| A1 | ✓ | ✓ | – | – | ✓ | logistic | – |
| A2 | ✓ | ✓ | ✓ | – | ✓ | logistic | – |
| A3 | ✓ | ✓ | – | ✓ | ✓ | logistic | – |
| Full-detector | ✓ | ✓ | ✓ | ✓ | ✓ | survival | – |
| Full-loop | ✓ | ✓ | ✓ | ✓ | ✓ | survival | ✓ |
| Null-span | ✓ | shuffled | ✓ | ✓ | ✓ | survival | – |
| Upper-bound | ✓ | ✓ | full attention | ✓ | noncausal | neural | oracle |

## 10. 运行顺序与停止规则

1. 本地 unit + synthetic integration。
2. GPU 小样本：32 questions × 4 samples，查 schema、显存和对齐。
3. 128 questions × 8 samples，完成非参数 gate。
4. 只有 gate 通过才扩到全量和第二模型。
5. 轻量 hazard OOF。
6. 只有轻量读出显示增量才训练 HGN/TCN 上界。
7. 先离线 replay policy value，再做真实 streaming generation。
8. 通过 calibration 后运行完整 intervention test；test 不回看调参。

停止条件：

- exact trace 有错位：停止全部下游。
- semantic anchor 不优于 qvec / shuffle：删除语义主张。
- Gram 对 baseline 无 OOF 增量：不做 Matrix-HMM。
- event 只在链尾触发：删除 local rupture 主张。
- intervention 净收益不为正：论文止于检测/机制，不声称闭环。

## 11. 计算预算

以 7B/8B、选 4–6 个层、只保存 prompt top-\(k\) evidence 和窗口统计为基准：

- 小样本 schema/gate：单卡约数小时；
- 128×8 开发集：约 1–2 GPU-day，取决于上下文长度和生成数；
- 全量三子集、两个模型：约 8–16 GPU-day；
- 轻量 hazard：CPU 数小时；
- HGN/TCN 上界：每 fold 1–4 GPU-hour；
- intervention：额外生成预算按触发 coverage 线性增长。

所有 attention 统计必须在线压缩；若实现退回完整 \(S^2\) attention dump，视为工程 gate 失败。

## 12. 结果表模板

主表至少包含：

| Method | Pair AUROC | Problem AUROC | Recall@5%FPR | Median delay | Brier | Cross-model |
|---|---:|---:|---:|---:|---:|---:|

干预表：

| Trigger | Wrong→Right | Correct→Wrong | Net gain | Extra tokens | Coverage |
|---|---:|---:|---:|---:|---:|

每个数值报告 problem-cluster bootstrap CI；同题方法比较使用 paired bootstrap 或 permutation test，并对主假设数量做校正。

## 13. 预注册式主结论边界

只有下列条件都满足，才可写“模型错误是证据耦合几何的局部断裂，并可用于干预”：

1. same-problem、grouped held-out 增量成立；
2. semantic anchor 优于随机/置乱 anchor；
3. boundary-free detector 不是链尾/长度代理；
4. hazard 校准有效；
5. 等计算干预净收益为正；
6. 至少跨一个模型家族或数据集复现。

否则应按最接近的已通过 gate 收缩结论，不用单个高 AUROC 覆盖负结果。
