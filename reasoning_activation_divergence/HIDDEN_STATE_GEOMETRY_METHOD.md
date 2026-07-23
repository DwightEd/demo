# Hidden-state discriminative geometry

## 1. 研究问题

本模块不再拟合“正常推理流形”。它直接检验：在相同 nuisance 和 observer 输出摘要
之外，原始 hidden trajectory 是否仍含有可跨数据集泛化的错误判别信息。

主统计量是 held-domain OOF log-loss 增量：

```text
Δhidden|output = NLL(output + nuisance) - NLL(hidden + output + nuisance)
```

正值表示 hidden 有增量；只有置信区间下界也大于 0，才算稳定证据。这是判别关联，
不是因果或功能性结论。

## 2. 真实数据与证据边界

输入是四个子数据集的：

```text
<data-root>/<dataset>/selected/trace.raw_residual_stream.npz
```

manifest 指向每条推理链的 `[response_token, stored_layer, hidden]` mmap shard。
程序只保留 response generator 匹配 `llama3.1-8b` 的记录，并验证 observer model。
entropy/NLL 若已嵌在该 manifest 中就直接读取；否则从同目录 `trace.npz` 按唯一
`chain_idx` 对齐，并逐条复核 dataset、generator、observer、label 与 step ranges。

需要明确区分：

- hidden 是 Llama-3.1-8B observer 对候选答案做 teacher-forcing replay 时保存的真实
  residual stream；
- 它不是原始在线生成时逐 token 缓存的 generation-matched hidden/KV trace；
- `token_entropy`、`token_nll` 是 logits 派生的 step summary；
- 当前 artifact 没有完整 vocabulary logits，本方法也不会伪造或退化补齐 logits。

第一版方法从每个已完成 reasoning step 的末 token 读取所有已存层状态。它没有把
4096 维 hidden 压成几个手工几何标量后再分类。

## 3. 两个互不混用的任务

### Whole chain

使用整条回答，包括首错后的状态，回答“完整轨迹中最多有多少可判别信息”。结果字段为：

```text
claim_scope = retrospective_information_ceiling
```

它可以反映错误后果，不能宣称能提前预警。

### Strict prefix

在预测第 `t` 步是否为首错时，只使用 `0..t-1` 步已经完成后的 hidden 和输出摘要。
风险集在首错处停止，首错为 step 0 的样本因缺少 prompt-end state 而明确记为
left-truncated，不用第一个响应 token 冒充。

```text
claim_scope = prospective_first_error_association
```

严格前缀获得增量也只支持 prospective association；功能性仍需 activation patching、
Jacobian/Fisher 或其他干预验证。

## 4. 首个插件方法：`raw_functional_probe`

每个 LODO fold 内依次执行：

1. 只用 outer-train chains 做 chain-balanced PCA；每条链给 PCA 相同数量的行，
   held domain 不参与投影或标准化。
2. 将 step × layer × 4096 状态投影为低维 channel，但保留 time 和 layer 轴。
3. whole-chain 使用固定 DCT time basis × DCT layer basis；strict-prefix 使用
   current/remote-history × DCT layer basis。
4. 用 rank-one tensor coefficient
   `time_or_history_factor ⊗ layer_factor ⊗ hidden_factor` 做 logistic probe，避免拟合
   完整高维系数张量。
5. 每个 hidden arm 都显式包含已拟合静态 arm 的精确参数解；非凸优化没有得到更低的
   同一训练目标时，原样回退到静态解，避免把优化差异误认为 hidden 增量。
6. 同 fold、同样本比较以下 arms：
   - `nuisance`：长度与位置控制；
   - `output_only`：nuisance + entropy/NLL summaries；
   - `hidden_only`：nuisance + hidden tensor；
   - `output_plus_hidden`：nuisance + output + hidden；
   - 多个 sample-keyed axis-permutation repeats：破坏共享 time/layer 对齐，但保留
     每条链的状态值集合。

这里的 `hidden_only` 和 `output_only` 名称表示“没有另一个主模态”，两者都保留相同
nuisance head。

固定 reversal 不是有效的轴顺序 null：若 train/test 同时反转，分类器可以学回这个
全局可逆变换。因此实现使用按 sample key 固定、可复现但跨样本不同的 permutation，
默认重复 3 个 seed。time-order 汇总只在 `visible_steps>=3` 的可辨识层报告；若该层为空，
诊断明确写为 `not_identifiable`，不会让主实验失败。

## 5. 评估与统计

- outer split 固定为四域 leave-one-dataset-out；
- 同一数据集内，同一 problem group 的多个 prefix rows 共享总权重；本地数值 ID 会先加
  dataset 限定，避免不同数据集复用 `0,1,...` 时发生假重叠；
- 仅当 artifact 真正提供 `problem_sha256:*` 时才审计跨域重题；否则 preflight 明确报告
  `cross_domain_problem_hash_audit.status=unavailable`，不会伪称已经完成跨域去重；
- 每条 eligible row 恰有一个 OOF prediction；
- 指标按 domain 分开报告，并计算等权 domain macro；
- hidden 增量和有序轴相对 null 的增量按 problem group 做 cluster bootstrap；
- 全部投影、标准化、模型参数和 null seed 都在 fold 边界内固定。

优先看 `results.json` 中两个任务各自的：

```text
summary.increments.hidden_given_output_summary_nll
summary.randomization_checks.time_axis_order
summary.randomization_checks.layer_axis_order
```

increment 的区间是固定训练集、固定模型后的 conditional test problem-group cluster
bootstrap；randomization checks 报告多 seed 敏感性范围，明确不是置信区间。当前 null arm
仍在全部 outer-train rows 上拟合、没有按 `visible_steps>=3` 重新拟合，因此结果字段明确标为
`sensitivity_only_not_axis_order_evidence`，不能当作干净的旋转/轴顺序因果证据。

## 6. 工程边界

```text
hidden_state_geometry/
  contracts.py       小型 typed records，不传递整段 metadata blob
  data.py            provenance gate、chain_idx 对齐与 mmap shard loader
  tasks.py           whole-chain / strict-prefix 风险集
  representation.py fold-only PCA、功能基与 axis null
  model.py           rank-one tensor logistic
  preprocessing.py  grouped weights 与 fold-only scaling
  method.py          方法输入/输出 protocol
  registry.py        方法注册表
  methods/           独立方法插件
  evaluation.py      通用 LODO、指标与 cluster bootstrap
  experiment.py      与具体方法解耦的编排和 artifact writer
  cli.py             preflight / foreground run 入口
```

新增方法时只需：

1. 在 `methods/` 新建一个文件，实现 `fit_predict(FoldInput) -> MethodFoldResult`；
2. 使用 `@register_method("method_name")` 注册；
3. 在 `methods/__init__.py::load_builtin_methods()` 增加一行 import。

数据加载、任务定义、LODO、bootstrap、CSV/NPZ 写出和 CLI 编排无需复制。非默认插件可
通过 CLI 的 `--method-config-json` 接收自己的配置对象。

## 7. 审计产物

每次 run 写出：

- `preflight.json`：每域 cohort、层、首个 shard shape 与证据类型；
- `results.json`：两个任务的 macro/per-domain 指标与增量区间；
- `oof_predictions.csv`：逐 chain/boundary 的 OOF 概率；
- `fold_audit.csv`：held domain、训练/测试 rows、groups 与 events；
- `model_factors.npz`：每个 task/fold 的 PCA 与 rank-one coefficient。
- `artifact_manifest.json`：共同 run ID 与上述文件的 SHA-256，防止混用半次运行产物。

preflight 会 mmap 打开全部 eligible shards 并核验 shape/count/schema，但不会把完整
4096 维数组载入内存。正式 run 才读取所需 step-end states。远端真实结果产生前，本地
合成测试结果不得当作研究结论。
