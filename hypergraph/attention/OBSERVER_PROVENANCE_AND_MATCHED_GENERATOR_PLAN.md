# Attention Hypergraph：Observer Provenance 与同生成器实验重构计划

## 1. 当前结论

本轮不改变 attention-row hypergraph 的图定义，先修复数据协议与运行入口。

当前 GSM8K 训练失败不是 GPU、图构造或类别划分问题，而是 provenance 校验把两种不同语义混在了一起：

- `model_name`、tokenizer、prompt protocol、attention layer/head、forward mode 等字段决定实际表示是否兼容；
- `generator_model`、`generator_model_commit` 描述原回答来自哪里；在 `observer` replay 中，它们不决定由固定 observer 模型计算出的表示空间。

ProcessBench 同一子集包含多个 generator。现有实现把 `generator_model` 纳入所有 replay 模式的唯一指纹，导致统一 Llama-3.1-8B observer 对多来源回答前向后仍被拒绝训练。

## 2. 研究问题与实验分层

### 2.1 主检测实验：generator-tag-matched reconstructed replay

仅选择 `generator == Llama-3.1-8B-Instruct` 的样本，用本地
`Meta-Llama-3.1-8B-Instruct` checkpoint 对冻结回答做 teacher-forcing forward。

该设置研究“名义同生成器回答在目标模型表示空间中的错误信号”。由于 ProcessBench 没有保存精确 checkpoint revision、原始 rendered prompt 和原始 token IDs，结果必须标为 reconstructed replay，不能声称恢复了原始生成轨迹。

官方数据的计划样本规模为：

| subset | total | matched | erroneous | correct |
|---|---:|---:|---:|---:|
| gsm8k | 400 | 61 | 38 | 23 |
| math | 1000 | 139 | 86 | 53 |
| olympiadbench | 1000 | 164 | 116 | 48 |
| omnimath | 1000 | 162 | 131 | 31 |
| total | 3400 | 526 | 371 | 155 |

### 2.2 辅助泛化实验：all-generator observer

全部 ProcessBench 回答通过同一个 observer 模型前向。该设置研究统一观察模型对跨来源错误的可检测性，不作为原生成模型的内部机制证据。

必须保留并报告 `generator_model`，用于：

- 每个 generator 的样本数与类别分布；
- 按 generator 分层的评估；
- leave-one-generator-out 泛化实验；
- 检查模型是否只学到来源模型或长度差异。

### 2.3 后续严格机制实验：model-native generation

由本地 checkpoint 重新生成答案并同步保存完整 prompt、token IDs、checkpoint hash、generation config 和在线 attention。该分支需要对新回答重新标注，不在本轮代码修复范围内。

## 3. 本轮代码范围

### M1：拆分 representation compatibility 与 source provenance

1. 保留完整的 `trace_method_provenance`，用于逐样本审计。
2. 新增显式的 representation provenance/fingerprint：
   - 始终包含 observer/replay 模型、tokenizer、prompt protocol、replay mode、dtype、提取方法以及实际 layer/head axis；
   - `same_generator` 模式继续包含并核验 generator identity 与 commit；
   - 经状态机验证的 `observer_counterfactual` 模式从兼容性 fingerprint 中排除 `generator_model` 和 `generator_model_commit`，但不从 trace metadata 中删除；未知或不完整状态继续 fail closed。
3. 输出 source-generator 分布，避免跨 generator 混合被静默隐藏。

### M2：加入显式 generator 选择

1. `inspect`、`build`、`train` 共用 `--generator-model` 数据过滤参数。
2. 过滤发生在 trace limit 之前，确保 `--limit` 表示筛选后的样本数。
3. 匹配使用数据集中记录的明确 generator tag，不通过模糊名称猜测模型身份。
4. 训练输出和 prediction CSV 保存 generator 字段。
5. shell pipeline 将 generator selector 写入配置门，并使用独立的 run 目录后缀，防止 matched-generator 与 all-generator 结果互相覆盖。

### M3：让 preflight 与 train 使用相同的 cohort gate

当前 `inspect` 只逐条构图，不执行整批 representation/provenance gate，因此会出现 `preflight=True` 后训练立即失败。

重构后，preflight 必须在启动任何 GPU fold 前检查：

- representation fingerprint 唯一性；
- layer/head axis 一致性；
- replay provenance 状态机；
- extraction shard scope；
- observer/unverified trace 的显式许可；
- generator 过滤后的样本数与类别分布。

### M4：拆分 extraction 与 training 代码哈希

现有 wrapper 用同一个 `method_code_sha256` 同时保护 trace 与 run；任何纯训练端修改都会错误地使有效 trace 无法复用。

改为：

- `extraction_code_sha256`：只覆盖生成 trace 语义的代码与参数；
- `training_code_sha256`：覆盖图构造、模型、目标和训练代码；
- 旧 trace 的内嵌 extraction method JSON 继续作为真实性依据；
- 不伪造或重写已有 trace provenance。

如旧 `pipeline_request.json` 只能识别合并哈希，迁移逻辑必须 fail closed，并提供一次性、可审计的兼容路径；不得静默改写。

## 4. 非目标

- 旧服务器缓存由 `scripts/migrate_artifacts_layout.sh` 原样移动到
  `data/attention_traces/`；迁移不改写 trace 内容。
- 不通过篡改 `.npz` 中的 `generator_model` 绕过校验。
- 不把 observer 结果表述成原生成模型的因果机制。
- 本轮不改变超边阈值、传播算子、pooling 或网络容量。
- 本轮不启动完整 GPU 实验；先完成代码、测试和独立审查。

## 5. 验收标准

### 单元与集成测试

1. 两条 observer trace 仅 `generator_model` 不同：representation fingerprint 相同，cohort gate 通过。
2. observer trace 的 replay model、tokenizer、prompt protocol 或 layer/head 不同：仍拒绝混合。
3. same-generator trace 的 generator identity/commit 不同：仍拒绝或在逐样本状态机中失败。
4. `--generator-model Llama-3.1-8B-Instruct` 只保留目标样本，且 limit 在过滤后生效。
5. inspect/preflight 能在训练前发现 cohort 不兼容。
6. prediction 与结果配置记录 generator selector 和实际 generator 分布。
7. shell pipeline 的 matched 与 all-generator 输出目录不会冲突。
8. 纯训练代码变更不会迫使重新抽取语义未变的 trace；抽取代码或参数变化仍会被拒绝复用。

### 运行门槛

- 目标 pytest 全部通过；
- shell syntax check 通过；
- ruff/compile 检查通过；
- 独立代码审查无阻断问题；
- 先运行 matched-generator 的最小 CPU/preflight，再提交两卡训练。

## 6. 服务器恢复策略

代码修复前不要重跑，因为会得到同一确定性错误。修复并同步服务器后：

1. 保留 audited attention traces；
2. 将失败的 `gsm8k_response_layer14` 运行目录移动到带时间戳的备份名称，而非立即删除；
3. 先运行 generator 分布/preflight；
4. 分别启动 matched-generator 与 all-generator observer，使用不同 `RUN_ROOT`；
5. 结果首先报告样本量、类别分布和 generator 分层指标，再比较超图模型性能。

## 7. 实施顺序

1. M1 provenance 拆分与测试；
2. M2 generator selector、输出审计与测试；
3. M3 统一 preflight/train cohort gate；
4. M4 哈希边界与旧 trace 复用策略；
5. 文档、CLI 示例和服务器恢复命令；
6. 全量本地验证与独立审查。
