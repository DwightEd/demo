# GPT-5.6 研究方案：CFEA

> Controlled First-Error Attribution and Repair
>
> 方案设计：GPT-5.6（独立方案代理，xhigh）
>
> 代码实施：Codex
> 日期：2026-08-05

## 1. 研究问题

项目最终要解决的是：在模型生成一个推理步骤之前或刚开始发生首错时，判断错误风险，并区分以下几种不能混为一谈的现象：

1. **预测**：内部状态能否提前预测下一步会错。
2. **中介/传播**：错误已经出现后，Attention 或 FFN 是否继续传递错误。
3. **修复**：替换某条内部路径是否能恢复正确 token 或正确步骤。
4. **根因**：哪一条内部路径的变化真正导致了首错。

预测、相关、修复和根因是四种不同强度的结论。项目不得由“可预测”直接升级为“机制根因”。

## 2. 关键识别结论

### 2.1 ProcessBench 能回答什么

ProcessBench 提供自然推理链和首错步骤标签，适合：

- 首错前风险检测；
- 链内首错定位；
- 首错后的错误传播分析；
- 在人工校正目标下做 repair / mediation 实验；
- 跨 GSM8K、MATH、OlympiadBench、OmniMath 的外部验证。

### 2.2 ProcessBench 不能单独回答什么

若错误链与校正链在目标 token 之前拥有完全相同的 token prefix，确定性模型在该 prefix 上的激活也完全相同。因此，自然校正 pair 不能识别“首错前 Attention 路由已经不同”或“FFN 参数知识已经不同”。

这类 pair 只能给出：

- 正确-vs-错误 token 的目标方向；
- 对错误状态做 patch 后能否 repair；
- 错误出现后的中介和传播证据。

它不能作为 root-cause donor。

### 2.3 根因需要什么

Attention/FFN 根因必须在**受控的 pre-decision counterfactual cohort**上识别：

- 两个条件只改变一个明确的语义变量；
- 在决策 token 前已经形成不同的、可验证的内部计算要求；
- 两条轨迹共享模板与对齐规则；
- 目标 token 可由程序或形式验证器精确计算；
- donor alignment 在实验前登记。

因此项目保留两种 pair：

- `target_correction`：自然错误与最小校正，只支持 target reference、mediation 或 repair。
- `controlled_root`：受控前决策反事实，才有资格进入 root-cause gate。

## 3. 方法名称

项目文件夹名：

`causal_first_error_attribution`

方法简称：

**CFEA — Controlled First-Error Attribution and Repair**

这个名字强调三点：首错、受控识别、归因与修复分离。旧的 `component_resolved_hazard` 不再作为推荐主方法。

## 4. 信息传递图

在决策位置 `q`，对每个层 `l`、Attention head `h` 和源 token `s` 建边：

[
e_{l,h,s	o q}=W_{O,l}^{(h)}left(a_{l,h,q,s}v_{l,h,s}ight).
]

边不是 attention weight 本身，而是写入 residual stream 的向量消息。以目标输出方向

[
d=W_U[y_{mathrm{desired}}]-W_U[y_{mathrm{baseline}}]
]

定义描述性边分数：

[
g_{l,h,s}=langle e_{l,h,s	o q},dangle.
]

图的节点包括：

- prompt token；
- separator token；
- 已完成推理步骤 token；
- 当前 decision token；
- Attention head；
- transformer block；
- 输出目标方向。

图用于提出“哪一层、哪一头、哪一来源 token 可能承载关键消息”的候选。它本身不是因果证据；因果效应必须由 rerun patch 测量。

## 5. Attention × FFN 因子干预

在预注册层上分别捕获 recipient 和 donor 的：

- block 输入 residual `r_pre`；
- Attention branch 输出 `A`；
- FFN branch 输出 `F`。

对 recipient 重跑四种组合：

[
M_{00}=M(A_r,F_r),quad
M_{10}=M(A_d,F_r),quad
M_{01}=M(A_r,F_d),quad
M_{11}=M(A_d,F_d).
]

结果用 desired-vs-baseline logit margin 计量。定义：

[
Delta_A=M_{10}-M_{00},
]

[
Delta_F=M_{01}-M_{00},
]

[
Delta_{AF}=M_{11}-M_{10}-M_{01}+M_{00}.
]

另做 `r_pre` patch，用于区分：

- 错误已经在更早层形成；
- Attention 本层路由贡献；
- FFN 本层更新贡献；
- Attention 与 FFN 的交互贡献。

自然校正上的 margin 改善只标记为 mediation/repair；受控 pair 才能进入 root-cause candidate。

## 6. 必须存在的对照

正式机制结论至少需要：

- random donor；
- wrong source token；
- wrong head 或 wrong layer；
- norm-matched noise；
- 同 problem/template 内未改变目标变量的 placebo；
- token-level rescue 和 step-level rescue。

效应必须超过对照，且按 domain → problem/template 分组 bootstrap 的置信区间通过零点，才能标记为 supported。缺少对照时，输出必须是 `controls_pending`，不能用描述性均值替代正式结论。

## 7. 首错检测任务

### 7.1 时间边界

候选步骤 `t` 的检测状态固定在该步骤首 token 之前：

[
q_t=mathrm{step_token_start}[t]-1.
]

送入模型的 input IDs 必须在 `q_t` 处物理截断，不能只依赖 causal mask。步骤 0 必须保留，不能因为缺少“上一步”而被删除。

### 7.2 评估单位

不能把所有 step 行混池后只报告 AUROC。主评估是每条链内部：

- First-error Top-1 localization；
- MRR；
- 正确链 false-alarm rate；
- 正确步骤 false-alarm rate；
- detection lead time；
- token/step repair rate。

训练/测试划分按 problem_hash 和 counterfactual sibling group 成组，并采用 leave-one-domain-out。相同问题、模板或反事实兄弟样本不能跨 split。

### 7.3 检测器与机制归因的关系

检测器可以使用首错前 residual/graph tensor，但其结论仍然是预测。机制实验负责解释哪些路径在受控条件下改变输出。二者可共享数据结构，不能共享结论标签。

不再把长度污染的手工标量作为核心证据，也不再把 Ridge/MLP 探针性能当作 Attention/FFN 根因。

## 8. 数据契约

推荐数据布局：

```text
RAGTruth/processbench_observer_llama31_full/
├── gsm8k/selected/
├── math/selected/
├── olympiadbench/selected/
├── omnimath/selected/
└── controlled_math/selected/
```

每个可运行域至少包含：

```text
selected/
├── trace.npz
└── causal_first_error_v1/
    └── onset_pairs_v1.jsonl
```

`trace.npz` 提供完整 token IDs、attention mask、prompt token count、step token ranges。ProcessBench 自然域还提供 `gold_error_step`；纯受控域不强制该字段。

每个 `controlled_root` pair 必须提供 recipient/counterfactual record、decision position、目标 token、template/condition、intervention variable 和 donor alignment。

## 9. 执行阶段

### Stage A：可识别性审计

不加载模型，只核验现有数据能支持哪类结论。若没有 `controlled_root`，root cause 必须报告 not identifiable。

### Stage B：双轨迹图提取

对自然 pair 提取 recipient 决策图；对受控 pair 同时提取 recipient 与 counterfactual 图。保存未来截断后的 input IDs、layer/head/source 边质量、边 margin proxy、Attention/FFN branch 和 provenance。

### Stage C：受控因子干预

仅对 `controlled_root` 运行 Attention×FFN×pre-state rerun。自然 pair 的干预进入 repair/mediation 轨道。

### Stage D：对照与正式 gate

加入随机 donor、错误 source/head/layer、等范数噪声和 placebo。按 domain → problem/template bootstrap，输出 individual effects 和群体置信区间。

### Stage E：首错监测

在 ProcessBench 上构造严格 prefix 状态，训练结构化 detector，并用链内定位、LODO 和 false alarm 评估。检测结论与因果结论分开报告。

## 10. 当前实现与未完成项

当前代码应做到：

- 无模型的 pair/data identifiability audit；
- 决策位置物理截断；
- recipient/counterfactual 双图提取；
- residual message 而非纯 attention weight 的图分数；
- Attention×FFN 四格 patch 和 pre-state patch；
- 结论 scope 强制区分；
- 链内首错定位与分组 split；
- 前台进度条和单一 shell 入口。

在以下条件满足前，项目不得声称已定位根因：

- 真实 `controlled_root` 数据已生成并验证；
- null controls 已实现和运行；
- token-level 与 step-level rescue 已验证；
- 多 problem/template、多 seed 的分组置信区间通过门槛。

## 11. 最小可证伪假设

- **H1 Routing**：受控 donor Attention patch 对目标 margin 的提升超过 random donor 与 wrong-source 对照，并产生 token/step rescue。
- **H2 FFN**：受控 donor FFN patch 的提升超过 random donor 与 norm-matched noise，并产生 rescue。
- **H3 Earlier state**：只有 pre-state patch 有效而 A/F 本层 patch 无效，说明错误已在更早层形成。
- **H4 Interaction**：`Δ_AF` 显著，说明 Attention 与 FFN 不能被解释为两个独立加性通道。
- **H0**：所有效应不超过对照；此时内部组件可能仍可用于预测，但不支持路由或 FFN 根因结论。
