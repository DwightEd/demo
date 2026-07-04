# 推理的谱几何：核心假设锚点

> 记录日期：2026-07-04  
> 目的：把外部论文 *The Spectral Geometry of Thought* 中有机制感的观察，整理成本项目后续实验的核心假设。本文档是研究锚点，不把外部结论直接当作本项目结论；所有 claim 都需要在 ProcessBench / 本地 hidden 数据上复验。

---

## 1. 背景动机

这条线不再只围绕模型输出的 chain-of-thought 文本，而是直接观察 Transformer 内部表征的谱结构如何随任务、层、token 和步骤变化。

外部工作最有价值的地方在于：它把谱结构当作跨模型、跨架构的内部机制信号，而不是单模型轶事。覆盖 Qwen、Pythia、Phi、Llama、DeepSeek-R1 等多个家族后，如果某类谱变化仍稳定存在，就更像是推理态的结构性特征。

本项目已有主线是 step-level first-error detection：给定一条多步数学推理链，定位第一个错误步骤。谱几何线的价值在于提供一个机制层解释：错误步骤是否对应表征从“局部协调的推理态”脱离，或者出现 step 边界附近的谱流断裂。

---

## 2. 外部观察转化为本项目假设

### H1：推理态伴随谱压缩

外部观察：多数模型在推理任务上出现更低的谱指数 `alpha`，即谱更压缩，主导方向更集中。

本项目假设：当模型进入有效推理态时，step 内 token cloud 的 Gram 谱会表现出更低 `alpha` 或更低有效谱维度。错误步骤可能不是简单“更压缩”或“更发散”，而是相对正确推理态出现可测的谱形状偏离。

可检验信号：
- `alpha(step tokens)`
- `dalpha = alpha_t - alpha_{t-1}`
- `pr / eff_rank`
- `rho_dir / rho_dir_c / rho_sub`

注意：本项目已有结果显示，一阶矩 κ 是稳定主信号；谱形状若要成立，必须证明它在 `[κ + logN]` 或更强基线之上有增量。

### H2：模型能力越强，谱压缩越明显

外部观察：模型越强，推理任务中的谱压缩现象越突出。

本项目假设：如果谱压缩是推理机制而非数据伪迹，那么它应随模型能力、任务难度或推理深度呈系统变化。

当前本项目限制：现有核心数据主要来自 Llama-3.1-8B-Instruct teacher-forcing。跨模型 claim 暂时不能写成本项目结论，只能作为后续扩展方向。

最低复验要求：
- 同一 pipeline 跑 base / instruct 或至少两个规模模型；
- 保持相同数据、相同 step 标签、相同层选择；
- 报告 bucket AUROC 和 GroupKFold 增量，避免把长度或难度当成能力效应。

### H3：指令微调可能造成谱反转

外部观察：base 模型中常见“推理 alpha 低于事实回忆”，instruction-tuned 后该关系可能翻转。

本项目假设：instruction tuning 改变的不只是输出风格，也可能改变模型调用知识和组织推理的内部通道。对错误检测而言，这意味着同一个谱指标在 base 与 instruct 上可能方向不同。

本项目写作约束：
- 不直接声称“instruction tuning 导致谱反转”，除非本地做了 base/instruct 对照；
- 若只使用 instruct 模型，应写成“在 instruction-tuned 审阅设定下，谱指标方向需要实证确定”；
- 所有 `alpha` / `rho_dir` / `eff_rank` 的方向都用 held-out detection sign 决定，不手动指定。

### H4：谱级联是局部协调，而不是全网共振

外部观察：token 级别存在谱级联，局部同步随层间距离衰减，推理任务中的同步更弱。

本项目假设：推理不是所有层同时共振，而是局部跨层协调逐步传递。错误或脆弱推理可能表现为跨层谱流同步下降、局部失同步或同步结构变浅。

当前代码对应：
- `spectral_flow.py::cascade_chain`
- 每层滑动 token window 计算 `spectral_alpha`
- 对 `alpha` 序列取差分
- 计算跨层 `alpha-gradient` 相关

当前结果字段：
- `cascade.rho_by_distance`：S6a，层距越远，同步是否衰减；
- `cascade.gold_boundary_sync`：S6c，gold 错误边界附近的局部同步能否链内定位错误步；
- `cascade.chain_sync`：S6b，链级平均同步是否区分正确/错误链。若缺失，需要记录 skipped reason。

### H5：谱标点应与推理步骤边界对齐

外部观察：谱相变信号可能和推理步骤边界对齐，像内部思维过程的“谱标点”。

本项目假设：如果 step 标签是真实推理结构的近似，那么错误发生点附近应出现可检测的谱边界信号。

重要修正：本项目数据已经有 `step_token_ranges`。因此谱标点实验应优先使用 step-native 或 boundary-native 设计，而不是只用固定 token sliding window。

推荐实验设计：
- step-native：每个 step 计算每层 `alpha / κ / eff_rank`，再看 step-to-step 差分；
- boundary-native：比较 step `t-1` 和 step `t` 的谱变化，直接对齐 gold error step；
- adaptive-window：若需要 token window，窗口长度应随相邻 step 长度自适应，而不是固定 `window=32`。

---

## 3. 与本项目已有主线的关系

本项目当前最稳的几何信号是：

```text
κ / resultant / coherence = step 内 token 单位方向的集中度
错误步骤通常 κ 更低，即方向更发散
```

谱几何线不能替代 κ，而应回答三个更具体的问题：

1. `alpha / eff_rank / rho_dir` 是否在 `[κ + logN]` 上有显著增量；
2. 跨层谱级联是否能定位 gold error boundary；
3. 谱形状是否解释 κ 失效的情形，例如长分类讨论、结构化发散或难任务。

如果这些问题的答案是否定的，谱几何仍可作为负结果：说明本数据中的可检测几何信号主要是一阶矩集中度，而不是跨层谱流结构。

---

## 4. 当前代码中的实验映射

| 假设 | 当前代码信号 | 粒度 | 备注 |
|---|---|---|---|
| H1 谱压缩 | `alpha`, `pr`, `rho_dir`, `rho_sub` | step-native | 直接按 `step_token_ranges` 切 step |
| H1 增量 | `inc_over_kexp_logn` | step-level OOF | 必须看 CI 是否跨 0 |
| H5 边界断裂 | `break_kap`, `break_alpha` | boundary token window | 依赖固定 `window`，需改进为 step-boundary native |
| H5 相位形状 | `conv_kap`, `conv_alpha` | chain-level sliding window | 当前受 token window 长度影响 |
| H4 谱级联 | `cascade.rho_by_distance` | token sliding window | 检测跨层同步随层距衰减 |
| H4/H5 局部失同步 | `cascade.gold_boundary_sync` | token sliding window + step boundary | gold 边界只用于定位，谱序列来自滑窗 |

---

## 5. 方法学红线

1. 不把 cross-problem AUROC 当主证据，必须报告 within-chain / bucket / GroupKFold。
2. 所有新谱信号必须过 `[κ + logN]`，更强版本应过 `[κ + logN + uncertainty]`。
3. S6 若继续使用 sliding window，必须报告有效链数、跳过原因和窗口覆盖率。
4. 对有 step 标签的数据，边界实验优先 step-native，不让固定 token window 决定样本是否进入。
5. 跨模型、base-vs-instruct、能力 scaling 只能作为待验证扩展，不能混进当前 Llama-3.1-8B 的结论。

---

## 6. 下一步实验建议

1. 重写 S6 的 step-native 版本：每个 step、每层一个 `alpha`，再计算跨层 step-delta 同步。
2. 给现有 S6 加 skipped diagnostics：链太短、finite sync 不足、类别不足、gold step 不可比较分别计数。
3. 对 `gsm8k_flow.json` 已有结果补表：S6a 层距相关、S6c gold-boundary top1、与随机期望差值。
4. 在 `math / omnimath` 上复验 `rho_dir / eff_rank` 是否只在难任务上补 κ。
5. 若拿到 base 模型 hidden，单独做 base-vs-instruct 的 `alpha` 方向检验。

---

## 7. 当前写作定位

可以写成：

> 推理的谱几何为本研究提供了一个机制锚点：推理可能对应内部表征从分散表示进入局部协调的低维谱态，而错误步骤则表现为这种协调的破裂或方向集中度下降。本项目在 ProcessBench 审阅设定下，将该思想操作化为两类可检验信号：一阶矩集中度 κ，以及谱形状 / 跨层谱流同步。现有证据支持 κ 是稳定主信号；谱形状和级联结构仍需在严格基线与 step-native 边界实验中验证。

---

## 8. StepFlow 论文与代码记录

外部锚点：*Reasoning Fails Where Step Flow Breaks*（ACL 2026 main，arXiv:2604.06695；代码 `external_repos/step-saliency`）。

这篇工作的出发点不是 residual geometry，而是 **step-level information flow**：

- **Step-Saliency**：把 token-level `attention x gradient` saliency 聚合成 step-to-step map，沿 question / thinking / summary 三段观察信息流。
- **Shallow Lock-in**：浅层过度关注当前 step，较少利用早先推理上下文。
- **Deep Decay**：深层逐渐丢失 thinking segment 的 saliency，summary 更依赖自身或最近几个 step。
- **StepFlow**：用两个 test-time actuator 修复 flow：
  - **OEB / Odds-Equal Bridge**：浅层 attention logits 上做组级 KL projection，把一部分注意力质量从 same-region 桥接到 question 或 analysis region。
  - **SMI / Step Momentum Injection**：深层在新 step 的第一个 token，把前一步最后一个小窗口的 value states 均值作为 residual momentum 注入。

代码对应：

- `scripts/analyze_step_saliency.py`：从文本中切 question / thinking / summary steps，聚合 token saliency 到 step matrix。
- `src/interventions/bridge_guard_oeb.py`：OEB 在 pre-softmax logits 上调整 same/bridge 两组注意力质量。
- `src/interventions/smi.py`：SMI 追踪 step 边界，在 `position_index == pending_step_start` 时对 head output 加 `alpha * mean(value_states[prev_step_tail])`。
- `src/interventions/attention_manager.py`：patch GPT-OSS eager attention，在 `on_pre_softmax / on_post_softmax / on_output` 三处挂 intervention。

对本项目的启发与边界：

1. StepFlow 证明了“step 之间的信息流断裂”可以是可干预对象，但它主要修的是 propagation / memory / carry-forward 类错误。
2. 它的干预是结构性修复，近似 always-on 或弱条件触发；它并不根据当前 step 的 residual geometry 判断“错误状态坏成哪一种”。
3. 它读的是 attention/saliency flow；本项目读的是 hidden-state residual geometry。两者不是替代关系，而是互补轴：

```text
flow axis     : 当前 step 是否从正确的历史位置取信息
geometry axis : 当前 step token cloud / residual state 是否处在健康推理态
```

因此，本项目不应复刻 OEB/SMI，而应提出 **flow-geometry mismatch**：

| flow | geometry | 可能病理 | 合理干预 |
|---|---|---|---|
| 断 | 坏 | step propagation break | bridge / momentum |
| 正常 | 坏 | 接了前文但内部状态发散 | truncate + recompute / branch |
| 断 | 稳 | 局部自洽但遗忘题设 | question-anchor / premise reset |
| 正常 | 稳 | 概念或公式错误 | verifier / abstain / external check |

这条线把几何信号从“报警器”升级为“病理分型器和干预选择器”。

---

## 9. κ_exp 与二阶矩：不是替代，而是矩分解

令一步内 token hidden 归一化后为单位方向：

```text
u_i = h_i / ||h_i||,    w_i >= 0,    sum_i w_i = 1
```

`spectral_flow.py::kappa_exp` 计算的是指数加权一阶矩合成长度：

```text
m = sum_i w_i u_i
κ_exp = ||m||
```

从 feature space 看，`κ_exp` 是一阶矩；但从 Gram 读法看，它已经是一个 pairwise 二次型：

```text
G_ij = <u_i, u_j>
κ_exp^2 = ||sum_i w_i u_i||^2
        = sum_i sum_j w_i w_j <u_i, u_j>
        = w^T G w
```

所以更准确的说法是：

> κ 不是普通意义上“二阶矩的退化情况”；κ 是方向分布的一阶均值长度，但它等价于 Gram 矩阵在权重向量 `w` 上的一个 rank-1 quadratic readout。

再定义未中心化二阶矩 / scatter：

```text
A = sum_i w_i u_i u_i^T
trace(A) = 1
C = A - m m^T
trace(C) = 1 - κ_exp^2
```

这给出一个干净分解：

```text
total directional energy = mean-direction energy + residual scatter energy
1                        = κ^2                  + trace(C)
```

解释：

- `κ^2` 读的是所有 token 是否朝同一个均值方向收束；
- `C` / `A` 的谱读的是剩余散布如何组织：各向同性散开、低维分支、双极轴向结构、多个簇等；
- 如果 token 方向近似单峰 vMF，`κ` 近似充分统计量，额外谱形状很难加；
- 如果 token 方向是 mixture / branching / bipolar / heavy-tail，`κ` 会把很多不同结构都压成同一个低值，此时二阶矩和高阶结构才可能补上。

因此，本项目下一步不应简单问“二阶矩是否超过 κ”，而应问：

```text
在 κ 相同或相近的 step 中，二阶矩能否区分
  correct structured branching
  vs error isotropic confusion
  vs error premise drift
  vs confident wrong coherence
```

推荐组合特征：

| 量 | 定义 | 机制含义 |
|---|---|---|
| `κ_exp` | `||sum w_i u_i||` | 均值方向集中度 |
| `lam1(A)` | scatter 最大轴 | 是否存在主轴 / 双极结构 |
| `eff_rank(A)` | scatter 有效秩 | 发散是低维结构化还是高维混乱 |
| `trace(C)` | `1 - κ^2` | 去均值后的总散布 |
| `lam1(C)` | centered 主散布轴 | 去掉均值后是否还有稳定分支 |
| `bipolarity` | high `lam1(A)` with low `κ` | 均值抵消但轴向强，可能是分支或正反两团 |
| `clusterability(G)` | Gram 上的团结构 | 低 κ 是多分支还是噪声 |

这也是“超图 / 高阶读法”的自然入口：κ 只读 `w^T G w` 这一条 rank-1 投影；二阶谱读 `G` 的 eigenstructure；超图读多个 token group 的 Gram minors / volumes / cluster consistency。

文章表述建议：

> We treat κ not as the final geometric signal, but as the rank-one mean component in a directional moment decomposition. Failures that κ cannot separate should appear in the residual scatter spectrum and higher-order Gram structure.
