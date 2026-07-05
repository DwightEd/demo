# 推理的谱几何：核心假设锚点

> 记录日期：2026-07-04  
> 目的：把外部论文 *The Spectral Geometry of Thought* 中有机制感的观察，整理成本项目后续实验的核心假设。本文档是研究锚点，不把外部结论直接当作本项目结论；所有 claim 都需要在 ProcessBench / 本地 hidden 数据上复验。

---

## 1. 背景动机

这条线不再只围绕模型输出的 chain-of-thought 文本，而是直接观察 Transformer 内部表征的谱结构如何随任务、层、token 和步骤变化。

外部工作最有价值的地方在于：它把谱结构当作跨模型、跨架构的内部机制信号，而不是单模型轶事。覆盖 Qwen、Pythia、Phi、Llama、DeepSeek-R1 等多个家族后，如果某类谱变化仍稳定存在，就更像是推理态的结构性特征。

本项目已有主线是 step-level first-error detection：给定一条多步数学推理链，定位第一个错误步骤。谱几何线的价值在于提供一个机制层解释：错误步骤是否对应表征从“局部协调的推理态”脱离，或者出现 step 边界附近的谱流断裂。

---

## 2. 外部论文实现与结论记录

外部锚点：*The Spectral Geometry of Thought: Phase Transitions, Instruction Reversal, Token-Level Dynamics, and Perfect Correctness Prediction in How Transformers Reason*（arXiv:2604.15350）。

这篇文章不是本项目要复刻的方法，而是说明“推理可以从 hidden activation 的谱动态中读出机制信号”。需要记录它做了什么、发现了什么，以及哪些地方不能照搬。

### 实现管线

- **对象**：每个模型、任务、层上的 hidden activation matrix `H_l in R^{T x d}`，其中 `T` 是 token 序列长度，`d` 是 hidden 维度。
- **谱量**：对 token 维中心化后的 `H_l` 做 SVD，得到奇异值序列 `s_k`，再拟合幂律：

```text
log s_k ≈ c - alpha * log k
```

- **论文内定义**：`alpha` 越高，奇异值衰减越快，方差更集中到少数方向；`alpha` 越低，谱更平，表示更分布式。
- **比较设计**：不是 first-error step detection，而是 reasoning tasks vs factual recall tasks、prompt vs response、base vs instruct、cross-model/cross-family 比较。
- **token dynamics**：用 token sliding window 逐 token 计算 `alpha`，再看 `alpha` 梯度、跨层梯度相关、层距衰减和 step boundary 对齐。
- **预测实验**：用 layer/phase 上的 `alpha` 特征训练 logistic regression，做 correctness prediction；这仍是监督预测，不是无监督机制证明。

### 主要结论

1. **Reasoning/factual spectral separation**：11 个模型、5 个架构家族中，多数模型在 reasoning vs factual recall 上有显著 `alpha` 差异。
2. **Instruction tuning reversal**：base 与 instruction-tuned 的 reasoning/factual `alpha` 关系可能反转，说明指令微调可能改变内部表示组织方式。
3. **Generation shift taxonomy**：prompt 到 response 的谱变化可分成 expansion、compression、equilibrium，不同架构族不同。
4. **Scaling**：在 Qwen base 家族内，reasoning/factual 谱差异随规模变化。
5. **Token-level spectral cascade**：逐 token `alpha` 梯度的跨层相关随层距衰减，邻近层同步更强、远层更独立。
6. **Step spectral punctuation**：`alpha` 梯度峰值与推理步骤边界、段落、连接词、计算结果等 token 位置对齐。
7. **Correctness prediction**：部分模型上 `alpha` 特征可以强预测最终正确性，但 OOD 结果变弱，说明泛化仍需谨慎。

### 不能照搬的地方

- **任务不同**：论文主比较是 reasoning vs factual recall；本项目当前数据是 ProcessBench first-error localization。不能把“reasoning 任务的谱差异”直接解释成“错误步骤谱差异”。
- **模型覆盖不同**：论文跨 11 模型；本项目当前主要是 Llama-3.1-8B-Instruct 的审阅/teacher-forcing 数据。不能写 base/instruct reversal 或 scaling。
- **alpha 方向不能偷换**：论文方法定义里 high `alpha` = faster decay / more concentrated；但它把 lower reasoning `alpha` 解释为 reasoning spectral compression/更分布式推理态。我们本地必须报告 raw direction，不能用 `-alpha` 去迎合叙事。
- **step boundary 来源不同**：论文用生成文本中的 token-level boundaries；本项目有人工/ProcessBench step labels，应优先做 step-native 或 boundary-native 转移模型。
- **预测不是机制**：高 AUC 只能说明谱读数可预测正确性，不等于 `alpha` 导致正确或错误。

因此，本项目应该借鉴的是它的 **实验结构**：跨任务对照、跨模型对照、逐 token/逐层动态、边界对齐、OOD 验证；而不是照搬 `alpha` 作为我们的核心创新。

---

## 3. 外部观察转化为本项目假设

### H1：推理态伴随谱压缩

外部观察：多数模型在推理任务上出现更低的谱指数 `alpha`；论文将其作为 reasoning/factual 的谱相变信号。注意，论文方法定义中 high `alpha` 表示谱衰减更快、能量更集中，因此“lower alpha = compression”的措辞不能直接搬到本项目。

本项目假设：当模型进入有效推理态时，step 内 token cloud 的 Gram/SVD 谱形状会进入某种可区分的动态 regime。错误步骤可能不是简单“更压缩”或“更发散”，而是相对健康推理转移出现可测的谱形状偏离。

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

## 4. 与本项目已有主线的关系

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

## 5. 当前代码中的实验映射

| 假设 | 当前代码信号 | 粒度 | 备注 |
|---|---|---|---|
| H1 谱压缩 | `alpha`, `pr`, `rho_dir`, `rho_sub` | step-native | 直接按 `step_token_ranges` 切 step |
| H1 增量 | `inc_over_kexp_logn` | step-level OOF | 必须看 CI 是否跨 0 |
| H5 边界断裂 | `break_kap`, `break_alpha` | boundary token window | 依赖固定 `window`，需改进为 step-boundary native |
| H5 相位形状 | `conv_kap`, `conv_alpha` | chain-level sliding window | 当前受 token window 长度影响 |
| H4 谱级联 | `cascade.rho_by_distance` | token sliding window | 检测跨层同步随层距衰减 |
| H4/H5 局部失同步 | `cascade.gold_boundary_sync` | token sliding window + step boundary | gold 边界只用于定位，谱序列来自滑窗 |

---

## 6. 方法学红线

1. 不把 cross-problem AUROC 当主证据，必须报告 within-chain / bucket / GroupKFold。
2. 所有新谱信号必须过 `[κ + logN]`，更强版本应过 `[κ + logN + uncertainty]`。
3. S6 若继续使用 sliding window，必须报告有效链数、跳过原因和窗口覆盖率。
4. 对有 step 标签的数据，边界实验优先 step-native，不让固定 token window 决定样本是否进入。
5. 跨模型、base-vs-instruct、能力 scaling 只能作为待验证扩展，不能混进当前 Llama-3.1-8B 的结论。

---

## 7. 下一步实验建议

1. 重写 S6 的 step-native 版本：每个 step、每层一个 `alpha`，再计算跨层 step-delta 同步。
2. 给现有 S6 加 skipped diagnostics：链太短、finite sync 不足、类别不足、gold step 不可比较分别计数。
3. 对 `gsm8k_flow.json` 已有结果补表：S6a 层距相关、S6c gold-boundary top1、与随机期望差值。
4. 在 `math / omnimath` 上复验 `rho_dir / eff_rank` 是否只在难任务上补 κ。
5. 若拿到 base 模型 hidden，单独做 base-vs-instruct 的 `alpha` 方向检验。

---

## 8. 当前写作定位

可以写成：

> 推理的谱几何为本研究提供了一个机制锚点：推理可能对应内部表征从分散表示进入局部协调的低维谱态，而错误步骤则表现为这种协调的破裂或方向集中度下降。本项目在 ProcessBench 审阅设定下，将该思想操作化为两类可检验信号：一阶矩集中度 κ，以及谱形状 / 跨层谱流同步。现有证据支持 κ 是稳定主信号；谱形状和级联结构仍需在严格基线与 step-native 边界实验中验证。

---

## 9. StepFlow 论文与代码记录

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

## 10. κ_exp 与二阶矩：不是替代，而是矩分解

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

---

## 11. 扩展路线：多通道机制相变审计

只沿几何谱继续堆标量不够。更强的研究问题应当是：

```text
推理首错发生时，hidden geometry、step flow、attention、logits uncertainty、方向锚定这些通道如何共同改变？
这些改变是同一种失败模式，还是可分型的机制综合征？
```

对应脚本：

```text
mechanism_phase_audit.py
```

它把已有 `full_*.npz` 中的信号组织成几个机制通道：

| 通道 | 代表信号 | 机制读法 |
|---|---|---|
| geometry | `resultant`, `coherence`, `cloud_D`, `geom_ae`, `step_direction_jump` | step 内表征是否从健康推理态发散 |
| uncertainty / logits | `tok_U_D`, `tok_U_C` 的 step mean/var | 模型是否知道自己不稳，还是 confident wrong |
| attention / flow | `stepattn` 中的 `q_frac`, `sink_frac`, `attn_entropy` | 当前 step 是否仍从题设和前文取信息 |
| anchor | `q_align`, `d_q_align_bad` | 当前推理方向是否偏离 question anchor |
| dynamics | `d_*`, `cz_*` | 不是看静态水平，而是看首错前后的跳变和因果异常 |
| mismatch | `flow_geometry_mismatch`, `confident_geom_bad`, `coherent_anchor_drift` | 把报警器升级为失败分型器 |

该脚本输出四类证据：

1. **event study**：把每条错误链按 gold first-error step 对齐，观察 `t=-3...+3` 的通道轨迹；
2. **within-chain localization**：在同一条链内部，错误步是否是某个信号最异常的位置，避免只做跨题 AUROC；
3. **OOF group increments**：比较 `confounds`、`geometry`、`uncertainty`、`attention`、`all`，看通道组合是否真的超过长度、位置、文本密度；
4. **syndrome counts**：统计首错步属于低 κ 发散、高不确定、confident low-kappa、anchor drift、flow break、coherent wrong 等哪类。

运行方式：

```bash
python mechanism_phase_audit.py --selftest --boot 100
python mechanism_phase_audit.py /path/to/full_gsm8k.npz --layer 14 --boot 500
python mechanism_phase_audit.py --dataset gsm8k --data_dir /gz-data/research/demo/data --layer 14 --boot 500
```

解释原则：

- 如果 `geometry` 强但 `uncertainty` 弱，说明模型内部表征坏了但 logits 未必承认错误，这是干预价值最高的区域；
- 如果 `attention/flow` 强于 geometry，故事更接近 StepFlow 的 memory / carry-forward break；
- 如果 `geometry + uncertainty` 已经饱和，而 `attention` 没有增量，说明当前数据的错误更像局部状态发散，不是前文传播断裂；
- 如果 `mismatch` 类特征有效，文章可以从“检测错误”上升到“识别错误病理并选择干预”。

这条线的目标不是把所有东西压成一个分数，而是找出首错步的动态机制指纹：何时是几何相变，何时是信息流断裂，何时是 confident wrong，何时需要 verifier 而不是继续桥接。

注意：这一节是扩展路线，不是本文的原始谱假设主线。主线仍然是 `alpha` 谱压缩、base/instruct 反转、跨层谱级联和 step 边界谱标点。

---

## 12. 外部谱线索的本地转译路线

外部论文给本项目提供的不是可直接照搬的三个指标，而是三个值得本地转译的问题：

1. **Reasoning/factual 谱区分**：推理任务与事实回忆任务的 hidden activation 谱是否处在不同 regime；这需要新增 factual recall 对照数据。
2. **Instruction reversal**：base 模型中常见 reasoning alpha 低于 factual recall；instruction-tuned 后该关系可能翻转，说明微调改变了内部调用和组织知识的通道。
3. **Spectral cascade and punctuation**：token/layer 级局部同步随层距指数衰减；推理任务同步更弱；相变信号与 step boundary 对齐，形成“谱标点”。

当前 ProcessBench `full_*.npz` 数据只包含推理链和错误步骤标签，因此它只能回答一个更窄的问题：**错误步骤是否偏离健康推理的谱动态**。

- 可以直接检验：step-native 谱形状、谱转移残差、跨层谱动态、gold first-error boundary 的谱异常定位；
- 不能直接检验：reasoning vs factual recall 的任务对照、base vs instruct 的谱反转、能力 scaling；
- 这些不能从 GSM8K 错误定位数据里硬讲，必须补 matched factual-recall hidden 和 matched base/instruct hidden。

已有探索脚本：

```text
spectral_hypothesis_audit.py
```

该脚本刻意不引入 attention/logits，只做谱线索的粗审计；它不是最终方法：

| 模块 | 输出 | 对应假设 |
|---|---|---|
| step-native alpha | `alpha_mean`, `d_alpha_mean` | 外部 `alpha` 线索的本地粗读法 |
| boundary punctuation | `phase_jump_l2`, `phase_desync_layer_std`, within-chain top1 | H3 的 step 标点 |
| cascade decay | `rho_by_distance`, `exp_decay_fit_abs_rho` | H3 的层距衰减 |
| testability report | `H1/H2 status` | 明确哪些外部 claim 还缺对照数据 |

运行方式：

```bash
python spectral_hypothesis_audit.py --selftest
python spectral_hypothesis_audit.py --dataset gsm8k --data_dir /gz-data/research/demo/data --layers 10 14 18 22
python spectral_hypothesis_audit.py /path/to/full_gsm8k.npz --hidden_dir /path/to/hidden/gsm8k --layers 10 14 18 22
```

解释红线：

- `spectral_alpha` 的本地实现是 raw exponent；结论必须报告 raw direction，不应使用 `compression_score_low_alpha` 去迎合外部叙事；
- 如果 `phase_jump_l2` 在 gold boundary 上 top1 明显高于随机期望，可以写“首错边界出现 step-native 谱标点”；
- 如果 `rho_by_distance` 随层距单调或指数衰减，才支持“局部协调、逐步传递，而非整网共振”；
- 如果只有错误定位上的 alpha 差异，不能写成“推理任务比事实回忆更压缩”；
- 如果没有 base/instruct 成对数据，不能写 instruction reversal，只能写成待验证假设。

---

## 13. 链级动态试验：发散是否失去健康约束

新的本地工作假设：

```text
错误推理不是“更发散”本身，而是发散失去健康约束。
正确推理也会发散，但通常会恢复、仍锚定题设、并符合健康推理的状态转移。
```

因此不再只把所有 step 打散做二分类，而是把每条推理链作为序列来分析：

| 假设 | 问题 | 信号 |
|---|---|---|
| recoverability | 高发散 step 后面是否重新收束 | `next_recovery_1`, `next_recovery_2`, `spread_cusum` |
| anchored divergence | 发散是否仍围绕题设/问题方向 | `anchor_loss`, `unanchored_divergence`, `unanchored_cusum` |
| calibration | 发散时模型是否知道自己不稳 | `uncertainty`, `confident_divergence`, `confident_cusum` |
| healthy transition | 当前转移是否像正确链里的正常转移 | `transition_surprise`, `transition_cusum` |
| online alarm | 能否按链实时报警，而不是离线挑错步 | per-chain FPR/recall/delay risk curve |

对应脚本：

```text
chain_dynamics_audit.py
```

核心建模：

1. 从 `full_*.npz` 读取每条链的 `resultant / q_align / U_D / step_direction_jump / geom_ae / cloud_D` 等 step 序列。
2. 构造 `spread = 1 - resultant`，并加入 causal z、leaky CUSUM、future recovery 等动态量。
3. 只用 **正确链** 拟合健康转移模型：

```text
y_t = A [y_{t-1}, logN_t, pos_t, 1] + noise
transition_surprise_t = Mahalanobis residual under correct-chain transition model
```

4. 在 held-out 链上打分，报告：
   - overall step/gold-error AUROC；
   - high-spread subset，即“正确也发散”的困难区域；
   - within-chain gold-step localization；
   - online single/CUSUM alarm 的 FPR、recall、delay。

运行方式：

```bash
python chain_dynamics_audit.py --selftest
python chain_dynamics_audit.py --dataset gsm8k --data_dir /gz-data/research/demo/data --layer 14
python chain_dynamics_audit.py /path/to/full_gsm8k.npz --layer 14
```

解释红线：

- `next_recovery_*` 是离线机制分析，不是实时 detector；
- 实时 detector 只能使用 causal features，例如 `cz_*`, `*_cusum`, `transition_surprise_t`；
- 如果 high-spread subset 中 `transition_surprise` 或 `unanchored_divergence` 仍有效，才说明我们区分了“健康发散”和“错误发散”；
- 如果只有 `spread` 有效，故事仍只是 κ/resultant 的复述；
- 如果 online FPR/recall 曲线可用，才进入干预实验：不同状态触发 re-anchor、truncate/regenerate、verifier 或 StepFlow-style bridge。

### 13.1 现阶段优化建议

GSM8K 首次结果显示：整体上 `spread/resultant` 仍最强，但在 high-divergence subset 中 `transition_surprise` 明显超过单纯 `spread`。这说明链级动态方向有价值，但当前方法还只是第一版，需要先做以下优化，再考虑 HMM/HSMM/LRSM。

| 问题 | 风险 | 当前优化 |
|---|---|---|
| 默认 transition model 太宽 | 多个 obs 同时进入导致支撑集变小，难判断哪一轴有效 | 加 `--obs_grid`，默认比较 `spread`, `spread+uncertainty`, `spread+anchor`, `spread+anchor+uncertainty`, `+jump` |
| position 混杂 | `pos` 在 within-chain 中过强，CUSUM 天然随时间积累 | 给关键分数加 `*_resid_ctrl`，用 cross-fit 线性模型去掉 `logN/pos` |
| 正确推理也会发散 | 全局 AUROC 可能只复述 κ/resultant | 单独报告 high-spread subset，检验“健康发散 vs 错误发散” |
| online 与 offline 混用 | `next_recovery_*` 看未来，不能实时干预 | 输出分为 online causal scores 与 offline mechanism scores |
| 方法过于手工 | 组合如 `z(spread)+z(anchor)` 只是粗读法 | 用 ablation 先确定有效轴，再升级到 latent state model |

`chain_dynamics_audit.py` v2 的组织：

```text
1. 构造链级状态轴：
   spread, d_spread, anchor_loss, uncertainty, confident/unanchored composites

2. 多个健康转移模型并行：
   y_t = A [y_{t-1}, logN_t, pos_t, 1] + noise
   transition_surprise__<obs-set>

3. 关键分数去混杂：
   score_resid_ctrl = score - E(score | logN, pos)

4. 三个读数表：
   overall_features
   high_spread_features
   control_residual_features

5. online risk curve：
   per-chain FPR / recall / delay / early_warn
```

下一步若 v2 结果支持 dynamic axes，再进入 LRSM：

```text
z_t in {stable, healthy_explore, recovery, unanchored_drift, confident_wrong, post_error}
p(z_t | z_{t-1})        # HMM/HSMM transition
p(y_t | z_t)            # emission over spread/anchor/uncertainty/flow/spectrum
alarm = P(bad state | y_<=t)
intervention = state-conditioned action
```

如果 v2 仍然只剩 `spread/resultant` 有效，则不应升级复杂模型，应回到数据/信号抽取层面寻找更有机制性的 anchor 或 flow 信号。

### 13.2 v2 三数据集结果：哪些方向现在可行

目前 `chain_dynamics_audit.py` v2 已在 GSM8K / MATH / OmniMath 的 L14 上跑出完整结果。结论不是“找到几个好标量”，而是更具体：

```text
错误推理的可检测性主要来自：
1. 发散程度 spread/resultant；
2. 发散是否失去题设锚定 anchor；
3. 发散时是否伴随不确定性/校准变化 uncertainty；
4. 当前状态转移是否不像正确链的健康转移 transition_surprise。
```

但这些信号的地位不同。

| 方向 | 当前证据 | 判断 |
|---|---|---|
| `spread/resultant` | GSM8K 约 0.77，MATH/OmniMath 约 0.70；仍是最强单轴之一 | **可作为基线，不是创新点**。它太接近 κ/resultant 主信号，只能说明错误更发散。 |
| `anchor_uncertainty` 组合 | OOF group 在三数据集都最稳：GSM8K 0.811，MATH 0.781，OmniMath 0.809 | **最可行的主方向**。故事应从“发散是否仍被锚定、是否被模型感知”展开。 |
| `transition_surprise` | GSM8K high-divergence subset 中最好，残差化后约 0.742；MATH/OmniMath 只有约 0.63/0.62 | **可行但数据集依赖**。它更像区分“健康探索 vs 错误漂移”的二级机制，而不是全局最强 detector。 |
| `uncertainty` | 在 MATH/OmniMath 的 high-divergence subset 中很强，约 0.675/0.663 | **可行**。复杂题里错误发散更像 calibration/uncertainty 问题，而不只是几何问题。 |
| online alarm | FPR 约 0.20 时 recall 多在 0.44--0.50，median delay 0--1 step，early warning 约 0.2--0.3 | **可作为实时 guard 的雏形**，但还不是强 early-warning。 |
| raw localization/CUSUM | `pos top1=1.0`，CUSUM 与时间天然相关 | **必须去混杂**。未经位置控制的定位结果不能作为机制证据。 |
| spectral alpha/cascade | 当前数据只能做 step/gold-error 审阅，不能测试 reasoning vs factual recall 或 base/instruct reversal | **外部锚点，不是本项目可直接复用的 claim**。 |

因此，当前可讲的研究主线应是：

```text
错误不是简单的“低 κ / 高 spread”，而是推理链进入一种
unanchored + miscalibrated + transition-surprising 的坏动态状态。
正确推理也会发散，但更可能保持题设锚定、带有不确定性提示，并能回到健康转移轨道。
```

这条主线比“找到一组标量”更强，因为它把每一步放回完整链中看：同样发散，健康探索和错误漂移应有不同的路径形状。

### 13.3 下一步优化：从标量表走向状态轨迹

当前 v2 仍然过于手工，主要问题是：

1. 多数表仍是 step-level 标量 AUROC，容易退化为 feature hunting。
2. `pos/logN` 是强混杂，尤其 localization 表中 `pos` 过强。
3. `transition_surprise` 只是一阶 Gaussian Markov residual，还不是隐状态模型。
4. online alarm 只是阈值触发，没有把不同坏状态映射到不同干预动作。

下一版代码要先做三件事，作为进入 HMM/HSMM/LRSM 之前的门槛：

| 优化 | 目的 | 成功判据 |
|---|---|---|
| residualized localization | 去掉 `logN/pos` 后看首错步是否仍能被定位 | 残差化表中 dynamic/transition/pattern 信号仍高于随机期望 |
| trajectory pattern features | 不只看当前标量，而看最近窗口的 drift/rise/volatility/persistence | `trajectory_pattern` 或 `sequence_state` OOF 显著超过 `anchor_uncertainty` |
| group increment test | 不只报告 AUROC，而报告相对强基线的 cluster bootstrap 增量 | `anchor_uncertainty -> sequence_state` 的增量稳定为正 |

对应代码组织为 `chain_dynamics_audit.py` v3：

```text
pat_<signal>_drift_wK      # 最近 K 步当前值相对窗口起点的漂移
pat_<signal>_rise_wK       # 最近 K 步正向爬升强度
pat_<signal>_vol_wK        # 最近 K 步波动
pat_<signal>_persist_wK    # 最近 K 步处于异常 causal-z 的比例

residual_localization      # 只看 score - E(score | logN, pos)
causal_pattern_localization
group_increments_vs_anchor_uncertainty
```

如果 v3 显示 `sequence_state` 稳定超过 `anchor_uncertainty`，再进入 latent state model：

```text
z_t in {
  stable,
  healthy_explore,
  recovery,
  unanchored_drift,
  confident_wrong,
  transition_break
}

emission: spread / anchor_loss / uncertainty / transition_surprise / pattern features
transition: p(z_t | z_{t-1})
online risk: P(bad state | y_<=t)
intervention: re-anchor, verifier, regenerate suffix, or ask model to bridge the broken step
```

如果 v3 不能超过 `anchor_uncertainty`，则不要急着上 HMM；应回到信号抽取层面，重点补 attention/logits/loss/verifier traces，寻找真正能解释“为什么这一步错”的机制通道。

---

## 14. Prompt-anchor 连通性：从 cosine anchor 到几何 lookback / 超图

当前 `anchor_uncertainty` 的强度不能过度解释。它里面的 `anchor_loss` 来自：

```text
anchor_loss_t = 1 - q_align_t
q_align_t = cosine(step_direction_t, qvec_layer)
```

也就是说，它本质上是 step pooled direction 与 question/prompt baseline vector 的余弦相似度。它不是 attention lookback ratio，也没有直接建模“当前 token 是否仍在读取题设”。因此它能说明“隐藏方向是否偏离题设基向量”，但不能说明：

- 当前 token 是否真的回看 prompt；
- 题设里的数字/变量/约束是否仍与当前推理片段连通；
- 自然生成中是否存在无需强制分步骤也能发现的 latent boundary。

新的工作假设：

```text
错误推理不是只在一个 step 标量上发散，
而是 response token 流与 prompt / 题设约束之间的连通性发生局部断裂。
这种断裂可以用 attention lookback、hidden 几何 lookback、以及 token-level 超图/图割共同读取。
```

### 14.1 两种 lookback

| 通道 | 读法 | 当前状态 |
|---|---|---|
| attention lookback | `q_frac` / prompt attention mass，类似 lookback ratio | 如果 `stepattn` 存在，可直接用；已有 `attn_audit.py` 可做增量测试 |
| hidden-geometry lookback | response token/window hidden 与 `qvec` 的几何连接强度 | 当前缺口；可直接用 hidden shard + `qvec` 计算 |

hidden-geometry lookback 不等同于 attention。它问的是：当前生成片段的表示是否仍落在题设诱导的方向/子空间附近。这个信号可以在没有 attention 的情况下运行，也能和 attention lookback 做正交性检验。

### 14.2 超图/图读法先做非参数诊断

数据量还不适合一上来训练 Hypergraph Neural Network。更合适的是先把超图作为结构化读法：

```text
nodes:
  response token windows, plus one virtual prompt node

edges / hyperedges:
  temporal edges          邻近窗口的顺序连接
  hidden-neighbor edges   hidden centroid 近邻连接
  prompt-anchor edges     window centroid 到 qvec 的连接
  optional step labels    只用于验证，不用于构图

scores:
  prompt_degree_ratio     当前窗口有多少连接回 prompt anchor
  local_conductance       相邻 token windows 之间的局部连通性
  boundary_break          局部图割变强 / 跨边界连接变弱
  anchor_drop             prompt-anchor strength 的突降
```

这对应一个 boundary-free 问题：不强制 prompt 输出 “Step 1/2/3”，只在自然 token 流上滑窗构图，然后问无监督边界分数能否贴近已有 step boundary；已有 step label 只作为验证标尺。

### 14.3 分阶段代码路线

**Phase A：非参数 prompt-anchor 超图诊断**

新增脚本：

```text
hypergraph_anchor_audit.py
```

输入：

```text
full_*.npz + data/hidden/<dataset>/<id>.npy
```

输出：

```text
1. step/gold-error AUROC：
   anchor_mean, anchor_min, prompt_degree_ratio, hidden_jump, boundary_break

2. high-divergence / high-break subset：
   是否能在“正确也会发散”的区域区分错误漂移

3. boundary-free step-boundary recovery：
   不使用 step label 构图，只用 step label 检查边界分数是否贴近真实 step start

4. within-chain first-error localization：
   判断首错是否落在 prompt-anchor 断裂或图割异常附近
```

成功判据：

- hidden-geometry lookback 对 `anchor_loss/q_align` 有增量，或至少在 residualized localization 中保留；
- boundary score 在 step boundary recovery 上明显高于随机；
- first-error 附近出现 anchor-drop / graph-cut 异常；
- 如果有 attention，attention lookback 与 geometry lookback 低相关且组合增益为正。

**Phase B：自然生成边界**

在不要求模型分步骤的 prompt style 上重抽样，保留 token hidden / logits / attention。用 Phase A 的 boundary score 自动切出 latent segments，再把 segments 映射回 verifier / intervention。

**Phase C：状态与干预**

只有当 Phase A/B 证明边界和 prompt-anchor 断裂稳定存在时，再进入轻量状态模型：

```text
healthy_reading -> grounded_transform -> unanchored_drift -> transition_break
```

干预不再对整条链重写，而是在自动边界后的 suffix 上执行：

- re-anchor to prompt constraints；
- regenerate suffix from last stable boundary；
- verify only the current latent segment；
- ask for a bridge between two disconnected segments。

### 14.4 适配 `hypergraph-hallucination`

本地项目 `D:\projects\research\hypergraph-hallucination` 的核心数据格式是：

```text
x                    # node features
he_incidence_index   # [2, incidence] = node id, hyperedge id
he_attr              # hyperedge attributes
he_mark              # prompt->response vs response->response mark
he_member_counts
y_token
response_idx
```

原项目的 `processed_hypergraph.py` 用 attention row 构造 response token 的超边；本项目暂时不能假设 attention 总是存在，因此 v1 适配为 hidden-geometry hypergraph：

```text
node 0:
  virtual prompt/question node

node 1..N:
  response sliding-window nodes

hyperedges:
  prompt-anchor edge     [prompt node, response window]
  temporal edge          [window_i, window_{i+1}]
  hidden-neighbor edge   [window_i, top-k hidden-neighbor windows]
```

节点特征使用小维度诊断向量，而不是 4096 维 hidden 原向量：

```text
prompt_flag,
geom_anchor_cos,
prompt_degree_ratio,
window_spread,
normalized_degree,
relative_position,
prev_hidden_jump,
prev_boundary_break
```

这样做有三个原因：

1. 数据少，先避免直接训练大 HGN；
2. 保留几何可解释性；
3. 远端若有 `torch` / `torch_geometric`，导出的 `.pt` 可以接原项目的 HypergraphLayer / HyperCHARM；本地无 `torch` 时导出 `.npz` 并做 schema + numpy message-passing smoke test。

验证命令：

```bash
python hypergraph_anchor_audit.py --selftest --top 10
python hypergraph_anchor_audit.py --selftest --top 5 --export_hypergraphs --export_limit 3 --overwrite_export
```

当前本地验证结果：

```text
project hypergraph: valid 80/80
message-passing smoke: 80/80
mean nodes: 18.4
mean hyperedges: 51.1
export format on local Windows: npz, because torch is unavailable
```

下一步判断标准：

- 如果真实数据上 `boundary_recovery.hidden_jump` 能恢复自然/人工 step boundary，说明无需强制分步骤也能发现 latent segment；
- 如果 `geom_anchor_loss / prompt_degree_ratio / entry_anchor_drop` 在 first-error localization 中保留，说明 prompt-anchor 断裂是机制候选；
- 如果导出的超图喂给原项目 HyperCHARM 后相对 `anchor_uncertainty` 有增量，才考虑进入训练式超图模型；
- 如果无增量，则保留超图作为诊断/边界发现工具，而不是主 detector。

### 14.5 完整 token-level HyperCHARM 训练版

前面的 `hypergraph_anchor_audit.py` 只是非参数诊断：它构图、导出兼容 schema、做 message-passing smoke test，并没有训练超图模型。为了真正检验“超图模型能否从 token hidden 中挖出非手工信号”，新增完整训练脚本：

```text
hypergraph_token_hgn.py
```

对照本地 `D:\projects\research\hypergraph-hallucination` 原项目：

| 项 | 原项目实现 | 本项目完整训练版 |
|---|---|---|
| 构图输入 | `attention` `[L,H,seq,seq]` | hidden shards `[R,4,4096]` + `qvec` |
| 节点 | token nodes | virtual prompt node + response token nodes |
| 节点特征 `x` | self-attention diagonal `[seq, L*H]` | selected multi-layer token hidden；默认保留 raw hidden 幅度 + unit direction，可选拼接 anchor/spread/jump 诊断量 |
| 超边 | response token/head 的 attention row，阈值选 members | prompt-anchor、causal temporal、hidden-neighbor hyperedges |
| `he_attr` | mean attention, max attention, head id | mean weight, max weight, edge kind, causal age/span |
| `he_mark` | prompt-cross vs response-only | prompt-anchor vs response-only |
| 预测粒度 | `train_hypergraph.py` 为 node/token logits；`train_hyper_newresponse.py` 为 graph-level | node/token logits，再聚合回 step first-error |
| loss | `BCEWithLogitsLoss(pos_weight)`，只在 response nodes 上算 | 同样使用 BCE + pos_weight，但只在 correct/pre-error/gold-error token 上算；post-error token mask 掉 |
| 消息传递 | `node2edge([x_node, he_mark]) -> mean per hyperedge -> edge2node([he_attr, agg]) -> mean per node -> LayerNorm -> residual` | 保持同一 HyperCHARM 机制 |
| 验证 | token AUROC/AUPR 或 response graph AUROC/AUPR | GroupKFold-by-chain OOF：token AUROC/AUPR、step AUROC/AUPR、graph AUROC/AUPR、within-chain first-error top1 |

关键设计选择：

```text
1. 不再使用 window 节点作为主训练对象，而是每个 response token 一个节点；
2. 默认读取多层 hidden：layers 10,14,18,22；
3. 默认 `x_mode=hidden_diag --hidden_form both`：raw hidden 幅度 + unit direction + 我们已知可用的 anchor/spread/jump 诊断通道；
4. 支持 `x_mode=hidden` / `diag` 与 `hidden_form=raw|unit|both` 做消融，判断模型能力来自 full hidden、方向几何还是手工信号；
5. 默认 `--causal`，hidden-neighbor 只连向过去 token，避免把未来信息泄漏进实时检测；
6. step 标签只用于 y/mask/evaluation，不用于构图。
```

训练命令：

```bash
python hypergraph_token_hgn.py --dataset gsm8k --data_dir /gz-data/research/demo/data --layers 10,14,18,22 --x_mode hidden_diag --hidden_form both --folds 5 --epochs 30 --batch_size 1 --hidden_dim 128 --gnn_layers 2 --output_dir outputs/hypergraph_hgn
```

必要消融：

```bash
python hypergraph_token_hgn.py --dataset gsm8k --data_dir /gz-data/research/demo/data --x_mode diag --folds 5 --epochs 30 --output_dir outputs/hypergraph_hgn
python hypergraph_token_hgn.py --dataset gsm8k --data_dir /gz-data/research/demo/data --x_mode hidden --hidden_form raw --folds 5 --epochs 30 --output_dir outputs/hypergraph_hgn
python hypergraph_token_hgn.py --dataset gsm8k --data_dir /gz-data/research/demo/data --x_mode hidden --hidden_form unit --folds 5 --epochs 30 --output_dir outputs/hypergraph_hgn
python hypergraph_token_hgn.py --dataset gsm8k --data_dir /gz-data/research/demo/data --x_mode hidden --hidden_form both --folds 5 --epochs 30 --output_dir outputs/hypergraph_hgn
python hypergraph_token_hgn.py --dataset gsm8k --data_dir /gz-data/research/demo/data --x_mode hidden_diag --hidden_form both --no-causal --folds 5 --epochs 30 --output_dir outputs/hypergraph_hgn
```

解释标准：

- 如果 `hidden_diag/both > diag`，说明 full multi-layer hidden 中确实有超出手工 anchor/spread/jump 的可学习结构；
- 如果 `hidden/raw` 强于 `hidden/unit`，说明幅度/能量通道有信息；如果 `hidden/unit` 强，说明几何方向本身更关键；
- 如果 `hidden/both ≈ hidden_diag/both` 且超过 `diag`，说明超图模型主要从 hidden 表征本身挖信号，而不是复述我们给的标量；
- 如果 `diag` 已经等于 `hidden_diag`，超图训练没有证明新表征，只是非线性组合已有标量；
- 如果 `--no-causal` 明显更强，必须标记为 offline upper bound，不能用于实时检测主张；
- 如果 step-level / localization 不升，只 graph-level 升，说明模型只学到链难度或全局错误倾向，不满足首错定位目标。

当前本地验证：

```text
python -m py_compile hypergraph_token_hgn.py  # passed
python hypergraph_token_hgn.py --help         # passed
local selftest/train not run: Windows env has no torch; run on GPU server
```

后续研究与优化建议：

1. 先在 GSM8K 跑 `diag / hidden / hidden_diag / hidden_diag --no-causal` 四组，确认是否存在真正的 full-hidden 增量；
2. 若有增量，再扩展到 MATH / OmniMath，检查是否在困难集更明显；
3. 若只有 graph AUROC 提升而 step/localization 不升，说明超图模型学到的是难度，不是动态推理失败；
4. 若 causal 模型有效，再接入在线干预；若只有 non-causal 有效，只能作为事后分析或 verifier。

---

## 15. 回到主线：轻量 reasoning-flow 验证套件

### 15.1 超图支线的阶段性判断

`hypergraph_token_hgn.py` 的完整训练版已经能跑起来，但从 GSM8K 前几折看，收益和代价不匹配：

```text
node_dim = 32778
fold validation node AUROC 大致在 0.70--0.80
validation AUPR 大致在 0.31--0.50
训练时间显著高于标量/序列方法
```

这说明超图模型目前更像一个重型读法，而不是主线 detector。它可以保留为上界或辅助诊断，但不应继续围绕它做主要叙事。尤其如果 step-level / within-chain localization 没有明显超过 `anchor_uncertainty`，就不能声称它挖到了新的动态推理机制。

关于 fold：当前脚本默认 `--folds 5`，使用 GroupKFold；不是写死，可以改成 `--folds 3` 做 smoke。每个 outer fold 的 train 部分再按 group 随机切约 `--val_frac 0.15` 做 validation。

完整 GSM8K 结果：

```text
===== hypergraph token HGN | full_gsm8k.npz =====
chains 395 | error chains 205 | layers [10, 14, 18, 22]
x_mode hidden_diag | hidden_form both | causal True

oof_node_auroc        0.6905
oof_node_aupr         0.3990
oof_step_auroc        0.7602
oof_step_aupr         0.3602
oof_graph_auroc       0.6778
oof_graph_aupr        0.6831
oof_loc_top1          0.8539
oof_loc_expected_top1 0.4629
```

解释：

- `OOF` 表示每个样本只在自己所在 outer fold 的 held-out test split 上被预测，再把 5 个 fold 拼起来评估；不是训练集效果。
- 训练目标只有 token-level `BCEWithLogitsLoss(pos_weight)`，并用 validation node AUPR early stopping；`step / graph / localization` 都不是训练目标，而是测试时从 token risk 聚合得到的评估。
- `oof_node_auroc/aupr`：token 是否属于 gold first-error step 的风险预测。它最细，但标签噪声最大，因为 step 内所有 token 被同标。
- `oof_step_auroc/aupr`：把一个 step 内 token risk max/mean pooling 成 step risk，再判断哪个 step 是 first-error。这是最贴近当前 first-error detection 的指标。
- `oof_graph_auroc/aupr`：把整条链的 step risk 再取 max，判断这条链是否有错。它更像 chain-level detector，不是定位。
- `oof_loc_top1`：在有错误链里，gold first-error step 的分数是否排第一。这个数高，说明模型会把风险集中到首错附近；但它不处理 correct chain，因此不能单独作为实时 detector 指标。

结论：

```text
超图 HGN 的 step AUROC 约 0.76，localization top1 很高，但 node/graph 指标一般；
考虑到 node_dim=32778、训练成本高、相对 anchor_uncertainty/sequence 标量增量有限，
它暂不应作为论文主线。它的价值是证明 hidden token graph 中确实有可定位信息，
但主线应转向更轻量、更可解释、可实时干预的 anchored-flow 机制。
```

### 15.2 主线重新定义

主线不再是“找更复杂模型”，而是验证一个更有机制感的命题：

```text
推理失败不是单纯的 hidden spread 变大，
而是 constraint-grounded flow 在在线生成过程中发生断裂：
正确困难推理可以发散，但它应保持题设锚定、可恢复、且状态转移像健康推理；
错误推理则表现为 unanchored drift / confident wrong / transition break。
```

现有 `anchor_loss = 1 - cosine(step_direction, qvec)` 只是最低阶近似。它有效说明方向对了，但太粗；后续应升级为多约束 anchor field、attention/logit traces、latent boundary 与局部干预。

### 15.3 新增验证脚本

新增：

```text
mainline_validation_suite.py
```

它复用 `chain_dynamics_audit.py` 的核心统计，不重复造指标，只做统一批量验证和摘要。每个 dataset/layer 输出：

```text
anchor_uncertainty AUROC
sequence_state AUROC
sequence_state over anchor_uncertainty 的 OOF cluster-bootstrap increment
dynamic_online / transition_ablation 增量
high-spread subset 中最强特征
residualized localization gain
online alarm 在指定 FPR 下的 recall / delay
recommendation: promote_sequence_state / intervention_ready / do_not_overfit / needs_signal_redesign
```

成功判据：

1. `sequence_state` 若不能稳定超过 `anchor_uncertainty`，不要急着上 HMM/超图/大模型；
2. high-spread subset 必须能区分健康发散与错误发散，否则只是复述 κ/resultant；
3. residualized localization 必须去掉 `logN/pos` 后仍有 gain；
4. online alarm 必须报告 FPR、recall、delay，不能只报 step AUROC；
5. 若 detection 增量有限，应把创新集中到 constraint anchor field 与局部干预，而不是继续堆分类器。

### 15.4 运行命令

本地自测：

```bash
python mainline_validation_suite.py --selftest --output_dir outputs/mainline_validation_selftest --n_boot 30 --folds 3
```

远端主线 smoke：

```bash
python mainline_validation_suite.py --datasets gsm8k --layers 14 --data_dir /gz-data/research/demo/data --max_chains 120 --folds 3 --n_boot 50 --output_dir outputs/mainline_validation_smoke
```

远端正式验证：

```bash
python mainline_validation_suite.py --datasets gsm8k,math,omnimath --layers 14 --data_dir /gz-data/research/demo/data --folds 5 --n_boot 200 --keep_full_results --output_dir outputs/mainline_validation
```

多层稳健性：

```bash
python mainline_validation_suite.py --datasets gsm8k,math,omnimath --layers 10,14,18,22 --data_dir /gz-data/research/demo/data --folds 5 --n_boot 200 --keep_full_results --output_dir outputs/mainline_validation_layers
```

下一步优化：

1. 如果 `sequence_state` 没有显著增量，转向 signal redesign：multi-anchor hidden geometry、attention lookback、logit/loss trace、verifier trace；
2. 如果 online alarm 可用但 AUROC 增量有限，优先做局部干预实验，而不是继续调 detector；
3. 如果某数据集只靠 uncertainty 强，说明复杂题中模型“知道自己不稳”，要区分 uncertainty-aware failure 与 confident wrong failure；
4. 若 residualized localization 仍强，再考虑 latent state model；否则 HMM/EM 只会把混杂包装成状态。

---

## 16. AAAI27 主线草案：AnchorFlow / Causal Anchor-Transport Field

### 16.1 为什么不能停在现有标量

目前起作用的量大致是：

| 量 | 当前含义 | 问题 |
|---|---|---|
| `spread/resultant` | 当前 step token hidden 是否发散 | 正确困难推理也会发散；容易只是难度信号 |
| `anchor_loss = 1 - cos(step_direction, qvec)` | step direction 是否偏离 prompt/question 基向量 | 只是单一 cosine，不能表达题设中的多个约束 |
| `uncertainty / EDIS-like entropy` | 模型 logits 是否不稳定或没把握 | EDIS 等前人已经做了动态不确定性；且 confident wrong 会漏 |
| `transition_surprise` | 当前状态转移是否不像正确链 | 目前只是一阶 Gaussian/Markov residual |
| `CUSUM` | 在线累计异常 | 是报警机制，不是推理机制 |
| `hypergraph HGN` | 从 token hidden graph 学 risk | 结果有定位信息，但成本高、增量不足 |

因此论文不能写成“我们组合了一堆标量”。更好的定位是：这些标量只是一个更高阶对象的投影读数。

### 16.2 核心命题

```text
Reasoning errors emerge as online breaks in a prompt-anchored transport field.

错误推理不是 uncertainty/spread/anchor_loss 单独变大，
而是当前生成轨迹与题目约束锚点之间的连通性发生相变：
正确困难推理可以发散，但仍会在题设约束之间传输、回锚和恢复；
错误推理则表现为 anchor mass collapse、constraint detachment、
wrong-anchor takeover 或 transition break。
```

工作名暂定：

```text
AnchorFlow
或
CAPF: Causal Anchor-Transport Phase Field
```

### 16.3 相关工作定位

已调研到的相邻工作：

| 工作 | 它做什么 | 我们避开的同质化 | 我们可接住的空白 |
|---|---|---|---|
| EDIS: Diagnosing LLM Reasoning via Entropy Dynamics | token entropy 的 burst/peak-valley/variance，trajectory-level reasoning quality | 不复刻 entropy spike score | 做 first-error / boundary-level hidden geometry break，尤其 confident-wrong |
| Reasoning Fails Where Step Flow Breaks / StepFlow | attention-gradient step saliency，发现 Shallow Lock-in / Deep Decay，并 test-time repair | 不直接照搬 step-saliency/OEB/SMI | 用 prompt anchors + hidden transport 解释“哪类约束断流”，并做局部 repair |
| Lookback Lens | attention lookback ratio 检测上下文幻觉 | 不只用 attention ratio 当分类器 | 把 attention lookback 作为 anchor transport 的一个观测通道 |
| ProcessBench / PRM | 监督式 step first-error 或 step reward | 不训练普通 PRM 复述标签 | 用 hidden/attention/logit 在线信号定位首错，PRM 只作 evaluator/verifier |
| Thought-ICS | 强制 thought boundary 后 self-localize + backtrack resample | 不依赖人工 step prompt | 做 boundary-free latent phase discovery，再局部 replay |
| MTI / CCD | 高熵/低置信 token 上做 test-time intervention | 不把高熵 token 当唯一触发器 | 设计 dual trigger：uncertainty trigger + anchor-flow break trigger |
| PRISM | semantic flow + hidden latent regime 的诊断 | 不停在事后诊断 | 加在线报警和局部干预闭环 |

这条线的差异化：从“标量异常检测”转成“prompt-relative structured field + boundary-free discovery + targeted repair”。

### 16.4 数学对象

从 prompt 中抽取 anchor 集合：

```text
A = {a_1, ..., a_K}
```

anchor 类型：

```text
entity/value anchor       数字、实体、变量
constraint anchor         条件、限制、否定词
operator anchor           比较、加减乘除、计数、比例、递推关系
target anchor             最终问题目标
commitment anchor         前面已生成并被认为稳定的中间结论
```

对生成到 token/window `t` 的 hidden states：

```text
X_t^l = [h_{t-w}^l, ..., h_t^l]
```

构造 hidden 到 anchor 的传输矩阵：

```text
C_ij^l(t) = 1 - cos(W_h h_i^l, W_a a_j^l)
Pi_t^l = Sinkhorn(C_t^l, epsilon)
z_t^l = colsum(Pi_t^l) in Delta^K
```

解释：

```text
Pi_t^l    当前 token/window 的表示质量如何分配到题设 anchors
z_t^l     当前推理状态在 anchor simplex 上的位置
G_t       anchor-anchor co-activation graph
C_t       anchor complex / topology，表示哪些约束被共同激活
```

已有标量变成投影：

```text
spread              hidden cloud 的局部体积投影
anchor_loss          K=1 的退化 anchor transport
uncertainty/EDIS     logits 侧的不稳定投影
transition_surprise  z_t / G_t 动力学残差的粗近似
CUSUM                phase break 的报警器，而不是机制本身
```

### 16.5 算法模块

**A. Prompt Anchor Parser**

MVP 先用规则 + LLM parser：

```text
输入：question / prompt
输出：anchor spans, anchor roles, anchor text, token indices
```

后续可训练轻量 tagger。必须保存 anchor 到 prompt token span 的映射，方便 attention lookback 和 intervention。

**B. Anchor Transport Encoder**

输入：

```text
per-token hidden states
prompt anchor hidden states
optional attention maps
logits / entropy traces
```

输出：

```text
Pi_t, z_t, anchor coverage, anchor entropy,
constraint detachment, wrong-anchor takeover,
anchor transition residual, topology break score
```

MVP 不必先上神经网络，可先用 Sinkhorn/softmax cosine transport + 手工 phase metrics。

**C. Boundary-Free Phase Discovery**

不用强制输出 Step 1/2/3，在自然 token 流上做 change point：

```text
phase_jump_t =
  ||z_t - z_{t-1}||
  + graph_distance(G_t, G_{t-1})
  + topo_break(C_t, C_{t-1})
  + hidden_transition_residual_t
```

已有 `step_token_ranges` 只作为 evaluation，不用于发现边界。

**D. Dual-Trigger Online Detector**

两类触发器：

```text
uncertainty trigger:
  entropy/EDIS/logit evidence 发现局部不确定

anchor-flow trigger:
  anchor coverage 下降、constraint detachment、transition break、
  confident wrong = anchor-flow break with low entropy
```

检测输出不是单一 risk，而是状态：

```text
healthy_exploration
grounded_compression
unanchored_drift
confident_wrong
transition_break
recovery
```

**E. Targeted Local Repair**

不要整条重跑，做局部干预：

```text
prompt repair:
  在当前边界插入短 re-anchor prompt，重申缺失约束/目标

micro-replay:
  从最近 clean latent boundary 生成 2-4 个短 continuation，只替换局部 step

attention repair:
  提高当前 token 对关键 anchor span 的 lookback / KV contribution

hidden repair:
  将当前 hidden 朝缺失 anchor subspace 做小幅 steering

verifier repair:
  只验证当前 latent segment，而不是整条 CoT
```

核心评价不只是 AUROC，而是：

```text
wrong-to-right flip rate
correct-to-wrong damage rate
token overhead
repair scope length
early warning delay
```

### 16.6 实验设计

**实验 1：AnchorFlow 是否超过现有标量**

对比：

```text
spread/resultant
anchor_loss cosine
uncertainty / EDIS
transition_surprise
scalar fusion / sequence_state
hypergraph HGN
AnchorFlow metrics
```

指标：

```text
step first-error AUROC/AUPR
within-chain first-error top1
residualized localization after logN/pos
high-spread subset AUROC
confident-wrong subset AUROC
cross-dataset transfer
```

**实验 2：Entropy spike vs Geometry break 对齐**

目标：证明不是 EDIS 复刻。

```text
在错误链上对齐 gold first-error step/token boundary；
比较 EDIS spike、entropy mean、anchor-flow break、hidden transition residual 谁更早/更准；
重点筛 confident-wrong: low entropy but wrong。
```

**实验 3：Boundary-free step discovery**

不用 step label 切分，只用 token trajectory 的 phase break：

```text
boundary F1 / tolerance-hit
phase purity
与人工 step_token_ranges 的 mutual information
first-error token interval overlap
```

**实验 4：Counterfactual step injection**

从正确 CoT 中注入微错误：

```text
number swap
operator swap
constraint deletion
wrong intermediate value
```

看 anchor-flow break 是否锁定注入点，避免只做自然错误相关性。

**实验 5：局部干预闭环**

在 detector 标记的位置做 repair：

```text
random repair
highest entropy repair
highest anchor_loss repair
highest CUSUM repair
StepFlow-style saliency repair
AnchorFlow repair
```

报告：

```text
final accuracy gain
wrong-to-right flip rate
correct-to-wrong damage
tokens per solved improvement
repair locality
```

### 16.7 消融

必要消融：

```text
full AnchorFlow
no multi-anchor: 单 qvec cosine
no transport: independent anchor cosines
no topology: 只用 z_t，不用 G_t/C_t
no attention lookback
no logits/EDIS
no transition dynamics
no boundary-free discovery: 使用人工 step boundary
no causal repair: 只检测不干预
random anchors
wrong anchors / shuffled prompt anchors
```

最关键的判据：

```text
如果 no topology 后 early-warning 下降，说明不是 anchor_loss 的复杂包装；
如果 no causal repair 后 detection 还行但 intervention 失败，说明因果干预是独立贡献；
如果 random anchors 接近 full，则 anchor parser/transport 没有机制意义，必须重做。
```

### 16.8 最小可行版本

MVP 不需要一开始实现所有复杂模块：

```text
1. anchor parser:
   规则/LLM 抽取 numbers/entities/goal/conditions

2. transport:
   用已有 hidden shards + anchor span hidden 计算 cosine/Sinkhorn transport

3. metrics:
   anchor coverage, anchor entropy, target mass, constraint detachment,
   transition residual, phase break

4. validation:
   接入 mainline_validation_suite，比较 anchor_uncertainty / sequence_state / AnchorFlow

5. intervention:
   先做 prompt repair 和 micro-replay，不做 hidden/attention steering
```

MVP 成功条件：

```text
step AUROC 相对 anchor_uncertainty 或 sequence_state 稳定 +0.03；
confident-wrong subset 有明显增量；
boundary-free phase break 能恢复 step boundary；
prompt repair 在 flagged wrong traces 上显著高于 random/high-entropy repair。
```

如果检测增量不足但干预有效，也可以把论文定位成：

```text
not a better score, but a better intervention trigger.
```
