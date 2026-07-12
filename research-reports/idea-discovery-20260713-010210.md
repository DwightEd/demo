# 方法重设计报告：Evidence-Coupled Geometry Hazard

生成时间：2026-07-13 01:02（Asia/Shanghai）
运行 ID：`demo-method-redesign-20260713`
阶段：idea-discovery

## 1. 结论先行

当前项目的问题不在于缺少更多几何指标，而在于三个层面尚未闭合：

1. **数据对象不忠实**：已有多样本抽取在生成时使用 chat template / few-shot prompt，却在隐藏态抽取时换成另一份手写 prompt；因此几何量不是原始生成轨迹的几何量。
2. **时间轴不统一**：step-level cloud 与 token-level uncertainty 曾被按共同最短长度直接拼接，第 \(t\) 个 token 与第 \(t\) 个 step 被错误视为同一事件。
3. **研究对象过于标量化**：静态 spread、entropy、single-qvec cosine、scalar HMM 都难以区分题目难度、长度、低置信正确与高置信错误，也没有形成可验证的在线干预闭环。

因此，下一版不应继续“加一个分数”，而应把研究对象改为一条可因果审计的系统：

```text
exact generation trace
  -> semantic prompt-evidence anchors
  -> compact causal lookback + anchor-conditioned Gram geometry
  -> boundary-free event discovery
  -> first-error survival hazard
  -> compute-matched micro-replay / repath / abstention
```

本报告暂用描述性名称 **Evidence-Coupled Geometry Hazard（ECGH）**。名称不是论文贡献；贡献必须由后述四个可证伪主张共同支撑。

## 2. 现有方案的关键问题

### 2.1 旧 spectral pipeline 已经失效

根 README 和多个 shell 入口仍引用已不存在的分析、绘图与 trajectory 脚本。当前仓库是若干实验岛，而不是从抽取到干预的单一可复现入口。旧文档中的“paper-ready”结论也早于后续同题多样本负结果，必须降级为历史结果。

### 2.2 早期高分受到问题难度与长度混杂

跨问题 AUROC 较高，并不能说明模型检测到了局部推理错误。同题多样本控制后，大部分信号下降到约 \(0.50\)–\(0.60\)，目前最稳的静态基线也只在中等水平。这说明旧信号很可能主要识别：

- 哪些问题更难；
- 哪些回答更长、step cloud 更大；
- 哪些格式与数据子集不同；
- 哪些链整体更分散，而不是错误首次发生的位置。

### 2.3 `anchor_uncertainty` 不是语义约束流

单个 `qvec` 与 response hidden 的 cosine 只给出全局方向相似度，不能回答“哪一条数字、实体、约束或目标被遗忘”。当前 AnchorFlow fallback 又把 qvec 人工分区并加文本抖动；它既不是真实 prompt-span hidden，也不构成有效的 shuffled-anchor 对照。

### 2.4 二阶矩值得保留，但只能做条件专家

对窗口表示 \(U_t\in\mathbb R^{n_t\times d}\)，二阶矩状态

\[
G_t=\frac{1}{n_t}U_tU_t^\top
\]

能表达平均向量和单一 spread 看不到的多方向竞争、谱尾增长和子空间旋转。但已有 scalar EM/HMM 接近随机，且 raw cloud volume 强烈依赖 \(n_t\)。所以不能把 \(G_t\) 再压成一个全局标量后宣称“潜状态发现”；必须在 prompt anchor 条件化和长度控制后，检验其对强基线的增量。

### 2.5 旧 attention hypergraph 不能直接复用

内部原型保存所有层、头的完整 \(S\times S\) attention，复杂度为

\[
O(LHS^2),
\]

长上下文时不可落地；固定阈值又不能跨长度、层和 head 比较。它还存在 token 标签错位、随机样本切分和图归一化问题。可复用的是 relation schema 与 HyperCHARM 上界模型，不是原数据流程。

## 3. 新方法对象

### 3.1 忠实的 `ReasoningTrace`

每条轨迹必须保存同一份生成上下文：

\[
\mathcal T=(p, x_{1:T}, I_p, I_x, O, H, A, M),
\]

其中 \(p\) 是完整渲染后的 prompt，\(I_p,I_x\) 是生成时实际 token IDs，\(O\) 是 tokenizer offsets，\(H\) 是选定层 hidden，\(A\) 是紧凑 attention 摘要，\(M\) 是模型 revision、tokenizer revision、采样参数和 seed。所有序列字段都必须声明时间轴：`prompt_token`、`response_token`、`window` 或 `step`。

若不能复用完全相同的输入 token，就应 fail fast，而不是悄悄重建 prompt。

### 3.2 语义 prompt anchors

从 prompt 文本解析四类锚点：数字/值、实体、约束、目标。对 anchor \(a_k\) 的字符区间，通过 offsets 得到 token span \(S_k\)，再用真实 prompt hidden 构造

\[
\mathbf a_k=\frac{1}{|S_k|}\sum_{i\in S_k}\mathbf h_i^{(\ell)}.
\]

没有 offsets 或 prompt hidden 时，只允许显式标记的 `qvec_fallback`，其结果只能用于 plumbing，不能进入语义主结果。

### 3.3 紧凑 evidence lookback

不保存完整 attention tensor；对每个 response token/window，仅保存与 prompt 证据有关的摘要：

\[
m_t^{(\ell,h)}=\sum_{j\in\text{prompt}}\alpha_{t,j}^{(\ell,h)},
\]

以及 prompt mass、归一化 entropy、top-\(k\) prompt token 及质量、head agreement、跨层 persistence、evidence-set churn。若每个 query 仅保留 \(k\) 个 prompt evidence token，存储复杂度由 \(O(LHS^2)\) 降到近似

\[
O(LHSk).
\]

### 3.4 anchor-conditioned Gram geometry

令当前因果窗口的 hidden cloud 为 \(H_t\)，活跃 anchor bank 为 \(A_t\)。用 SVD/QR 得到 anchor 子空间投影 \(P_{A_t}\)，定义残差云

\[
R_t=H_t(I-P_{A_t}),
\qquad
G_t=\frac{1}{n_t}R_tR_t^\top.
\]

从 \(G_t\) 提取：残差能量、effective rank、谱尾质量、regularized log-det、主子空间漂移。这些量描述“当前推理中无法由题设证据解释的表示体积”，而不是 raw hidden spread。

二阶矩专家只在以下含混区域启用：

\[
\mathcal R=\{t:\text{spread}_t\text{ high},\;\text{entropy}_t\text{ low}\},
\]

即模型表示分裂但输出已经过度承诺的 confident-error 候选区。

### 3.5 Detection–Expression coupling gap

《The Phenomenology of Hallucinations》提出内部检测信号可能位于低输出敏感方向。这里不把 coupling gap 当作独立“万能分数”，而把它作为状态解释变量。若 \(J_t\) 是局部 logit/readout Jacobian 的低秩近似，定义可见性

\[
v_t=\frac{\|J_t r_t\|_2}{\|r_t\|_2+\varepsilon},
\]

其中 \(r_t\) 是 anchor-residual rupture direction。高残差、低 evidence lookback、低 \(v_t\)、低 entropy 的组合，对应“内部偏离已出现，但输出仍强制承诺”。

### 3.6 无人工 step 边界的因果事件发现

每个时间点只使用过去窗口计算稳健标准化变化：

\[
c_t=
w_1 z(\Delta \text{lookback}_t)+
w_2 z(\Delta \text{transport}_t)+
w_3 z(\Delta \text{Gram}_t)+
w_4 z(\Delta \text{visibility}_t).
\]

基准中心与尺度只能由 \(1{:}t-1\) 的 median/MAD 估计；不使用整条链统计，也不把 gold step boundary 用于建图。局部峰值形成 discovered segment，人工 step 仅用于评价“事件是否落在首错附近”。

### 3.7 首错 survival hazard

不再把错误 step 后的所有 token 当作正样本。令 \(T\) 为首次错误事件，学习离散 hazard：

\[
q_t=P(T=t\mid T\ge t,\mathcal H_{\le t}),
\qquad
S_t=\prod_{s\le t}(1-q_s).
\]

正确链是右删失样本；错误发生后的位置不进入风险集。训练损失为

\[
\mathcal L_{	ext{hazard}}
=-sum_i\left[
\sum_{t<T_i}\log(1-q_{i,t})+
\mathbf 1[\delta_i=1]\log q_{i,T_i}
\right].
\]

这使模型输出天然对应“现在是否首次越界”，适合在线触发和检测延迟评价。

### 3.8 干预必须评估净收益

在校准后的 hazard 超阈值时，从最后安全事件边界做 compute-matched micro-replay，并插入最相关、但最近 lookback 已下降的 prompt anchors。可选动作是 `replay`、`repath`、`abstain`。净效用写为

\[
U=Delta\text{accuracy}
-\lambda_c\Delta\text{tokens}
-\lambda_d P(\text{correct}\to\text{wrong})
-\lambda_a P(\text{false alarm}).
\]

所有被触发的正确链都必须进入 damage 统计；不能只在已知错误链上做 repair。

## 4. 四个论文级可证伪主张

### Claim A：语义证据耦合优于题目难度代理

在同一问题的多次采样内，真实 prompt-span anchor + compact lookback 对首错 hazard 有增量，并优于 qvec、随机 anchor、shuffled span、长度、entropy、spread。

### Claim B：条件二阶几何识别“过度承诺式错误”

在 high-spread / low-entropy 子集，anchor-residual Gram 动态对 base hazard 有稳定 OOF 增量；若仅跨问题有效，则该主张失败。

### Claim C：boundary-free 事件能提前或贴近首错定位

不读取人工 step boundary 的情况下，变化事件在固定正确链 FPR 下具有可接受 recall 和 delay，并显著优于随机位置、entropy peak、链尾启发式与 supervised step-only upper bound 的可部署部分。

### Claim D：触发器带来正的干预净收益

在等 token budget 下，ECGH-triggered micro-replay 相比 random、entropy、max-spread、self-consistency 触发获得更高 wrong-to-right，同时控制 correct-to-wrong 与 abstention cost。

任何一个模块单独提高 AUROC 都不足以支持完整论文主张。

## 5. 与近期工作的差异边界

- **Lookback Lens** 已证明 context-vs-generation attention ratio 可检测上下文幻觉；本方法必须证明多语义 anchor、条件几何、首错 hazard 与干预的增量，不能把 prompt mass 本身当创新。
- **Hidden-State Transport Geometry（2026）** 已研究 step-level hidden transport 与 first-error localization；本方法的必要差异是 exact-generation trace、prompt-evidence anchors、boundary-free causal events、right-censored hazard 与闭环 intervention。
- **GeoFaith** 已将潜在几何、entropy dynamics、检测与 RL 组成闭环；本方法不能声称“几何 + 干预”本身新颖，重点应是证据约束耦合、同题控制与首错生存建模。
- **StepFlow** 已使用 attention-gradient step saliency 和 intervention；本方法要避免依赖标注 step，且比较非梯度 compact lookback 与其上界。
- **CausalGaze** 已使用 hidden + attention causal graph 和 counterfactual intervention；因而“构图 + 因果干预”不是独立创新。我们的图/超图只作为结构化上界，核心可部署对象仍是紧凑因果状态和 hazard。
- **The Phenomenology of Hallucinations** 提供 detection–expression gap 假说；本项目应以同题生成、首错定位和 held-out coverage–accuracy 实证检验，不复用其跨数据集类型边界作为标签。

## 6. 备选方向与淘汰理由

| 方向 | 价值 | 当前结论 |
|---|---|---|
| 单独 readout-coupling gap | 机制解释强 | 与 Phenomenology 高重叠，且单标量不足；保留为辅助状态 |
| Matrix-HMM / Wishart state model | 可表达矩阵状态 | 当前非参数增量尚未成立；先过 gate，再决定是否上潜状态模型 |
| 旧 attention hypergraph 扩容 | 结构表达强 | 存储不可承受、数据错位、创新弱；只保留 compact lookback 和 HyperCHARM 上界 |
| 全局 AnchorFlow score | 工程简单 | 容易退化为 qvec/长度代理；淘汰单分数叙事 |
| ECGH 完整系统 | 能同时验证机制、定位与干预 | 选定；按最小可证伪顺序实现 |

## 7. 流程可执行性判断

| 流程 | 当前状态 | 本轮目标 |
|---|---|---|
| skills / research state | 已走通 | 80/80 skills 与共享工具校验通过 |
| 本地数学与基础测试 | 部分走通 | 修复负特征值、fold 泄漏并增加 integration assertions |
| GPU exact trace 抽取 | 入口存在但结果不忠实 | 复用生成 token IDs，保存 prompt offsets/span hidden |
| compact lookback | 未实现 | 新增纯 NumPy 摘要与 GPU hook 接口 |
| semantic anchor | 仅 fallback scaffold | 接入真实 prompt-span hidden，fallback 明示 |
| conditional Gram | 只有研究计划 | 实现透明非参数版本和 synthetic gate |
| boundary-free event | 未实现 | 实现仅用过去的 robust change detector |
| first-error hazard | 未实现 | 实现风险集、右删失、group OOF 评价 |
| hypergraph upper bound | 条件可跑 | 沿用 `demo` 的强化版，不复制旧内部仓库 |
| intervention | 非流式原型 | 先实现动作构造与离线 policy value，再做真实 streaming |

## 8. Go / no-go 规则

先完成 P0 数据与时间轴修复，然后按顺序过门：

1. **Trace gate**：随机抽样逐 token 比对生成 IDs 与抽取 IDs，必须 100% 一致；所有跨时间轴融合必须有显式映射。
2. **Anchor gate**：真实 anchor 相比 qvec、random、span-shuffle 在 same-problem grouped OOF 中有稳定增量；否则停止语义主张。
3. **Gram gate**：在强基线之后 \(\Delta\text{AUROC}\) 或固定 FPR recall 有 bootstrap CI 下界大于 0，且不是链尾/长度效应；否则二阶矩降级为分析工具。
4. **Localization gate**：在固定 5% 正确链 FPR 下报告 event recall、median signed delay 和 false-alarm position；若只在链尾触发，则淘汰 rupture claim。
5. **Intervention gate**：等 compute 下净修复率为正，且 correct-to-wrong damage 不高于预设上限；否则只保留检测论文，不声称闭环成功。

## 9. 当前风险

- 本机没有项目 GPU 依赖，真实模型抽取只能做静态审查和远端命令交付；本地只能确认 CPU/synthetic 路径。
- prompt attention 是否比 hidden-anchor transport提供额外信息仍是实证问题。
- ProcessBench 的 step labels 不是 token 级真实因果时刻，delay 应报告区间和敏感性分析。
- “boundary-free”不能只靠无监督峰值命名；必须展示相对人工 step、随机边界和链尾启发式的比较。
- 若真实 attention hook 增加过多显存，应只保留 prompt top-\(k\) rows / online summaries，绝不恢复完整 \(S^2\) dump。

## 10. 决策

进入 experiment-bridge。实现优先级为：

```text
P0 exact trace + explicit time axes
 -> P1 semantic anchor / compact lookback / conditional Gram / causal event
 -> P1 first-error hazard + grouped evaluation
 -> P2 streaming hook and compute-matched intervention
```

二阶矩、超图与 readout gap 都保留，但分别是条件专家、结构上界和解释变量；它们不再各自竞争成为新的单标量主方法。

## 参考工作

- [The Phenomenology of Hallucinations](https://arxiv.org/abs/2603.13911)
- [GeoFaith](https://arxiv.org/abs/2605.26893)
- [StepFlow](https://arxiv.org/abs/2604.06695)
- [Lookback Lens](https://aclanthology.org/2024.emnlp-main.84/)
- [Where Does Reasoning Break? Step-Level Hallucination Detection via Hidden-State Transport Geometry](https://arxiv.org/abs/2605.13772)
- [The Spectral Geometry of Thought](https://arxiv.org/abs/2604.15350)
- [The Geometry of Reason](https://arxiv.org/abs/2601.00791)
- [LAFaCT](https://aclanthology.org/2026.acl-long.312/)
