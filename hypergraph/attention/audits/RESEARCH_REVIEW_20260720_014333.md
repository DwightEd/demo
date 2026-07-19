# Attention 超图方法评审

**日期**：2026-07-20
**评审者**：GPT-5.6-Sol ultra（fresh same-family，provisional）
**结论**：CONDITIONAL GO——可推送实现并启动 4-sample GPU pilot；真实实验与归因控制完成前不可作效果/机制结论。

## 1. 与原版的关系

当前忠实默认保留了原仓库 `processed_hypergraph.py` 与 `train_hypergraph.py` 的核心定义：

- 完整 prompt+response 轴上每个 token 是一个节点；
- 节点特征为所有 layer/head 的 self-attention diagonal；
- 对每个 response receiver、layer、head，用严格 `attention > tau` 选择历史 source，并加入 receiver center；
- edge attributes 为 attention mean、max 与 normalized flattened head id；
- uniform incidence、symmetric node→hyperedge→node HyperCHARM、residual 和 token logit。

独立随机构图对拍为 200/200；默认数值模型对拍最大误差约 `2.1e-7`。当前代码没有再用
hidden-kNN 冒充原 attention 超图，也没有额外“主拓扑”。

## 2. 为什么旧改法效果可能不好

此前把 response hidden similarity/kNN 当主拓扑，实际同时改变了 node axis、关系来源、边选择、
传播和监督粒度；它不再是原方法的相邻变体，因此“别人同样做 QA hallucination detection”并不能
保证迁移有效。对本任务，hidden similarity 更像内容/状态相似性，未直接表达一个 query 把
attention mass 路由到哪些历史 token；它可以做节点内容，但不能替代 attention 拓扑后仍称原方法。

即使恢复 attention，固定阈值仍受长度与 receiver 位置混杂：历史 key 增多后 attention mass 被重新
分配，过阈值成员数、零边率和度数会系统变化；step 越长，取最大 step score 时也有更多“命中机会”。
所以模型可能学到 length/position hazard，而不是幻觉机制。

## 3. 机制与声明边界

attention 适合作为 routing proxy，不等于因果解释。当前 target alignment 是
`same_index_post_emission`：row `i` 已读入 token `i`，所以它是离线/post-token diagnostic，
不是 token 生成前预测。receiver-only 只阻止完整图中的未来消息回流，不能把 same-index 变成
pre-emission。observer teacher forcing 只能说明观察模型如何表征给定文本，不能反推原生成器机制。

hidden 只作为显式 node-feature innovation。若 `activation_only` 或 combined 特征优于 diagonal，
但没有固定维瓶颈/随机投影容量对照，只能称“输入内容与模型容量的联合差异”。

## 4. 当前框架内可调整的组件

| 组件 | 忠实默认 | 已实现的独立变体 | 必要控制 |
|---|---|---|---|
| topology | threshold attention-row | top-k、cumulative mass、layer/head subset | length/position density curve |
| source | all past | prompt-only、response-only | 相同 receiver/feature |
| node feature | attention diagonal | activation-only、combined | 固定维/容量匹配 |
| incidence | uniform | raw/normalized attention | attention sink 与 mass |
| propagation | symmetric | receiver-only | offline 与 prefix 任务分开 |
| message | source-only | receiver/source interaction | 参数量匹配 |
| operator | hyperedge set | directed pairwise | degree/lag-matched rewiring |
| preprocessing | per-graph z-score | none | prefix 任务禁用未来统计 |
| objective | exact token | first-error step、response | 不跨粒度伪造标签 |
| readout | token logit | mean/logmeanexp | length-only baseline |

## 5. 最小有效实验序列

1. A0-C：receiver-only + preprocessing none + `model-layers=0`。
2. A1-C：只加入 attention hypergraph message passing。
3. A7-P：同 support/属性/参数的 directed pairwise 对照。
4. degree/lag-matched rewiring：判断收益是否来自真实 attention relation。
5. length-only/relative-position classifier 与预注册长度分层。
6. A2/A3/A4/A5/A6 每次只改变一个组件；hidden 变体必须容量匹配。
7. 多 seed，并对同一 problem 的模型差值做 paired bootstrap 95% CI。

## 6. Claims Matrix

| 观察结果 | 允许声明 |
|---|---|
| A1 不超过 A0 | 当前 node feature 已解释结果；无图增益 |
| A1>A0，但不超过 rewiring | 图结构/正则可能有用，不能归因真实 attention relation |
| A1>rewiring，但不超过 pairwise | attention relation 有诊断价值，未证明高阶超边增益 |
| A1 同时超过 A0、rewiring、pairwise，CI 下界过预注册 margin | 可称 attention-row hypergraph 有 held-out 增量 |
| combined hidden 更好但未容量匹配 | 只能称 feature+capacity 联合改善 |
| next-token aligned/prefix intervention 也成立 | 才可讨论更接近生成机制的预测；仍不能自动称因果 |

当前允许：忠实实现原 attention-row HyperCHARM；提供可审计的 post-emission/teacher-forced
diagnostic pipeline。当前不允许：超图提高检测、attention 揭示致幻因果机制、在线首错预测、
hidden 优于 attention。

`review_independence=same-family`
`acceptance_status=provisional`
