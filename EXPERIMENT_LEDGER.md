# 推理隐状态研究实验总账

更新日期：2026-07-14

## 1. 文档目的

本文整理当前仓库中能够辨认的全部主要实验族、远端运行记录和对应结论。它不是按脚本数量罗列，而是按研究假设合并重复实现，并区分：

- **已记录结果**：仓库中有结果文件或正式记录；
- **终端记录**：已在远端运行并由终端输出确认，但原始产物尚未全部进入仓库；
- **仅实现**：代码和方法说明存在，但没有可用于下结论的真实数据结果；
- **历史结论**：在较弱协议下成立，但已被同题控制、长度位置残差化或冻结跨数据集验证收紧。

证据等级：

| 等级 | 含义 |
|---|---|
| A | 同题多采样或冻结跨数据集验证，含分组交叉验证、bootstrap/null、长度位置控制中的至少两项 |
| B | 单数据集受控实验，协议基本可信，但尚未跨数据集或同题复现 |
| C | 探索性、监督上界、小样本、位置敏感或仅有终端摘要 |
| U | 只有实现或数据协议，尚无决定性真实结果 |

当前结论以本总账为准。旧的进展报告保留研究过程价值，但其中“长度已被完全排除”“几何有大幅独立增量”等表述已被更严格实验部分取代。

## 2. 一页结论

1. **hidden state 确实包含可学习的正确性信息。** 泄漏安全的监督 probe、PCA probe 和结构化 HGN 通常达到约 $0.71$ 到 $0.76$ AUROC。但这只证明表征可分，不证明存在某个低维流形、稳定 basin 或可解释几何机制。
2. **最稳定的无监督几何现象是方向一致性下降。** 对一步内 token 方向，错误样本通常有更小 resultant、更大 spread。原始跨问题 AUROC 约 $0.70$ 到 $0.77$，同题多采样约 $0.63$ 到 $0.66$。
3. **spread 不是干净、充分或普适的错误变量。** 冻结跨数据集后，raw spread 的 macro/min AUROC 为 $0.726/0.702$，长度位置残差化后降为 $0.600/0.521$。它是弱到中等的生理标记，而不是独立错误机制。
4. **“错误前发生动态发散或相变”没有得到支持。** jump、slope、curvature、HSMM、path kernel、预测动力学和多数 CUSUM 变体均未稳定超过静态 level；报警通常偏晚或位于链尾。
5. **复杂几何没有自动带来信息。** HCR-Holo、layer-time holonomy/Wilson loop、简单流形密度、VAE 重构几何、直接 Gram/谱尾和 token-stream 谱指标都未显示可靠的独立增量。
6. **残差谱秩只支持一个局部子类型。** spread 与高 residual effective rank 同时出现时，得到 OR $7.23$、FPR $0.052$、recall $0.283$ 的高风险碎片化子类；但联合全局 AUROC 低于 spread，不能称为更强检测器。
7. **logit uncertainty/entropy 不是“最后剩下的唯一东西”，但它是不可删除的强通道。** 在 response 级和较难任务上，熵轨迹通常比动态几何稳定；在局部 first-error 定位上，静态 spread 有时更强。二者作用粒度不同。
8. **当前最稳的融合基线不是纯 anchor。** `anchor_uncertainty` 实际是 spread、anchor loss、uncertainty、step length、relative position 的 OOF 线性模型；其 $0.811/0.781/0.809$ 不能解释成“问题向量相似度”的机制效果。
9. **目前最值得保留的新信号是条件联合 innovation。** 正确链上拟合 spread、anchor loss、uncertainty 的转移律后，joint surprise 的长度位置残差化 macro/min AUROC 为 $0.647/0.621$。但它尚未证明能提升完整融合基线。
10. **低维正确流形、NTS 法向逃逸、真实 premise-source flow 和 attention/ICR 因果链尚未被否证。** 它们缺少决定性运行或必要数据，必须标为未裁决，不能与已失败的 toy 几何混为一谈。

## 3. 数据与评估协议

### 3.1 数据资产

| 设置 | 数据 | 规模与用途 | 关键限制 |
|---|---|---|---|
| ProcessBench GSM8K | `data/features/full_gsm8k.npz` | 395 条链；全部约 2072 步；first-error 任务去除 post-error 后 1560 行、205 个正例 | 一题通常只有一条回答；首错位置、step 长度和难度混杂强 |
| ProcessBench MATH | `data/features/full_math.npz` | 982 条链、约 4339 步；跨数据集复现 | 同上 |
| ProcessBench OmniMath | `data/features/full_omnimath.npz` | 998 条链、约 4788 步；较难任务复现 | 同上 |
| 历史 OlympiadBench | `full_olympiadbench` 系列 | 997 条链、约 5816 步；旧 SMCD 对照 | 不在当前冻结三数据集主线中 |
| GSM8K 5-shot 多采样 | `data/gsm8k_v2_5shot.npz` | 原始 2646 条；`answer_format_ok` 后 2035 条、1756 正确、279 错误、94 个 contrastive problems | 旧文件无精确 token ID 对齐 |
| GSM8K custom 多采样 | `data/gsm8k_v2_custom.npz` | 原始 3600 条；有效 3452 条、2920 正确、532 错误、300 题、147 个 contrastive problems | 旧文件只有 `legacy_cloud_order`，无法做 lexical nuisance 确证 |

标签约定：`is_correct=1` 表示正确；`gold_error_step=-1` 表示整条 ProcessBench 链正确。2026-07-03 前构建的部分 trajectory cache 曾有标签翻转和 token range 切片错误，不应继续使用。

### 3.2 五种不能混用的评估目标

| 目标 | 回答的问题 | 主要混杂 |
|---|---|---|
| pooled first-error AUROC | 所有候选步骤中首错是否得分更高 | 问题难度、位置、长度 |
| within-chain rank | 同一错误链内能否找到首错步 | 首错天然较晚；`pos` 可达到 1.0 |
| response AUROC/AUPRC | 整条回答最终是否错误 | 长度、题目难度、格式失败 |
| same-problem paired AUROC | 同一道题的错误采样是否高于正确采样 | 最能控制题目难度，但没有精确首错标签 |
| pre-error future | 当前正确前缀能否预言未来出错 | 标签结构和剩余链长，难度最高 |

因此，跨问题 $0.80+$ 不能替代同题 $0.60+$，within-chain $0.90$ 也不能自动解释成机制定位。

## 4. 核心变量

一步 $t$、层 $\ell$ 内有 token hidden states $h_{t,i}^{(\ell)}$。令

\[
u_{t,i}^{(\ell)}=\frac{h_{t,i}^{(\ell)}}{\lVert h_{t,i}^{(\ell)}\rVert_2},
\qquad
R_{t,\ell}=\left\lVert
\frac{\sum_i w_i u_{t,i}^{(\ell)}}{\sum_i w_i}
\right\rVert_2,
\qquad
\operatorname{spread}_{t,\ell}=1-R_{t,\ell}.
\]

这里 $R$ 是 mean resultant length。它只描述方向集中度，不说明这些方向是否指向正确 premise，也不等于推理流形本身。

令 $\widetilde w_i=w_i/\sum_j w_j$，则残差散射满足

\[
C=\sum_i \widetilde w_i u_i u_i^\top-\mu\mu^\top,
\qquad
1=\lVert\mu\rVert_2^2+\operatorname{tr}(C).
\]

因此，`residual_energy` 与 spread 高度同源；真正可能新增的信息只能来自 $C$ 的谱形状，而不是其总能量。

## 5. 全部实验族总表

| ID | 实验族 | 最关键结果 | 当前判决 | 等级/来源 |
|---|---|---|---|---|
| E01 | 监督 probe / PCA / sparse probe | honest within AUROC 约 0.71；PCA25 最高约 0.735；mid ensemble 0.764 | hidden state 可分，但不是无监督机制 | B/记录 |
| E02 | 难度分解与随机切分审计 | random split 约 0.90，group/within 约 0.71；难度方向移除后变化很小或下降 | 证明早期高 AUC 有严重难度膨胀 | B/记录 |
| E03 | 单步 resultant/spread | GSM8K L14 raw 0.772；MATH/Omni/Olympiad 约 0.703 | 保留为弱到中等静态标记 | A/记录 |
| E04 | norm 与单位球消融 | norm 0.727；unit sphere 0.776；增量 +0.049 ([0.024,0.080]) | 方向信息真实，但仍可能含长度难度 | B/记录 |
| E05 | 历史多层 κ + EDIS 分类器 | OURS-LDA 0.840/0.797/0.802/0.803；旧 strict 增量 +0.110/+0.059/+0.032/+0.046 | 历史上界；被后续严格协议收紧，不作当前主张 | C/历史 |
| E06 | response 几何与熵 | min-κ 0.740/0.698/0.658/0.640；EDIS 0.736/0.731/0.812/0.739；GDIS 0.555–0.619 | 熵在困难/response 级更稳，几何偏局部 | B/记录 |
| E07 | κ drop、jump、shape、cross-step | 相对 drop 0.715/0.678/0.667/0.668，始终弱于绝对 κ；增量约 0–0.003 | 退役为主检测器 | B/记录 |
| E08 | 同题多采样静态 spread | 5-shot 0.634；custom late 0.659、max 0.639 | 最可信的 spread 证据，但效果中等 | A/记录 |
| E09 | temporal rupture / phase transition | GSM8K raw spread event 0.790，但 response z/jump/break 多为 0.51–0.57；44% 无清晰事件 | 不支持普适相变或早期 rupture | B/终端+记录 |
| E10 | residual-rank morphology | 控制长度、κ、位置后 `res_eff_rank` AUROC 0.602，差值 CI 为 ([1.983,5.528]) | 支持“碎片化”局部机制，非强检测器 | B/记录 |
| E11 | spread + residual rank 联合象限 | dual-high OR 7.23，recall 0.283，FPR 0.052；joint AUROC 0.714 < spread 0.772 | 保留为高精度错误子型 | B/记录 |
| E12 | step Gram / second moment / spectral tail | baseline 0.685；best Gram 0.660；spectral tail 0.634，增量 -0.051 ([-0.080,-0.020]) | 退役直接谱尾主张 | A/记录 |
| E13 | token-stream κ/alpha/effective rank | baseline 0.668；best group 0.670，增量 +0.002 ([-0.023,0.027])；alarm recall 0.269@FPR 0.049 | 退役为在线检测器；仅保留形态描述 | A/记录 |
| E14 | scalar HSMM / latent EM | 0.538；censor80 0.506；静态 spread 0.682 | 退役 | A/记录 |
| E15 | path kernel / functional shape | best witness 0.668、shape 约 0.615；静态 0.683 | 退役 | A/记录 |
| E16 | hypergraph token HGN | node 0.691、step 0.760、graph 0.678；top1 0.854 | 监督关系读取上界；非几何机制、未控位置 | C/记录 |
| E17 | qvec AnchorFlow fallback | 加入 baseline 仅 +0.001 至 +0.003，且 shuffled-kind 相近 | 退役 fallback；真实 prompt spans 未测试 | B/记录 |
| E18 | constraint anchor flow | baseline 0.831，baseline+anchor 0.829；`pos` within 1.000 | 无增量，单特征高分被位置污染 | C/终端 |
| E19 | regex premise ledger | custom 0.562 vs spread 0.659；5-shot 0.548 vs 0.634 | 退役 regex 语义约束 | A/记录 |
| E20 | first-error step/token delta、角度、曲率 | step 最好 residualized AUC 0.595，q=0.312；token 约 0.454–0.540，均不显著 | 明确负结果 | A/原始输出 |
| E21 | 同题静态/动态流形几何 | 静态 support residual 0.610；动态 late residual 0.593；dynamic-static -0.039 ([-0.066,-0.011]) | 静态优于动态，不支持演化优势 | A/终端 |
| E22 | debiased directional consensus | raw global 0.662；length residual 0.581；fixed-window residual 0.593；debiased-raw 0 | 信号存在，去偏估计器无增益 | A/终端 |
| E23 | predictive state geometry | ordered Mahalanobis 0.480；shuffle 0.461；fixed consensus 0.646；全部 gate FAIL | 退役当前 reduced-rank 预测动力学 | A/终端 |
| E24 | HCR-Holo | first-error HCR 约 0.52；response 最好约 0.53；step length 0.707 | 退役当前 holonomy/closure 实现 | C/终端 |
| E25 | layer-time transport/Wilson/LID | first-error transport 0.566、holonomy 0.510；response fiber rank 0.643；reliable Wilson 覆盖仅 3% | 兼容性负基线，不支持曲率主张 | B/终端 |
| E26 | CTG / transport chart | CTG 0.646 pooled、0.700 within；胜 random/permuted，但加入 controls 后下降 | 有结构但无检测增量 | C/终端 |
| E27 | reasoning-state hazard geometry | geometry 0.781/0.764；controls 0.809/0.831；controls+geometry 0.806/0.803 | 几何主要重复 controls | C/终端 |
| E28 | VAE latent separatrix | VAE uncertainty 0.48–0.61；监督 hazard 0.770、energy 0.738；centroid 0.736 | latent 可分，但性能来自监督 readout，不验证流形机制 | C/终端 |
| E29 | frozen cross-dataset transition | raw spread macro/min 0.726/0.702；residual 0.600/0.521；joint surprise residual 0.647/0.621 | 当前最严格主结果；条件 innovation 弱保留 | A/记录 |
| E30 | full sequence feature stack | anchor baseline 0.811/0.781/0.809；sequence 0.790/0.779/0.816；增量 -0.021/-0.002/+0.007 | 不再堆叠 broad sequence features | A/记录 |
| E31 | NTS normal/tangent escape | Gate 2 代码和协议存在，缺少 canonical 决定性结果 | 未裁决 | U/实现 |
| E32 | conditional tangent escape + output cotangent | 普通切空间部分可跑；缺少 exact `step_output_cotangent` | 强 gate 未测试 | U/实现 |
| E33 | prompt-control ICR / attention graph | 抽取与插件框架存在，ProcessBench 真正 ICR/attention 谱结果缺失 | 未裁决 | U/实现 |
| E34 | prefix innovation / transition tube / tube refinement | 已实现并有 selftest；没有进入统一结果记录的真实主结果 | 未裁决，不得写成成功或失败 | U/实现 |
| E35 | exact whole-chain HS/ME | 代码实现了严格 whole-chain Gram 与 prefix adaptation；未见统一真实结果 | 未裁决；step Gram 的失败不能替代 exact protocol | U/实现 |

## 6. 详细结果与解释

### 6.1 表征可分性：有信息，但不是几何发现

早期 probe 系列得到：

- honest within-problem probe：约 $0.707$ 到 $0.715$；
- PCA25：约 $0.720$ 到 $0.735$；
- mid-band ensemble：$0.753$，meta learner：$0.764$；
- random split cross-problem probe：约 $0.896$ 到 $0.898$。

最后一组远高于 group/within 结果，说明问题难度泄漏足以制造接近 $0.90$ 的表面性能。监督 probe 的有效结论仅是：

> Llama-3.1-8B 的 teacher-forced hidden state 中存在可被标签监督读取的错误相关信息。

它不能推出正确推理天然占据一个低维流形，也不能说明模型“意识到自己错了”。

### 6.2 静态方向一致性：真实但有限

ProcessBench GSM8K L14：

| 信号 | AUROC | 正常步均值 | 首错步均值 |
|---|---:|---:|---:|
| spread | 0.772 | 0.366 | 0.407 |
| resultant | 0.772 | 0.634 | 0.593 |
| transition surprise | 0.720 | 6.788 | 22.306 |
| d_spread | 0.702 | -0.010 | 0.023 |

单位球消融说明方向不只是 norm：unit-sphere 相对 norm-only 提升 $+0.049$，CI 下界为正。但更严格的同题和跨数据集结果同时说明，这种方向变化仍与长度、题目难度和步骤功能纠缠。

最诚实的表述是：

> 错误步骤平均具有更弱的 token 方向一致性；该现象可重复，但其独立判别力通常只是中等，并且随任务和控制协议显著下降。

### 6.3 历史 SMCD 主表为何不能直接沿用

旧 SMCD 多层特征 + LDA 记录了较高数字：

| 数据集 | OURS-LDA | EDIS | 旧 strict 增量 |
|---|---:|---:|---:|
| GSM8K | 0.840 | 0.719 | +0.110 |
| MATH | 0.797 | 0.717 | +0.059 |
| OmniMath | 0.802 | 0.754 | +0.032 |
| Olympiad | 0.803 | 0.753 | +0.046 |

这些数字在当时的 GroupKFold/bucket 协议下成立，但它们是多层、多统计量、监督分类器与单个 EDIS 分数的组合比较。后续发现：

- cross-problem 与同题结果差距很大；
- bucket 并未充分消除连续长度和位置结构；
- `anchor_uncertainty` 还显式含 `logN` 与 `pos`；
- 冻结 residual spread 的 worst-dataset AUROC 只有 $0.521$。

所以旧主表可作为历史检测上界，不能再作为“纯几何有巨大独立增量”的论文证据。

### 6.4 response 级：熵与几何粒度不同

四数据集 response AUROC：

| 信号 | GSM8K | MATH | OmniMath | Olympiad |
|---|---:|---:|---:|---:|
| min-κ | 0.740 | 0.698 | 0.658 | 0.640 |
| EDIS/logit entropy | 0.736 | 0.731 | 0.812 | 0.739 |
| GDIS 动态几何 | 0.555–0.619 | - | - | - |

这不是“几何完全没用、只剩熵”，而是：

- spread 更接近局部 step token-cloud 的组织程度；
- entropy 更直接读取模型输出分布的不确定性，整链和困难任务上更强；
- 对 confident-wrong，二者都可能漏检，因为错误可以既低熵又方向集中。

### 6.5 动态、相变和早期预警：总体为负

`trajectory_phase_transition_audit.py` 的 raw spread 在首错步上为 $0.790$，但这是当前 level 对首错的区分，不是动态突变的独立贡献。其 z-score、jump、break、shock 聚合到 response 后多为 $0.51$ 到 $0.57$。spread 模式中：

- stable-prefix-break：26.6%；
- gradual drift：11.0%；
- persistently unstable prefix：11.0%；
- isolated jump：7.3%；
- no clear geometry event：44.0%。

同题 temporal audit 也显示 `cloud_spread.level_late=0.659`，而 local contrast 只有约 $0.579$。这说明动态方法通常只是重新编码“某一步本身较难/较散”，没有发现普适先兆。

冻结三数据集在线报警只在 FPR 约 $0.10$ 到 $0.12$ 时召回约 $0.30$ 到 $0.37$，不具备部署质量。

### 6.6 residual rank：机制子类，不是全局提升

在控制 length、κ 和 position 后，首错步 residual effective rank 仍有 AUROC $0.602$，均值差 bootstrap CI 为 $[1.983,5.528]$。联合象限结果：

| 状态 | error rate | OR | 说明 |
|---|---:|---:|---|
| dual high spread + rank | 0.453 | 7.23 | 高精度碎片化首错子型 |
| spread only | 0.175 | 1.56 | 一般方向失配 |
| rank only | 0.124 | 0.93 | 秩本身不构成风险 |
| low-low | 0.065 | 0.28 | 低风险 |

但全局比较为：spread $0.772$、joint raw $0.754$、joint strict $0.714$。所以可以说“rank 解释了部分低 κ 的形态”，不能说“谱秩提高了检测”。

### 6.7 直接谱、Gram 和 token-stream：没有新增判别力

同题控制下：

- static baseline：$0.685$；
- best token-matrix/Gram group：$0.660$；
- spectral-tail：$0.634$；
- spectral-tail increment：(-0.051)，CI ([-0.080,-0.020])。

token-stream 分支中，length+entropy+static baseline 为 $0.668$，best stream group 为 $0.670$，增量 CI 跨零。有效秩呈现 expand-then-compress 的描述性形态，但正确和错误轨迹都出现，不能作为 correctness 机制。

这否定的是“当前 hidden-cloud 谱标量能增加检测”，不是所有谱几何。attention graph Laplacian、exact whole-chain HS/ME 与 reference-relative topology 仍是不同对象。

### 6.8 first-error 微分几何：明确负结果

已提交原始输出使用 correct-chain 学得的 nuisance residualizer，并按 chain length、relative position、event-step length 匹配。

step 级最优为 L14 `relative_delta_norm`：AUROC $0.595$、paired $d_z=0.208$、q-value $0.312$。token 级所有 offset-0 结果约为 $0.454$ 到 $0.540$，多重校正后均不显著。

token 内定位 AUROC 虽可到 $0.779$，但 top1 仅 $0.10$、平均 rank 约 30，说明它不是可用的精确定位器。由此应停止把简单位移范数、转角或 Menger curvature 当作“模型在首错处发生几何相变”的证据。

### 6.9 同题多采样：静态强于动态

3452 个有效 custom samples 的结果：

| 信号 | within | cross |
|---|---:|---:|
| response chars control | 0.578 | 0.719 |
| static support | 0.603 | 0.607 |
| static support length residual | 0.610 | 0.629 |
| dynamic support late | 0.589 | 0.690 |
| dynamic support late length residual | 0.593 | 0.617 |

动态减静态为 (-0.030)，CI ([-0.063,0.005])；残差化后为 (-0.039)，CI ([-0.066,-0.011])。因此“整链演化比静态状态更有信息”在当前对象上被直接否定。

### 6.10 预测状态几何：完整 gate 失败

correct-only reduced-rank predictor 的同题结果：

| 信号 | within AUROC |
|---|---:|
| ordered predictive Mahalanobis | 0.480 |
| static density | 0.466 |
| shuffled future null | 0.461 |
| same-problem mismatch null | 0.463 |
| fixed-window consensus | 0.646 |

ordered 相对 shuffle 只有 $+0.019$，CI 跨零；相对 fixed consensus 为 $-0.166$，CI 全负。所有 decision gate 失败。这说明现有 token cloud 的线性 reduced-rank predictability 不是错误轨迹的有效状态模型。

### 6.11 HCR、holonomy、layer-time：数学结构存在不等于任务信号

HCR-Holo 的 first-error、closure、commutator 和 basin 指标均约 $0.52$，response 最好约 $0.53$。Layer-time compatibility audit 的结果为：

- first-error transport residual：$0.566$；
- time rewire：$0.565$；
- holonomy：$0.510$；
- response fiber rank：$0.643$；
- response LID：$0.638$；
- reliable Wilson 仅覆盖 3% responses。

这些量可作为对 state array 的几何描述，但尚无证据表明它们对应推理正确性。尤其在 sparse layers 与 legacy step pooling 上构造 Wilson loop，不能称为已验证的 gauge curvature。

CTG 相对 random subspace 和 permuted steps 有增量，说明 transport 构造捕捉到某种有序结构；但 `controls+ctg` 不超过 controls，因此它不是新增错误信息。

### 6.12 VAE latent：低维投影并非关键步骤

full GSM8K VAE/latent 结果：

| 指标 | first-error AUROC | pre-error future AUROC |
|---|---:|---:|
| supervised hazard | 0.770 | 0.479 |
| supervised energy | 0.738 | 0.429 |
| VAE posterior uncertainty | 0.605 | 0.614 |
| VAE decoder uncertainty | 0.537 | 0.605 |
| VAE reconstruction NLL | 0.480 | 0.602 |
| VAE KL | 0.328 | 0.485 |

centroid separability 为 $0.736$，但 estimated ID 为 first-error $8.922$、non-error $9.377$，并不支持“错误必然更高维”。`hazard/energy` 是用错误标签训练的监督 readout，不能拿来证明 VAE 自己发现了错误流形。VAE 在这里是压缩器，不是理论必需项。

### 6.13 监督结构模型：证明信息上限，不证明机制

HGN 的 step AUROC $0.760$、first-error top1 $0.854$ 看起来很强，但它读取 token-step-layer 关系并接受监督。类似地，latent hazard、CTG readout 和普通 probe 都可能学习位置、长度、题型或文本结构。

它们的价值是：

> 完整 hidden representation 中还有 spread 以外的可学习信息。

它们不能直接支持：

> 正确推理沿某个低维几何对象稳定演化。

要把前者升级为后者，至少需要同题分组、位置长度匹配、null construction、跨任务冻结和因果干预。

### 6.14 当前冻结跨数据集结论

三数据集 L14：

| 数据集 | fused anchor baseline | broad sequence | 增量 |
|---|---:|---:|---:|
| GSM8K | 0.811 | 0.790 | -0.021 |
| MATH | 0.781 | 0.779 | -0.002 |
| OmniMath | 0.809 | 0.816 | +0.007 |

冻结标量：

| 信号 | raw macro/min | length-position residual macro/min |
|---|---:|---:|
| d_spread | 0.662/0.638 | 0.540/0.504 |
| spread | 0.726/0.702 | 0.600/0.521 |
| direction jump | 0.517/0.500 | 0.560/0.526 |
| spread CUSUM | 0.518/0.491 | 0.555/0.528 |
| joint CUSUM | 0.593/0.579 | 0.637/0.628 |
| spread surprise | 0.521/0.461 | 0.579/0.560 |
| joint surprise | 0.609/0.574 | 0.647/0.621 |

joint 使用正确链拟合：

\[
z_t=A z_{t-1}+B c_t+b+\varepsilon_t,
\quad
z_t=[\operatorname{spread}_t,\operatorname{anchorloss}_t,
\operatorname{uncertainty}_t],
\quad
c_t=[\log(1+N_t),\operatorname{relpos}_t],
\]

并以

\[
r_t=\varepsilon_t^\top\Sigma_{\varepsilon}^{-1}\varepsilon_t
\]

作为 transition surprise。当前只支持：

> 首错步骤比单纯高 spread 更稳定地违反正确链的联合转移规律。

尚不支持：

> 该 innovation 对完整融合基线有稳定 AUROC/AUPRC 增量。

## 7. 已被否证、应停止重复的主张

| 主张 | 结论 |
|---|---|
| 错误推理普遍在首错前持续发散 | 否；动态通常弱于静态，事件模式不统一 |
| 简单转角、曲率或 delta norm 能定位首错 | 否；匹配与多重校正后接近 chance |
| 谱熵、有效秩或 Gram 尾部天然优于 spread | 否；同题增量为负 |
| VAE 低维 latent 自然把正确/错误分开 | 否；无监督 VAE 指标弱，强项来自监督 head |
| holonomy/Wilson loop 已经捕捉推理曲率 | 否；效果近 chance 或覆盖不足 |
| scalar HMM/HSMM 能发现错误状态转移 | 否；明显低于静态 spread |
| 只要对长度分桶就已完全排除长度混杂 | 否；连续残差和同题控制仍使结果下降 |
| `anchor_uncertainty=0.81` 证明问题锚定机制 | 否；它是含长度、位置、spread、entropy 的融合分类器 |
| 高 within-chain localization 自动代表机制 | 否；位置单独可达 1.0，token top1 可能接近 0 |

## 8. 仍然成立的发现

### 8.1 强度排序

1. **监督 hidden-state readout**：约 $0.71$ 到 $0.76$，证明可分信息存在。
2. **融合 controls + geometry + uncertainty**：约 $0.78$ 到 $0.81$，是当前检测基线，不是单一机制。
3. **raw static spread**：跨数据集约 $0.70$ 到 $0.77$，同题约 $0.63$ 到 $0.66$。
4. **严格控制后的 spread**：平均仍高于 chance，但 worst dataset 接近 $0.52$。
5. **dynamic geometry**：大多 $0.50$ 到 $0.60$，很少超过静态。
6. **复杂 manifold/holonomy/VAE 无监督量**：当前大多 near chance 或无增量。

### 8.2 当前最可信的科学叙事

不是：

> 正确推理位于已被证明的低维稳定流形，错误会从流形上发散。

而是：

> 错误推理常伴随一步内 token 方向一致性降低；部分首错还表现为高秩残差碎片化。该现象是局部、任务相关且受长度/步骤功能影响的弱生理信号。模型输出不确定性提供另一条互补通道。把二者放入正确链条件化转移模型后，可以得到可跨数据集复现的 joint innovation，但其独立检测增量仍待证明。

## 9. 尚未裁决、不能被旧负结果替代的方向

| 方向 | 为什么尚未裁决 | 决定性实验 |
|---|---|---|
| NTS 法向逃逸 | Gate 2 的 NTS vs isotropic REMA 结果未进入总账 | 同 bank、三重残差化、跨数据集 CI |
| output-sensitive tangent escape | 缺少 exact downstream cotangent | 保存 `step_output_cotangent`，测试法向量是否进入 logit-sensitive 方向 |
| 真实 premise/source flow | qvec 与 regex 都不是 token-span 因果归因 | prompt span attention、patching/knockout、同题 counterfactual |
| ICR residual mismatch | 目前主要完成框架和 synthetic tests | ProcessBench 真实 attention + compact ICR，比较 controls 与 spread/entropy |
| attention graph spectrum | hidden Gram 不能替代 attention Laplacian | 保存 attention graph，测试 HFER/smoothness 的同题增量 |
| exact whole-chain HS/ME | step-local Gram 的失败不等于 whole-chain protocol | whole-chain与prefix两级、长度/同题/perturbation normalization |
| reference-relative manifold | 当前方法多为 source-free、自身轨迹几何 | 同题 correct rollout teacher manifold，严格留一和跨题 student 蒸馏 |
| 因果“模型意识到错误” | 所有现有结果主要是相关性 | activation/attention patching 能否改变错误概率并保持语义控制 |

## 10. 下一阶段的硬门槛

在继续提出新理论前，统一要求：

1. **先过同题或跨数据集冻结门槛**，禁止只报 pooled AUROC；
2. **显式比较长度、位置、step count、entropy 与随机/null 子空间**；
3. **报告增量而非单独分数**：

\[
\Delta\operatorname{AUROC}
=\operatorname{AUROC}(\text{baseline}+s)
-\operatorname{AUROC}(\text{baseline});
\]

4. **bootstrap 单位必须是 problem/chain**，不能把步骤当独立样本；
5. **区分检测、机制、边界三种贡献**：不提高 AUC 的量只有在解释错误子型或排除理论时才值得保留；
6. **需要因果声称时必须做 intervention**；attention 图或 hidden 相关性本身不够；
7. **禁止把监督 readout 的效果归因给前面的 VAE、流形或几何变换**；必须单独报告无监督几何量；
8. **结果不通过预注册 gate 就停止调参**，避免继续堆叠 toy 标量。

## 11. 结果来源索引

仓库内主要记录：

- [`md/progress/results_summary.md`](md/progress/results_summary.md)：早期 probe、PCA、difficulty、SPE、temporal 结果。
- [`md/progress/AAAI27_PAPER_PROGRESS_REPORT.md`](md/progress/AAAI27_PAPER_PROGRESS_REPORT.md)：spread、sphere、同题多采样、HGN 的阶段性汇总。
- [`md/progress/SMCD_SUMMARY.md`](md/progress/SMCD_SUMMARY.md)：旧 SMCD/EDIS、多数据集和 response 级历史结果。
- [`md/insights/2026-07-06-hypothesis-evidence-matrix.md`](md/insights/2026-07-06-hypothesis-evidence-matrix.md)：已退役与未测试方向矩阵。
- [`md/progress/plans/2026-07-07-directional-dispersion-mechanism-audit.md`](md/progress/plans/2026-07-07-directional-dispersion-mechanism-audit.md)：residual-rank、joint quadrant、constraint-anchor 结果。
- [`md/insights/MULTISAMPLE_DATA_AND_METHOD_NOTES.md`](md/insights/MULTISAMPLE_DATA_AND_METHOD_NOTES.md)：同题数据、temporal、tube 与 latent-state 协议。
- [`md/guides/DATA.md`](md/guides/DATA.md)：canonical 文件、字段与历史缓存风险。
- [`prompt_control_flow/RESULT_CROSS_DATASET_TRANSITION_2026-07-14.md`](prompt_control_flow/RESULT_CROSS_DATASET_TRANSITION_2026-07-14.md)：当前冻结跨数据集主结果。
- [`outputs/first_error_geometry/full_gsm8k/step/summary.md`](outputs/first_error_geometry/full_gsm8k/step/summary.md)：step event 原始结果。
- [`outputs/first_error_geometry/full_gsm8k/token/summary.md`](outputs/first_error_geometry/full_gsm8k/token/summary.md)：token event 原始结果。

以下数字目前主要来自远端终端摘要，尚缺统一提交的 raw JSON/CSV：trajectory phase、same-problem dynamic/static、directional consensus、predictive state、HCR-Holo、layer-time geometry、CTG、reasoning-state hazard、VAE latent separatrix。它们已纳入本总账。表中的 A-C 表示实验协议强度，`终端` 表示产物来源；只有完整 bootstrap/null 摘要的终端结果可暂按相应协议评级，论文引用前仍应归档 raw JSON/CSV。

## 12. 最终判决

这些实验并非全部白费。它们已经排除了一个很大的无效设计空间：仅靠更多曲率、谱熵、VAE、HMM、holonomy 或动态聚合，无法把一个受长度/难度影响的 spread 信号变成可靠机制。当前真正站得住的增量知识有三条：

1. hidden state 中有错误相关信息，但监督可分性远强于无监督几何解释；
2. 方向一致性下降是可重复但有限的局部现象，残差高秩只刻画其中一个碎片化子型；
3. 错误更像对“正确链条件化联合转移律”的 violation，而不是普适的几何发散。

这把下一阶段问题收敛为：**能否找到一个具有真实来源语义或输出因果敏感性的状态变量，并在同题、长度位置控制和跨数据集冻结下，对现有融合基线提供正的增量？** 在回答这个问题之前，不应再把新数学名词本身当作创新。
