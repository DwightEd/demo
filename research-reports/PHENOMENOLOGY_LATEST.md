# 《The Phenomenology of Hallucinations》机制拆解与项目启示

论文：[arXiv:2603.13911](https://arxiv.org/abs/2603.13911)，v1 预印本。本文把论文观察、作者解释和我们对实验设计的判断分开陈述。

## 1. 核心观点

论文将幻觉重述为一个“内部检测已经发生，但没有正确表达到输出”的失配过程：

```text
Detection -> Fracture -> Breach
```

- **Detection**：论文在其数据集标签下观察到 factual 与 impossible/long-tail 输入的隐藏几何逐层分开。
- **Fracture**：论文观察到某些拓扑统计量变化，并把 \(\beta_0\) 增长解释为不确定状态“碎裂”成多个局部簇；碎裂是作者的机制解释，不是统计量本身。
- **Breach**：低输出敏感度是测量结果；“交叉熵强制承诺、MLP 联想泄漏导致幻觉”则是作者对该结果的机制解释。

最直观的理解是：模型可能有“烟雾传感器”，但传感器与“警报器/拒答器”之间的连线很弱。论文的价值不在于证明模型具有人的主观自知，而在于提出一个可干预的工程假设：应测试内部不确定性方向是否存在、是否能被可靠读出、是否因果影响输出。

## 2. 实验对象

语言模型部分比较三类输入：

- 容易且高正确率的事实问题；
- 错误前提或上下文不足的 impossible 问题；
- TruthfulQA / 长尾 PopQA 等高失败率问题。

模型覆盖 Llama、Qwen、Mistral、Pythia、OLMo；另以 PixArt-\(\Sigma\) 检查扩散模型中是否也有类似高维不稳定表示。

这里有一个非常重要的构念风险：论文在相当程度上按“数据集类型”定义 factual / impossible / hallucination，而不是在同一道题、同一生成配置下按实际输出正确与否配对。因此它可能同时测到了主题、风格、难度、长度和 OOD，而不只是幻觉机制。

## 3. 表示边界方向

在第 \(\ell\) 层，令事实样本与不确定样本的隐藏均值分别为 \(\mu_\ell^{\mathcal F}\) 和 \(\mu_\ell^{\mathcal U}\)。论文定义单位边界方向

\[
b_\ell=
\frac{\mu_\ell^{\mathcal U}-\mu_\ell^{\mathcal F}}
{\|\mu_\ell^{\mathcal U}-\mu_\ell^{\mathcal F}\|_2}.
\]

它回答的是“从事实数据中心走向不确定数据中心的方向是什么”。作者观察到类中心距离通常随层数增长，认为网络主动放大了两类输入的表示差异。

但它不是天然的“幻觉方向”：只要两组数据在主题或 prompt 形式上不同，中心差也会变大。论文第 5 节已经加入随机正交、范数匹配方向控制；更完整的验证仍需要 leave-topic-out、同题正确/错误配对、多个随机种子，以及 held-out 估计 \(b_\ell\)。

## 4. 局部内在维度：为何未知状态更“散”

论文使用近邻距离估计局部内在维度（LID）：

\[
\widehat{\operatorname{LID}}(x)
=-\left[
\frac{1}{k}\sum_{i=1}^{k}\log\frac{r_i(x)}{r_k(x)}
\right]^{-1},
\]

其中 \(r_i(x)\) 是第 \(i\) 个近邻距离。LID 高意味着局部样本需要更多自由方向才能描述。

作者解释为：熟悉事实被训练压缩成稳定、低维的吸引域；未知或冲突输入会激活多个不完整、互相竞争的特征，因此表示更高维、更弥散。

这个机制直觉合理，但论文中“普遍达到 2–3 倍”的表述不宜照搬。Table 6 的 Hallucination/Factual LID 比约为 \(1.22\)–\(1.41\)，Impossible/Factual 约为 \(1.39\)–\(1.61\)；明确达到 \(2.5\times\) 的是 PixArt 示例 \(12.5/5.0\)。LID 还对 \(k\)、距离度量、样本密度和类别混合高度敏感。

## 5. 谱与拓扑：Fracture 如何被量化

论文还用协方差/奇异值谱衡量表示能量是否分散。由于正文在协方差特征值与归一化奇异值之间表述不完全一致，这里用一般谱量 \(s_i\) 表示：

\[
p_i=\frac{s_i}{\sum_j s_j},
\]

则谱熵和有效维数可写为

\[
H_{\mathrm{spec}}=-\sum_i p_i\log p_i,
\qquad
D_{\mathrm{eff}}=\exp(H_{\mathrm{spec}}).
\]

谱越平，说明表示分散在更多方向。论文的一些表格把数百量级数值称为 entropy，实际更像 \(\exp(H)\) 后的有效维数；复现时必须明确到底报告 \(H\) 还是 \(D_{\mathrm{eff}}\)。

作者还使用 Vietoris–Rips filtration 的持久同调：\(\beta_0\) 表示连通分支数，\(\beta_1\) 表示环。观察到不确定样本的 \(\beta_0\) 较高后，作者称其发生“拓扑碎裂”。

这一结论依赖 filtration 半径、持续性阈值和归一化。没有完整曲线时，单独报告某个尺度下的 \(\beta_0\) 很难说明机制；来自多个语义主题的样本本来就会产生多个分支。更不能由非零 \(\beta_1\) 直接推出“线性不可读”，尤其论文同时报告了强线性探针。

## 6. Breach：内部方向为何影响不了输出

论文用多种局部敏感度近似检验沿 \(b_\ell\) 的扰动是否改变输出分布。为直观理解（这是本文的线性化，不是论文直接估计完整 Jacobian 奇异谱），若 logits 为 \(z=f(h)\)，局部变化为

\[
\Delta z\approx J_f(h)\,\Delta h.
\]

如果 \(b_\ell\) 落在局部 readout 的低增益方向，即使内部投影很强，\(J_f(h)b_\ell\) 仍很小。论文实际对 unembedding \(W_U\) 做 SVD，并另行计算 KL sensitivity 与 Jacobian amplification；据此提出 low-sensitivity subspace：模型在隐藏空间中“检测”了异常，但异常方向对词表输出不敏感。

论文用 baseline 与单侧扰动分布之间的 symmetric KL 近似 Fisher sensitivity：

\[
F(b_\ell)\approx\epsilon^{-2}
D_{\mathrm{KL}}^{\mathrm{sym}}
\!\left(p(\cdot\mid h_\ell),p(\cdot\mid h_\ell+\epsilon b_\ell)\right),
\qquad
D_{\mathrm{KL}}^{\mathrm{sym}}(p,q)
=\frac12\left[D_{\mathrm{KL}}(p\|q)+D_{\mathrm{KL}}(q\|p)\right].
\]

这里的数值实现需要警惕：若 \(\epsilon=10^{-3}\) 且直接用 FP16，对范数很大的 hidden state 来说扰动可能接近数值分辨率，差分会消失或不稳定。复现应在 FP32/FP64 中扫描多个 \(\epsilon\)，并与范数匹配随机方向比较。

论文还把 unembedding/readout 矩阵做 SVD，区分高、低输出敏感子空间，并测不确定性方向落在哪一侧。不同章节的 low-sensitivity ratio 分母定义似有差异，截断维数也需要预注册，否则容易通过选阈值放大结论。

## 7. Attention 与 MLP 的机制解释

论文比较 attention 与 MLP 更新对不确定性方向的对齐，观察到部分模型中 MLP 对齐明显更强。作者的解释是：attention 更接近上下文检索和校正，MLP 储存的联想先验在证据不足时继续完成模式，少数高峰激活最终支配输出。

这是有启发性的组件归因，但不能仅凭 cosine 对齐断言因果主导。更强设计应分别 patch/ablate attention 与 MLP 更新，做路径特异 intervention，并匹配层、范数和 token 位置。

## 8. “Simplex Vertex Attractor”解释

对 one-hot 标签 \(y\)，交叉熵为

\[
L(z,y)=-\log\frac{e^{z_y}}{\sum_j e^{z_j}},
\qquad
\nabla_z L=p-y.
\]

作者认为该梯度持续把预测推向概率单纯形的顶点；若训练中没有明确的拒答或未知目标，不确定输入最终也会落入某个词表决策盆地，形成强制承诺。

应把它看作机制假说，而不是交叉熵的定理。对数据分布的期望风险，交叉熵最优解可以是非退化的条件分布；关键问题是训练数据、偏好优化和解码是否奖励合适的拒答/校准行为，而不是 one-hot 形式本身必然导致幻觉。

## 9. 三类干预

### 9.1 Readout bypass

训练隐藏态探针识别不确定输入，再直接提高 “unsure” 类输出；论文报告 Llama/Qwen 拒答率约为 \(100\%/99.75\%\)。拒答率很高并不意外：只要 logit 偏置足够大，输出就会被强制改变。真正应报告的是 held-out detector 表现、事实问题误拒率、coverage–accuracy 曲线与干预幅度。

### 9.2 Boundary steering

向事实样本注入不确定性方向后，论文报告 Llama/Qwen 输出变化率约为 \(99.75\%/89.36\%\)。论文已经比较随机正交且范数匹配的方向，但控制并未完全消除非特异扰动：例如 Llama early flip 为 \(0.98\) 对 \(0.90\)，Qwen mid 为 \(0.27\) 对 \(0.26\)，late 为 \(0.16\) 对 \(0.15\)。真正仍缺的是多随机种子、双向剂量曲线、明确的扰动幅度 \(\alpha\)、语义保持与事实性结果。

### 9.3 PCA manifold repair

把异常表示投影回事实子空间并没有稳定修复循环输出：Llama loop 约由 \(9.85\%\) 变为 \(12.88\%\)，Qwen 约为 \(1.77\%\to1.77\%\)。作者据此认为幻觉不是简单加性噪声。更谨慎的说法是：全局线性 PCA 投影不是足够的修复算子；这不排除局部、条件化或非线性 manifold repair。

## 10. 论文最可信与最薄弱的部分

最可信的是：

- 将“内部可检测性”和“输出可表达性”明确拆开；
- 同时检查几何、输出敏感度、组件与干预；
- 给出了可以被随机方向、held-out 探针和 causal patching 继续检验的机制框架。

最薄弱的是：

- 数据集类型与真实生成错误混淆；
- 训练/测试划分、topic holdout 和阈值细节不够清楚；
- LID、拓扑、各向同性与 Fisher 的若干强表述超过表格直接支持范围；例如 Table 6 的 Qwen-2.5 isotropy 为 factual \(0.582\)、hallucination \(0.511\)、impossible \(0.406\)，不支持“不确定输入普遍更 isotropic”。正文称 instruction tuning 抑制 Fisher，但 Table 5 的 Llama-3.2-3B hallucination Fisher 从 base \(0.049\) 变为 instruct \(0.459\)；Table 3、Table 5 与正文的数量级/聚合方式也未解释清楚；
- “提高 unsure logit 后拒答”不能单独证明内部信号已经自然连接到拒答；
- 扩散模型的计数/文字失败更像能力不足，不一定与语言事实幻觉共享同一认识论对象。

因此，论文支持的是“detection–expression gap 值得作为可检验假说”，还不足以无条件证明“模型知道自己不知道”。

## 11. 对 `demo` 的直接启示

项目不应简单复刻论文的跨数据集 boundary vector。更强的检验是：

1. 同一道题采样多条正确/错误推理，控制共享的题目难度；
2. 在实际生成上下文中提取 prompt anchor、hidden 与 logit/readout；
3. 定位首次错误前后的 anchor-residual rupture；
4. 测 rupture direction 的输出可见性，而不是只测类中心距离；
5. 用 survival hazard 处理正确链右删失与错误后 mask；
6. 在所有触发链上做等预算 micro-replay，报告净收益。

这使论文的宏观假说变成一个更严格的局部检验：

\[
\text{evidence detachment}
\rightarrow
\text{conditional geometric rupture}
\rightarrow
\text{low readout visibility / overcommitment}
\rightarrow
\text{first-error hazard}
\rightarrow
\text{intervention utility}.
\]

若任何箭头不能在同题 held-out 数据上成立，就应断开该机制链，而不是用最终分类分数替代中间证据。
