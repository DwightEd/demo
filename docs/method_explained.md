# CTC 方法组件详解

> 本文档逐个拆解开题报告（v11）中 CTC（Conditional Trajectory Coherence）框架的每个方法组件：做什么、怎么实现、基于什么理论。

## 整体流水线

CTC 用一句话概括：**在正确推理轨迹上学会"下一步隐状态应该长什么样"，然后对新轨迹逐步检查"这一步是否偏离了预期"。**

```
原始隐状态 r_j (4096维)
    ↓ Layer 1: HARP投影
低维表征 z_j (200维)
    ↓ Layer 2: Causal Transformer预测
条件分布 p(z_j | z_{<j})  →  surprise分数 s_j
    ↓ Layer 2: CUSUM + CTM聚合
前缀累积分数 S_j
    ↓ Layer 3: ICAD校准
p-value  →  报警/不报警  →  (如果报警) 梯度干预
```

---

## Layer 1: HARP 投影

### 做什么

把 4096 维的隐状态压到 200 维。

### 怎么做

拿模型最后一层的"反嵌入矩阵" $W_{\text{unemb}}$（把隐状态映射到词表概率的矩阵，shape = [vocab_size × hidden_dim]），对它做 SVD 分解：

$$W_{\text{unemb}} = U \Sigma V^\top$$

$V$ 的每一列是一个"方向"。奇异值大的方向对应"生成输出词"最重要的方向，奇异值小的方向对应"与输出词无关但模型内部在用"的方向。HARP 取**最后 5% 的奇异值对应的方向**（约 200 个），认为这些就是"推理子空间"——模型在做推理计算时用的方向，不直接影响输出词。

投影操作就是矩阵乘法：$z_j = R^\top r_j$，其中 $R$ 是那 200 个方向拼成的矩阵。

### 理论来源

**HARP** (Hu et al., ICLR 2026)。实证发现：在这个子空间里做幻觉检测，AUROC 比在原始 4096 维空间里高 15-20 个百分点。直觉是推理相关的信号集中在这些"输出正交"方向上，投影过去等于去噪。

### 代码实现

```python
# 一次性离线计算
U, S, Vt = torch.linalg.svd(model.lm_head.weight)  # lm_head就是W_unemb
cutoff = int(0.95 * len(S))  # 取后5%
R = Vt[cutoff:].T  # shape [hidden_dim, ~200]

# 在线每步投影
z_j = R.T @ r_j  # 4096 → 200
```

---

## Layer 2a: Causal Transformer + Gaussian Head

### 做什么

给定前面所有步的低维表征 $z_1, ..., z_{j-1}$，预测第 $j$ 步应该是什么分布。

### 怎么做

一个小型 2 层 Transformer（不是大模型本身，是另一个参数量约 200K 的小网络），用 causal mask（只能看过去），输出一个 128 维的 context 向量 $c_j$。然后接一个"Gaussian Cholesky head"——两个线性层，一个输出均值 $\hat{\mu}_j$（200 维），一个输出下三角矩阵 $\hat{L}_j$（用来构造协方差矩阵 $\hat{\Sigma}_j = \hat{L}_j \hat{L}_j^\top$）。

这样就得到了一个完整的多元高斯分布 $\mathcal{N}(\hat{\mu}_j, \hat{\Sigma}_j)$，描述"根据前面的推理步，下一步的隐状态应该在哪个区域"。

### 理论来源

- **TraDE** (Fakoor et al., 2020)：用 causal Transformer 做连续变量的密度估计，是这个架构的直接来源。核心思想是把 autoregressive 模型从离散 token 推广到连续向量。
- **Cholesky 参数化**：用下三角矩阵保证协方差矩阵正定——数值线性代数的标准做法。

### 训练

只在**正确推理轨迹**上训练，loss 就是负对数似然：

$$\mathcal{L} = \sum_j -\log p(z_j \mid z_{<j})$$

意思是让模型学会"正确推理时下一步通常往哪走"。不需要错误样本的标签。

### 为什么选 Causal Transformer 而非 Neural Process

- Neural Process 的核心优势是跨任务的 amortized inference，在本研究的单任务大数据设定（PRM800K 约 48 万正确步样本）下不适用。
- TNP 的 permutation invariance over context 对**有序**推理步是有害归纳偏置——推理步严格有序，模型应直接利用这一结构。

---

## Layer 2b: Surprise 分数 $s_j$

### 做什么

衡量"实际的 $z_j$ 有多偏离预测分布"。

### 怎么做

直接算负对数似然：

$$s_j = -\log p(z_j \mid z_{<j}) = \frac{1}{2}(z_j - \hat{\mu}_j)^\top \hat{\Sigma}_j^{-1} (z_j - \hat{\mu}_j) + \frac{1}{2}\log\det\hat{\Sigma}_j + \text{const}$$

第一项就是 **Mahalanobis 距离的平方**——"$z_j$ 离预测均值有多远，用预测的协方差做归一化"。如果预测说"下一步应该在这个椭球里"，实际跑到了椭球外面，这个距离就大，$s_j$ 就高。

### 为什么叫 surprise

信息论里，$-\log p(x)$ 就是观测到 $x$ 的"惊讶程度"。正确步 → 模型预测准 → surprise 低；幻觉步 → 偏离预测 → surprise 高。

### 与位置判别的核心区别

已有方法（GeoReason、TRACED 等）是问"$z_j$ 这个点本身像不像异常点"。CTC 问的是"**在 $z_1,...,z_{j-1}$ 之后，$z_j$ 是否合理**"。

类比：在正常对话里突然说一句"今天天气真好"——单独看这句话完全正常（位置判别看不出问题），但如果前文在讨论微积分，这句话就很突兀（条件判别能捕捉）。

数学基础：条件 KL 散度严格不小于边际 KL 散度——条件信号在原理上严格强于位置信号。BCE 在数学上无法区分的两类情形（边际罕见但上下文合理 vs 边际常见但上下文跳变）在条件预测框架下天然分离。

---

## Layer 2c: CUSUM + CTM 双轨聚合

### 问题

单步 $s_j$ 有噪声，需要多步信号累积才能做出可靠判断。但 DeepSeek-R1 等模型会"自我纠错"（先犯错再回头改），如果用简单的 running-max（取历史最高分），纠错后分数降不下来，产生大量假阳性。

### Track 1: CUSUM (Page, 1954)

累积和检测，工业质控里用了 70 年的方法：

$$S_j = \max(0,\; S_{j-1} + s_j - k)$$

$k$ 是一个"正常水平"参考值（用校准集上的中位数 + 0.5 标准差）。

- 当 $s_j > k$（surprise 高于正常）→ $S_j$ 上升 → 累积证据
- 当 $s_j < k$（surprise 恢复正常）→ $S_j$ 下降 → **自动遗忘之前的 spike**
- 当 $S_j$ 降到 0 就彻底重置

这正好解决了 self-correction 问题：模型先犯错（$s_j$ 飙升），再纠正（$s_j$ 回落），CUSUM 自然衰减，不会永久保留假阳性。Running-max 做不到这一点。

**理论保证：** Lorden (1971) 证明 CUSUM 在"平均检测延迟 vs 虚警率"这个 tradeoff 上是 **minimax 最优**的。

### Track 2: Conformal Test Martingale (Vovk, 2021)

先把 $s_j$ 通过校准集转成 p-value $p_j$（"在正常情况下，看到比这更极端的 $s_j$ 的概率是多少"），然后用 betting function 把 p-values 累乘：

$$M_j = \prod_{i=1}^{j} f(p_i)$$

如果所有步都正常，$p_i$ 服从均匀分布，$M_j$ 在 1 附近波动。如果幻觉发生，$p_i$ 偏小，$M_j$ 指数增长。

**理论保证：Ville 不等式** $\Pr[\sup_j M_j \ge 1/\delta] \le \delta$——不管在第几步停下来看，$M_j$ 超过阈值的概率都被控制住（分布无关、anytime-valid）。

### 为什么要两个

| | CUSUM | CTM |
|---|---|---|
| 擅长 | 突变型幻觉（某步突然偏离） | 渐变型幻觉（逐渐漂移，每步偏离一点点） |
| self-correction | 自然衰减 | 累乘结构也会因好 p-value 缓慢回落 |
| 理论保证 | minimax 最优检测延迟 | anytime-valid 停车证书 |

两个取**并集**报警：任一触发就报警。

---

## Layer 3a: ICAD-CPR 校准

### 问题

$s_j$ 的绝对值没有统计意义——不知道"$s_j = 15$"算高还是不高。需要把它转成有统计保证的 p-value。

### 怎么做

Inductive Conformal Anomaly Detection (Laxhammar & Falkman, 2014)：

1. 留出一批正确轨迹作为校准集，计算它们每步的 $s_j$，存下来
2. 对新轨迹的每步 $s_j$，数一下校准集里有多少 $s$ 比它大：

$$p_j = \frac{1 + |\{s_{\text{cal}} \ge s_j\}|}{n_{\text{cal}} + 1}$$

3. 如果 $p_j \le \varepsilon$（比如 0.05），说明"在正常轨迹中，这种程度的 surprise 极其罕见"→ 报警

**CPR（Conformal Path Reasoning）** 处理一个技术细节：每条轨迹有多步，如果对每步单独算 p-value 会有多重检验问题。CPR 取整条链的 path-max $s_* = \max_j s_j$，恢复跨查询的交换性。

### 理论保证

有限样本下虚警率 $\le \varepsilon$，不需要任何分布假设——这是 conformal prediction 的核心优势。

### 双层校准结构

- **L1 ICAD-CPR**：步级边际虚警率证书（有限样本 $\le \varepsilon$）
- **L2 CTM**：序列级 anytime-valid 停车证书（Ville 不等式）
- 超参仅需 3 个：$\varepsilon$、$\delta$、CUSUM 阈值 $k$

CTM 同时担任**聚合**（Track 2）与**校准**双重角色——这是设计简洁性的核心来源。

---

## Layer 3b: 梯度干预

### 做什么

检测到幻觉后，把模型的隐状态"推回"正确方向。

### 怎么做

$$\Delta r_j = -\eta \cdot \nabla_{r_j} s_j$$

$s_j$ 是 surprise 分数，$\nabla_{r_j} s_j$ 是 surprise 对当前隐状态的梯度——"隐状态往哪个方向动，surprise 上升最快"。取负号就是"往 surprise 下降最快的方向走"，即**推向训练分布的 mode**（正确推理区域）。

这个梯度是通过对小 Transformer（$\Phi_\psi$）做一次反向传播得到的，不需要对大模型本身反向传播。

然后通过 **KV-cache 重写**把修正后的 $r_j'$ 写回去——在目标层重新算 $W_K r_j'$ 和 $W_V r_j'$，替换原来的 key/value，后续生成基于修正后的表征继续。

### 理论来源

- KV-cache 重写的工程做法来自 **Reasoning as Trajectories** (Sun et al., 2026)，他们用 rank-32 低秩更新。
- 梯度方向的选择是 NLL 对数似然梯度的标准推论——$-\nabla s_j$ 指向训练分布的 mode。
- 因果对照（basin / random / orthogonal 三方向）参考 **Hallucination Basins** (Cherukuri & Varshney, 2026)。

---

## 训练辅助技巧

### Scheduled Sampling (Bengio et al., 2015)

训练小 Transformer 时，正常是用真实的 $z_{j-1}$ 作为输入（teacher forcing）。但推理时是用模型自己的预测。这个 gap 会导致误差累积（exposure bias）。

解决方法：训练时以线性递增概率（最高 30%）用模型自己的预测 $\hat{\mu}_{j-1}$ 替代真实输入，让模型适应自己的误差。

### InfoNCE 对比学习 (van den Oord et al., 2018)

防止小 Transformer 的 context 向量 $c_j$ 坍缩到一个点（hypersphere collapse）。加一个对比损失，让 $c_j$ 和对应的 $z_j$ 相近，和其他 $z_k$ 远离。权重 $\lambda_{\text{ctr}} = 0.1$。

---

## 组件来源汇总

| 组件 | 直接来源 | 我们的用法 |
|------|---------|-----------|
| HARP 投影 | Hu et al. ICLR 2026 | 原封使用，作为降维预处理 |
| Causal Transformer 密度估计 | TraDE (Fakoor et al. 2020) | 将输入从表格数据换成推理步隐状态 |
| Gaussian Cholesky head | 标准参数化 | 无改动 |
| Surprise $s_j = -\log p$ | 信息论基础概念 | 作为步级异常分数 |
| CUSUM | Page 1954, Lorden 1971 | 无改动，用于 self-correction-robust 聚合 |
| Conformal Test Martingale | Vovk 2021 | 兼任聚合 + 校准双重角色 |
| ICAD-CPR | Laxhammar & Falkman 2014 | 无改动，用于步级 p-value 校准 |
| KV-cache 梯度干预 | Reasoning as Trajectories 2026 | 干预方向从全局均值换成 $-\nabla s_j$ |
| Scheduled Sampling | Bengio et al. 2015 | 标准使用，缓解 exposure bias |
| InfoNCE | van den Oord et al. 2018 | 标准使用，防止表征坍缩 |

**坦率评估：** 每个组件都是现有方法。CTC 的贡献是**组合方式**和**应用场景**（步级条件检测 + self-correction 鲁棒），而不是单个组件的创新。真正的原创性需要来自**经验发现**——例如 AFC（Attention-FFN 协同度）假设，如果被 Q1 pilot 验证，才是属于自己的贡献。
