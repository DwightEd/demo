# 几何维度线 — 指标、代码、结果汇总 (2026-06-10)

ProcessBench gsm8k + 采样 K=12 (v2_5shot)，Llama-3.1-8B，teacher-forcing 抽取。
判别口径以 **题内配对 AUROC + 只看答案 + 格式控住**（`within_ans` / `--format_ok_only`）为准。

## 1. 指标定义、公式、代码位置

抽取总入口：`extract_features.py` → `extract_chain()`（每条链一次前向，读 hidden_states + logits）。

### A. 论文不确定性（逐 token，trace channel）
| 指标 | 含义 | 公式 | 代码 |
|---|---|---|---|
| `U_D` | 分布不确定性 = 全词表预测熵 | `−Σ_v p(v) log p(v)` | `features/uncertainty.py::_entropy_committal` |
| `U_C` | 委身不确定性 = 实际下一 token 的伯努利方差 | `p(v_t)·(1−p(v_t))` | 同上 |
| `U_E` | 认知不确定性 = 梯度模长平方（本次跳过，24G OOM） | `‖∇_θ p(v_t)‖²` | `features/uncertainty.py::epistemic_grad_norms` |

### B. 逐向量几何（对每步 exp 池化向量 z；raw 不标准化）
池化：`utils/step_vector.py::step_vector(mode="step_exp")`（Lu et al. Eq.6，后 token 加权，**不** L2 归一化以保模长）。
计算：`features/geometry.py::vector_features()`，能量分布 `p_d = z_d²/‖z‖²`。
| 指标 | 含义 | 公式 | 代码 |
|---|---|---|---|
| `norm` | 激活模长（幅度） | `‖z‖₂` | `vector_features` |
| `pr` | Rényi α=2 有效维（峰主导） | `(Σz²)²/Σz⁴ = (Σp_d²)⁻¹` | `participation_ratio` |
| `ae` | Rényi α=1（Shannon）有效维 | `exp(−Σ p_d log p_d)` | `renyi_eff_dim(α=1)` |
| `ed_half` | Rényi α=0.5（偏 bulk）有效维 | `(Σ √p_d)²` | `renyi_eff_dim(α=0.5)` |
| `e50/e90` | 覆盖 50%/90% 能量的最少维数 | 排序累计 | `energy_width` |
| `ae_robust` | 去掉 top-4 massive 维后的 AE | — | `ae_robust` |
| `anom_k5/k10` | 异常激活维数 | `#{d: |z_d|>k·median|z|}` | `anom_count` |

### C. 点云谱（对每步 token 云 H_j ∈ R^{n×d}，做 SVD）
计算：`utils/spectral.py::step_layer_spectral_summary()`。**注意 cloud_D ≤ 该步 token 数 n → 受长度污染。**
| 指标 | 含义 | 公式 | 代码 |
|---|---|---|---|
| `cloud_D` | 点云有效秩（谱熵） | `exp(−Σ p_i log p_i)`, `p_i=σ_i²/Σσ²` | `effective_rank` |
| `cloud_V` | 谱能量 | `Σσ_i²` | `spectral_energy` |
| `cloud_C` | 顶端集中度 | `σ_1²/Σσ_i²` | `top_concentration` |

### D. 整链非线性内在维数（长度鲁棒，不随点数增长）
对整链所有 response token 池成一个云 (R~200, d=4096)。
| 指标 | 含义 | 代码 |
|---|---|---|
| `id_mle` | Levina-Bickel MLE-kNN 内在维数 | `utils/spectral.py::cim_tle_intrinsic_dim` |
| `id_twonn` | TwoNN 内在维数（Facco 2017） | `features/geometry.py::twonn_dim` |

### E. 轨迹汇总（Tracing 风格，对任一逐步/逐 token 序列）
`features/trace_profile.py::profile()` → `µ_early`(前25%)、`µ_mid`(中50%)、`µ_late`(后25%)、`slope`、`r²`。

## 2. 分析脚本
| 脚本 | 干什么 |
|---|---|
| `analyze_features.py` | 链级判别力（within/cross AUROC + Cohen d）；`--format_ok_only` 控格式；`--corr` 维度 vs 熵相关矩阵（raw + 题内残差） |
| `localize_signal.py` | 步级定位（用 ProcessBench `gold_error_step`）+ 出错步对齐曲线 + 正确/错误轨迹表；`step_ntokens` 长度对照 |
| `probe_features.py` | Tracing 式题内 GroupKFold 逻辑回归；static vs slope/r² 消融，paper vs geom 消融 |
| `spe_analysis.py` | 子空间泄漏 SPE（已搁置） |
| `plot_trajectories.py` | 轨迹/对齐/单条样例的 PNG 图 |

## 3. 结果（截至 2026-06-10）

### 链级（采样 v2_5shot，format-ok，题内，只看答案）
- **最强 = `U_D` 均值 ~0.615**（论文特征），`U_C` ~0.60。
- 我们的几何全部 ~0.40–0.57：`pr_L16_late 0.565`、`cloud_D ~0.548`、`ed_half ~0.547`、`norm 0.40`（错误更低）、`cloud_C 0.42`（错误更不集中）。
- `n_steps` 0.464（长度非信号）。Mahalanobis（旧主线）~0.657。
- **方向一致**：错误链 熵↑、有效维↑、模长↓、集中度↓ = 都"更分散"，**但各自都弱**。
- **`id_mle_L8` = 0.581（最强几何）**：长度鲁棒（× n_resp_tokens 题内相关 0.02，对比 cloud_D 0.24）+ 难度无关（within 0.581 > cross 0.522）+ 方向对。`id_twonn_L16` 0.562。
  → 你的"换好估计器"赌注在 *干净性* 上赢了，但绝对值仍 < U_D。

### 维度 vs 熵 相关性（`--corr`，题内残差）
- `UD_mid × id_mle_L8 = 0.41`、× id_mle_L16 = 0.33 → **中度冗余**（熵高↔内在维高，同一个"分散"的两个窗口）。
- `id_twonn` × UD_mid = 0.14（最正交，但 id_twonn 最弱）。
- **结论：维度和熵中度相关、部分冗余，不是正交 → 互补空间很小。**

### Tracing 式探针（含 id_dim/cloud，format-ok）
- `paper static 0.634`（最强）；`paper + id_dim 0.567`（**加几何反而降**）；`ALL static 0.608`（259 特征 < 6 个 paper 特征）。
- 几何 cross 0.66–0.71 全是难度膨胀（within 0.56–0.59）。
- **几何对 U_D 零增量甚至负增量。**

### 步级定位（ProcessBench，gold 出错步）
- `cloud_D` loc_auroc **0.72**，但 `step_ntokens` 单独 **0.70** → **cloud_D = 步长伪装**，几何净增 +0.02（噪声内）。
- 其它：`norm` 0.28（出错步模长更低，翻转 0.72）、`anom_k10` 0.61、`U_D` 0.58。都弱。

### SPE 子空间泄漏（已搁置）
- 干净 within ~0.58–0.60，方向对（错误泄漏更多），但没打过 Mahalanobis；`meanSPE_cor` 在 k=100 仍 0.28 → 健康子空间高维，"低维流形"不成立。

### Tracing 式探针（合并 + slope/r²）
- `paper static 0.634`（最强）；`paper +dyn 0.631`（slope/r² **不加强**）；`ALL +dyn 0.612`（几何 **拖累**）。
- **几何相对 U_D/U_C 无增量。**

## 4. 最终判断（几何线收口）
"错误推理占据更高维/更发散"在 population 层面**真且显著（Cohen d≈0.35）**，但：
1. **效应小** → 难度+格式+长度控住后逐条 AUROC ~0.55–0.58（信息上限，非方法问题）；
2. **最干净的几何信号 = 整链非线性内在维数 `id_mle`（0.581，难度无关、长度鲁棒）**，但
3. **它与输出熵 U_D 中度冗余（题内 r≈0.41）** → 组合零增量（探针 paper+id 0.567 < paper 0.634）。

→ **几何/维度不是独立检测信号，而是"不确定性的另一种度量"**：错误推理在输出分布（熵）和内部表示（内在维）上同步"更分散"，两者耦合。真正活着的强信号是 `U_D`（熵 0.634）和 Mahalanobis（位移 0.657）。

**可写的正面结论（discussion）**：内部表示维度 ↔ 预测熵 耦合（r≈0.41，难度/长度均无关），是同一"分散"的两个窗口；`id_mle` 提供了难度无关、长度鲁棒的几何度量。
**待最后确认**：极简 2-vs-4 特征探针（`min: UD+UC` vs `min: +id_mle`）排除过拟合即定案。
**方向决定**：几何线收口 → 转"记忆 vs 证据冲突，从哪步走偏"（用已验证的步级提取 + U_D/Mahalanobis 信号）。

## 5. 几何线最终审计判决 (2026-06-11)

步级、ProcessBench 过程级标签（正=首错步 205，负=正确链步+错链首错前步 970），8 层
(2,6,10,14,18,22,26,30)，交叉拟合 V*，链配对 bootstrap n=500。`layer_coupling.py` / `resid_audit.py`。

### (A) "谱场"/跨层协调：证伪
阶梯 (i)单层→(ii)池化→(ii')逐层打分→(iii)线性袋子→(iv-a)=(iii)+c，三对照 = rand-quad / smooth-c。
| 度量 | (iii) | (iv-a) | (iv-a)−(iii) | −rand-quad | −smooth-c | AUROC(−c) |
|---|---|---|---|---|---|---|
| cloud_D | 0.714 | 0.713 | −0.001 ns | +0.000 ns | −0.001 ns | 0.485 |
| ed_half | 0.715 | 0.713 | −0.002 ns | −0.000 ns | +0.000 ns | 0.505 |
| norm | 0.791 | 0.791 | −0.001 ns | +0.002 ns | +0.001 ns | 0.523 |
- **协调度 c 在线性袋子之上零增量**（所有 gap 95% CI 含 0），**c 不区分错误步**（AUROC≈0.5，cloud_D 反向）。
- `smooth-c` 对照确认：即便有微弱跨层结构，也分不出它与残差流平滑性。
- → **"谱场" = 一袋逐层特征，核心概念证伪。跨层协调降级为分析章节。**

### (B) 步级几何"定位" ≈ 长度/位置混淆
`resid_audit`（E[M|n_tok, j/T, density] 在正确步上拟合，残差替代）：
- cloud_D：**nuisance-only AUROC = 0.711 ≈ raw 0.714** → 首错步=更长/更靠后/公式更密，平凡事实贡献几乎全部判别力；残差化后诚实几何 ~0.63（且 < 混淆单独 0.71，几何是长度的 noisy 代理）。
- ed_half：浅层(2,6) raw 0.61→resid 0.51（是混淆）；**深层(14–30) raw~0.53→resid~0.58（不降反升，长度掩盖了一点真信号）**——微弱但真。
- norm：步级 0.791 最高（待补 resid_audit；按模式预计大半是长度）。

### 几何线五条审计结论（可写论文）
1. **谱场证伪**：跨层协调 c 零增量、不区分错步（3 度量×3 对照）。
2. **步级几何定位 ≈ 长度/位置混淆**：nuisance-only 0.71 ≈ raw 0.71，残差 ~0.63。
3. **CIM 轨迹级证伪**：D+V=0.509=chance（模型级 Spearman 0.9 ≠ 轨迹级）。
4. **几何 ⊥ 不确定性冗余**：id_mle×U_D 题内 r=0.41，组合零增量。
5. **链级几何 = 难度膨胀**：cross>within，控住后塌到 ~0.55–0.58。

**定位**：不是"找不到信号"，是**严格证明了几何信号是什么、不是什么**——干净的负面/审计论文。真正活着的检测信号是 U_D（熵 0.634）与 Mahalanobis（位移 0.657）。深层 ed_half 残差 ~0.58 是唯一去混淆后还活着的几何残量，弱。

## 6. 修正：步级混淆分解 → norm 是独立机制信号 (2026-06-11)

`resid_audit.py` 三段式分解（步级、ProcessBench 过程标签、按链 bootstrap CI）：
`raw AUROC = 表层混淆 + 不确定性 + 几何增量`，每段单独检验显著性。混淆 = n_tok / j/T / density
（实测 **n_tok=长度是唯一主混淆**：err 73.7 vs good 49.4，单独 AUROC 0.708；pos 0.524、density 0.551 弱）。

| 成分（步级 AUROC） | 值 | 增量(95% CI) |
|---|---|---|
| 表层混淆（长度主导） | 0.711 | — |
| + 不确定性 U_D/U_C | 0.775 | +0.064 |
| **+ 中层 norm（几何）** | **0.832** | **+0.057 [0.033, 0.082] 显著** |

- **norm = 唯一扛过完整分解的几何信号**：在长度+位置+密度+**熵+委身**全控住后仍 **+0.057 显著**。
  中层 14 峰值（raw 0.775→resid 0.660），**方向与长度相反**（出错步更长却中层模长更低）→ 不是长度/不是熵的翻版 = **独立机制信号**。
- **cloud_D 增量仅 +0.021**（控熵后），各层平坦 → ≈ 长度，机制论断撑不起。
- **关键重构**：活下来的**不是"维度/弥散"锚点（证伪），是"幅度"（norm）**。论文叙事应为
  "**维度假设失败，但中层激活幅度携带一个独立于不确定性的步级幻觉信号**"。
  （norm 一直是最强几何特征：链级 Cohen's d=−0.69 全表最大；现在才严格隔离出独立性。）
- **方法论贡献**：三段式混淆分解（表层 + 不确定性 + 几何），机制论断只建在显著增量上，
  混淆/不确定性作诚实标注的成分保留在融合检测器里（0.832）。

**待办（norm 深挖）**：逐层扫描确认中层定位；跨 ProcessBench config（math/olympiad/omnimath）；
与 Mahalanobis 关系；机制解释（为何中层模长在错步下降）。另一条活线：梯度谱场（H100）。

## 7. 再修正：norm 下降 = 方向弥散（不是幅度）→ 弥散锚点干净复活 (2026-06-13)

第 6 节说"幅度 norm 是独立信号"。把 norm **逐 token 拆开**后真相更深一层——**不是 token 变弱，是池化时方向抵消更多 = 弥散**，锚点（最初被有效秩证伪的那个）以干净形式复活，估计器从"有效秩"换成"方向相干度"。

### 关键拆解（`norm_decomp.py --per_token`，gsm8k layer 14）
| 量 | err | good | 读数 |
|---|---|---|---|
| `mean_tok_norm`（逐 token 模长均值） | 8.09 | 8.15 | **几乎相等 → token 个体不弱** |
| `norm`（exp 池化后模长） | 4.81 | 5.18 | **池化后缩水 → 抵消更多** |
- 个体一样强、池化缩水 = **错步 token 方向更发散、池化时互相抵消** = 弥散。不是"激活变弱"。

### 两个弥散估计器（步级，混淆+U_D 全控，链 bootstrap CI）
| 指标 | 定义 | 增量(over 长度+位置+密度+U_D) | 桶内 L14 | raw L14 | 长度相关 |
|---|---|---|---|---|---|
| `coherence` | `‖池化‖ / mean_t‖h_t‖`（除整体尺度） | **+0.062 [0.039, 0.087] 显著** | 0.717 | 0.778 | −0.831 |
| `resultant` | `‖exp池化(单位化 token)‖ ∈[0,1]`（纯方向） | **+0.059 [0.038, 0.083] 显著** | 0.708 | 0.772 | −0.832 |
- **两个独立估计器数值几乎重合**（CI 重叠）→ 互证。resultant 把**全部逐 token 幅度除掉、只留方向**,信号纹丝不动 → **弥散是真方向现象,不是任何幅度假象**。

### massive activation：测过——不是独立信号,也不污染 coherence
`norm_decomp` 数据驱动定 massive 维 [290,291,682,3266,4055]：
- `massive_frac` AUROC = **0.577 近随机**（err 0.36 vs good 0.38）→ **错步并未对 massive 维做特别的事**，massive 不是该单列的独立信号。
- `norm_massive`(0.709)/`norm_bulk`(0.733) 看着有区分度，但是**全谱按比例一起缩**的"搭便车"，非 massive 特异。
- `coherence` = 0.59 vs 0.64（**不是 ≈1**）→ massive 没把弥散云伪装成相干，**未污染 coherence**。
- 次要线（弱）：`massive_frac` 桶内 0.627 > raw 0.577 → 固定长度下错步能量略离开 sink 维，远不及 coherence，仅一句话提及。

### 机制论断（逐项钉死,所有混淆排除）
> **首错步的 token 云方向更发散——这一步没有单一主导的相干方向。** 纯几何信号:
> ❌token变弱(mean_tok_norm 平) ❌massive(frac 近随机+resultant 免疫) ❌幅度方差(resultant 除尽幅度仍在)
> ❌长度(桶内 0.708) ❌输出熵(U_D 之上 +0.059 显著) ❌密度(残差化后仍在) → ✅**方向弥散**

### 叙事最终形态（CIM 逐步操作化）
- **最初假设"错误推理更弥散"**对，但**估计器错了**:有效秩 cloud_D 在 n≪d 下退化为长度（第 5 节证伪）。
- **正确估计器 = 方向相干度 `resultant`**（无量纲∈[0,1]、纯方向、构造上免疫幅度与 massive,审稿人挑不出毛病）;`coherence` 作伴随互证。
- **on/off 受约束流形的逐步操作化**:正确步 token 共享主导方向(on-manifold,相干);首错步方向发散(off-manifold,弥散)。
- 第 6 节"幅度(norm)"的措辞**修正**为"方向弥散,norm 只是被幅度/massive 污染的代理"。

**论文主特征 = `resultant`**;coherence 互证;norm 是发现脚手架。诚实保留:原始 0.77 大半是长度(corr −0.83),干净机制效应中等(增量 +0.06、桶内 0.71)。

## 8. 跨数据集 gate + R1 融合检测器 (2026-06-14, `fuse_detector.py`)

ProcessBench 四 config(olympiadbench 待补),Llama-3.1-8B teacher-forcing,步级过程标签。

### resultant 跨 config:过 gate,但难度梯度明显(4/4 config)
| config | 增量(over 混淆+U_D) | 桶内 L14 | raw L14 | 长度相关 |
|---|---|---|---|---|
| gsm8k(易) | **+0.059** [0.038,0.083] | **0.708** | 0.772 | −0.832 |
| math(中) | **+0.048** [0.035,0.061] | **0.620** | 0.703 | −0.837 |
| olympiad(难) | **+0.039** [0.027,0.049] | **0.588** | 0.703 | −0.843 |
| omnimath(难) | **+0.027** [0.019,0.035] | **0.569** | 0.702 | −0.843 |
- 四个**都显著、桶内都 >0.55** → gate 过。**信号随难度单调衰减**(桶内 0.708→0.620→0.588→0.569,四点一条线,贴近地板)。
- **论文写 difficulty-graded but uniformly significant,不能写 uniformly robust。** 机制假设:难题上正确步本身也更发散(gap 缩小)+ 难数据混淆更强(omnimath/olympiad baseline ~0.78-0.80、n_tok 单独 ~0.71)。

### R1 融合(GroupKFold-by-chain,混淆+熵进基线)
| config | baseline | best single | FUSED | FUSED−base | FUSED−single |
|---|---|---|---|---|---|
| gsm8k | 0.775 | coherence@14 0.822 | 0.847 | +0.073 | +0.026 显著 |
| math | 0.751 | norm@14 0.791 | 0.805 | +0.055 | +0.015 显著 |
| olympiad | 0.780 | cloud_D@10 0.783 | 0.818 | +0.038 | **+0.035 显著** |
| omnimath | 0.796 | cloud_D@10 0.799 | 0.835 | +0.039 | **+0.036 显著** |

**两个硬结论(决定 ML 路线)**:
1. **最佳单标量跨 config 不稳定**:coherence(gsm8k)/norm(math)/cloud_D(olympiad,omnimath) 跟着难度换 → **没有单一手工标量普适**。
2. **越难单标量越崩、越需结构**:两个难 config(olympiad/omnimath)best single 仅比 baseline +0.003(几乎死),FUSED 救回 ~+0.038,`FUSED−single` 在难 config 最大(+0.035/+0.036)。
- 配合 gsm8k 上 **GBM 全面 < logit**(非线性无效)→ **瓶颈在表征,不在分类器,也不在"选哪个标量"**。
- → **R2(token 云学习池化/富表征)被坐实,尤其为难数据。** 在手工标量上堆更大模型=浪费。

## 9. R2 富表征(学习池化)= 否定;检测-数字轴封顶 (2026-06-15)

`set_pool_detector.py` (DCSE: SAB 自注意力 + K 多原型浓度,JL 投影云 k=256,层 10/14,GroupKFold-by-chain)。
| config | baseline(nuis+U 的MLP) | +云 MEAN | +云 DCSE | DCSE−base |
|---|---|---|---|---|
| gsm8k | 0.795 | 0.833 | 0.843 | +0.048 显著;DCSE−mean +0.009 **ns** |
| omnimath(决定性) | **0.822** | 0.799 ❌ | 0.793 ❌ | **−0.029**(负) |
- **决定性测(omnimath,标量崩得最惨处)否定**:连 MEAN 池化都 < baseline → 不是 DCSE 过拟合,是**投影云没带来比 5 个混淆+熵特征更多的可用信号**。
- **手工标量 > 学习云**:fuse FUSED 0.835 > 学习云 0.793 → **标量是高效摘要,不是"太简单";富表征在此数据规模榨不出更多。池化瓶颈假设证伪。**
- 诚实:JL k=256 可能损方向信息,但标量在全 d=4096 上也仅 +0.04 封顶 → 云里大概率无更多。
- **结论:检测-AUROC 轴封顶(R1 融合 +0.02~0.04、R2 富表征 ≤0 都没捅破),信号结构性上限 ~0.6 干净 / ~0.83 含混淆。停止优化该数字。**

**方向转轴(不再优化检测数字)**:① 时间轴=前兆/早期预警(事件研究,现有数据);② 效用轴=选择性预测/best-of-N 重排;③ 机制轴=模块归因(attn vs FFN 注入弥散,GPU 钩子)。论文骨架候选:严格审计(eff-rank/norm 是长度/幅度伪影,方向弥散是唯一幸存真信号)+ 早期预警动力学。
