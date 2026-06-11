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
