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
- `id_mle/id_twonn` **待测**（需 `--intrinsic_dim` 重抽）。

### 步级定位（ProcessBench，gold 出错步）
- `cloud_D` loc_auroc **0.72**，但 `step_ntokens` 单独 **0.70** → **cloud_D = 步长伪装**，几何净增 +0.02（噪声内）。
- 其它：`norm` 0.28（出错步模长更低，翻转 0.72）、`anom_k10` 0.61、`U_D` 0.58。都弱。

### SPE 子空间泄漏（已搁置）
- 干净 within ~0.58–0.60，方向对（错误泄漏更多），但没打过 Mahalanobis；`meanSPE_cor` 在 k=100 仍 0.28 → 健康子空间高维，"低维流形"不成立。

### Tracing 式探针（合并 + slope/r²）
- `paper static 0.634`（最强）；`paper +dyn 0.631`（slope/r² **不加强**）；`ALL +dyn 0.612`（几何 **拖累**）。
- **几何相对 U_D/U_C 无增量。**

## 4. 当前判断
"错误推理占据更高维/更发散"在 population 层面**真且显著（Cohen d≈0.35）**，但**效应小**：难度+格式+长度控住后逐条 AUROC ~0.55。真正活着的强信号是 `U_D`（熵）和 Mahalanobis（位移），都是"不确定性/距离"，不是"维度"。

**两个未决实验**：(1) 长度鲁棒内在维数 `id_twonn/id_mle` 能否把逐条信号拉上去；(2) 维度 vs 熵的题内相关（冗余 or 互补）。两者出来即可定：几何收口 or 再战。
