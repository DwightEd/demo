# 特征计算方式 与 单标量检测表现

ProcessBench(gsm8k/math/olympiad/omnimath),Llama-3.1-8B,teacher-forcing,步级首错检测
(正=gold 首错步,负=正确链步+错链首错前步)。所有 AUROC 为**原始池化**(未控混淆),层 14 除非注明。

---

## 1. 特征家族与计算方式

记一步的 token 云 `H = [h_1,...,h_n]`,`h_t ∈ R^d`(d=4096),exp 池化权重
`w_t = e^{(t-1)/(n-1)} / Σ e^{·}`(后 token 更重,Lu et al. Eq.6),池化向量 `z = Σ w_t h_t`。

### A. 浓度/幅度家族(`features/geometry.py`, `utils/step_vector.py`)
| 特征 | 公式 | 测什么 |
|---|---|---|
| `norm` | `‖z‖ = ‖Σ w_t h_t‖` | 池化向量幅度(发现脚手架,d=−0.69 最强单特征) |
| `resultant` | `‖Σ w_t ĥ_t‖ ∈[0,1]`, `ĥ_t=h_t/‖h_t‖` | **纯方向浓度**(先单位化→免疫幅度、无量纲) |
| `coherence` | `‖z‖ / mean_t‖h_t‖` | 池化/逐token = 对齐(= resultant 被幅度方差污染的近似) |
| `mean_tok_norm` | `mean_t ‖h_t‖` | 逐 token 平均模长 |
| `resultant_bulk` | resultant 但先把 top-massive 维置 0 | 去 massive 后的方向浓度(测是否 massive 驱动) |
| `resultant_unif` | resultant 但权重均匀 | exp vs 均权(测加权是否重要) |
| `norm_bulk` | `‖z‖` 但 massive 维置 0 | bulk 幅度 |

→ **norm = resultant = coherence 是同一个信号**(corr 0.94–0.97,互相增量 ns;因 mean_tok_norm 近恒定 → norm ≈ const·resultant)。resultant 是其无量纲严格版。

### B. 谱/维度家族(`utils/spectral.py`,对 token 云 H 做 SVD,奇异值 σ_i)
| 特征 | 公式 | 测什么 |
|---|---|---|
| `cloud_D` | `exp(−Σ p_i log p_i)`, `p_i=σ_i²/Σσ²` | **有效秩**(秩 ≤ n_tok → **结构上=长度**) |
| `cloud_V` | `Σ σ_i²` | 谱能量 |
| `cloud_C` | `σ_1²/Σσ²` | 顶端集中度 |
| `pr` | `(Σ z_d²)² / Σ z_d⁴`(池化向量) | Rényi α=2 有效维(峰主导) |
| `ae` | `exp(−Σ p_d log p_d)`, `p_d=z_d²/‖z‖²` | Rényi α=1 有效维(池化向量) |

### C. 不确定性家族(`features/uncertainty.py`,逐 token,步内平均)
| 特征 | 公式 | 测什么 |
|---|---|---|
| `U_D` | `−Σ_v p(v) log p(v)` | 全词表预测熵(分布不确定性) |
| `U_C` | `p(v_t)·(1−p(v_t))` | 委身不确定性(实际下一 token 的伯努利方差) |

### D. 混淆变量
| 特征 | 计算 | 测什么 |
|---|---|---|
| `n_tok` | 步内 token 数 | **长度**(错步更长:err 73.7 vs good 49.4) |
| `pos` | `j/(T−1)` | 步在链中的位置 |
| `density` | 非字母字符占比 | 公式/数字密度 |

---

## 2. 单标量特征检测表现(原始池化 AUROC)

| 特征 | gsm8k | omnimath | 真信号? |
|---|---|---|---|
| **coherence** | **0.778** | 0.706 | ✅ 浓度(三名一物) |
| **norm** | 0.775 | 0.698 | ✅ 同上 |
| **resultant** | 0.772 | 0.702 | ✅ 同上(严格版) |
| `resultant_bulk`(去massive) | 0.752 | — | ✅ 去 massive 仍在 |
| `norm_bulk` | 0.744 | — | ✅ |
| **cloud_D** | 0.714 | **0.732** | ❌ = 长度(见 §3) |
| **n_tok(长度)** | 0.708 | **0.717** | ⚠️ 真但是混淆 |
| `ae`/`pr`(逐token池化) | 0.717 | 0.604 | ❌ 同轴/长度 |
| `U_C`(委身) | 0.605 | 0.664 | ✅ 不确定性(链间) |
| `U_D`(熵) | 0.619 | 0.648 | ✅ 不确定性(链间) |
| `ae`/`pr`(池化向量) | ~0.58 | ~0.52 | ❌ 死 |
| `massive_frac` | 0.577 | — | ❌ 近随机 |
| `mean_tok_norm` | 0.573 | 0.573 | ❌ 个体不弱(锚点反证) |
| **融合(全部, logistic)** | **0.847** | **0.835** | 最强(但含长度) |

(历史链级 Mahalanobis:0.657 答案口径 / 0.83 strict 口径,不同管线。)

---

## 3. 长度混淆:谁是长度影子

`corr(特征, log n_tok)`(Spearman)+ 控长度后还剩多少:

| 特征 | corr(·, 长度) | 桶内/链内去长度 | 控长度后判决 |
|---|---|---|---|
| **cloud_D** | **+0.99** | 链内⊥长度 = **0.51(随机)** | **= 长度**(秩≤n_tok,数学卡死)。死 |
| resultant | −0.83 | 桶内 0.708 / 链内⊥nt **0.574** | 有长度之外真成分。活 |
| norm | −0.71 | 桶内 0.707 | 活 |
| coherence | −0.83 | 桶内 0.717 | 活 |

**关键事实(回答"正确长步有效秩也高吗")**:
有效秩只看 token 数、**不分对错** → 长的正确步和长的错误步有效秩**一样高** → 固定长度后 cloud_D ≈ 随机(0.51)。**cloud_D 区分对错完全靠"错步恰好更长",不是几何。**
反之 resultant 先单位化(免疫长度的秩天花板),固定长度后仍 0.57–0.71 → 真几何。

**几何信号在 [长度+位置+密度+U_D+U_C] 之上的真增量**(resid_audit):
| 特征 | 增量(gsm8k) |
|---|---|
| resultant | **+0.059 [.038,.083] 显著** |
| norm | +0.057 显著 |
| cloud_D | +0.021(≈长度,机制撑不起) |

---

## 4. 结论
- **真几何信号只有一个**:方向浓度(`norm`=`resultant`=`coherence`),bulk、非 massive、⊥ 不确定性,单打 ~0.77、控长度后真增量 +0.057;
- **cloud_D / pr / ae** = 长度(秩≤n_tok)或同轴冗余,死;
- **n_tok 长度** 本身是真预测子(错步更长)但是混淆,不是几何;
- **融合 0.847** 最强,但大头是长度——做检测器可用,做"几何"论断必须控长度。
