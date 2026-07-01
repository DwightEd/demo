# OMniMath 结果分析和汇报指南

## `data_loading_gpu` 计算的是什么？

该脚本为每个 reasoning **step** 计算以下几何特征：

### 特征说明

| 特征 | 数学定义 | 物理意义 | 假设 |
|------|----------|----------|------|
| **kappa** | $\kappa = \|\frac{1}{n}\sum \hat{u}_i\|$ | 向量集中度（一阶矩） | 错误step的$\kappa$更低（更发散） |
| **eff_rank** | $r_{eff} = \exp(-\sum \lambda_i \log \lambda_i)$ | 有效秩（二阶矩） | 错误step的$r_{eff}$更高（更多维度活跃） |
| **spectral_entropy** | $H = -\sum \lambda_i \log \lambda_i$ | 谱熵 | 错误step的$H$更高（分布更均匀） |
| **eigenvalues** | $\lambda_1, \dots, \lambda_{10}$ | Scatter矩阵前10特征值 | 错误step衰减更慢 |

其中 $\hat{u}_i = h_i / \|h_i\|$ 是归一化token向量，$\lambda_i$ 是 Scatter矩阵 $S = \frac{1}{n}\hat{U}^T\hat{U}$ 的特征值。

## 如何运行分析

### 在远程服务器（数据所在位置）

```bash
cd /gz-data/research/demo/
python analyze_results.py
```

### 在本地Windows（如果数据已同步）

```bash
python analyze_results.py --cache-dir "F:/path/to/cache/omnimath" --npz-path "F:/path/to/full_omnimath.npz"
```

## 输出内容

### 1. 控制台报告

```
================================================================================
OMniMath 几何特征分析报告
================================================================================

数据集: omnimath
总轨迹数: XXXX
正确: XXXX, 错误: XXXX

--------------------------------------------------------------------------------
关键发现汇总
--------------------------------------------------------------------------------

【Kappa - 向量集中度】
假设: 错误step的kappa更低（向量更发散）
  Layer 10: ✓ 确认
    正确=0.XXXX, 错误=0.XXXX, d=X.XXX, p=0.XXXX
  ...

【Effective Rank - 有效秩】
假设: 错误step的eff_rank更高
  ...

【Spectral Entropy - 谱熵】
假设: 错误step的熵更高
  ...

--------------------------------------------------------------------------------
总体结论
--------------------------------------------------------------------------------

显著性结果 (p<0.05):
  Kappa: X/4 层显著
  Eff_Rank: X/4 层显著
  Entropy: X/4 层显著

大效应量 (|d| > 0.8):
  Kappa: X/4 层
  Eff_Rank: X/4 层
  Entropy: X/4 层
```

### 2. LaTeX表格（可直接复制到论文）

```latex
\begin{table}[h]
\centering
\caption{Step-level Kappa分布（正确 vs 错误）}
\label{tab:kappa_distribution}
...
\end{table}
```

### 3. JSON结果文件

```json
{
  "metadata": {...},
  "layer_results": [
    {
      "layer": 10,
      "kappa": {"cohens_d": ..., "p_value": ...},
      "eff_rank": {...},
      "entropy": {...}
    },
    ...
  ]
}
```

## 如何解读结果

### Cohen's d 效应量标准

- |d| < 0.2: 无意义
- 0.2 ≤ |d| < 0.5: 小效应
- 0.5 ≤ |d| < 0.8: 中效应
- |d| ≥ 0.8: **大效应**（强区分能力）

### p值显著性

- p < 0.05: 统计显著
- p < 0.01: 高度显著

### AUC（分类能力）

- AUC = 0.5: 无区分能力
- AUC > 0.6: 有一定区分能力
- AUC > 0.7: 较强区分能力
- AUC > 0.8: 强区分能力

## 假设验证清单

- [ ] **H1**: 错误step的kappa显著低于正确step
  - 检查: error_mean < correct_mean 且 p < 0.05

- [ ] **H2**: 错误step的eff_rank显著高于正确step
  - 检查: error_mean > correct_mean 且 p < 0.05

- [ ] **H3**: 错误step的spectral_entropy显著高于正确step
  - 检查: error_mean > correct_mean 且 p < 0.05

## 结果解读示例

### 场景1: 假设全部成立

```
✓ H1成立: kappa在所有层显著，d=1.2（大效应）
✓ H2成立: eff_rank在3/4层显著，d=0.9（大效应）
✓ H3成立: entropy在所有层显著，d=1.0（大效应）

结论: 几何特征能有效区分正确/错误step，支持manifold geometry假设
```

### 场景2: 部分假设成立

```
✓ H1成立: kappa显著，d=0.6（中效应）
✗ H2不成立: eff_rank无显著差异
✓ H3成立: entropy显著，d=0.7（中效应）

结论: 一阶矩（kappa）和谱熵有效，二阶矩（eff_rank）需进一步研究
```

### 场景3: 假设全部不成立

```
✗ 所有假设不成立: 特征无显著差异或方向反转

结论: 需要重新审视假设或检查数据质量
```
