# 动态监测 vs 静态分析说明

## 两种分析方法对比

### 1. 静态分析 (`analyze_results.py`)

**目的**: 统计正确vs错误的特征分布差异

**分析方法**:
- 收集所有正确step的特征值（kappa, eff_rank, entropy）
- 收集所有错误step的特征值
- 计算Cohen's d（效应量）、p值（显著性）、AUC

**输出**:
```
Layer 14:
  正确steps: 1500, 错误steps: 500
  Kappa: 正确=0.823, 错误=0.651, d=1.2**, p<0.001
  Eff_Rank: 正确=2.3, 错误=3.8, d=0.9**, p<0.001
```

**局限**: 只看整体分布，没有考虑时间序列特性

---

### 2. 动态分析 (`analyze_dynamics.py`)

**目的**: 分析推理过程中的特征变化，实现在线监测

**分析方法**:
1. **因果z-score**: 只用历史数据计算 `z[t] = (s[t] - mean(s[:t])) / std(s[:t])`
2. **相变检测**: 检测z-score超过阈值（|z| > 2）的时间点
3. **错误预测**: 分析gold_error_step前的特征变化

**输出**:
```
动态监测分析报告
==================

Kappa动态分析:
  正确轨迹平均触发次数: 0.5
  错误轨迹平均触发次数: 2.3
  ✓ 错误轨迹显著更多相变

错误预测分析:
  错误step前kappa平均变化: -0.152
  kappa下降比例: 78.3%
  ✓ 大多数错误前有kappa下降信号

在线监测示例:
  Step    Kappa        Z-Score      Status
  0       0.823        nan          Normal
  1       0.791        -0.32        Normal
  2       0.652        -1.82        ⚠️ PHASE CHANGE
  3       0.521        -2.45        ⚠️ PHASE CHANGE
  4       0.489        -2.91        ❌ ERROR STEP

建议阈值: z < -1.5 (kappa下降)
提前预警: 2 steps
```

---

## 核心差异

| 维度 | 静态分析 | 动态分析 |
|------|----------|----------|
| 数据视角 | 看所有steps的分布 | 看每条轨迹的时序 |
| 关键问题 | "错误step的kappa是否更低？" | "错误前kappa是否下降？" |
| 约束 | 无因果约束 | 因果约束（只用历史） |
| 应用场景 | 离线评估、论文统计 | 在线监测、实时预警 |
| 输出形式 | Cohen's d, p值 | z-score轨迹、触发阈值 |

---

## 在线监测原理

### 因果z-score

```python
def causal_z_score(seq):
    """只使用历史数据，满足online约束"""
    z = np.full(len(seq), np.nan)
    for t in range(2, len(seq)):
        history = seq[:t]  # 只用t之前的数据
        z[t] = (seq[t] - history.mean()) / history.std()
    return z
```

### 触发规则

```python
# kappa下降触发
if causal_z_score(kappa_seq)[t] < -1.5:
    trigger_warning()  # 检测到几何相变

# 响应
if warning_triggered:
    intervene()  # 干预：重采样、压缩路径等
```

---

## 现有的动态监测实现

| 文件 | 方法 | 粒度 | 触发信号 |
|------|------|------|----------|
| `resp_cusum.py` | CUSUM | response-level | EMA-kappa偏离baseline |
| `online_intervene.py` | resultant/causal_z | step-level | ||exp-pooled unit vectors|| |
| `analyze_dynamics.py` | z-score/phase | step-level | kappa/eff_rank/entropy变化 |

---

## 如何运行

```bash
cd /gz-data/research/demo/
git pull origin main

# 静态分析
python analyze_results.py

# 动态分析
python analyze_dynamics.py

# 或一键运行
bash run_analysis.sh
```
