# Trajectory of Thought: Geometric Phase Transitions in Multi-Step Reasoning

完整的轨迹几何相变检测框架，用于AAAI 2027论文投稿。

## 核心假设

**错误推理是几何轨迹的相变，不是单点异常**

- 正确推理：步骤间的表示空间连续、平滑演化，几何轨迹保持低维流形结构
- 错误推理：几何轨迹发生"相变"——要么断裂（信息流中断），要么扭曲（进入错误子空间）

## 三层架构

### Layer 1: Step-wise Geometry（每步的几何特征）

从真实hidden states计算：
- **标量特征**：κ（方向集中度）、eff_rank（有效秩）、spectral_entropy（谱熵）
- **几何特征**：principal_directions（主方向）、eigenvalues（特征值）

### Layer 2: Trajectory Geometry（步骤间几何关系）

真正的几何特征（基于特征向量）：
- `principal_direction_rotation()`：主方向旋转角度
- `subspace_drift()`：子空间漂移程度
- `projection_residual()`：投影残差

标量动态演化：
- `scalar_evolution_smoothness()`：标量演化平滑度
- `scalar_trend_consistency()`：标量趋势一致性

组合指标：
- `local_smoothness()`：相邻步骤的几何相似度
- `global_coherence()`：首尾步骤的关联强度
- `spectral_evolution_stability()`：谱形状演化稳定性

### Layer 3: Phase Transition Detection（相变检测）

- `detect_shallow_lockin()`：Shallow Lock-in（浅层信息流锁定）
- `detect_deep_decay()`：Deep Decay（深层信息衰减）

## 文件结构

```
demo/
├── data_loading.py          # 数据加载与几何特征计算
├── trajectory_geometry.py    # 轨迹几何分析（Layer 2）
├── phase_transition.py       # 相变检测（Layer 3）
├── validation.py             # H1-H4验证实验
├── main.py                   # 主入口
└── TRAJECTORY_README.md      # 本文档
```

## 使用方法

### 基本用法

```bash
python main.py \
    --npz_path /gz-data/research/demo/data/features/full_omnimath.npz \
    --hidden_dir /gz-data/research/demo/data/hidden/omnimath/ \
    --output_dir ./trajectory_results \
    --layers 14
```

### 完整参数

```bash
python main.py \
    --npz_path /gz-data/research/demo/data/features/full_omnimath.npz \
    --hidden_dir /gz-data/research/demo/data/hidden/omnimath/ \
    --output_dir ./trajectory_results \
    --layers 10 14 18 22 \
    --min_steps 3 \
    --n_bootstrap 5000 \
    --n_top_components 10 \
    --drop_threshold 0.15 \
    --stability_threshold 0.5 \
    --entropy_threshold 0.7 \
    --detect_method standard \
    --verbose
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--npz_path` | 必需 | full_*.npz文件路径 |
| `--hidden_dir` | 必需 | hidden shards目录路径 |
| `--output_dir` | ./trajectory_results | 输出目录 |
| `--layers` | [14] | 分析的层列表 |
| `--min_steps` | 3 | 最少步骤数过滤 |
| `--n_bootstrap` | 5000 | Bootstrap迭代次数 |
| `--n_top_components` | 10 | 计算的主成分数 |
| `--drop_threshold` | 0.15 | Shallow Lock-in下降阈值 |
| `--stability_threshold` | 0.5 | Deep Decay稳定性阈值 |
| `--entropy_threshold` | 0.7 | Deep Decay谱熵阈值 |
| `--detect_method` | standard | 检测方法（standard/adaptive） |
| `--skip_validation` | False | 跳过H1-H4验证 |

## 验证假设

### H1: 轨迹平滑度区分error vs correct
- 假设：错误推理的轨迹smoothness低于正确推理
- 检验：Mann-Whitney U test (one-tailed)
- 指标：Cohen's d, 95% CI

### H2: Shallow Lock-in在error中更频繁
- 假设：错误推理中检测到Shallow Lock-in的比例更高
- 检验：Fisher's exact test
- 指标：Odds Ratio

### H3: Deep Decay在error中更频繁
- 假设：错误推理中检测到Deep Decay的比例更高
- 检验：Fisher's exact test
- 指标：Odds Ratio

### H4: 轨迹检测器优于单步检测器
- 假设：轨迹级指标优于单步指标
- 检验：Bootstrap AUROC比较
- 指标：ΔAUROC, p-value

## 输出格式

### JSON结果文件

```json
{
  "L14": {
    "metadata": {
      "layer": 14,
      "n_trajectories": 998,
      "n_correct": 500,
      "n_error": 498
    },
    "lockin_stats": {
      "total": 998,
      "correct_detected": 50,
      "error_detected": 124,
      "correct_total": 500,
      "error_total": 498,
      "correct_rate": 0.10,
      "error_rate": 0.25
    },
    "decay_stats": {...}
  },
  "validation": {
    "L14_H1": {
      "hypothesis": "H1",
      "metric": "smoothness",
      "error_mean": 0.650,
      "correct_mean": 0.780,
      "cohens_d": 0.450,
      "p_value": 0.0001,
      "significant": true
    },
    ...
  }
}
```

### 控制台输出

```
================================================================================
Trajectory of Thought: Geometric Phase Transitions
================================================================================
...
================================================================================
VALIDATION SUMMARY
================================================================================
Test         Metric                    Error      Correct    Diff        Test             p-value    Sig?
----------------------------------------------------------------------------------------------------
L14_H1       smoothness               0.650      0.780      0.130       d=0.45           0.0001     *
L14_H2       shallow_lockin            25.0%      10.0%      15.0%       OR=3.12          0.0050     *
L14_H3       deep_decay               19.7%      11.0%      8.7%        OR=2.85          0.0080     *
L14_H4       trajectory_vs_stepwise  N/A        N/A        0.065       Δ=0.065          0.0230     *
----------------------------------------------------------------------------------------------------
* p < 0.05
================================================================================
```

## 数据要求

### NPZ文件
必须包含以下字段：
- `problem_ids`: 问题ID数组
- `is_correct_strict`: 正确性标签（**1=correct, 0=error**；写入端 extract_features._pb_record，真值锚点 gold_error_step<0 ⟺ correct。2026-07-03 前的文档/代码曾误记为 0=correct）
- `stepcloud`: (N, T, 33, 9) 步骤特征数组
- `step_token_ranges`: (N, T, 2) 每步的token范围

### Hidden Shards
目录结构：
```
/gz-data/research/demo/data/hidden/<subset>/
├── <subset>-0.npy
├── <subset>-1.npy
└── ...
```

文件格式：
- Shape: (R, 4, 4096)
- R: 响应中的token总数
- 4: layers [10, 14, 18, 22]
- 4096: hidden dimension

## 实现细节

### 真实的谱特征计算

```python
# 1. 提取token hidden states
H_i^l = hidden[start_i:end_i, layer_idx, :]  # (n_i, 4096)

# 2. 归一化每个token向量
û_j = h_j / ||h_j||

# 3. 计算Scatter Matrix（二阶矩）
S_i^l = (1/n_i) * Σ û_j ⊗ û_j

# 4. 特征值分解
λ_1 ≥ λ_2 ≥ ... ≥ λ_d

# 5. 提取特征
- κ_i^l = ||mean(û)||  ∈ [0,1]
- eff_rank_i^l = exp(-Σ λ_k log λ_k)
- spectrum_i^l = [λ_1, ..., λ_10]
```

### 几何相似度计算

```python
geometric_sim(g1, g2) =
    0.20 * κ_sim +
    0.15 * eff_rank_sim +
    0.25 * rotation_sim +
    0.20 * drift_sim +
    0.20 * spectrum_sim
```

## 依赖

- numpy
- scipy
- scikit-learn
- tqdm
- dataclasses（Python 3.7+）

## 参考文献

该框架结合了以下论文的思想：

1. **Spectral Geometry**：分析表示空间的谱结构
2. **Step Flow**：分析步骤间的信息流

我们的创新点：
- 将两篇论文的思想结合：谱方法 + 步骤演化
- 提出几何轨迹视角：推理是一条曲线，错误是曲线的"扭结"
- 无监督、可解释、步骤级的检测方法

---

**版本**: v1.0
**状态**: 实现完成，待实验验证
