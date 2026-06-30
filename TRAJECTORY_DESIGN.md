# 方向2完整实现：Cross-Step Geometric Coherence - 轨迹几何相变检测

## 📋 文档信息

- **Title**: "Trajectory of Thought: Geometric Phase Transitions in Multi-Step Reasoning"
- **目标会议**: AAAI 2027
- **创建日期**: 2026-06-30
- **状态**: 设计阶段，待实现

---

## 🎯 核心研究假设

> **错误推理是几何轨迹的相变，不是单点异常**

### 假设阐述

- **正确推理**：步骤间的表示空间连续、平滑演化，几何轨迹保持低维流形结构
- **错误推理**：几何轨迹发生"相变"——要么断裂（信息流中断），要么扭曲（进入错误子空间）

### 关键洞察

1. 单步骤的κ/eff_rank无法区分"合法发散"vs"错误发散"
2. 但步骤间的几何轨迹形状可以区分：
   - 正确推理 = 平滑曲线（渐变）
   - 错误推理 = 突变/分叉（相变）

---

## 📚 与两篇论文的定位对话

| 维度 | Spectral Geometry | Step Flow | 我们的工作 |
|------|------------------|-----------|-----------|
| 分析单元 | 单点（整个回答的谱） | 步骤间的注意力流 | 步骤间的几何轨迹 |
| 核心发现 | 谱分布预测正确性 | 信息流断裂模式 | 几何轨迹的平滑性=推理健康度 |
| 方法 | 谱统计 | 梯度加权注意力 | 轨迹的几何拓扑分析 |
| 局限 | 不看步骤演化 | 需要backward pass | 无监督、可解释、步骤级 |

### Our Novelty

1. 将两篇论文的思想结合：谱方法（Spectral Geometry）+ 步骤演化（Step Flow）
2. 提出几何轨迹视角：推理是一条曲线，错误是曲线的"扭结"
3. 无监督、可解释、步骤级的检测方法

---

## 🏗️ 方法架构：三层分析（完整版）

### Layer 1: Step-wise Geometry (每步的几何特征)

#### 1.1 真正的几何信息（基于特征向量）

**关键洞察**：特征值（λ）描述散布强度，特征向量（V）描述主方向。**轨迹是主方向的演化**。

```
对于步骤i、层l：

Scatter Matrix: S_i^l = (1/n_i) Σ û_j ⊗ û_j

特征分解: S_i^l = V_i^l Λ_i^l (V_i^l)^T
  - Λ_i^l = diag(λ_1, ..., λ_d): 特征值（描述强度）
  - V_i^l = [v_1, ..., v_d]: 特征向量（描述方向）
```

**几何特征（基于特征向量）**：
```python
# 主成分方向
principal_directions_i = V_i[:, :k]  # 前k个主成分向量

# 方向集中度（κ）
kappa_i = ||mean(û_i)||

# 有效秩
eff_rank_i = exp(-Σ λ_j log λ_j)
```

#### 1.2 标量特征的动态演化

即使标量本身是冗余的，它们的**动态变化模式**是有价值的：

```python
# 标量序列（随步骤演化）
kappa_sequence = [κ_0, κ_1, κ_2, ...]        # 方向集中度的演化
eff_rank_sequence = [r_0, r_1, r_2, ...]    # 有效秩的演化
entropy_sequence = [H_0, H_1, H_2, ...]     # 谱熵的演化
norm_sequence = [||μ_0||, ||μ_1||, ...]       # 质心范数的演化
```

**动态变化的统计量**：
- 均值、方差、最小值、最大值
- 变化率：Δ_i = value_{i+1} - value_i
- 累积变化：total_change = |value_last - value_first|

#### 1.3 完整的数据结构

```python
@dataclass
class StepGeometry:
    """单步骤的完整几何描述"""
    step_id: int
    layer: int
    n_tokens: int

    # === 标量特征 ===
    kappa: float              # 方向集中度
    eff_rank: float           # 有效秩
    spectral_entropy: float   # 谱熵
    norm: float               # 质心范数

    # === 真正的几何特征（特征向量）===
    principal_directions: np.ndarray  # (d, k) 前k个主成分向量
    eigenvalues: np.ndarray          # (d,) 所有特征值
    scatter_matrix: np.ndarray       # (d, d) scatter matrix
```

---

#### 数据来源
```
Hidden shards: /gz-data/research/demo/data/hidden/<subset>/<chain_id>.npy
格式: (R, 4, 4096)
  - R: 响应中的token总数
  - 4: layers [10, 14, 18, 22]
  - 4096: hidden dimension

Step token ranges: step_token_ranges[i] = [(start_1, end_1), (start_2, end_2), ...]
```

#### 特征计算流程

对于步骤i、层l：

```
1. 提取token hidden states:
   H_i^l = hidden[start_i:end_i, layer_idx, :]  # (n_i, 4096)

2. 归一化每个token向量:
   û_j = h_j / ||h_j||,  j = 1, ..., n_i

3. 计算Scatter Matrix (二阶矩):
   S_i^l = (1/n_i) * Σ_{j=1}^{n_i} û_j ⊗ û_j  # (4096, 4096)

4. 特征值分解:
   λ_1 ≥ λ_2 ≥ ... ≥ λ_d,  Σ λ_k = 1

5. 提取几何特征:
   - κ_i^l = ||mean(û)||  ∈ [0,1]  (方向集中度，一阶矩)
   - eff_rank_i^l = exp(-Σ λ_k log λ_k)  ∈ [1, n_i]  (有效秩，二阶矩)
   - spectrum_i^l = [λ_1, ..., λ_10]  (前10个特征值)
   - spectral_entropy_i^l = -Σ λ_k log λ_k  (谱熵)
```

#### 数据结构
```python
@dataclass
class StepGeometry:
    """单个步骤在单层的完整几何描述"""
    step_id: int           # 步骤索引
    layer: int             # 层索引
    n_tokens: int          # 该步骤的token数

    # 一阶矩
    kappa: float            # 方向集中度

    # 二阶矩
    eff_rank: float         # 有效秩
    spectrum: np.ndarray    # 谱分布 (前10个特征值)
    spectral_entropy: float # 谱熵

    # 辅助
    norm: float            # 平均范数
```

---

### Layer 2: Trajectory Geometry (步骤间几何关系)

#### 2.1 真正的几何轨迹分析（基于特征向量）

##### 指标1: Principal Direction Rotation（主方向旋转）

**核心思想**：错误推理中，主成分方向会突然旋转。

```python
def principal_direction_rotation(g1: StepGeometry, g2: StepGeometry, k: int = 5) -> float:
    """计算主成分方向的旋转角度

    Returns:
        平均旋转角度（弧度）
    """
    V1 = g1.principal_directions[:, :k]  # (d, k)
    V2 = g2.principal_directions[:, :k]  # (d, k)

    # 计算每个主成分的旋转角度
    angles = []
    for i in range(k):
        v1_i = V1[:, i]
        v2_i = V2[:, i]
        cos_theta = np.dot(v1_i, v2_i)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        angles.append(theta)

    return np.mean(angles)  # 平均旋转角度
```

##### 指标2: Subspace Drift（子空间漂移）

**核心思想**：错误推理中，步骤间的子空间夹角会突然增大。

```python
def subspace_drift(g1: StepGeometry, g2: StepGeometry, k: int = 5) -> float:
    """计算子空间之间的漂移程度

    使用：主子空间之间的最大夹角
    """
    V1 = g1.principal_directions[:, :k]
    V2 = g2.principal_directions[:, :k]

    # 计算子空间之间的角度（使用Grassmann manifold距离）
    # 简化：计算两个投影矩阵的Frobenius范数距离
    P1 = V1 @ V1.T  # (d, d) 投影矩阵
    P2 = V2 @ V2.T

    drift = np.linalg.norm(P1 - P2, ord='fro')
    return drift
```

##### 指标3: Projection Residual（投影残差）

**核心思想**：错误推理中，点云到主子空间的距离会增大（信息丢失）。

```python
def projection_residual(geometry: StepGeometry, k: int = 5) -> float:
    """计算点云到主子空间的投影残差

    残差 = ||I - P_k|| * H_i，其中P_k是前k个主成分的投影矩阵
    """
    V = geometry.principal_directions[:, :k]
    P = V @ V.T
    I = np.eye(V.shape[0])
    residual = np.linalg.norm(I - P, ord='fro')

    return residual
```

#### 2.2 标量动态演化分析

##### 指标4: Scalar Evolution Smoothness（标量演化平滑度）

```python
def scalar_evolution_smoothness(scalar_sequence: List[float]) -> Tuple[float, np.ndarray]:
    """标量序列的平滑度

    检测标量的演化是否平滑
    """
    if len(scalar_sequence) < 2:
        return np.nan, np.array([])

    diffs = np.diff(scalar_sequence)
    smoothness = 1.0 / (1.0 + np.var(diffs))

    return smoothness, diffs
```

##### 指标5: Scalar Trend Consistency（标量趋势一致性）

```python
def scalar_trend_consistency(scalar_sequence: List[float]) -> float:
    """标量序列的趋势一致性

    正确推理：单调趋势（如κ递增）
    错误推理：趋势被打断
    """
    if len(scalar_sequence) < 3:
        return np.nan

    # 拟合线性趋势
    x = np.arange(len(scalar_sequence))
    slope, _ = np.polyfit(x, scalar_sequence, 1)

    # 计算R²
    y_pred = slope * x + np.mean(scalar_sequence)
    ss_res = np.sum((scalar_sequence - y_pred) ** 2)
    ss_tot = np.sum((scalar_sequence - np.mean(scalar_sequence)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return r2
```

#### 2.3 组合相似度（更新版）

```python
def geometric_sim(g1: StepGeometry, g2: StepGeometry) -> float:
    """组合多个几何维度的相似度

    包括：
    - 标量相似：κ, eff_rank
    - 几何相似：主方向旋转、子空间漂移
    - 谱形状相似：JS散度
    """
    # 标量部分
    kappa_sim = 1.0 - min(abs(g1.kappa - g2.kappa), 1.0)
    eff_rank_sim = 1.0 - abs(g1.eff_rank - g2.eff_rank) / max(g1.eff_rank, g2.eff_rank, 1.0)

    # 几何部分
    rotation_sim = 1.0 - (2/np.pi) * principal_direction_rotation(g1, g2)
    drift_sim = 1.0 / (1.0 + subspace_drift(g1, g2))

    # 谱形状
    spectrum_sim = 1.0 - jensenshannon(g1.eigenvalues[:10], g2.eigenvalues[:10])

    return (0.2 * kappa_sim + 0.15 * eff_rank_sim +
            0.25 * rotation_sim + 0.2 * drift_sim +
            0.2 * spectrum_sim)
```

#### 2.4 核心指标总结

| 指标 | 类型 | 检测内容 | 预期（正确推理） |
|------|------|----------|----------------|
| Principal Direction Rotation | 几何 | 主方向旋转角度 | 小（平滑） |
| Subspace Drift | 几何 | 子空间漂移程度 | 小（稳定） |
| Projection Residual | 几何 | 信息丢失程度 | 低（信息保留） |
| κ Evolution Smoothness | 标量动态 | κ演化平滑度 | 高（单调递增） |
| eff_rank Trend | 标量动态 | eff_rank趋势 | 一致（无突变） |

---

#### 2.1 几何相似度计算

```python
def geometric_sim(g1: StepGeometry, g2: StepGeometry,
                   kappa_weight: float = 0.4,
                   eff_rank_weight: float = 0.3,
                   spectrum_weight: float = 0.3) -> float:
    """计算两个步骤之间的几何相似度

    组合三个维度：
    - kappa_sim: 方向一致性（1 - |κ1 - κ2|）
    - eff_rank_sim: 分散程度相似性
    - spectrum_sim: 谱形状相似性（1 - JS散度）
    """
    # kappa相似度
    kappa_sim = 1.0 - min(abs(g1.kappa - g2.kappa), 1.0)

    # eff_rank相似度
    max_er = max(g1.eff_rank, g2.eff_rank, 1.0)
    eff_rank_sim = 1.0 - abs(g1.eff_rank - g2.eff_rank) / max_er

    # spectrum相似度
    s1 = g1.spectrum / (g1.spectrum.sum() + 1e-12)
    s2 = g2.spectrum / (g2.spectrum.sum() + 1e-12)
    spectrum_sim = 1.0 - jensenshannon(s1, s2)

    return (kappa_weight * kappa_sim +
            eff_rank_weight * eff_rank_sim +
            spectrum_weight * spectrum_sim)
```

#### 2.2 三个核心指标

##### 指标1: Local Smoothness（局部平滑度）

```python
def local_smoothness(geometry_sequence: List[StepGeometry]) -> Tuple[float, np.ndarray]:
    """相邻步骤的几何相似度

    正确推理：smoothness高（渐进变化）
    错误推理：smoothness低（突变）

    Returns:
        (全局平滑度, 每个transition的平滑度序列)
    """
    if len(geometry_sequence) < 2:
        return np.nan, np.array([])

    smoothness_values = []
    for i in range(len(geometry_sequence) - 1):
        sim = geometric_sim(geometry_sequence[i], geometry_sequence[i + 1])
        smoothness_values.append(sim)

    return float(np.mean(smoothness_values)), np.array(smoothness_values)
```

##### 指标2: Global Coherence（全局连贯性）

```python
def global_coherence(geometry_sequence: List[StepGeometry],
                     n_early: int = 3,
                     n_late: int = 3) -> float:
    """首尾步骤的关联强度

    正确推理：首尾连贯（始终围绕问题展开）
    错误推理：首尾断裂（后期遗忘问题）

    关键洞察：后期步骤是否还"记住"早期推理
    """
    if len(geometry_sequence) < max(n_early, n_late):
        return np.nan

    # 早期步骤的平均谱
    early_spectra = [g.spectrum for g in geometry_sequence[:n_early] if g.spectrum.size > 0]
    early_spectrum = np.mean(early_spectra, axis=0)
    early_spectrum = early_spectrum / (early_spectrum.sum() + 1e-12)

    # 后期步骤的平均谱
    late_spectra = [g.spectrum for g in geometry_sequence[-n_late:] if g.spectrum.size > 0]
    late_spectrum = np.mean(late_spectra, axis=0)
    late_spectrum = late_spectrum / (late_spectrum.sum() + 1e-12)

    # 连贯度 = 1 - JS散度
    return 1.0 - jensenshannon(early_spectrum, late_spectrum)
```

##### 指标3: Spectral Evolution Stability（谱演化稳定性）

```python
def spectral_evolution_stability(geometry_sequence: List[StepGeometry]) -> float:
    """谱形状沿步骤的演化稳定性

    正确推理：谱形状渐进演化（稳定）
    错误推理：谱形状跳变（不稳定）

    计算方法：
    1. 计算相邻谱的"变化率"：Δ_i = ||spectrum_{i+1} - spectrum_i||
    2. 稳定性 = 1 / (1 + var(Δ))
    """
    if len(geometry_sequence) < 3:
        return np.nan

    spectra = [g.spectrum for g in geometry_sequence if g.spectrum.size > 0]
    if len(spectra) < 3:
        return np.nan

    spectra = np.array(spectra)

    # 计算相邻谱的变化率
    diffs = np.diff(spectra, axis=0)           # (T-1, 10)
    diff_magnitudes = np.linalg.norm(diffs, axis=1)  # (T-1,)

    # 稳定性 = 1 / (1 + 变化率的方差)
    variance = np.var(diff_magnitudes)
    return 1.0 / (1.0 + variance)
```

---

### Layer 3: Phase Transition Detection (相变检测)

#### 3.1 Shallow Lock-in 检测（浅层信息流锁定）

**假设**：在浅层（L10, L14, L18），错误推理会出现步骤间的几何相似度突然集中在当前步骤（自强化）

```python
def detect_shallow_lockin_trajectory(trajectory: ReasoningTrajectory,
                                     shallow_layers: List[int] = [10, 14, 18],
                                     drop_threshold: float = 0.15,
                                     min_window: int = 2) -> Dict:
    """检测Shallow Lock-in：浅层平滑度突降

    检测逻辑：
    1. 计算coherence profile：每一步与之前所有步骤的平均相似度
    2. 检测是否有突然下降（> threshold）

    Returns:
        {
            'detected': bool,
            'layer': int or None,
            'lockin_step': int,  # 发生lock-in的步骤
            'drop_magnitude': float,  # 下降幅度
            'coherence_before': float,
            'coherence_after': float,
        }
    """
    for layer in shallow_layers:
        geom_sequence = trajectory.get_geometry_sequence(layer)
        if len(geom_sequence) < 3:
            continue

        # 计算coherence profile
        coherence_profile = []
        for i in range(1, len(geom_sequence)):
            sim_to_past = [geometric_sim(geom_sequence[i], geom_sequence[j])
                          for j in range(i)]
            coherence_profile.append(np.mean(sim_to_past))

        # 检测突然下降
        for i in range(min_window, len(coherence_profile) - min_window):
            before = np.mean(coherence_profile[i-min_window:i])
            after = np.mean(coherence_profile[i:i+min_window])

            if before - after > drop_threshold:
                return {
                    'detected': True,
                    'layer': layer,
                    'lockin_step': i + 1,
                    'drop_magnitude': before - after,
                    'coherence_before': before,
                    'coherence_after': after,
                }

    return {'detected': False, 'layer': None, 'lockin_step': -1}
```

#### 3.2 Deep Decay 检测（深层信息衰减）

**假设**：在深层（L18, L22），错误推理会出现谱演化失稳 + 晚期谱熵低（信息丢失）

```python
def detect_deep_decay_trajectory(trajectory: ReasoningTrajectory,
                                 deep_layers: List[int] = [18, 22],
                                 stability_threshold: float = 0.5,
                                 entropy_threshold: float = 0.7) -> Dict:
    """检测Deep Decay：深层谱演化失稳

    检测逻辑：
    1. 计算谱演化稳定性
    2. 计算后期步骤的平均谱熵
    3. 检测：stability < threshold AND late_entropy < threshold

    Returns:
        {
            'detected': bool,
            'layer': int or None,
            'stability': float,
            'late_entropy': float,
        }
    """
    for layer in deep_layers:
        geom_sequence = trajectory.get_geometry_sequence(layer)
        if len(geom_sequence) < 4:
            continue

        # 计算谱演化稳定性
        stability = spectral_evolution_stability(geom_sequence)
        if not np.isfinite(stability):
            continue

        # 计算后期步骤的平均谱熵
        late_spectra = [g.spectrum for g in geom_sequence[-3:] if g.spectrum.size > 0]
        if not late_spectra:
            continue
        late_entropy = np.mean([spectral_entropy(s) for s in late_spectra])

        # Decay信号：稳定性低 AND 晚期谱熵低（信息丢失）
        if stability < stability_threshold and late_entropy < entropy_threshold:
            return {
                'detected': True,
                'layer': layer,
                'stability': stability,
                'late_entropy': late_entropy,
            }

    return {'detected': False, 'layer': None, 'stability': np.nan, 'late_entropy': np.nan}
```

---

## 🔬 验证实验设计（H1-H4）

### H1: 轨迹平滑度区分error vs correct

**假设**: 错误推理的轨迹smoothness低于正确推理

**检验方法**: Mann-Whitney U test (one-tailed: error < correct)

**预期结果**:
- mean(smoothness_error) < mean(smoothness_correct)
- Cohen's d > 0.3 (中效应)
- p < 0.001

**Bootstrap置信区间**: 95% CI for mean difference

```python
def run_validation_h1(trajectories: List[ReasoningTrajectory],
                      layer: int = 14,
                      n_bootstrap: int = 5000) -> ValidationResult:
    """验证H1"""
    # 1. 计算每条链的smoothness
    correct_smoothness = [compute_trajectory_metrics(t, layer)['smoothness']
                           for t in trajectories if t.is_correct]
    error_smoothness = [compute_trajectory_metrics(t, layer)['smoothness']
                        for t in trajectories if not t.is_correct]

    # 2. Mann-Whitney U test
    stat, pval = stats.mannwhitneyu(error_smoothness, correct_smoothness,
                                     alternative='less')

    # 3. Cohen's d
    cohens_d = (mean(correct) - mean(error)) / pooled_std

    # 4. Bootstrap CI
    mean_diff, ci_low, ci_high = bootstrap_mean_diff(correct, error, n_bootstrap)

    return ValidationResult(...)
```

### H2: Shallow Lock-in模式在error中更频繁

**假设**: 错误推理中检测到Shallow Lock-in的比例更高

**检验方法**: Fisher's exact test (alternative='greater')

**预期结果**:
- lockin_error / n_error > lockin_correct / n_correct
- Odds Ratio > 2
- p < 0.01

### H3: Deep Decay模式在error中更频繁

**假设**: 错误推理中检测到Deep Decay的比例更高

**检验方法**: Fisher's exact test (alternative='greater')

**预期结果**:
- decay_error / n_error > decay_correct / n_correct
- Odds Ratio > 2
- p < 0.01

### H4: 基于轨迹的检测器优于基于单步几何的检测器

**假设**: 轨迹级指标（smoothness + coherence + stability）优于单步指标（mean kappa）

**检验方法**: Paired t-test on per-chain AUROC 或 Bootstrap比较

**预期结果**:
- AUROC_trajectory > AUROC_stepwise
- Δ > 0.05 (有意义的提升)
- p < 0.05

---

## 📁 实现文件结构

### 需要删除的文件（有错误或简化）
```
❌ trajectory_phase_transition.py   # 使用构造的spectrum
❌ analyze_auroc.py                  # 有语法错误
❌ diagnose_detection.py             # 诊断脚本
❌ simple_auroc.py                   # 简化版
❌ check_raw_auroc.py                # 原始特征检查
❌ run_phase_transition.sh           # 旧的执行脚本
❌ TRAJECTORY_PHASE_README.md        # 旧的文档
```

### 需要创建的文件
```
✅ data_loading.py                 # 数据加载与特征计算
✅ step_geometry.py                # Layer 1: Step-wise Geometry
✅ trajectory_geometry.py          # Layer 2: Trajectory Geometry
✅ phase_transition.py             # Layer 3: Phase Transition Detection
✅ validation.py                    # 验证实验 (H1-H4)
✅ main.py                         # 主入口
✅ README.md                       # 使用说明
```

---

## 📝 实现详细步骤

### 步骤1: data_loading.py

**功能**: 加载数据并计算每步的真实几何特征

**类和函数**:
```python
class StepGeometry:
    """单步骤几何特征数据类"""
    step_id: int
    layer: int
    n_tokens: int
    kappa: float
    eff_rank: float
    spectrum: np.ndarray      # (10,)
    spectral_entropy: float

class ReasoningTrajectory:
    """完整推理链"""
    chain_id: int
    problem_id: int
    is_correct: bool
    steps: Dict[int, Dict[int, StepGeometry]]  # {layer: {step: StepGeometry}}

def load_hidden_shard(npz_path: str, chain_id: int, hidden_dir: str) -> np.ndarray:
    """加载单个hidden shard
    
    Args:
        npz_path: NPZ文件路径
        chain_id: 链ID
        hidden_dir: hidden目录路径
    
    Returns:
        hidden: (R, 4, 4096) array or None
    """

def compute_step_geometry(hidden: np.ndarray,
                          step_range: Tuple[int, int],
                          layer_idx: int,
                          layer_id: int) -> StepGeometry:
    """从hidden states计算步骤几何特征
    
    Args:
        hidden: (R, 4, 4096) hidden states
        step_range: (start, end) token范围
        layer_idx: 层索引（在4层中）
        layer_id: 层ID（实际层号）
    
    Returns:
        StepGeometry对象
    """

def load_all_trajectories(npz_path: str,
                         hidden_dir: str) -> Tuple[List[ReasoningTrajectory], Dict]:
    """加载所有推理链并计算几何特征
    
    Args:
        npz_path: full_*.npz路径
        hidden_dir: hidden目录路径
    
    Returns:
        (trajectories, metadata)
    """
```

**测试代码**:
```python
# Test: 加载单个链
trajs, meta = load_all_trajectories("data/features/full_omnimath.npz",
                                    "data/hidden/omnimath/")
print(f"Loaded {len(trajs)} chains")
print(f"First chain has {len(trajs[0].steps)} steps")

# Test: 检查第一步L14的几何特征
step_geom = trajs[0].get_step_geometry(step_id=0, layer=14)
print(f"κ = {step_geom.kappa:.3f}")
print(f"eff_rank = {step_geom.eff_rank:.2f}")
print(f"spectrum = {step_geom.spectrum}")
```

---

### 步骤2: step_geometry.py

**功能**: Layer 1的数据结构定义（已在data_loading.py中）

**无需额外文件**，数据结构定义整合在data_loading.py中

---

### 步骤3: trajectory_geometry.py

**功能**: Layer 2的三个核心指标

**函数签名**:
```python
def geometric_sim(g1: StepGeometry, g2: StepGeometry,
                   kappa_weight: float = 0.4,
                   eff_rank_weight: float = 0.3,
                   spectrum_weight: float = 0.3) -> float:
    """计算两个步骤的几何相似度"""

def local_smoothness(geometry_sequence: List[StepGeometry]) -> Tuple[float, np.ndarray]:
    """局部平滑度: (全局, 序列)"""

def global_coherence(geometry_sequence: List[StepGeometry],
                     n_early: int = 3,
                     n_late: int = 3) -> float:
    """全局连贯度"""

def spectral_evolution_stability(geometry_sequence: List[StepGeometry]) -> float:
    """谱演化稳定性"""

def compute_trajectory_metrics(trajectory: ReasoningTrajectory,
                               layer: int) -> Dict[str, float]:
    """计算单个轨迹的所有指标
    
    Returns:
        {
            'smoothness': float,
            'coherence': float,
            'stability': float,
            'n_steps': int,
        }
    """
```

**测试代码**:
```python
# Test: 计算单链的指标
from data_loading import load_all_trajectories
from trajectory_geometry import compute_trajectory_metrics

trajs, _ = load_all_trajectories(...)
metrics = compute_trajectory_metrics(trajs[0], layer=14)
print(f"Smoothness: {metrics['smoothness']:.3f}")
print(f"Coherence: {metrics['coherence']:.3f}")
print(f"Stability: {metrics['stability']:.3f}")
```

---

### 步骤4: phase_transition.py

**功能**: Layer 3的相变检测

**函数签名**:
```python
def detect_shallow_lockin(trajectory: ReasoningTrajectory,
                         shallow_layers: List[int] = [10, 14, 18],
                         drop_threshold: float = 0.15) -> Dict:
    """检测Shallow Lock-in"""

def detect_deep_decay(trajectory: ReasoningTrajectory,
                     deep_layers: List[int] = [18, 22],
                     stability_threshold: float = 0.5,
                     entropy_threshold: float = 0.7) -> Dict:
    """检测Deep Decay"""

def detect_phase_transition(trajectory: ReasoningTrajectory,
                           layer: int = 14) -> Dict:
    """组合检测信号
    
    Returns:
        {
            'has_transition': bool,
            'transition_type': 'none' | 'lockin' | 'decay' | 'both',
            'lockin_result': Dict,
            'decay_result': Dict,
        }
    """
```

**测试代码**:
```python
# Test: 检测相变
from phase_transition import detect_shallow_lockin, detect_deep_decay

lockin_result = detect_shallow_lockin(trajs[0], shallow_layers=[14])
print(f"Lock-in detected: {lockin_result['detected']}")

decay_result = detect_deep_decay(trajs[0], deep_layers=[14, 18])
print(f"Decay detected: {decay_result['detected']}")
```

---

### 步骤5: validation.py

**功能**: 运行H1-H4验证实验

**函数签名**:
```python
@dataclass
class ValidationResult:
    hypothesis: str
    metric: str
    error_mean: float
    correct_mean: float
    cohens_d: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_error: int
    n_correct: int
    interpretation: str

def bootstrap_mean_diff(arr1: np.ndarray, arr2: np.ndarray,
                       n_bootstrap: int = 5000) -> Tuple[float, float, float]:
    """Bootstrap计算均值差异的CI"""

def run_validation_h1(trajectories: List[ReasoningTrajectory],
                      layer: int = 14) -> ValidationResult:
    """H1: smoothness区分"""

def run_validation_h2(trajectories: List[ReasoningTrajectory],
                      layer: int = 14) -> ValidationResult:
    """H2: Shallow Lock-in频率"""

def run_validation_h3(trajectories: List[ReasoningTrajectory],
                      layer: int = 14) -> ValidationResult:
    """H3: Deep Decay频率"""

def run_validation_h4(trajectories: List[ReasoningTrajectory],
                      layer: int = 14) -> ValidationResult:
    """H4: 轨迹 vs 单步"""

def run_all_validations(trajectories: List[ReasoningTrajectory],
                       layers: List[int] = [14],
                       output_dir: str = None) -> Dict[str, ValidationResult]:
    """运行所有验证"""
```

---

### 步骤6: main.py

**功能**: 主入口，解析参数并运行实验

**用法**:
```bash
python main.py \
    --npz_path /gz-data/research/demo/data/features/full_omnimath.npz \
    --hidden_dir /gz-data/research/demo/data/hidden/omnimath/ \
    --output_dir ./trajectory_results \
    --layers 14 \
    --n_bootstrap 5000
```

---

## 🧪 实验运行计划

### 阶段1: 单数据集测试
```bash
# 在omnimath上测试
cd /gz-data/research/demo/
python main.py \
    --npz_path data/features/full_omnimath.npz \
    --hidden_dir data/hidden/omnimath/ \
    --output_dir ./results/omnimath \
    --layers 14
```

### 阶段2: 多层分析
```bash
# 在多个层上分析
python main.py \
    --npz_path data/features/full_omnimath.npz \
    --hidden_dir data/hidden/omnimath/ \
    --output_dir ./results/omnimath_multilayer \
    --layers 10 14 18 22
```

### 阶段3: 多数据集验证
```bash
# 在所有数据集上运行
for dataset in gsm8k math omnimath; do
    python main.py \
        --npz_path data/features/full_${dataset}.npz \
        --hidden_dir data/hidden/${dataset}/ \
        --output_dir ./results/${dataset}
done
```

---

## 📊 预期输出格式

### JSON结果文件
```json
{
  "metadata": {
    "subset": "omnimath",
    "n_chains": 998,
    "n_correct": 500,
    "n_error": 498,
    "layers": [14]
  },
  "results": {
    "L14_H1": {
      "hypothesis": "H1",
      "metric": "smoothness",
      "error_mean": 0.65,
      "correct_mean": 0.78,
      "cohens_d": 0.45,
      "p_value": 0.0001,
      "ci_lower": 0.10,
      "ci_upper": 0.16,
      "n_error": 498,
      "n_correct": 500
    },
    "L14_H2": {
      "hypothesis": "H2",
      "metric": "shallow_lockin",
      "error_mean": 0.25,
      "correct_mean": 0.10,
      "cohens_d": NaN,
      "p_value": 0.005,
      ...
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
Loading: data/features/full_omnimath.npz
Loading hidden shards from: data/hidden/omnimath/
Loaded 998 chains (500 correct, 498 error)
Computing step geometry from hidden states...

================================================================================
Layer 14 Analysis
================================================================================

[H1] smoothness: error=0.650 < correct=0.780
  Cohen's d=0.450, p=0.0001, 95%CI=[0.100, 0.160]

[H2] Shallow Lock-in: OR=3.12, p=0.0050
  Error: 124/498 vs Correct: 50/500

[H3] Deep Decay: OR=2.85, p=0.0080
  Error: 98/498 vs Correct: 55/500

[H4] Trajectory vs Stepwise: Δ=0.065
  Trajectory AUROC=0.785 > Stepwise AUROC=0.720

================================================================================
Summary
================================================================================
Test            Metric                    AUROC    Cohen's d  p-value
--------------------------------------------------------------------------------
L14_H1          smoothness               0.721    0.450      0.0001 *
L14_H2          shallow_lockin            N/A      N/A        0.0050 *
L14_H3          deep_decay                N/A      N/A        0.0080 *
L14_H4          trajectory_vs_stepwise    0.785    0.065      0.0230 *
--------------------------------------------------------------------------------
* p < 0.05
================================================================================
```

---

## ✅ 实现检查清单

### 数据加载
- [ ] 正确加载NPZ文件
- [ ] 正确关联hidden shards
- [ ] 处理缺失的hidden文件
- [ ] 正确提取步骤token范围
- [ ] 计算真实的scatter matrix
- [ ] 特征值分解正确

### Layer 1
- [ ] κ计算正确（||mean(û)||）
- [ ] eff_rank计算正确（exp(-Σλlogλ)）
- [ ] spectrum存储正确（前10个特征值）
- [ ] spectral_entropy计算正确

### Layer 2
- [ ] geometric_sim组合三个维度
- [ ] local_smoothness返回全局和序列
- [ ] global_coherence使用JS散度
- [ ] spectral_evolution_stability计算方差

### Layer 3
- [ ] detect_shallow_lockin检测coherence突降
- [ ] detect_deep_decay检测stability+entropy条件
- [ ] 阈值可配置
- [ ] 返回详细的检测信息

### 验证
- [ ] H1使用Mann-Whitney U test
- [ ] H2/H3使用Fisher's exact test
- [ ] H4比较AUROC
- [ ] Bootstrap CI正确计算
- [ ] 结果保存为JSON
- [ ] 控制台输出清晰

---

## 🚀 实现优先级

### P0（必须实现）
1. data_loading.py - 数据加载和特征计算
2. trajectory_geometry.py - Layer 2三个指标
3. validation.py - H1验证
4. main.py - 主入口

### P1（重要）
1. phase_transition.py - Layer 3检测
2. validation.py - H2-H4验证

### P2（优化）
1. 多层并行计算
2. 可视化工具
3. 交互式调试模式

---

## 📝 待确认事项

在开始实现前，需要确认：

1. **Hidden目录结构**: `/gz-data/research/demo/data/hidden/<subset>/` 是否正确？
2. **Hidden文件命名**: `<subset>-<chain_id>.npy` 格式是否正确？
3. **层映射**: Hidden中的4层对应[10, 14, 18, 22]是否正确？
4. **阈值参数**:
   - `drop_threshold` (Shallow Lock-in): 0.15是否合适？
   - `stability_threshold` (Deep Decay): 0.5是否合适？
   - `entropy_threshold` (Deep Decay): 0.7是否合适？
5. **权重参数**:
   - `kappa_weight`: 0.4
   - `eff_rank_weight`: 0.3
   - `spectrum_weight`: 0.3
   是否需要调整？

---

**文档版本**: v1.0
**最后更新**: 2026-06-30
**状态**: 待确认后开始实现
