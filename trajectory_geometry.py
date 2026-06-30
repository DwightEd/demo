#!/usr/bin/env python3
"""轨迹几何分析：Layer 2 - Trajectory Geometry

包含：
1. 真正的几何特征（基于特征向量）
   - Principal Direction Rotation（主方向旋转）
   - Subspace Drift（子空间漂移）
   - Projection Residual（投影残差）

2. 标量动态演化
   - Scalar Evolution Smoothness（标量演化平滑度）
   - Scalar Trend Consistency（标量趋势一致性）

3. 组合相似度和核心指标
   - geometric_sim: 组合多维度相似度
   - local_smoothness: 相邻步骤的几何相似度
   - global_coherence: 首尾步骤的关联强度
   - spectral_evolution_stability: 谱形状演化稳定性
"""

import numpy as np
from scipy.spatial.distance import jensenshannon
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from data_loading import StepGeometry, ReasoningTrajectory


# =============================================================================
# 真正的几何特征（基于特征向量）
# =============================================================================

def principal_direction_rotation(g1: StepGeometry, g2: StepGeometry, k: int = 5) -> float:
    """计算主成分方向的旋转角度

    核心思想：错误推理中，主成分方向会突然旋转。

    Args:
        g1, g2: 两个步骤的几何特征
        k: 使用前k个主成分

    Returns:
        平均旋转角度（弧度），范围[0, π/2]
    """
    if g1.principal_directions.size == 0 or g2.principal_directions.size == 0:
        return np.nan

    V1 = g1.principal_directions[:, :k]  # (d, k)
    V2 = g2.principal_directions[:, :k]  # (d, k)

    k_actual = min(k, V1.shape[1], V2.shape[1])
    if k_actual == 0:
        return np.nan

    angles = []
    for i in range(k_actual):
        v1_i = V1[:, i]
        v2_i = V2[:, i]

        cos_theta = np.dot(v1_i, v2_i)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        angles.append(theta)

    return float(np.mean(angles)) if angles else np.nan


def subspace_drift(g1: StepGeometry, g2: StepGeometry, k: int = 5) -> float:
    """计算子空间之间的漂移程度

    核心思想：错误推理中，步骤间的子空间夹角会突然增大。

    使用：主子空间之间的投影矩阵距离

    Args:
        g1, g2: 两个步骤的几何特征
        k: 使用前k个主成分

    Returns:
        漂移程度，越大表示漂移越大
    """
    if g1.principal_directions.size == 0 or g2.principal_directions.size == 0:
        return np.nan

    V1 = g1.principal_directions[:, :k]
    V2 = g2.principal_directions[:, :k]

    k_actual = min(k, V1.shape[1], V2.shape[1])
    if k_actual == 0:
        return np.nan

    V1 = V1[:, :k_actual]
    V2 = V2[:, :k_actual]

    # 计算投影矩阵
    P1 = V1 @ V1.T  # (d, d)
    P2 = V2 @ V2.T

    # Frobenius范数距离
    drift = np.linalg.norm(P1 - P2, ord='fro')

    return float(drift)


def projection_residual(geometry: StepGeometry, k: int = 5) -> float:
    """计算点云到主子空间的投影残差

    核心思想：错误推理中，点云到主子空间的距离会增大（信息丢失）。

    Args:
        geometry: 步骤几何特征
        k: 主成分数

    Returns:
        残差，越大表示信息丢失越多
    """
    if geometry.principal_directions.size == 0:
        return np.nan

    V = geometry.principal_directions[:, :k]
    k_actual = min(k, V.shape[1])

    if k_actual == 0:
        return np.nan

    V = V[:, :k_actual]
    P = V @ V.T
    d = V.shape[0]
    I = np.eye(d)

    # 残差 = ||I - P||_F
    residual = np.linalg.norm(I - P, ord='fro')

    return float(residual)


# =============================================================================
# 标量动态演化
# =============================================================================

def scalar_evolution_smoothness(scalar_sequence: List[float]) -> Tuple[float, np.ndarray]:
    """标量序列的平滑度

    检测标量的演化是否平滑。
    正确推理：平滑演化；错误推理：突变。

    Args:
        scalar_sequence: 标量值序列

    Returns:
        (平滑度, 差分序列)
        平滑度 = 1 / (1 + var(diff))，范围[0, 1]
    """
    if len(scalar_sequence) < 2:
        return np.nan, np.array([])

    arr = np.array(scalar_sequence)
    diffs = np.diff(arr)

    # 移除NaN
    valid_diffs = diffs[~np.isnan(diffs)]

    if len(valid_diffs) == 0:
        return np.nan, diffs

    variance = np.var(valid_diffs)
    smoothness = 1.0 / (1.0 + variance)

    return float(smoothness), diffs


def scalar_trend_consistency(scalar_sequence: List[float]) -> float:
    """标量序列的趋势一致性（R²）

    核心思想：正确推理有单调趋势（如κ递增），错误推理趋势被打断。

    Args:
        scalar_sequence: 标量值序列

    Returns:
        R²值，范围[0, 1]，越高表示趋势越一致
    """
    if len(scalar_sequence) < 3:
        return np.nan

    arr = np.array(scalar_sequence)
    valid_mask = ~np.isnan(arr)
    arr = arr[valid_mask]

    if len(arr) < 3:
        return np.nan

    x = np.arange(len(arr))

    # 拟合线性趋势
    coeffs = np.polyfit(x, arr, 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    # 计算R²
    y_pred = slope * x + intercept
    ss_res = np.sum((arr - y_pred) ** 2)
    ss_tot = np.sum((arr - np.mean(arr)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else np.nan

    r2 = 1.0 - ss_res / ss_tot
    return float(r2)


def extract_scalar_sequence(geometry_sequence: List[StepGeometry],
                             feature: str) -> List[float]:
    """从几何序列中提取标量特征序列

    Args:
        geometry_sequence: 步骤几何序列
        feature: 特征名 ('kappa', 'eff_rank', 'spectral_entropy', 'norm')

    Returns:
        标量值序列
    """
    if feature == 'kappa':
        return [g.kappa for g in geometry_sequence if np.isfinite(g.kappa)]
    elif feature == 'eff_rank':
        return [g.eff_rank for g in geometry_sequence if np.isfinite(g.eff_rank)]
    elif feature == 'spectral_entropy':
        return [g.spectral_entropy for g in geometry_sequence if np.isfinite(g.spectral_entropy)]
    elif feature == 'norm':
        return [g.norm for g in geometry_sequence if np.isfinite(g.norm)]
    else:
        raise ValueError(f"Unknown feature: {feature}")


# =============================================================================
# 组合相似度
# =============================================================================

def geometric_sim(g1: StepGeometry, g2: StepGeometry,
                 kappa_weight: float = 0.2,
                 eff_rank_weight: float = 0.15,
                 rotation_weight: float = 0.25,
                 drift_weight: float = 0.2,
                 spectrum_weight: float = 0.2) -> float:
    """组合多个几何维度的相似度

    包括：
    - 标量相似：κ, eff_rank
    - 几何相似：主方向旋转、子空间漂移
    - 谱形状相似：JS散度

    Args:
        g1, g2: 两个步骤的几何特征
        kappa_weight: κ相似度的权重
        eff_rank_weight: eff_rank相似度的权重
        rotation_weight: 主方向旋转的权重
        drift_weight: 子空间漂移的权重
        spectrum_weight: 谱形状的权重

    Returns:
        相似度，范围[0, 1]，越高越相似
    """
    # 标量部分
    kappa_sim = 1.0 - min(abs(g1.kappa - g2.kappa), 1.0)
    max_er = max(g1.eff_rank, g2.eff_rank, 1.0)
    eff_rank_sim = 1.0 - abs(g1.eff_rank - g2.eff_rank) / max_er

    # 几何部分
    rotation = principal_direction_rotation(g1, g2)
    if np.isfinite(rotation):
        rotation_sim = 1.0 - (2 / np.pi) * rotation
    else:
        rotation_sim = 0.5

    drift = subspace_drift(g1, g2)
    if np.isfinite(drift):
        drift_sim = 1.0 / (1.0 + drift)
    else:
        drift_sim = 0.5

    # 谱形状
    if g1.eigenvalues.size >= 2 and g2.eigenvalues.size >= 2:
        s1 = g1.eigenvalues[:10]
        s2 = g2.eigenvalues[:10]
        s1 = s1 / (s1.sum() + 1e-12)
        s2 = s2 / (s2.sum() + 1e-12)
        spectrum_sim = 1.0 - jensenshannon(s1, s2)
    else:
        spectrum_sim = 0.5

    # 组合
    total_weight = kappa_weight + eff_rank_weight + rotation_weight + drift_weight + spectrum_weight
    sim = (kappa_weight * kappa_sim +
           eff_rank_weight * eff_rank_sim +
           rotation_weight * rotation_sim +
           drift_weight * drift_sim +
           spectrum_weight * spectrum_sim) / total_weight

    return float(sim)


def geometric_sim_scalar_only(g1: StepGeometry, g2: StepGeometry) -> float:
    """仅基于标量的几何相似度（用于fallback）

    当没有特征向量时使用。
    """
    # κ相似度
    kappa_sim = 1.0 - min(abs(g1.kappa - g2.kappa), 1.0)

    # eff_rank相似度
    max_er = max(g1.eff_rank, g2.eff_rank, 1.0)
    eff_rank_sim = 1.0 - abs(g1.eff_rank - g2.eff_rank) / max_er

    # 谱相似度
    if g1.eigenvalues.size >= 2 and g2.eigenvalues.size >= 2:
        s1 = g1.eigenvalues[:10]
        s2 = g2.eigenvalues[:10]
        s1 = s1 / (s1.sum() + 1e-12)
        s2 = s2 / (s2.sum() + 1e-12)
        spectrum_sim = 1.0 - jensenshannon(s1, s2)
    else:
        spectrum_sim = 0.5

    # 等权组合
    return (kappa_sim + eff_rank_sim + spectrum_sim) / 3.0


# =============================================================================
# 核心轨迹指标
# =============================================================================

def local_smoothness(geometry_sequence: List[StepGeometry]) -> Tuple[float, np.ndarray]:
    """局部平滑度：相邻步骤的几何相似度

    正确推理：smoothness高（渐进变化）
    错误推理：smoothness低（突变）

    Args:
        geometry_sequence: 步骤几何序列

    Returns:
        (全局平滑度, 每个transition的平滑度序列)
    """
    if len(geometry_sequence) < 2:
        return np.nan, np.array([])

    smoothness_values = []
    for i in range(len(geometry_sequence) - 1):
        g1 = geometry_sequence[i]
        g2 = geometry_sequence[i + 1]

        # 尝试使用完整相似度
        if g1.principal_directions.size > 0 and g2.principal_directions.size > 0:
            sim = geometric_sim(g1, g2)
        else:
            sim = geometric_sim_scalar_only(g1, g2)

        smoothness_values.append(sim)

    if not smoothness_values:
        return np.nan, np.array([])

    return float(np.mean(smoothness_values)), np.array(smoothness_values)


def global_coherence(geometry_sequence: List[StepGeometry],
                     n_early: int = 3,
                     n_late: int = 3) -> float:
    """全局连贯度：首尾步骤的关联强度

    核心思想：后期步骤是否还"记住"早期推理
    正确推理：首尾连贯（始终围绕问题展开）
    错误推理：首尾断裂（后期遗忘问题）

    Args:
        geometry_sequence: 步骤几何序列
        n_early: 早期步骤数
        n_late: 后期步骤数

    Returns:
        连贯度，范围[0, 1]，越高表示越连贯
    """
    if len(geometry_sequence) < max(n_early, n_late):
        return np.nan

    # 早期步骤的平均谱
    early_spectra = [g.eigenvalues[:10] for g in geometry_sequence[:n_early]
                     if g.eigenvalues.size >= 2]
    if not early_spectra:
        return np.nan

    early_spectrum = np.mean(early_spectra, axis=0)
    early_spectrum = early_spectrum / (early_spectrum.sum() + 1e-12)

    # 后期步骤的平均谱
    late_spectra = [g.eigenvalues[:10] for g in geometry_sequence[-n_late:]
                    if g.eigenvalues.size >= 2]
    if not late_spectra:
        return np.nan

    late_spectrum = np.mean(late_spectra, axis=0)
    late_spectrum = late_spectrum / (late_spectrum.sum() + 1e-12)

    # 连贯度 = 1 - JS散度
    coherence = 1.0 - jensenshannon(early_spectrum, late_spectrum)

    return float(coherence)


def spectral_evolution_stability(geometry_sequence: List[StepGeometry]) -> float:
    """谱形状演化稳定性

    核心思想：错误推理的谱形状会跳变
    正确推理：谱形状渐进演化（稳定）
    错误推理：谱形状跳变（不稳定）

    Args:
        geometry_sequence: 步骤几何序列

    Returns:
        稳定性，范围[0, 1]，越高越稳定
    """
    if len(geometry_sequence) < 3:
        return np.nan

    spectra = [g.eigenvalues[:10] for g in geometry_sequence if g.eigenvalues.size >= 2]
    if len(spectra) < 3:
        return np.nan

    spectra = np.array(spectra)

    # 计算相邻谱的变化率
    diffs = np.diff(spectra, axis=0)           # (T-1, 10)
    diff_magnitudes = np.linalg.norm(diffs, axis=1)  # (T-1,)

    # 移除NaN
    diff_magnitudes = diff_magnitudes[~np.isnan(diff_magnitudes)]

    if len(diff_magnitudes) == 0:
        return np.nan

    # 稳定性 = 1 / (1 + 变化率的方差)
    variance = np.var(diff_magnitudes)
    stability = 1.0 / (1.0 + variance)

    return float(stability)


# =============================================================================
# 综合指标计算
# =============================================================================

@dataclass
class TrajectoryMetrics:
    """轨迹的所有几何指标"""
    smoothness: float = np.nan
    coherence: float = np.nan
    stability: float = np.nan

    # 标量动态指标
    kappa_smoothness: float = np.nan
    eff_rank_smoothness: float = np.nan
    entropy_smoothness: float = np.nan

    kappa_trend: float = np.nan
    eff_rank_trend: float = np.nan

    n_steps: int = 0

    # 几何特征详情
    mean_rotation: float = np.nan
    mean_drift: float = np.nan


def compute_trajectory_metrics(trajectory: ReasoningTrajectory,
                               layer: int = 14) -> TrajectoryMetrics:
    """计算单个轨迹的所有指标

    Args:
        trajectory: 推理轨迹
        layer: 分析的层

    Returns:
        TrajectoryMetrics对象
    """
    geom_sequence = trajectory.get_geometry_sequence(layer)

    if len(geom_sequence) == 0:
        return TrajectoryMetrics(n_steps=0)

    metrics = TrajectoryMetrics(n_steps=len(geom_sequence))

    # 核心三个指标
    smoothness, _ = local_smoothness(geom_sequence)
    metrics.smoothness = smoothness

    coherence = global_coherence(geom_sequence)
    metrics.coherence = coherence

    stability = spectral_evolution_stability(geom_sequence)
    metrics.stability = stability

    # 标量动态指标
    kappa_seq = extract_scalar_sequence(geom_sequence, 'kappa')
    if kappa_seq:
        k_smooth, _ = scalar_evolution_smoothness(kappa_seq)
        metrics.kappa_smoothness = k_smooth
        k_trend = scalar_trend_consistency(kappa_seq)
        metrics.kappa_trend = k_trend

    eff_rank_seq = extract_scalar_sequence(geom_sequence, 'eff_rank')
    if eff_rank_seq:
        e_smooth, _ = scalar_evolution_smoothness(eff_rank_seq)
        metrics.eff_rank_smoothness = e_smooth
        e_trend = scalar_trend_consistency(eff_rank_seq)
        metrics.eff_rank_trend = e_trend

    entropy_seq = extract_scalar_sequence(geom_sequence, 'spectral_entropy')
    if entropy_seq:
        h_smooth, _ = scalar_evolution_smoothness(entropy_seq)
        metrics.entropy_smoothness = h_smooth

    # 几何特征详情
    rotations = []
    drifts = []
    for i in range(len(geom_sequence) - 1):
        rot = principal_direction_rotation(geom_sequence[i], geom_sequence[i + 1])
        dr = subspace_drift(geom_sequence[i], geom_sequence[i + 1])
        if np.isfinite(rot):
            rotations.append(rot)
        if np.isfinite(dr):
            drifts.append(dr)

    if rotations:
        metrics.mean_rotation = float(np.mean(rotations))
    if drifts:
        metrics.mean_drift = float(np.mean(drifts))

    return metrics


def compute_all_metrics(trajectories: List[ReasoningTrajectory],
                       layer: int = 14,
                       verbose: bool = False) -> List[TrajectoryMetrics]:
    """计算所有轨迹的指标

    Args:
        trajectories: 轨迹列表
        layer: 分析的层
        verbose: 显示进度

    Returns:
        指标列表
    """
    metrics_list = []

    iterator = trajectories
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(trajectories, desc=f"Computing metrics (L{layer})")

    for traj in iterator:
        if traj.has_layer(layer):
            metrics = compute_trajectory_metrics(traj, layer)
            metrics_list.append(metrics)
        else:
            metrics_list.append(TrajectoryMetrics(n_steps=0))

    return metrics_list


# =============================================================================
# 辅助函数
# =============================================================================

def print_metrics_summary(metrics_list: List[TrajectoryMetrics],
                          is_correct_list: List[bool]):
    """打印指标统计摘要"""
    metrics_array = np.array([
        [m.smoothness, m.coherence, m.stability,
         m.kappa_smoothness, m.eff_rank_smoothness]
        for m in metrics_list
    ])

    correct_mask = np.array(is_correct_list)
    error_mask = ~correct_mask

    # 移除NaN行
    valid_mask = ~np.isnan(metrics_array).any(axis=1)
    metrics_valid = metrics_array[valid_mask]
    correct_valid = correct_mask[valid_mask]

    print("\n" + "=" * 80)
    print("Metrics Summary")
    print("=" * 80)
    print(f"{'Metric':<20} {'Correct':>10} {'Error':>10} {'Diff':>10}")
    print("-" * 80)

    names = ['Smoothness', 'Coherence', 'Stability',
             'Kappa Smooth', 'Eff_rank Smooth']

    for i, name in enumerate(names):
        col = metrics_valid[:, i]

        if correct_valid.sum() > 0:
            c_mean = col[correct_valid].mean()
        else:
            c_mean = np.nan

        if (~correct_valid).sum() > 0:
            e_mean = col[~correct_valid].mean()
        else:
            e_mean = np.nan

        diff = c_mean - e_mean if np.isfinite(c_mean) and np.isfinite(e_mean) else np.nan

        print(f"{name:<20} {c_mean:>10.4f} {e_mean:>10.4f} {diff:>10.4f}")

    print("=" * 80)


if __name__ == "__main__":
    # 示例用法
    from data_loading import load_all_trajectories, get_trajectory_with_min_steps

    npz_path = "/gz-data/research/demo/data/features/full_omnimath.npz"
    hidden_dir = "/gz-data/research/demo/data/hidden/omnimath/"

    print("Loading trajectories...")
    trajectories, metadata = load_all_trajectories(
        npz_path=npz_path,
        hidden_dir=hidden_dir,
        verbose=True,
    )

    # 过滤出至少有3个步骤的轨迹
    filtered = get_trajectory_with_min_steps(trajectories, min_steps=3, layer=14)
    print(f"\nFiltered to {len(filtered)} trajectories with ≥3 steps")

    # 计算指标
    metrics_list = compute_all_metrics(filtered, layer=14, verbose=True)

    # 打印摘要
    is_correct_list = [t.is_correct for t in filtered]
    print_metrics_summary(metrics_list, is_correct_list)
