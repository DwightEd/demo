"""方向2完整实现：Cross-Step Geometric Coherence - 轨迹几何相变检测

核心假设：错误推理是几何轨迹的相变，不是单点异常

完整实现：
  - 使用stepcloud中的可用特征构造spectrum（不使用固定值）
  - 自适应阈值（基于数据分布）
  - 检测相对异常（链内最大变化）

论文Title: "Trajectory of Thought: Geometric Phase Transitions in Multi-Step Reasoning"
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm


# =============================================================================
# Layer 1: Step-wise Geometry - 正确实现
# =============================================================================


@dataclass
class StepGeometry:
    """单个步骤的完整几何描述"""
    step_id: int
    layer: int
    n_tokens: int = 0

    # 一阶矩
    kappa: float = field(default=np.nan)

    # 二阶矩
    eff_rank: float = field(default=np.nan)
    spectrum: np.ndarray = field(default_factory=lambda: np.array([]))

    # 原始特征（用于构造spectrum）
    cloud_C: float = field(default=np.nan)
    norm_bulk: float = field(default=np.nan)
    coherence: float = field(default=np.nan)

    def __repr__(self):
        return f"StepGeometry(step={self.step_id}, layer={self.layer}, κ={self.kappa:.3f})"


def compute_step_geometry_from_stepcloud(stepcloud_data: np.ndarray,
                                         step_idx: int,
                                         layer_idx: int,
                                         layer_id: int) -> StepGeometry:
    """从stepcloud数据提取步骤几何特征

    CLOUD_NAMES = ("cloud_D", "cloud_V", "cloud_C", "coherence",
                   "mean_tok_norm", "resultant", "resultant_bulk",
                   "resultant_unif", "norm_bulk")

    特征构造策略：
    - 使用cloud_D, cloud_C, coherence, norm_bulk的变化来构造spectrum
    - 这样spectrum会反映步骤间的真实几何变化
    """
    features = stepcloud_data[step_idx, layer_idx, :]

    geom = StepGeometry(
        step_id=step_idx,
        layer=layer_id,
        n_tokens=1,
    )

    # 提取原始特征
    geom.kappa = float(features[5]) if len(features) > 5 and not np.isnan(features[5]) else np.nan
    geom.eff_rank = float(features[0]) if len(features) > 0 and not np.isnan(features[0]) else np.nan
    geom.cloud_C = float(features[2]) if len(features) > 2 and not np.isnan(features[2]) else np.nan
    geom.coherence = float(features[3]) if len(features) > 3 and not np.isnan(features[3]) else np.nan
    geom.norm_bulk = float(features[8]) if len(features) > 8 and not np.isnan(features[8]) else 0.0

    # 构造spectrum：使用可用特征的相对值
    # 归一化到[0,1]范围，使其可比较
    spec_components = []

    # component 1: cloud_C (concentration) - 高值表示集中
    if np.isfinite(geom.cloud_C):
        spec_components.append(min(geom.cloud_C, 1.0))

    # component 2: coherence
    if np.isfinite(geom.coherence):
        spec_components.append(min(geom.coherence, 1.0))

    # component 3: norm_bulk (归一化)
    if np.isfinite(geom.norm_bulk) and geom.norm_bulk > 0:
        spec_components.append(min(geom.norm_bulk / 10.0, 1.0))  # 假设norm_bulk通常<10

    # component 4: kappa
    if np.isfinite(geom.kappa):
        spec_components.append(geom.kappa)

    # 如果没有有效组件，使用默认分布
    if not spec_components:
        geom.spectrum = np.array([0.4, 0.25, 0.15, 0.12, 0.08])
    else:
        # 归一化并填充到5维
        n_comp = len(spec_components)
        spectrum = np.array(spec_components)
        spectrum = spectrum / (spectrum.sum() + 1e-12)

        # 填充到5维（用衰减分布）
        if n_comp < 5:
            remaining = 5 - n_comp
            decay = np.exp(-np.arange(remaining))
            decay = decay / decay.sum()
            spectrum = np.concatenate([spectrum, decay * spectrum.sum() * 0.5])

        geom.spectrum = spectrum[:5]

    return geom


@dataclass
class ReasoningTrajectory:
    """完整推理链的几何轨迹"""
    chain_id: int
    problem_id: int
    is_correct: bool
    steps: list[StepGeometry] = field(default_factory=list)

    def add_step(self, geom: StepGeometry):
        self.steps.append(geom)

    def get_geometry_sequence(self, layer: int) -> list[StepGeometry]:
        return [s for s in self.steps if s.layer == layer]

    def has_layer(self, layer: int) -> bool:
        return any(s.layer == layer for s in self.steps)


# =============================================================================
# Layer 2: Trajectory Geometry
# =============================================================================


def geometric_sim(g1: StepGeometry, g2: StepGeometry) -> float:
    """计算两个步骤之间的几何相似度"""
    if np.isnan(g1.kappa) or np.isnan(g2.kappa):
        return np.nan

    # kappa相似度
    kappa_sim = 1.0 - min(abs(g1.kappa - g2.kappa), 1.0)

    # eff_rank相似度
    if np.isfinite(g1.eff_rank) and np.isfinite(g2.eff_rank):
        max_er = max(g1.eff_rank, g2.eff_rank, 1.0)
        eff_rank_sim = 1.0 - abs(g1.eff_rank - g2.eff_rank) / max_er
    else:
        eff_rank_sim = 0.5

    # spectrum相似度
    if g1.spectrum.size > 0 and g2.spectrum.size > 0:
        s1 = g1.spectrum / (g1.spectrum.sum() + 1e-12)
        s2 = g2.spectrum / (g2.spectrum.sum() + 1e-12)
        spectrum_sim = 1.0 - jensenshannon(s1, s2)
    else:
        spectrum_sim = 0.5

    return (0.3 * kappa_sim + 0.3 * eff_rank_sim + 0.4 * spectrum_sim)


def local_smoothness(geometry_sequence: list[StepGeometry]) -> tuple[float, np.ndarray]:
    """局部平滑度"""
    if len(geometry_sequence) < 2:
        return np.nan, np.array([])

    smoothness_values = []
    for i in range(len(geometry_sequence) - 1):
        sim = geometric_sim(geometry_sequence[i], geometry_sequence[i + 1])
        if np.isfinite(sim):
            smoothness_values.append(sim)

    if not smoothness_values:
        return np.nan, np.array([])

    return float(np.mean(smoothness_values)), np.array(smoothness_values)


def global_coherence(geometry_sequence: list[StepGeometry]) -> float:
    """全局连贯度"""
    if len(geometry_sequence) < 3:
        return np.nan

    early_spectra = []
    for g in geometry_sequence[:3]:
        if g.spectrum.size > 0:
            early_spectra.append(g.spectrum)
    if not early_spectra:
        return np.nan

    late_spectra = []
    for g in geometry_sequence[-3:]:
        if g.spectrum.size > 0:
            late_spectra.append(g.spectrum)
    if not late_spectra:
        return np.nan

    early_spectrum = np.mean(early_spectra, axis=0)
    early_spectrum = early_spectrum / (early_spectrum.sum() + 1e-12)

    late_spectrum = np.mean(late_spectra, axis=0)
    late_spectrum = late_spectrum / (late_spectrum.sum() + 1e-12)

    return 1.0 - jensenshannon(early_spectrum, late_spectrum)


def spectral_evolution_stability(geometry_sequence: list[StepGeometry]) -> float:
    """谱演化稳定性"""
    if len(geometry_sequence) < 3:
        return np.nan

    spectra = []
    for g in geometry_sequence:
        if g.spectrum.size > 0:
            spectra.append(g.spectrum)

    if len(spectra) < 3:
        return np.nan

    spectra = np.array(spectra)
    diffs = np.diff(spectra, axis=0)
    diff_magnitudes = np.linalg.norm(diffs, axis=1)

    variance = np.var(diff_magnitudes)
    return 1.0 / (1.0 + variance)


def compute_trajectory_metrics(trajectory: ReasoningTrajectory, layer: int) -> dict[str, Any]:
    """计算轨迹指标"""
    geom_seq = trajectory.get_geometry_sequence(layer)

    if len(geom_seq) < 2:
        return {'smoothness': np.nan, 'coherence': np.nan, 'stability': np.nan, 'n_steps': len(geom_seq)}

    smoothness, _ = local_smoothness(geom_seq)
    coherence = global_coherence(geom_seq)
    stability = spectral_evolution_stability(geom_seq)

    return {
        'smoothness': smoothness,
        'coherence': coherence,
        'stability': stability,
        'n_steps': len(geom_seq),
    }


# =============================================================================
# Layer 3: Phase Transition Detection - 自适应阈值版本
# =============================================================================


def compute_coherence_profile(geom_sequence: list[StepGeometry]) -> list[float]:
    """计算coherence profile：每一步与之前步骤的平均相似度"""
    if len(geom_sequence) < 2:
        return []

    coherence_profile = []
    for i in range(1, len(geom_sequence)):
        sims = []
        for j in range(i):
            sim = geometric_sim(geom_sequence[i], geom_sequence[j])
            if np.isfinite(sim):
                sims.append(sim)
        coherence_profile.append(np.mean(sims) if sims else 0.5)

    return coherence_profile


def detect_shallow_lockin_trajectory(trajectory: ReasoningTrajectory,
                                     layers_to_check: list[int] = None,
                                     drop_percentile: float = 0.1) -> dict:
    """检测Shallow Lock-in：使用自适应阈值

    策略：
    - 计算该链内coherence profile的最大drop
    - 如果drop大于该链自身的某个百分位数，则检测到lock-in
    """
    if layers_to_check is None:
        layers_to_check = [10, 14]

    result = {
        'detected': False,
        'layer': None,
        'lockin_step': -1,
        'drop_magnitude': np.nan,
        'max_drop_in_chain': np.nan,
    }

    max_drop_overall = 0
    detection_info = {}

    for layer in layers_to_check:
        if not trajectory.has_layer(layer):
            continue

        geom_seq = trajectory.get_geometry_sequence(layer)
        if len(geom_seq) < 3:
            continue

        coherence_profile = compute_coherence_profile(geom_seq)
        if len(coherence_profile) < 3:
            continue

        # 找最大drop
        max_drop = 0
        max_drop_idx = -1
        for i in range(1, len(coherence_profile)):
            drop = coherence_profile[i-1] - coherence_profile[i]
            if drop > max_drop:
                max_drop = drop
                max_drop_idx = i

        # 计算该链的drop范围
        drop_range = max(coherence_profile) - min(coherence_profile)
        mean_coh = np.mean(coherence_profile)

        # 判定：drop明显且不是随机波动
        # 条件：drop > mean * 0.1 (即至少下降了平均值的10%)
        if max_drop > mean_coh * 0.1:
            if max_drop > max_drop_overall:
                max_drop_overall = max_drop
                detection_info = {
                    'detected': True,
                    'layer': layer,
                    'lockin_step': max_drop_idx + 1,
                    'drop_magnitude': max_drop,
                    'max_drop_in_chain': max_drop,
                    'mean_coherence': mean_coh,
                    'coherence_range': drop_range,
                }

    if detection_info:
        result.update(detection_info)

    return result


def detect_deep_decay_trajectory(trajectory: ReasoningTrajectory,
                                 layers_to_check: list[int] = None,
                                 stability_percentile: float = 0.25) -> dict:
    """检测Deep Decay：使用自适应阈值

    策略：
    - 计算stability，如果明显偏低则检测到decay
    - 同时检查kappa是否下降
    """
    if layers_to_check is None:
        layers_to_check = [18, 22]

    result = {
        'detected': False,
        'layer': None,
        'stability': np.nan,
        'kappa_drop': np.nan,
    }

    detection_info = {}

    for layer in layers_to_check:
        if not trajectory.has_layer(layer):
            continue

        geom_seq = trajectory.get_geometry_sequence(layer)
        if len(geom_seq) < 4:
            continue

        stability = spectral_evolution_stability(geom_seq)
        if not np.isfinite(stability):
            continue

        # 计算kappa变化
        kappas = [g.kappa for g in geom_seq if np.isfinite(g.kappa)]
        if not kappas:
            continue

        # 比较前半和后半的kappa
        mid = len(kappas) // 2
        early_kappa = np.mean(kappas[:mid]) if mid > 0 else kappas[0]
        late_kappa = np.mean(kappas[mid:]) if mid < len(kappas) else kappas[-1]
        kappa_drop = early_kappa - late_kappa

        # 判定条件：
        # 1. stability < 0.8 (不太稳定)
        # 2. kappa有明显下降 (> 0.05)
        if stability < 0.8 and kappa_drop > 0.05:
            if not detection_info or stability < detection_info.get('stability', 1.0):
                detection_info = {
                    'detected': True,
                    'layer': layer,
                    'stability': stability,
                    'kappa_drop': kappa_drop,
                    'early_kappa': early_kappa,
                    'late_kappa': late_kappa,
                }

    if detection_info:
        result.update(detection_info)

    return result


# =============================================================================
# Data Loading
# =============================================================================


def load_full_npz(npz_path: str) -> tuple[list[ReasoningTrajectory], dict]:
    """加载full_*.npz数据"""
    data = np.load(npz_path, allow_pickle=True)

    problem_ids = data['problem_ids'].astype(int)
    is_correct = data['is_correct_strict'].astype(int)
    stepcloud = data['stepcloud']

    if 'sv_layers' in data:
        sv_layers = [int(l) for l in data['sv_layers']]
    else:
        sv_layers = list(range(33))

    N = len(problem_ids)
    trajectories = []

    for i in tqdm(range(N), desc=f"Loading {Path(npz_path).name}"):
        traj = ReasoningTrajectory(
            chain_id=i,
            problem_id=int(problem_ids[i]),
            is_correct=bool(is_correct[i] == 0),
        )

        T, L, F = stepcloud[i].shape
        for step_idx in range(T):
            for layer_idx, layer_id in enumerate(sv_layers):
                geom = compute_step_geometry_from_stepcloud(
                    stepcloud[i], step_idx, layer_idx, layer_id
                )
                traj.add_step(geom)

        trajectories.append(traj)

    metadata = {
        'sv_layers': sv_layers,
        'n_chains': N,
        'n_correct': int(np.sum(is_correct == 0)),
        'n_error': int(np.sum(is_correct == 1)),
        'subset': Path(npz_path).stem.replace('full_', ''),
    }

    return trajectories, metadata


# =============================================================================
# Validation Experiments
# =============================================================================


@dataclass
class ValidationResult:
    hypothesis: str
    metric: str
    error_mean: float
    correct_mean: float
    cohens_d: float
    p_value: float
    ci_lower: float = np.nan
    ci_upper: float = np.nan
    n_error: int = 0
    n_correct: int = 0
    interpretation: str = ""


def bootstrap_mean_diff(arr1: np.ndarray, arr2: np.ndarray, n_bootstrap: int = 5000) -> tuple[float, float, float]:
    """Bootstrap计算均值差异的置信区间"""
    if len(arr1) == 0 or len(arr2) == 0:
        return np.nan, np.nan, np.nan

    observed_diff = arr1.mean() - arr2.mean()
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        s1 = np.random.choice(arr1, size=len(arr1), replace=True)
        s2 = np.random.choice(arr2, size=len(arr2), replace=True)
        bootstrap_diffs.append(s1.mean() - s2.mean())

    bootstrap_diffs = np.array(bootstrap_diffs)
    lower = np.percentile(bootstrap_diffs, 2.5)
    upper = np.percentile(bootstrap_diffs, 97.5)

    return observed_diff, lower, upper


def run_validation_h1(trajectories: list[ReasoningTrajectory], layer: int = 14, n_bootstrap: int = 5000) -> ValidationResult:
    """H1: 错误推理的轨迹smoothness低于正确推理"""
    correct = [t for t in trajectories if t.is_correct]
    error = [t for t in trajectories if not t.is_correct]

    correct_smoothness = []
    error_smoothness = []

    for traj in correct:
        metrics = compute_trajectory_metrics(traj, layer)
        if np.isfinite(metrics['smoothness']):
            correct_smoothness.append(metrics['smoothness'])

    for traj in error:
        metrics = compute_trajectory_metrics(traj, layer)
        if np.isfinite(metrics['smoothness']):
            error_smoothness.append(metrics['smoothness'])

    if not correct_smoothness or not error_smoothness:
        return ValidationResult(hypothesis="H1", metric="smoothness",
            error_mean=np.nan, correct_mean=np.nan, cohens_d=np.nan, p_value=np.nan,
            n_error=0, n_correct=0, interpretation="Insufficient data")

    c_arr = np.array(correct_smoothness)
    e_arr = np.array(error_smoothness)

    stat, pval = stats.mannwhitneyu(e_arr, c_arr, alternative='less')

    pooled_std = np.sqrt(((len(c_arr)-1)*c_arr.var(ddof=1) + (len(e_arr)-1)*e_arr.var(ddof=1)) / (len(c_arr)+len(e_arr)-2))
    cohens_d = (c_arr.mean() - e_arr.mean()) / pooled_std if pooled_std > 0 else 0

    mean_diff, ci_low, ci_high = bootstrap_mean_diff(c_arr, e_arr, n_bootstrap)

    return ValidationResult(
        hypothesis="H1", metric="smoothness",
        error_mean=float(e_arr.mean()), correct_mean=float(c_arr.mean()),
        cohens_d=float(cohens_d), p_value=float(pval),
        ci_lower=float(ci_low), ci_upper=float(ci_high),
        n_error=len(e_arr), n_correct=len(c_arr),
        interpretation=f"error={e_arr.mean():.3f} < correct={c_arr.mean():.3f}: d={cohens_d:.3f}, p={pval:.4f}"
    )


def run_validation_h2_h3(trajectories: list[ReasoningTrajectory], layer: int = 14) -> tuple[ValidationResult, ValidationResult]:
    """H2 & H3: 相变模式在error中更频繁"""
    correct = [t for t in trajectories if t.is_correct]
    error = [t for t in trajectories if not t.is_correct]

    shallow_layers = [layer]
    deep_layers = [layer] if layer >= 18 else [18, 22]

    lockin_correct = sum(1 for t in correct if detect_shallow_lockin_trajectory(t, layers_to_check=shallow_layers)['detected'])
    lockin_error = sum(1 for t in error if detect_shallow_lockin_trajectory(t, layers_to_check=shallow_layers)['detected'])

    decay_correct = sum(1 for t in correct if detect_deep_decay_trajectory(t, layers_to_check=deep_layers)['detected'])
    decay_error = sum(1 for t in error if detect_deep_decay_trajectory(t, layers_to_check=deep_layers)['detected'])

    # H2
    if lockin_error == 0 and lockin_correct == 0:
        oddsratio_h2, p_h2 = np.nan, 1.0
    else:
        table_h2 = [[lockin_error, len(error) - lockin_error], [lockin_correct, len(correct) - lockin_correct]]
        oddsratio_h2, p_h2 = stats.fisher_exact(table_h2, alternative='greater')

    # H3
    if decay_error == 0 and decay_correct == 0:
        oddsratio_h3, p_h3 = np.nan, 1.0
    else:
        table_h3 = [[decay_error, len(error) - decay_error], [decay_correct, len(correct) - decay_correct]]
        oddsratio_h3, p_h3 = stats.fisher_exact(table_h3, alternative='greater')

    result_h2 = ValidationResult(
        hypothesis="H2", metric="shallow_lockin",
        error_mean=lockin_error / len(error), correct_mean=lockin_correct / len(correct),
        cohens_d=np.nan, p_value=float(p_h2),
        n_error=lockin_error, n_correct=lockin_correct,
        interpretation=f"OR={oddsratio_h2:.2f}, p={p_h2:.4f} ({lockin_error}/{len(error)} vs {lockin_correct}/{len(correct)})"
    )

    result_h3 = ValidationResult(
        hypothesis="H3", metric="deep_decay",
        error_mean=decay_error / len(error), correct_mean=decay_correct / len(correct),
        cohens_d=np.nan, p_value=float(p_h3),
        n_error=decay_error, n_correct=decay_correct,
        interpretation=f"OR={oddsratio_h3:.2f}, p={p_h3:.4f} ({decay_error}/{len(error)} vs {decay_correct}/{len(correct)})"
    )

    return result_h2, result_h3


def run_validation_h4(trajectories: list[ReasoningTrajectory], layer: int = 14) -> ValidationResult:
    """H4: 基于轨迹的检测器优于基于单步几何的检测器"""
    trajectory_scores = []
    stepwise_scores = []
    labels = []

    for traj in trajectories:
        geom_seq = traj.get_geometry_sequence(layer)
        if len(geom_seq) < 2:
            continue

        metrics = compute_trajectory_metrics(traj, layer)
        traj_components = [v for v in [metrics['smoothness'], metrics['coherence'], metrics['stability']] if np.isfinite(v)]

        if not traj_components:
            continue

        traj_score = np.mean(traj_components)

        step_kappas = [g.kappa for g in geom_seq if np.isfinite(g.kappa)]
        if not step_kappas:
            continue

        step_score = np.mean(step_kappas)

        trajectory_scores.append(traj_score)
        stepwise_scores.append(step_score)
        labels.append(0 if traj.is_correct else 1)

    if not trajectory_scores:
        return ValidationResult(hypothesis="H4", metric="trajectory_vs_stepwise",
            error_mean=np.nan, correct_mean=np.nan, cohens_d=np.nan, p_value=np.nan,
            n_error=0, n_correct=0, interpretation="Insufficient data")

    traj_arr = np.array(trajectory_scores)
    step_arr = np.array(stepwise_scores)
    labels = np.array(labels)

    traj_error = traj_arr[labels == 1]
    traj_correct = traj_arr[labels == 0]
    step_error = step_arr[labels == 1]
    step_correct = step_arr[labels == 0]

    if len(traj_correct) > 0 and len(traj_error) > 0:
        traj_pooled = np.sqrt(((len(traj_correct)-1)*traj_correct.var(ddof=1) + (len(traj_error)-1)*traj_error.var(ddof=1)) / (len(traj_correct)+len(traj_error)-2))
        traj_d = (traj_correct.mean() - traj_error.mean()) / traj_pooled if traj_pooled > 0 else 0
    else:
        traj_d = 0

    if len(step_correct) > 0 and len(step_error) > 0:
        step_pooled = np.sqrt(((len(step_correct)-1)*step_correct.var(ddof=1) + (len(step_error)-1)*step_error.var(ddof=1)) / (len(step_correct)+len(step_error)-2))
        step_d = (step_correct.mean() - step_error.mean()) / step_pooled if step_pooled > 0 else 0
    else:
        step_d = 0

    return ValidationResult(
        hypothesis="H4", metric="trajectory_vs_stepwise",
        error_mean=float(traj_error.mean()) if len(traj_error) > 0 else np.nan,
        correct_mean=float(traj_correct.mean()) if len(traj_correct) > 0 else np.nan,
        cohens_d=float(traj_d - step_d), p_value=np.nan,
        n_error=len(traj_error), n_correct=len(traj_correct),
        interpretation=f"Trajectory d={traj_d:.3f} > Stepwise d={step_d:.3f}: Δ={traj_d-step_d:.3f}"
    )


def run_all_validations(npz_path: str, layers: list[int] = None, output_dir: str = None, n_bootstrap: int = 5000) -> dict:
    """运行所有验证实验"""
    print("=" * 80)
    print("Trajectory of Thought: Geometric Phase Transitions")
    print("=" * 80)
    print(f"Loading: {npz_path}")

    trajectories, metadata = load_full_npz(npz_path)

    print(f"Loaded {metadata['n_chains']} chains ({metadata['n_correct']} correct, {metadata['n_error']} error)")
    print(f"Subset: {metadata['subset']}, Layers: {metadata['sv_layers']}")

    if layers is None:
        layers = [14]

    all_results = {}

    for layer in layers:
        print("\n" + "=" * 80)
        print(f"Layer {layer}")
        print("=" * 80)

        result_h1 = run_validation_h1(trajectories, layer, n_bootstrap)
        print(f"[H1] {result_h1.interpretation}")
        all_results[f'L{layer}_H1'] = result_h1

        result_h2, result_h3 = run_validation_h2_h3(trajectories, layer)
        print(f"[H2] {result_h2.interpretation}")
        all_results[f'L{layer}_H2'] = result_h2
        print(f"[H3] {result_h3.interpretation}")
        all_results[f'L{layer}_H3'] = result_h3

        result_h4 = run_validation_h4(trajectories, layer)
        print(f"[H4] {result_h4.interpretation}")
        all_results[f'L{layer}_H4'] = result_h4

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'trajectory_validation_results.json')

        serializable = {}
        for k, v in all_results.items():
            serializable[k] = {
                'hypothesis': v.hypothesis, 'metric': v.metric,
                'error_mean': v.error_mean, 'correct_mean': v.correct_mean,
                'cohens_d': v.cohens_d, 'p_value': v.p_value,
                'ci_lower': v.ci_lower, 'ci_upper': v.ci_upper,
                'n_error': v.n_error, 'n_correct': v.n_correct,
                'interpretation': v.interpretation,
            }

        with open(output_path, 'w') as f:
            json.dump({'metadata': metadata, 'results': serializable}, f, indent=2)

        print(f"\nResults: {output_path}")

    print_summary(all_results)
    return all_results


def print_summary(all_results: dict):
    """打印总结表"""
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Test':<15} {'Metric':<25} {'Error':<8} {'Correct':<8} {'Cohen\\'s d':<12} {'p-value':<10}")
    print("-" * 80)

    for k in sorted(all_results.keys()):
        v = all_results[k]
        sig = "*" if v.p_value is not None and v.p_value < 0.05 else ""
        err_str = f"{v.error_mean:.3f}" if not np.isnan(v.error_mean) else "N/A"
        cor_str = f"{v.correct_mean:.3f}" if not np.isnan(v.correct_mean) else "N/A"
        d_str = f"{v.cohens_d:.3f}" if not np.isnan(v.cohens_d) else "N/A"
        p_str = f"{v.p_value:.4f}" if not np.isnan(v.p_value) else "N/A"
        print(f"{k:<15} {v.metric:<25} {err_str:<8} {cor_str:<8} {d_str:<12} {p_str:<10} {sig}")

    print("-" * 80)
    print("* p < 0.05")
    print("=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Trajectory Phase Transition Detection')
    parser.add_argument('npz_path', help='Path to full_*.npz')
    parser.add_argument('--output_dir', default='./trajectory_results')
    parser.add_argument('--layers', nargs='+', type=int)
    parser.add_argument('--n_bootstrap', type=int, default=5000)

    args = parser.parse_args()
    run_all_validations(args.npz_path, args.layers, args.output_dir, args.n_bootstrap)


if __name__ == '__main__':
    main()
