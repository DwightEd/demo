"""几何验证实验套件：基于todo list的完整实现

实验目标：
1. 验证二阶矩(eff_rank/spectrum)在难任务上的普适性
2. 检测Shallow Lock-in（浅层κ+eff_rank双降）
3. 检测Deep Decay（深层α衰减加速）
4. 构建κ+eff_rank非线性联合检测器

与"The Shape of Reasoning"的关联：
- 该文使用拓扑数据分析(TDA)中的持久同伦(Persistent Homology)
- 核心思想：推理轨迹的"形状"(拓扑结构)包含质量信息
- 我们借鉴：用谱形状和步骤间距离刻画推理轨迹的几何形状
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy import stats
from scipy.linalg import eigh
from scipy.optimize import curve_fit
from scipy.spatial.distance import jensenshannon


# =============================================================================
# Part 1: 核心几何特征（一阶+二阶矩）
# =============================================================================


@dataclass
class StepGeometry:
    """单个步骤的完整几何描述

    包含：
    - 一阶矩：κ（方向集中度）
    - 二阶矩：eff_rank（散布熵）、spectrum（谱形状）、α（衰减指数）
    """
    step_id: int
    layer: int
    n_tokens: int

    # 一阶矩
    kappa: float = field(default=np.nan)

    # 二阶矩
    eff_rank: float = field(default=np.nan)
    spectrum: np.ndarray = field(default_factory=lambda: np.array([]))
    decay_alpha: float = field(default=np.nan)

    # 位置
    norm: float = field(default=np.nan)
    centroid: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ChainGeometry:
    """完整推理链的几何表示

    用于：
    - 检测Shallow Lock-in
    - 检测Deep Decay
    - 计算步骤间连贯性
    """
    steps: list[StepGeometry] = field(default_factory=list)
    is_correct: bool = False
    problem_id: int = -1

    def add_step(self, geom: StepGeometry):
        self.steps.append(geom)

    def get_steps_by_layer(self, layer: int) -> list[StepGeometry]:
        return [s for s in self.steps if s.layer == layer]


def compute_step_geometry(H: np.ndarray, step_id: int, layer: int,
                          use_exp_weights: bool = True) -> StepGeometry:
    """从token云计算步骤几何特征

    Args:
        H: (n, d) token hidden states
        step_id: 步骤索引
        layer: 层索引
        use_exp_weights: 是否使用exp权重（后段token更重要）

    Returns:
        StepGeometry对象
    """
    H = np.asarray(H, dtype=np.float64)
    n, d = H.shape

    geom = StepGeometry(step_id=step_id, layer=layer, n_tokens=n)

    # === 一阶矩：κ ===
    # 归一化token
    H_norm = H / np.linalg.norm(H, axis=1, keepdims=True)

    # exp权重（后段更重要）
    if use_exp_weights and n > 1:
        pos = np.arange(n) / (n - 1)
        weights = np.exp(pos)
    else:
        weights = np.ones(n)

    weights = weights / weights.sum()

    # 加权平均方向
    mean_dir = (weights[:, None] * H_norm).sum(axis=0)
    geom.kappa = float(np.linalg.norm(mean_dir))

    # === 二阶矩：散布矩阵 ===
    # S = (1/n)Σ ûᵀû
    scatter = (H_norm.T @ H_norm) / n

    # 特征值分解
    try:
        eigenvals = eigh(scatter, eigvals_only=True)
        eigenvals = np.sort(eigenvals)[::-1]  # 降序
        eigenvals = eigenvals / eigenvals.sum()  # 归一化
    except:
        eigenvals = np.ones(d) / d

    geom.eff_rank = float(np.exp(-np.sum(eigenvals * np.log(eigenvals + 1e-12))))
    geom.spectrum = eigenvals[:5]  # 存前5个

    # === 衰减指数α ===
    if eigenvals.size >= 5:
        i = np.arange(1, min(10, len(eigenvals)) + 1)
        lam = eigenvals[:len(i)]
        try:
            log_lam = np.log(lam + 1e-12)
            log_i = np.log(i)
            alpha, _ = np.polyfit(log_i, log_lam, 1)
            geom.decay_alpha = -alpha
        except:
            geom.decay_alpha = np.nan
    else:
        geom.decay_alpha = np.nan

    # === 位置信息 ===
    geom.norm = float(np.linalg.norm(H.mean(axis=0)))
    geom.centroid = H.mean(axis=0)

    return geom


# =============================================================================
# Part 2: 谱形状分析（借鉴TDA思想）
# =============================================================================


def compute_spectral_shape_vector(eigenvals: np.ndarray, k: int = 10) -> dict:
    """计算完整的谱形状描述

    借鉴TDA思想：用多尺度特征描述"形状"

    Returns:
        dict: {
            'top_ratios': 前5个特征值占比,
            'decay_alpha': 衰减指数,
            'elbow_position': 拐点位置,
            'entropy': 谱熵,
            'gini': 基尼系数,
        }
    """
    eig = eigenvals[eigenvals > 1e-12]
    if eig.size == 0:
        return {'top_ratios': np.zeros(5), 'decay_alpha': np.nan,
                'elbow_position': np.nan, 'entropy': np.nan, 'gini': np.nan}

    # 归一化
    eig = eig / eig.sum()

    result = {}

    # 1. 前k个占比
    result['top_ratios'] = eig[:5]

    # 2. 衰减指数
    if eig.size >= 5:
        i = np.arange(1, min(10, len(eig)) + 1)
        lam = eig[:len(i)]
        try:
            log_lam = np.log(lam + 1e-12)
            log_i = np.log(i)
            alpha, _ = np.polyfit(log_i, log_lam, 1)
            result['decay_alpha'] = -alpha
        except:
            result['decay_alpha'] = np.nan
    else:
        result['decay_alpha'] = np.nan

    # 3. 拐点位置（最大差值位置）
    if eig.size > 2:
        diffs = eig[:-1] - eig[1:]
        elbow = np.argmax(diffs) / eig.size
        result['elbow_position'] = elbow
    else:
        result['elbow_position'] = np.nan

    # 4. 谱熵
    result['entropy'] = float(np.sum(-eig * np.log(eig + 1e-12)))

    # 5. 基尼系数（不平等程度）
    sorted_eig = np.sort(eig)
    n = len(sorted_eig)
    gini = np.sum(2 * np.arange(1, n + 1) - n - 1) * sorted_eig / (n * np.sum(sorted_eig))
    result['gini'] = gini

    return result


def compare_spectral_shapes(spec1: dict, spec2: dict) -> float:
    """比较两个谱形状的相似度

    借鉴TDA的Wasserstein距离思想
    """
    # JS散度
    p = spec1['top_ratios']
    q = spec2['top_ratios']
    js = jensenshannon(p, q)

    # 衰减指数差
    alpha_diff = abs(spec1.get('decay_alpha', 0) - spec2.get('decay_alpha', 0))

    # 组合分数
    return float(js + alpha_diff * 0.1)


# =============================================================================
# Part 3: 步骤间几何距离
# =============================================================================


def bures_distance(cov1: np.ndarray, cov2: np.ndarray) -> float:
    """Bures距离（Wasserstein-2的shape部分）

    Bures²(A,B) = tr(A) + tr(B) - 2tr((A^{1/2}BA^{1/2})^{1/2})
    """
    # 对称化
    A = (cov1 + cov1.T) / 2
    B = (cov2 + cov2.T) / 2

    # A的平方根
    try:
        w, V = eigh(A)
        w = np.clip(w, 0, None)
        A_sqrt = (V * np.sqrt(w)[None, :]) @ V.T

        # 中间矩阵
        M = A_sqrt @ B @ A_sqrt
        w, _ = eigh((M + M.T) / 2)
        w = np.clip(w, 0, None)
        cross = np.sum(np.sqrt(w))

        val = np.trace(A) + np.trace(B) - 2 * cross
        return float(np.sqrt(max(val, 0)))
    except:
        return np.nan


def w2_distance(mu1: np.ndarray, cov1: np.ndarray,
                mu2: np.ndarray, cov2: np.ndarray) -> float:
    """完整Wasserstein-2距离（位置+形状）"""
    pos = np.sum((mu1 - mu2) ** 2)
    bur = bures_distance(cov1, cov2)
    return float(np.sqrt(max(pos + bur ** 2, 0)))


def compute_step_distances(chain: ChainGeometry, layer: int) -> dict:
    """计算步骤间距离矩阵

    Returns:
        dict: {
            'w2_matrix': (T,T) Wasserstein距离,
            'bures_matrix': (T,T) Bures距离,
            'mean_w2': 平均W2距离,
            'disconnection_score': 断连分数,
        }
    """
    steps = chain.get_steps_by_layer(layer)
    if len(steps) < 2:
        return {'w2_matrix': np.array([]), 'bures_matrix': np.array([]),
                'mean_w2': np.nan, 'disconnection_score': np.nan}

    T = len(steps)
    w2_mat = np.zeros((T, T))
    bures_mat = np.zeros((T, T))

    # 简化：假设每步的协方差可以用谱形状估计
    for i in range(T):
        for j in range(i + 1, T):
            # 用谱差异作为Bures近似
            spec_i = compute_spectral_shape_vector(
                np.concatenate([steps[i].spectrum, [1 - steps[i].spectrum.sum()]]))
            spec_j = compute_spectral_shape_vector(
                np.concatenate([steps[j].spectrum, [1 - steps[j].spectrum.sum()]]))

            js = jensenshannon(spec_i['top_ratios'], spec_j['top_ratios'])
            bures_mat[i, j] = js
            bures_mat[j, i] = js

            # W2 = 位置 + Bures
            if steps[i].centroid.size > 0 and steps[j].centroid.size > 0:
                pos = np.sum((steps[i].centroid - steps[j].centroid) ** 2)
                w2_mat[i, j] = np.sqrt(pos + js ** 2)
                w2_mat[j, i] = w2_mat[i, j]

    # 断连分数：与前一歩的平均距离
    off_diag = w2_mat[~np.eye(T, dtype=bool)]
    mean_w2 = off_diag.mean() if off_diag.size > 0 else np.nan

    # 相邻步的平均距离
    adj_w2 = np.mean([w2_mat[i, i + 1] for i in range(T - 1)])

    return {
        'w2_matrix': w2_mat,
        'bures_matrix': bures_mat,
        'mean_w2': mean_w2,
        'adjacent_w2': adj_w2,
        'disconnection_score': adj_w2 if np.isfinite(adj_w2) else np.nan,
    }


# =============================================================================
# Part 4: Shallow Lock-in检测
# =============================================================================


def detect_shallow_lockin(chain: ChainGeometry, layer: int,
                          threshold_kappa: float = 0.1,
                          threshold_eff_rank: float = 0.5) -> dict:
    """检测Shallow Lock-in模式

    模式特征：
    - κ突然下降（集中度降低）
    - eff_rank同时下降（散布熵降低）
    - 两者同时发生表示信息流断裂

    Returns:
        dict: {
            'has_lockin': 是否检测到lock-in,
            'lockin_step': 发生的步骤,
            'kappa_drop': κ下降量,
            'eff_rank_drop': eff_rank下降量,
        }
    """
    steps = chain.get_steps_by_layer(layer)
    if len(steps) < 3:
        return {'has_lockin': False, 'lockin_step': -1,
                'kappa_drop': np.nan, 'eff_rank_drop': np.nan}

    kappas = [s.kappa for s in steps if np.isfinite(s.kappa)]
    eff_ranks = [s.eff_rank for s in steps if np.isfinite(s.eff_rank)]

    if len(kappas) < 3 or len(eff_ranks) < 3:
        return {'has_lockin': False, 'lockin_step': -1,
                'kappa_drop': np.nan, 'eff_rank_drop': np.nan}

    # 计算移动平均变化
    kappa_diff = np.diff(kappas)
    eff_rank_diff = np.diff(eff_ranks)

    # 寻找同时下降的位置
    lockin_step = -1
    max_drop = 0

    for i in range(len(kappa_diff)):
        if kappa_diff[i] < -threshold_kappa and eff_rank_diff[i] < -threshold_eff_rank:
            drop = abs(kappa_diff[i]) + abs(eff_rank_diff[i])
            if drop > max_drop:
                max_drop = drop
                lockin_step = i

    return {
        'has_lockin': lockin_step >= 0,
        'lockin_step': lockin_step,
        'kappa_drop': float(kappa_diff[lockin_step]) if lockin_step >= 0 else np.nan,
        'eff_rank_drop': float(eff_rank_diff[lockin_step]) if lockin_step >= 0 else np.nan,
    }


# =============================================================================
# Part 5: Deep Decay检测
# =============================================================================


def detect_deep_decay(chain: ChainGeometry, layer: int,
                     alpha_threshold: float = 1.5) -> dict:
    """检测Deep Decay模式

    模式特征：
    - 谱衰减指数α过大（>1.5表示快速衰减）
    - 高能量集中在少数几个方向
    - 后续步骤缺乏早期推理的信息

    Returns:
        dict: {
            'has_decay': 是否检测到decay,
            'mean_alpha': 平均衰减指数,
            'decay_steps': 衰减严重的步骤列表,
        }
    """
    steps = chain.get_steps_by_layer(layer)
    alphas = [s.decay_alpha for s in steps if np.isfinite(s.decay_alpha)]

    if not alphas:
        return {'has_decay': False, 'mean_alpha': np.nan, 'decay_steps': []}

    mean_alpha = np.mean(alphas)

    # 找出α过大的步骤
    decay_steps = [i for i, s in enumerate(steps)
                  if np.isfinite(s.decay_alpha) and s.decay_alpha > alpha_threshold]

    return {
        'has_decay': len(decay_steps) > len(alphas) / 2,  # 过半数步骤
        'mean_alpha': mean_alpha,
        'decay_steps': decay_steps,
    }


# =============================================================================
# Part 6: 联合检测器（κ + eff_rank，非线性组合）
# =============================================================================


@dataclass
class DetectionResult:
    """检测结果"""
    risk_score: float  # 0-1，越高越危险
    risk_level: str    # 'low', 'medium', 'high'
    kappa_signal: float
    eff_rank_signal: float
    pattern_detected: list[str]  # 检测到的模式


class KappaEffRankDetector:
    """κ + eff_rank 联合检测器

    设计原则：
    1. 非线性组合（逻辑与：两者都异常才报警）
    2. 可解释（每个信号都有清晰含义）
    3. 无监督（不需要标签学习）
    """

    def __init__(self,
                 kappa_low_threshold: float = 0.5,
                 eff_rank_low_threshold: float = 2.0,
                 eff_rank_high_threshold: float = 10.0):
        """
        Args:
            kappa_low_threshold: κ低于此值表示分散
            eff_rank_low_threshold: eff_rank低于此值表示过度集中
            eff_rank_high_threshold: eff_rank高于此值表示过度分散
        """
        self.kappa_low = kappa_low_threshold
        self.eff_rank_low = eff_rank_low_threshold
        self.eff_rank_high = eff_rank_high_threshold

    def detect(self, chain: ChainGeometry, layer: int) -> DetectionResult:
        """检测单条推理链"""
        steps = chain.get_steps_by_layer(layer)
        if not steps:
            return DetectionResult(0.5, 'unknown', np.nan, np.nan, [])

        # 计算平均信号
        kappas = [s.kappa for s in steps if np.isfinite(s.kappa)]
        eff_ranks = [s.eff_rank for s in steps if np.isfinite(s.eff_rank)]

        if not kappas or not eff_ranks:
            return DetectionResult(0.5, 'unknown', np.nan, np.nan, [])

        mean_kappa = np.mean(kappas)
        mean_eff_rank = np.mean(eff_ranks)

        # 检测模式
        patterns = []

        # 模式1：κ低 + eff_rank高 = 各向同性扩散（可疑）
        if mean_kappa < self.kappa_low and mean_eff_rank > self.eff_rank_high:
            patterns.append('isotropic_diffusion')

        # 模式2：κ高 + eff_rank低 = 过度集中（可能lock-in）
        elif mean_kappa > 0.7 and mean_eff_rank < self.eff_rank_low:
            patterns.append('over_concentrated')

        # 模式3：两者都中等 = 健康推理
        elif 0.5 < mean_kappa < 0.8 and self.eff_rank_low < mean_eff_rank < self.eff_rank_high:
            patterns.append('healthy_reasoning')

        # 检测Lock-in
        lockin = detect_shallow_lockin(chain, layer)
        if lockin['has_lockin']:
            patterns.append('shallow_lockin')

        # 检测Decay
        decay = detect_deep_decay(chain, layer)
        if decay['has_decay']:
            patterns.append('deep_decay')

        # 计算风险分数（非线性）
        if 'isotropic_diffusion' in patterns or 'shallow_lockin' in patterns:
            risk = 0.8 + 0.2 * (0.5 - mean_kappa)
        elif 'over_concentrated' in patterns:
            risk = 0.6
        elif 'healthy_reasoning' in patterns:
            risk = 0.2
        else:
            risk = 0.5

        # 风险等级
        if risk > 0.7:
            level = 'high'
        elif risk > 0.4:
            level = 'medium'
        else:
            level = 'low'

        return DetectionResult(
            risk_score=np.clip(risk, 0, 1),
            risk_level=level,
            kappa_signal=mean_kappa,
            eff_rank_signal=mean_eff_rank,
            pattern_detected=patterns,
        )


# =============================================================================
# Part 7: 批量验证实验
# =============================================================================


@dataclass
class ValidationResults:
    """验证实验结果"""
    feature_name: str
    error_mean: float
    correct_mean: float
    cohens_d: float
    p_value: float
    n_error: int
    n_correct: int


def run_validation_experiment(chains: list[ChainGeometry],
                            layers: list[int],
                            output_dir: str | None = None) -> dict[str, ValidationResults]:
    """运行验证实验

    比较error vs correct在几何特征上的差异

    Returns:
        dict: {feature_name: ValidationResults}
    """
    results = {}

    # 分组
    correct_chains = [c for c in chains if c.is_correct]
    error_chains = [c for c in chains if not c.is_correct]

    for layer in layers:
        print(f"\n=== Layer {layer} ===")

        # 提取特征
        features_to_test = [
            ('kappa', lambda c: np.mean([s.kappa for s in c.get_steps_by_layer(layer)
                                       if np.isfinite(s.kappa)])),
            ('eff_rank', lambda c: np.mean([s.eff_rank for s in c.get_steps_by_layer(layer)
                                         if np.isfinite(s.eff_rank)])),
            ('decay_alpha', lambda c: np.mean([s.decay_alpha for s in c.get_steps_by_layer(layer)
                                              if np.isfinite(s.decay_alpha)])),
        ]

        for feat_name, feat_func in features_to_test:
            # 计算值
            correct_vals = [feat_func(c) for c in correct_chains]
            error_vals = [feat_func(c) for c in error_chains]

            # 过滤NaN
            correct_vals = [v for v in correct_vals if np.isfinite(v)]
            error_vals = [v for v in error_vals if np.isfinite(v)]

            if not correct_vals or not error_vals:
                continue

            # 统计检验
            correct_arr = np.array(correct_vals)
            error_arr = np.array(error_vals)

            # Mann-Whitney U
            stat, pval = stats.mannwhitneyu(error_arr, correct_arr, alternative='two-sided')

            # Cohen's d
            pooled_std = np.sqrt(((len(correct_arr)-1)*correct_arr.var(ddof=1) +
                                 (len(error_arr)-1)*error_arr.var(ddof=1)) /
                                (len(correct_arr)+len(error_arr)-2))
            cohens_d = (error_arr.mean() - correct_arr.mean()) / pooled_std if pooled_std > 0 else 0

            key = f'L{layer}_{feat_name}'
            results[key] = ValidationResults(
                feature_name=key,
                error_mean=float(error_arr.mean()),
                correct_mean=float(correct_arr.mean()),
                cohens_d=float(cohens_d),
                p_value=float(pval),
                n_error=len(error_arr),
                n_correct=len(correct_arr),
            )

            # 打印
            direction = ">" if error_arr.mean() > correct_arr.mean() else "<"
            print(f"{feat_name:15s}: error{direction}correct "
                  f"| error={error_arr.mean():.3f}, correct={correct_arr.mean():.3f} "
                  f"| d={cohens_d:.3f}, p={pval:.4f}")

    # 保存结果
    if output_dir:
        output_path = Path(output_dir) / 'geometry_validation_results.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serializable_results = {
            k: {
                'feature_name': v.feature_name,
                'error_mean': v.error_mean,
                'correct_mean': v.correct_mean,
                'cohens_d': v.cohens_d,
                'p_value': v.p_value,
                'n_error': v.n_error,
                'n_correct': v.n_correct,
            }
            for k, v in results.items()
        }

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\nResults saved to {output_path}")

    return results


# =============================================================================
# Part 8: 与TDA的关联（借鉴"The Shape of Reasoning"）
# =============================================================================


def compute_topological_signature(chain: ChainGeometry, layer: int) -> dict:
    """计算推理轨迹的"拓扑签名"

    借鉴TDA思想：
    1. 持久性：特征在不同尺度下的持久性
    2. Betti数：拓扑特征的数量
    3. 形状描述：轨迹的整体形状

    对于我们的几何框架：
    - 用谱形状的持久性（不同k下的top-k占比）
    - 用步骤间距离的持久性（不同尺度下的连通性）
    """
    steps = chain.get_steps_by_layer(layer)
    if len(steps) < 3:
        return {}

    # 1. 谱持久性：不同k下top-k占比的变化
    spectra = np.array([s.spectrum for s in steps if s.spectrum.size > 0])
    if spectra.size == 0:
        return {}

    k_values = [1, 2, 3, 5]
    persistence = {}
    for k in k_values:
        if spectra.shape[1] >= k:
            top_k_ratios = spectra[:, :k].sum(axis=1)
            persistence[f'top{k}_persistence'] = {
                'mean': float(top_k_ratios.mean()),
                'std': float(top_k_ratios.std()),
                'min': float(top_k_ratios.min()),
                'max': float(top_k_ratios.max()),
            }

    # 2. 步骤间距离的"持久性"
    dists = compute_step_distances(chain, layer)
    if dists['w2_matrix'].size > 0:
        w2_flat = dists['w2_matrix'][np.triu_indices_from(dists['w2_matrix'], k=1)]
        persistence['distance_persistence'] = {
            'mean': float(np.mean(w2_flat)),
            'std': float(np.std(w2_flat)),
        }

    # 3. "Betti数"类比：有效独立方向数
    eff_ranks = [s.eff_rank for s in steps if np.isfinite(s.eff_rank)]
    if eff_ranks:
        persistence['betti_analog'] = {
            'mean_eff_rank': float(np.mean(eff_ranks)),
            'min_eff_rank': float(np.min(eff_ranks)),
            'max_eff_rank': float(np.max(eff_ranks)),
        }

    return persistence


# =============================================================================
# 主函数
# =============================================================================


def main():
    """示例用法"""
    print(__doc__)
    print("""
验证实验检查清单：

□ 数据准备
  □ 从NPZ加载ChainGeometry对象
  □ 确认label（is_correct）
  □ 确认layers

□ 特征提取
  □ 每步骤的κ、eff_rank、spectrum
  □ 谱衰减指数α
  □ 步骤间距离矩阵

□ 模式检测
  □ Shallow Lock-in（κ+eff双降）
  □ Deep Decay（α过大）

□ 统计验证
  □ error vs correct的差异检验
  □ Cohen's d效应量
  □ bootstrap CI

□ 输出
  □ 完整结果JSON
  □ 每层每特征的统计报告
    """)


if __name__ == '__main__':
    main()
