"""Geometric verification experiment: Step Flow → Spectral Geometry migration.

核心假设：Step Flow描述的信息流断裂模式可以在几何角度观测到。

实验设计：
1. 谱形状特征 - 不只是标量，而是完整的谱分布
2. 步骤间几何距离 - Bures/W2距离捕捉表示空间漂移
3. 流形结构度量 - 曲率、相干性、各向异性
4. 非线性聚合 - 将几何特征映射到检测空间

与Step Flow模式的对应：
- Shallow Lock-in → 浅层谱突然坍缩（eff_rank下降，top_concentration上升）
- Deep Decay → 深层谱衰减加速（α指数过大）
- 步骤自强化 → 对角block谱集中度高
- Summary孤立 → W2(thinking, summary)过大
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class StepCloud:
    """单个步骤在特定层的token云表示"""

    def __init__(self, H: np.ndarray, step_id: int, layer: int):
        """
        Args:
            H: (n, d) token hidden states
            step_id: 步骤索引
            layer: 层索引
        """
        self.H = H
        self.step_id = step_id
        self.layer = layer
        self.n_tokens, self.dim = H.shape

        # 缓存计算结果
        self._mu = None
        self._cov = None
        self._eigenvals = None
        self._eigenvecs = None

    @property
    def mu(self) -> np.ndarray:
        """质心（位置）"""
        if self._mu is None:
            self._mu = self.H.mean(axis=0)
        return self._mu

    @property
    def cov(self) -> np.ndarray:
        """协方差矩阵"""
        if self._cov is None:
            Hc = self.H - self.mu
            self._cov = (Hc.T @ Hc) / max(self.n_tokens - 1, 1)
        return self._cov

    @property
    def eigenvals(self) -> np.ndarray:
        """特征值（降序）"""
        if self._eigenvals is None:
            eigvals = np.linalg.eigvalsh(self.cov)
            self._eigenvals = np.sort(eigvals)[::-1]
        return self._eigenvals

    @property
    def eigenvecs(self) -> np.ndarray:
        """特征向量（按特征值降序）"""
        if self._eigenvecs is None:
            eigvals, eigvecs = np.linalg.eigh(self.cov)
            idx = np.argsort(eigvals)[::-1]
            self._eigenvals = eigvals[idx]
            self._eigenvecs = eigvecs[:, idx]
        return self._eigenvecs


class ChainGeometry:
    """完整推理链的几何表示"""

    def __init__(self, step_clouds: dict[int, dict[int, StepCloud]],
                 is_correct: bool, problem_id: int):
        """
        Args:
            step_clouds: {step_id: {layer: StepCloud}}
            is_correct: 是否正确
            problem_id: 问题ID
        """
        self.step_clouds = step_clouds
        self.is_correct = is_correct
        self.problem_id = problem_id
        self.steps = sorted(step_clouds.keys())
        self.layers = sorted(list(next(iter(step_clouds.values())).keys()))

    def get_cloud(self, step: int, layer: int) -> StepCloud | None:
        return self.step_clouds.get(step, {}).get(layer)


# ---------------------------------------------------------------------------
# Spectral shape features (not just scalars!)
# ---------------------------------------------------------------------------


def spectral_decay_rate(eigenvals: np.ndarray, k: int = 20,
                        eps: float = 1e-12) -> float:
    """拟合谱衰减指数：λ_i ∝ i^(-α)

    Args:
        eigenvals: 特征值（降序）
        k: 使用前k个特征值拟合
        eps: 数值下界

    Returns:
        α指数（越大衰减越快）
    """
    eig = eigenvals[eigenvals > eps][:k]
    if eig.size < 5:
        return np.nan

    i = np.arange(1, len(eig) + 1)
    log_eig = np.log(eig + eps)
    log_i = np.log(i)

    # 线性拟合：log(λ) = log(c) - α*log(i)
    slope, _ = np.polyfit(log_i, log_eig, 1)
    return -slope  # α = -slope


def spectral_shape_vector(eigenvals: np.ndarray, k: int = 10,
                          eps: float = 1e-12) -> np.ndarray:
    """归一化的谱形状向量（用于比较不同步骤/层的谱形状）

    包含：
    - 前k个特征值的相对比例
    - 衰减指数
    - 拐点位置
    """
    eig = eigenvals[eigenvals > eps]
    if eig.size == 0:
        return np.full(k + 2, np.nan)

    # 1. 前k个特征值的相对比例
    top_k = np.zeros(k)
    top_k[:min(k, eig.size)] = eig[:k] / (eig.sum() + eps)

    # 2. 衰减指数
    alpha = spectral_decay_rate(eig)

    # 3. 拐点位置（最大差值位置）
    if eig.size > 2:
        diffs = eig[:-1] - eig[1:]
        elbow = np.argmax(diffs) / eig.size
    else:
        elbow = np.nan

    return np.concatenate([top_k, [alpha, elbow]])


def spectral_concentration(eigenvals: np.ndarray,
                           eps: float = 1e-12) -> dict[str, float]:
    """多尺度谱集中度（不只是top_concentration）

    Returns:
        dict: {
            'c1': λ_1 / Σλ (top集中度),
            'c2': (λ_1+λ_2) / Σλ (top-2集中度),
            'c5': (λ_1+...+λ_5) / Σλ (top-5集中度),
            'eff_rank': exp(熵),
            'entropy': 谱熵,
        }
    """
    eig = eigenvals[eigenvals > eps]
    if eig.size == 0:
        return {k: np.nan for k in ['c1', 'c2', 'c5', 'eff_rank', 'entropy']}

    total = eig.sum()
    cdf = np.cumsum(eig) / total

    p = eig / total
    entropy = -np.sum(p * np.log(p + eps))
    eff_rank = np.exp(entropy)

    return {
        'c1': cdf[0] if eig.size > 0 else np.nan,
        'c2': cdf[1] if eig.size > 1 else np.nan,
        'c5': cdf[4] if eig.size > 4 else np.nan,
        'eff_rank': eff_rank,
        'entropy': entropy,
    }


# ---------------------------------------------------------------------------
# Step-to-step geometric distances
# ---------------------------------------------------------------------------


def bures_distance(cov1: np.ndarray, cov2: np.ndarray,
                  eps: float = 1e-12) -> float:
    """Bures距离（Wasserstein-2 between Gaussians的shape部分）

    Bures^2(A,B) = tr(A) + tr(B) - 2tr((A^{1/2}BA^{1/2})^{1/2})
    """
    # 对称化确保PSD
    A = (cov1 + cov1.T) / 2
    B = (cov2 + cov2.T) / 2

    # A的平方根
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 0, None)
    A_sqrt = (V * np.sqrt(w)[None, :]) @ V.T

    # 中间矩阵的特征值
    M = A_sqrt @ B @ A_sqrt
    w, _ = np.linalg.eigh((M + M.T) / 2)
    w = np.clip(w, 0, None)

    cross = np.sum(np.sqrt(w))
    val = np.trace(A) + np.trace(B) - 2 * cross
    return float(np.sqrt(max(val, 0)))


def gaussian_w2_distance(mu1: np.ndarray, cov1: np.ndarray,
                         mu2: np.ndarray, cov2: np.ndarray) -> float:
    """完整的Wasserstein-2距离（位置+形状）

    W2^2 = ||μ1-μ2||^2 + Bures^2(Σ1, Σ2)
    """
    pos = np.sum((mu1 - mu2) ** 2)
    bur = bures_distance(cov1, cov2)
    return float(np.sqrt(max(pos + bur ** 2, 0)))


def spectral_js_divergence(eig1: np.ndarray, eig2: np.ndarray,
                          eps: float = 1e-12) -> float:
    """谱分布的JS散度（比较谱形状）"""
    # 归一化为概率分布
    p1 = eig1 / (eig1.sum() + eps)
    p2 = eig2 / (eig2.sum() + eps)

    return jensenshannon(p1, p2)


def compute_step_distances(chain: ChainGeometry,
                          layer: int) -> dict[str, np.ndarray]:
    """计算步骤间距离矩阵

    Returns:
        dict: {
            'w2': (T,T) Wasserstein距离矩阵,
            'bures': (T,T) Bures距离矩阵,
            'js': (T,T) JS散度矩阵,
        }
    """
    steps = chain.steps
    T = len(steps)

    w2_mat = np.zeros((T, T))
    bures_mat = np.zeros((T, T))
    js_mat = np.zeros((T, T))

    clouds = [chain.get_cloud(s, layer) for s in steps]

    for i in range(T):
        for j in range(i + 1, T):
            if clouds[i] is None or clouds[j] is None:
                continue

            # W2距离
            w2_ij = gaussian_w2_distance(
                clouds[i].mu, clouds[i].cov,
                clouds[j].mu, clouds[j].cov
            )
            w2_mat[i, j] = w2_ij
            w2_mat[j, i] = w2_ij

            # Bures距离
            bur_ij = bures_distance(clouds[i].cov, clouds[j].cov)
            bures_mat[i, j] = bur_ij
            bures_mat[j, i] = bur_ij

            # JS散度
            js_ij = spectral_js_divergence(clouds[i].eigenvals,
                                           clouds[j].eigenvals)
            js_mat[i, j] = js_ij
            js_mat[j, i] = js_ij

    return {'w2': w2_mat, 'bures': bures_mat, 'js': js_mat}


# ---------------------------------------------------------------------------
# Layer-wise patterns (Shallow Lock-in & Deep Decay detection)
# ---------------------------------------------------------------------------


def detect_shallow_lockin(chain: ChainGeometry,
                          layer_band: str = 'shallow',
                          n_shallow: int = 4) -> dict[str, float]:
    """检测Shallow Lock-in模式

    指标：
    - shallow_eff_rank_drop: 浅层有效秩下降程度
    - shallow_concentration_spike: 浅层集中度突增
    - self_reinforcement: 对角block质量
    """
    if layer_band == 'shallow':
        layers = chain.layers[:n_shallow]
    else:  # 'deep'
        layers = chain.layers[-n_shallow:]

    metrics = {'eff_rank': [], 'c1': [], 'c5': []}

    for layer in layers:
        for step in chain.steps:
            cloud = chain.get_cloud(step, layer)
            if cloud is None:
                continue
            conc = spectral_concentration(cloud.eigenvals)
            metrics['eff_rank'].append(conc['eff_rank'])
            metrics['c1'].append(conc['c1'])
            metrics['c5'].append(conc['c5'])

    if not metrics['eff_rank']:
        return {'shallow_eff_rank_drop': np.nan,
                'shallow_concentration_spike': np.nan}

    # 与前半段比较（模拟问题context）
    mid = len(metrics['eff_rank']) // 2
    early_rank = np.nanmean(metrics['eff_rank'][:mid])
    late_rank = np.nanmean(metrics['eff_rank'][mid:])
    early_c1 = np.nanmean(metrics['c1'][:mid])
    late_c1 = np.nanmean(metrics['c1'][mid:])

    return {
        'shallow_eff_rank_drop': early_rank - late_rank,  # 正值=下降
        'shallow_concentration_spike': late_c1 - early_c1,  # 正值=上升
    }


def detect_deep_decay(chain: ChainGeometry,
                     n_deep: int = 4) -> dict[str, float]:
    """检测Deep Decay模式

    指标：
    - decay_alpha: 深层衰减指数
    - thinking_summary_disconnect: thinking段与summary段的距离
    """
    deep_layers = chain.layers[-n_deep:]

    alphas = []
    for layer in deep_layers:
        for step in chain.steps[:-1]:  # 排除summary
            cloud = chain.get_cloud(step, layer)
            if cloud is None:
                continue
            alpha = spectral_decay_rate(cloud.eigenvals)
            if np.isfinite(alpha):
                alphas.append(alpha)

    if not alphas:
        return {'decay_alpha': np.nan}

    return {
        'decay_alpha': np.mean(alphas),
        'decay_alpha_std': np.std(alphas),
    }


# ---------------------------------------------------------------------------
# Non-linear aggregation: Manifold mapping
# ---------------------------------------------------------------------------


class GeometricEmbedding:
    """将几何特征非线性映射到检测空间

    方法：
    1. 核方法：将谱形状映射到RKHS
    2. 流形学习：将多维特征降维到健康-疾病轴
    3. 组合得分：多种几何信号的加权非线性组合
    """

    def __init__(self, method: str = 'kernel'):
        self.method = method
        self.reference = None  # 健康基线（从correct样本学习）

    def fit_reference(self, correct_chains: list[ChainGeometry],
                      layer: int):
        """从正确样本学习健康基线"""
        if self.method == 'kernel':
            # 收集参考谱形状
            ref_shapes = []
            for chain in correct_chains:
                for step in chain.steps:
                    cloud = chain.get_cloud(step, layer)
                    if cloud is not None:
                        shape = spectral_shape_vector(cloud.eigenvals)
                        if np.all(np.isfinite(shape)):
                            ref_shapes.append(shape)
            if ref_shapes:
                self.reference = np.vstack(ref_shapes)
                self.ref_mean = self.reference.mean(axis=0)
                self.ref_cov = np.cov(self.reference.T)

    def score_deviation(self, shape: np.ndarray) -> float:
        """计算偏离健康基线的程度（马氏距离）"""
        if self.reference is None:
            return np.nan

        diff = shape - self.ref_mean
        try:
            mahal = diff @ np.linalg.inv(self.ref_cov) @ diff.T
            return float(np.sqrt(max(mahal, 0)))
        except np.linalg.LinAlgError:
            return float(np.linalg.norm(diff))

    def aggregate_signal(self, chain: ChainGeometry,
                        layer: int) -> dict[str, float]:
        """聚合几何检测信号

        Returns:
            dict: {
                'shape_mahal': 谱形状偏离度,
                'step_disconnection': 步骤间平均距离,
                'lockin_score': lock-in模式强度,
                'decay_score': decay模式强度,
                'combined': 非线性组合得分,
            }
        """
        signals = {}

        # 1. 谱形状偏离
        shapes = []
        for step in chain.steps:
            cloud = chain.get_cloud(step, layer)
            if cloud is not None:
                shape = spectral_shape_vector(cloud.eigenvals)
                if np.all(np.isfinite(shape)):
                    shapes.append(shape)

        if shapes and self.reference is not None:
            mahals = [self.score_deviation(s) for s in shapes]
            signals['shape_mahal'] = np.mean(mahals)
        else:
            signals['shape_mahal'] = np.nan

        # 2. 步骤间连接性
        dists = compute_step_distances(chain, layer)
        # 排除对角线（自距离）
        off_diag = dists['w2'][~np.eye(dists['w2'].shape[0], dtype=bool)]
        signals['step_disconnection'] = np.mean(off_diag) if off_diag.size > 0 else np.nan

        # 3. 模式强度
        lockin = detect_shallow_lockin(chain, layer_band='shallow')
        decay = detect_deep_decay(chain)

        signals['lockin_score'] = (lockin['shallow_concentration_spike'] -
                                   lockin['shallow_eff_rank_drop'])
        signals['decay_score'] = decay['decay_alpha']

        # 4. 非线性组合（sigmoid归一化后相乘）
        def sigmoid(x): return 1 / (1 + np.exp(-x))

        components = []
        if np.isfinite(signals['shape_mahal']):
            components.append(sigmoid(signals['shape_mahal'] - 2))  # >2显著偏离
        if np.isfinite(signals['step_disconnection']):
            components.append(sigmoid(signals['step_disconnection'] -
                                    signals['step_disconnection']))  # 相对于自身
        if np.isfinite(signals['lockin_score']):
            components.append(sigmoid(signals['lockin_score']))
        if np.isfinite(signals['decay_score']):
            components.append(sigmoid((signals['decay_score'] - 1.5) * 2))  # α>1.5加速衰减

        if components:
            signals['combined'] = np.prod(components)
        else:
            signals['combined'] = np.nan

        return signals


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def load_chain_data(npz_path: str) -> tuple[list[ChainGeometry], dict]:
    """从NPZ文件加载ChainGeometry对象

    期望的NPZ结构（兼容extract_features.py输出）：
    - stepgeom: (N, T, L, F) per-step几何特征
    - layers_used: 使用的层列表
    - problem_ids: 问题ID
    - is_correct_strict: 正确性标签
    """
    data = np.load(npz_path, allow_pickle=True)

    chains = []
    meta = {
        'layers': [int(l) for l in data['layers_used']],
        'problems': data['problem_ids'].astype(int),
        'labels': data['is_correct_strict'].astype(int),
    }

    # 这里需要根据实际的NPZ结构适配
    # 如果有原始hidden states，需要从那里构建StepCloud

    return chains, meta


def run_verification_experiment(npz_path: str, output_dir: str):
    """运行几何验证实验

    1. 加载correct和error chains
    2. 计算每条chain的几何特征
    3. 比较error vs correct的几何差异
    4. 输出统计结果
    """
    chains, meta = load_chain_data(npz_path)

    if not chains:
        print("No chains loaded. Need to implement data loading.")
        return

    # 分组
    correct = [c for c in chains if c.is_correct]
    error = [c for c in chains if not c.is_correct]

    print(f"Loaded {len(correct)} correct, {len(error)} error chains")

    # 选择分析层（建议：early=L4, mid=L16, late=L28）
    layers_to_analyze = [meta['layers'][len(meta['layers']) // 3],
                         meta['layers'][len(meta['layers']) // 2]]

    results = {}

    for layer in layers_to_analyze:
        print(f"\n=== Analyzing layer {layer} ===")

        # 学习健康基线
        embedder = GeometricEmbedding(method='kernel')
        embedder.fit_reference(correct, layer)

        # 计算所有chain的信号
        correct_signals = []
        error_signals = []

        for chain in correct:
            sig = embedder.aggregate_signal(chain, layer)
            correct_signals.append(sig)

        for chain in error:
            sig = embedder.aggregate_signal(chain, layer)
            error_signals.append(sig)

        # 统计检验
        for key in ['shape_mahal', 'step_disconnection', 'lockin_score',
                    'decay_score', 'combined']:
            c_vals = [s[key] for s in correct_signals if np.isfinite(s[key])]
            e_vals = [s[key] for s in error_signals if np.isfinite(s[key])]

            if not c_vals or not e_vals:
                continue

            # Mann-Whitney U检验
            stat, pval = stats.mannwhitneyu(e_vals, c_vals,
                                            alternative='greater')
            # Cohen's d
            pooled_std = np.sqrt(((len(c_vals)-1)*np.var(c_vals, ddof=1) +
                                  (len(e_vals)-1)*np.var(e_vals, ddof=1)) /
                                 (len(c_vals)+len(e_vals)-2))
            cohens_d = (np.mean(e_vals) - np.mean(c_vals)) / pooled_std

            print(f"{key}: error>{' ' if np.mean(e_vals)>np.mean(c_vals) else '<'}correct "
                  f"| U={stat:.1f}, p={pval:.4f}, d={cohens_d:.3f}")

            results[f'L{layer}_{key}'] = {
                'error_mean': float(np.mean(e_vals)),
                'correct_mean': float(np.mean(c_vals)),
                'p_value': float(pval),
                'cohens_d': float(cohens_d),
            }

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'geometry_verification_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Geometric verification: Step Flow → Spectral Geometry'
    )
    parser.add_argument('npz', help='Path to features NPZ file')
    parser.add_argument('--output', default='./results',
                        help='Output directory')
    args = parser.parse_args()

    run_verification_experiment(args.npz, args.output)


if __name__ == '__main__':
    main()
