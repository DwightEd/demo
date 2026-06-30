"""基于AAAI2027实验结果的研究路线图

核心发现：二阶矩谱(eff_rank/spectrum)在难任务上超越一阶矩κ

验证实验规划：
1. 在math/olympiad上确认二阶矩泛化性
2. 构建κ+eff_rank联合检测器（两个可解释无监督标量）
3. 与Step-Saliency对齐：Shallow Lock-in→谱坍缩，Deep Decay→α过大
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from typing import NamedTuple


# =============================================================================
# 核心验证：κ vs eff_rank在不同任务上的表现
# =============================================================================


class StepCloudGeometry:
    """步骤内token云的完整几何描述

    不只是标量，而是保留一阶+二阶矩的完整信息
    """

    def __init__(self, H: np.ndarray):
        """
        Args:
            H: (n, d) token hidden states (already projected if needed)
        """
        self.H = H
        self.n, self.d = H.shape
        self._mu = None
        self._scatter = None
        self._eigenvals = None

    @property
    def mu(self) -> np.ndarray:
        """质心（位置）"""
        if self._mu is None:
            self._mu = self.H.mean(axis=0)
        return self._mu

    @property
    def scatter_matrix(self) -> np.ndarray:
        """二阶矩：S = (1/n)Σ ûᵀû (Bingham/Watson充分统计量)

        其中 û = h/‖h‖ 是单位token向量
        """
        if self._scatter is None:
            # 每个token归一化
            H_norm = self.H / np.linalg.norm(self.H, axis=1, keepdims=True)
            self._scatter = (H_norm.T @ H_norm) / self.n
        return self._scatter

    @property
    def eigenvals(self) -> np.ndarray:
        """散布矩阵的特征值（降序，和为1）"""
        if self._eigenvals is None:
            eigvals = eigh(self.scatter_matrix, eigvals_only=True)
            self._eigenvals = np.sort(eigvals)[::-1]
        return self._eigenvals

    # ========== 一阶矩 ==========
    def kappa(self, weights: np.ndarray | None = None) -> float:
        """κ = ‖Σ w_t û_t‖ / Σ w_t

        第一阶矩：单位向量的加权平均长度
        range: [0, 1], 1=完全对齐, 0=各向同性
        """
        H_norm = self.H / np.linalg.norm(self.H, axis=1, keepdims=True)
        if weights is None:
            # 默认exp权重（后段token更重要）
            pos = np.arange(self.n) / max(self.n - 1, 1)
            weights = np.exp(pos)
        weights = weights / weights.sum()

        mean_direction = (weights[:, None] * H_norm).sum(axis=0)
        return float(np.linalg.norm(mean_direction))

    # ========== 二阶矩 ==========
    def eff_rank(self, eps: float = 1e-12) -> float:
        """有效秩 = exp(−Σ λ_i log λ_i)

        二阶矩：散布矩阵的特征值熵
        range: [1, n], 1=单方向, n=各向同性
        """
        lam = self.eigenvals
        lam = lam[lam > eps]
        if lam.size == 0:
            return 1.0
        return float(np.exp(-np.sum(lam * np.log(lam + eps))))

    def spectral_shape(self, k: int = 5) -> np.ndarray:
        """谱形状向量（前k个特征值）

        用于区分"各向同性扩散"vs"低秩结构化"
        """
        return self.eigenvals[:k]

    def decay_alpha(self, k: int = 10) -> float:
        """拟合衰减指数 λ_i ∝ i^(-α)"""
        lam = self.eigenvals[:k]
        if lam.size < 5:
            return np.nan
        i = np.arange(1, len(lam) + 1)
        log_lam = np.log(lam + 1e-12)
        log_i = np.log(i)
        alpha, _ = np.polyfit(log_i, log_lam, 1)
        return -alpha

    # ========== 完整描述 ==========
    def full_descriptor(self) -> dict:
        """返回(一阶, 二阶, 谱形状)完整描述"""
        return {
            # 一阶
            'kappa': self.kappa(),
            # 二阶
            'eff_rank': self.eff_rank(),
            # 谱形状
            'spectrum_top5': self.spectral_shape(5),
            'decay_alpha': self.decay_alpha(),
            # 位置
            'norm': float(np.linalg.norm(self.mu)),
        }


# =============================================================================
# 联合检测器：κ + eff_rank（两个可解释无监督标量）
# =============================================================================


class KappaEffRankDetector:
    """基于一阶+二阶矩的联合检测器

    设计原则：
    1. 不做线性聚合（两个信号机制不同）
    2. 无监督（不需要标签）
    3. 可解释（κ=集中度, eff_rank=分散形状）
    """

    def __init__(self):
        self.kappa_threshold = None
        self.eff_rank_threshold = None

    def fit_healthy(self, clouds: list[StepCloudGeometry]):
        """从正确样本学习健康基线"""
        kappas = [c.kappa() for c in clouds]
        eff_ranks = [c.eff_rank() for c in clouds]

        # 使用百分位数作为阈值
        self.kappa_threshold = np.percentile(kappas, 25)  # 25%分位
        self.eff_rank_threshold = np.percentile(eff_ranks, 75)  # 75%分位

    def predict_score(self, cloud: StepCloudGeometry) -> dict:
        """返回两个独立的分数 + 组合决策

        组合逻辑（非线性）：
        - κ低 AND eff_rank低 → 错误概率高
        - κ低 OR eff_rank低 → 需关注
        - 两者都正常 → 健康
        """
        k = cloud.kappa()
        er = cloud.eff_rank()

        # 归一化分数（相对于阈值）
        kappa_score = (k - self.kappa_threshold) if self.kappa_threshold else np.nan
        eff_rank_score = (self.eff_rank_threshold - er) if self.eff_rank_threshold else np.nan

        # 非线性组合（逻辑与）
        if np.isfinite(kappa_score) and np.isfinite(eff_rank_score):
            # 低κ AND 高eff_rank分散 → 最可疑
            combined = float((kappa_score < 0) & (eff_rank_score < 0))
        else:
            combined = np.nan

        return {
            'kappa': k,
            'eff_rank': er,
            'kappa_score': kappa_score,
            'eff_rank_score': eff_rank_score,
            'combined': combined,
            'risk_level': self._risk_level(kappa_score, eff_rank_score),
        }

    def _risk_level(self, ks: float, ers: float) -> str:
        """风险等级分类"""
        if not np.isfinite(ks) or not np.isfinite(ers):
            return 'unknown'
        if ks < 0 and ers < 0:
            return 'high'
        elif ks < 0 or ers < 0:
            return 'medium'
        else:
            return 'low'


# =============================================================================
# Step-Saliency对齐：检测Shallow Lock-in和Deep Decay
# =============================================================================


def detect_shallow_lockin(clouds: list[StepCloudGeometry]) -> float:
    """检测Shallow Lock-in：κ突然下降 + eff_rank突降

    正确推理：κ维持高水平，eff_rank稳定
    Lock-in错误：κ坍缩到低值，eff_rank同时下降
    """
    if len(clouds) < 3:
        return np.nan

    kappas = [c.kappa() for c in clouds]
    eff_ranks = [c.eff_rank() for c in clouds]

    # 计算变化率（前半 vs 后半）
    mid = len(clouds) // 2
    early_k = np.mean(kappas[:mid])
    late_k = np.mean(kappas[mid:])
    early_er = np.mean(eff_ranks[:mid])
    late_er = np.mean(eff_ranks[mid:])

    # Lock-in信号：κ下降 AND eff_rank下降
    kappa_drop = early_k - late_k
    er_drop = early_er - late_er

    # 组合分数（逻辑与）
    return float((kappa_drop > 0.1) and (er_drop > 0.5))


def detect_deep_decay(clouds: list[StepCloudGeometry]) -> float:
    """检测Deep Decay：深层谱衰减加速

    正确推理：α ≈ 1（幂律衰减）
    Decay错误：α >> 1（快速衰减）
    """
    alphas = [c.decay_alpha() for c in clouds if np.isfinite(c.decay_alpha())]

    if not alphas:
        return np.nan

    mean_alpha = np.mean(alphas)

    # α > 1.5 表示加速衰减
    return float(mean_alpha > 1.5)


# =============================================================================
# 验证实验模板
# =============================================================================


def run_validation_experiment():
    """验证二阶矩在难任务上的普适性

    待确认：
    1. math上eff_rank是否仍显著
    2. olympiad上eff_rank是否仍显著
    3. κ+eff_rank联合是否超过单独效果
    """
    print("""
验证实验检查清单：

□ 数据准备
  □ math子集respcloud重抽（omnimath已验证）
  □ olympiad子集respcloud重抽
  □ 确认长度bucketed

□ 指标计算
  □ 每步骤的κ、eff_rank、spectrum_top5
  □ 步骤间的Bures/W2距离
  □ 谱衰减指数α

□ 统计检验
  □ eff_rank单独AUROC（预期>κ）
  □ spectrum vector AUROC（预期+0.02）
  □ κ+eff_rank联合（预期非线性增益）

□ 与Step-Saliency对齐
  □ Shallow Lock-in → κ+eff_rank双降
  □ Deep Decay → α指数过大
  □ 步骤自强化 → 对角block集中度

□ 结果记录
  □ 四子集完整表格
  □ bootstrap CI
  □ 长度bucket分解
    """)


if __name__ == '__main__':
    run_validation_experiment()
