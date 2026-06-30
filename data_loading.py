#!/usr/bin/env python3
"""数据加载：从hidden states计算真实的几何特征

Hidden shards: /gz-data/research/demo/data/hidden/<subset>/<chain_id>.npy
格式: (R, 4, 4096)
  - R: 响应中的token总数
  - 4: layers [10, 14, 18, 22]
  - 4096: hidden dimension
"""

import numpy as np
from scipy.linalg import eigh
from scipy.stats import entropy as scipy_entropy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 常量定义
# =============================================================================

HIDDEN_LAYERS = [10, 14, 18, 22]  # Hidden shards中的层


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class StepGeometry:
    """单步骤的完整几何描述（包含标量和特征向量信息）

    Attributes:
        step_id: 步骤索引
        layer: 层索引
        n_tokens: 该步骤的token数

        # 标量特征（一阶矩和二阶矩）
        kappa: 方向集中度 ||mean(û)||
        eff_rank: 有效秩 exp(-Σ λ log λ)
        spectral_entropy: 谱熵 -Σ λ log λ
        norm: 平均token范数

        # 真正的几何特征（基于特征向量）
        principal_directions: (d, k) 前k个主成分向量
        eigenvalues: (d,) 所有特征值（降序）
        scatter_matrix: (d, d) scatter matrix（可选，内存大时可不存）
    """
    step_id: int
    layer: int
    n_tokens: int

    # 标量特征
    kappa: float = np.nan
    eff_rank: float = np.nan
    spectral_entropy: float = np.nan
    norm: float = np.nan

    # 真正的几何特征（特征向量）
    principal_directions: np.ndarray = field(default_factory=lambda: np.array([]))
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    scatter_matrix: np.ndarray = field(default_factory=lambda: np.array([]))

    # 辅助特征
    spectrum_top10: np.ndarray = field(default_factory=lambda: np.array([]))
    lambda1: float = np.nan  # 最大特征值
    lambda2: float = np.nan  # 第二大特征值
    spectral_gap: float = np.nan  # λ1 - λ2


@dataclass
class ReasoningTrajectory:
    """完整推理链

    Attributes:
        chain_id: 链索引
        problem_id: 问题ID
        is_correct: 是否正确
        n_steps: 步骤数
        step_ranges: 每步的token范围 [(start1, end1), ...]
        hidden: (R, 4, 4096) hidden states或None
        steps: {layer_id: {step_id: StepGeometry}}
    """
    chain_id: int
    problem_id: int
    is_correct: bool
    n_steps: int
    step_ranges: List[Tuple[int, int]] = field(default_factory=list)
    hidden: Optional[np.ndarray] = None
    steps: Dict[int, Dict[int, StepGeometry]] = field(default_factory=dict)

    def has_layer(self, layer: int) -> bool:
        """检查是否有该层的几何特征"""
        return layer in self.steps and len(self.steps[layer]) > 0

    def get_geometry_sequence(self, layer: int) -> List[StepGeometry]:
        """获取某层的几何特征序列（按step_id排序）"""
        if not self.has_layer(layer):
            return []
        step_geoms = self.steps[layer]
        return [step_geoms[i] for i in sorted(step_geoms.keys()) if step_geoms[i].n_tokens > 0]


# =============================================================================
# 几何特征计算（真实实现）
# =============================================================================

def compute_step_geometry(hidden: np.ndarray,
                         step_range: Tuple[int, int],
                         layer_idx: int,
                         layer_id: int,
                         step_id: int,
                         n_top_components: int = 10,
                         store_scatter: bool = False) -> Optional[StepGeometry]:
    """从hidden states计算步骤几何特征

    Args:
        hidden: (R, 4, 4096) hidden states
        step_range: (start, end) token范围
        layer_idx: 层索引（在4层中）
        layer_id: 层ID（实际层号）
        step_id: 步骤ID
        n_top_components: 保存前n个主成分
        store_scatter: 是否存储scatter matrix（内存大）

    Returns:
        StepGeometry or None（如果步骤为空）
    """
    start, end = step_range
    if end <= start:
        return None

    # 提取该步骤在该层的hidden states
    H = hidden[start:end, layer_idx, :]  # (n_tokens, 4096)
    n_tokens, d = H.shape

    if n_tokens == 0:
        return None

    eps = 1e-12

    # 归一化每个token向量
    H_norm = H / (np.linalg.norm(H, axis=1, keepdims=True) + eps)

    # 一阶矩：kappa = ||mean(û)||
    mu = H_norm.mean(axis=0)
    kappa = float(np.linalg.norm(mu))

    # 二阶矩：scatter matrix
    S = (H_norm.T @ H_norm) / n_tokens  # (d, d)

    # 特征值分解
    try:
        eigvals = eigh(S, eigvals_only=True)
    except:
        # 如果分解失败，返回基本信息
        return StepGeometry(
            step_id=step_id,
            layer=layer_id,
            n_tokens=n_tokens,
            kappa=kappa,
            norm=float(H.mean()),
        )

    eigvals = np.sort(eigvals)[::-1]  # 降序
    eigvals = eigvals / (eigvals.sum() + eps)  # 归一化

    # 计算标量特征
    lam = eigvals[eigvals > eps]
    if len(lam) > 0:
        eff_rank = float(np.exp(-np.sum(lam * np.log(lam + eps))))
        spec_entropy = float(scipy_entropy(eigvals + eps))
    else:
        eff_rank = 1.0
        spec_entropy = 0.0

    # 计算特征向量（如果需要几何信息）
    principal_directions = np.array([])
    if n_tokens >= n_top_components:
        try:
            eigvals_full, eigvecs = eigh(S, subset_by_index=[d-n_top_components, d-1])
            # eigh返回升序，需要反转
            eigvals_full = eigvals_full[::-1]
            eigvecs = eigvecs[:, ::-1]
            principal_directions = eigvecs  # (d, n_top_components)
        except:
            principal_directions = np.array([])

    # 构造StepGeometry
    geom = StepGeometry(
        step_id=step_id,
        layer=layer_id,
        n_tokens=n_tokens,
        kappa=kappa,
        eff_rank=eff_rank,
        spectral_entropy=spec_entropy,
        norm=float(H.mean()),
        eigenvalues=eigvals,
        spectrum_top10=eigvals[:10],
        lambda1=float(eigvals[0]) if len(eigvals) > 0 else np.nan,
        lambda2=float(eigvals[1]) if len(eigvals) > 1 else np.nan,
        spectral_gap=float(eigvals[0] - eigvals[1]) if len(eigvals) > 1 else np.nan,
    )

    if principal_directions.size > 0:
        geom.principal_directions = principal_directions

    if store_scatter:
        geom.scatter_matrix = S

    return geom


def compute_all_layers_geometry(hidden: np.ndarray,
                                step_ranges: List[Tuple[int, int]],
                                n_top_components: int = 10) -> Dict[int, Dict[int, StepGeometry]]:
    """计算所有层、所有步骤的几何特征

    Args:
        hidden: (R, 4, 4096) hidden states
        step_ranges: 每步的token范围
        n_top_components: 保存的主成分数

    Returns:
        {layer_id: {step_id: StepGeometry}}
    """
    all_geometry = {}

    for layer_idx, layer_id in enumerate(HIDDEN_LAYERS):
        layer_geometry = {}

        for step_id, (start, end) in enumerate(step_ranges):
            if end <= start:
                continue

            geom = compute_step_geometry(
                hidden=hidden,
                step_range=(start, end),
                layer_idx=layer_idx,
                layer_id=layer_id,
                step_id=step_id,
                n_top_components=n_top_components,
            )

            if geom is not None:
                layer_geometry[step_id] = geom

        if layer_geometry:
            all_geometry[layer_id] = layer_geometry

    return all_geometry


# =============================================================================
# 数据加载函数
# =============================================================================

def get_hidden_filename(chain_id: int, subset: str) -> str:
    """获取hidden文件名"""
    return f"{subset}-{chain_id}.npy"


def load_hidden_shard(hidden_dir: str, chain_id: int, subset: str) -> Optional[np.ndarray]:
    """加载单个hidden shard

    Args:
        hidden_dir: hidden目录路径
        chain_id: 链ID
        subset: 数据集名称

    Returns:
        (R, 4, 4096) array or None
    """
    filename = get_hidden_filename(chain_id, subset)
    filepath = Path(hidden_dir) / filename

    if not filepath.exists():
        return None

    try:
        hidden = np.load(filepath)
        return hidden
    except Exception as e:
        print(f"Warning: Failed to load {filepath}: {e}")
        return None


def load_all_trajectories(npz_path: str,
                         hidden_dir: str,
                         n_top_components: int = 10,
                         store_scatter: bool = False,
                         verbose: bool = True) -> Tuple[List[ReasoningTrajectory], Dict]:
    """加载所有推理链并计算几何特征

    Args:
        npz_path: full_*.npz路径
        hidden_dir: hidden目录路径
        n_top_components: 保存的主成分数
        store_scatter: 是否存储scatter matrix
        verbose: 显示进度条

    Returns:
        (trajectories, metadata)
    """
    # 加载NPZ数据
    data = np.load(npz_path, allow_pickle=True)

    problem_ids = data['problem_ids']
    is_correct_strict = data['is_correct_strict']
    stepcloud = data['stepcloud']

    # 获取step token ranges
    if 'step_token_ranges' in data:
        step_token_ranges = data['step_token_ranges']
    else:
        # 如果没有step_token_ranges，无法继续
        raise ValueError("NPZ文件必须包含step_token_ranges字段")

    # 获取hidden文件名（如果有的话）
    if 'hidden_files' in data:
        hidden_files = data['hidden_files']
    else:
        # 从subset推断
        subset = Path(npz_path).stem.replace('full_', '')
        hidden_files = np.array([get_hidden_filename(i, subset) for i in range(len(problem_ids))])

    # 推断subset
    subset = Path(npz_path).stem.replace('full_', '')

    # 创建轨迹列表
    trajectories = []
    n_correct = 0
    n_error = 0

    iterator = range(len(problem_ids))
    if verbose:
        iterator = tqdm(iterator, desc="Loading trajectories")

    for i in iterator:
        # 加载hidden shard
        hidden = None
        hidden_path = Path(hidden_dir) / hidden_files[i]
        if hidden_path.exists():
            try:
                hidden = np.load(hidden_path)
            except:
                pass

        # 获取step ranges
        ranges = step_token_ranges[i]
        if ranges is None:
            ranges = []

        # 创建轨迹
        traj = ReasoningTrajectory(
            chain_id=i,
            problem_id=int(problem_ids[i]),
            is_correct=bool(is_correct_strict[i] == 0),
            n_steps=len(ranges),
            step_ranges=ranges,
            hidden=hidden,
        )

        # 计算几何特征
        if hidden is not None and len(ranges) > 0:
            try:
                all_geom = compute_all_layers_geometry(
                    hidden=hidden,
                    step_ranges=ranges,
                    n_top_components=n_top_components,
                )
                traj.steps = all_geom
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to compute geometry for chain {i}: {e}")

        trajectories.append(traj)

        if traj.is_correct:
            n_correct += 1
        else:
            n_error += 1

    # 元数据
    metadata = {
        'subset': subset,
        'n_chains': len(trajectories),
        'n_correct': n_correct,
        'n_error': n_error,
        'layers': HIDDEN_LAYERS,
        'npz_path': npz_path,
        'hidden_dir': hidden_dir,
    }

    return trajectories, metadata


# =============================================================================
# 辅助函数
# =============================================================================

def get_trajectory_with_min_steps(trajectories: List[ReasoningTrajectory],
                                  min_steps: int = 3,
                                  layer: int = 14) -> List[ReasoningTrajectory]:
    """过滤出至少有min_steps个步骤的轨迹（在指定层）"""
    filtered = []
    for traj in trajectories:
        if traj.has_layer(layer):
            n_geom = len(traj.get_geometry_sequence(layer))
            if n_geom >= min_steps:
                filtered.append(traj)
    return filtered


def print_trajectory_stats(trajectories: List[ReasoningTrajectory],
                          metadata: Dict,
                          layer: int = 14):
    """打印轨迹统计信息"""
    print("=" * 80)
    print("Trajectory Statistics")
    print("=" * 80)
    print(f"Subset: {metadata['subset']}")
    print(f"Total chains: {metadata['n_chains']}")
    print(f"  Correct: {metadata['n_correct']}")
    print(f"  Error: {metadata['n_error']}")
    print(f"Layers: {metadata['layers']}")

    # 检查有多少轨迹有该层的几何特征
    n_with_layer = sum(1 for t in trajectories if t.has_layer(layer))
    print(f"\nLayer {layer}:")
    print(f"  Chains with geometry: {n_with_layer}/{metadata['n_chains']}")

    if n_with_layer > 0:
        # 计算平均步骤数
        step_counts = [len(t.get_geometry_sequence(layer)) for t in trajectories if t.has_layer(layer)]
        print(f"  Mean steps: {np.mean(step_counts):.2f}")
        print(f"  Min steps: {min(step_counts)}")
        print(f"  Max steps: {max(step_counts)}")

        # 抽样检查第一个轨迹的几何特征
        first_valid = next((t for t in trajectories if t.has_layer(layer)), None)
        if first_valid:
            geom_seq = first_valid.get_geometry_sequence(layer)
            if geom_seq:
                print(f"\nSample (chain {first_valid.chain_id}):")
                g = geom_seq[0]
                print(f"  Step 0: κ={g.kappa:.3f}, eff_rank={g.eff_rank:.2f}, "
                      f"entropy={g.spectral_entropy:.3f}, n_tokens={g.n_tokens}")
                if hasattr(g, 'principal_directions') and g.principal_directions.size > 0:
                    print(f"  Principal directions shape: {g.principal_directions.shape}")
                if g.eigenvalues.size >= 3:
                    print(f"  Eigenvalues top3: {g.eigenvalues[:3]}")

    print("=" * 80)


# =============================================================================
# 主函数（用于测试）
# =============================================================================

if __name__ == "__main__":
    # 示例用法
    npz_path = "/gz-data/research/demo/data/features/full_omnimath.npz"
    hidden_dir = "/gz-data/research/demo/data/hidden/omnimath/"

    print("Loading trajectories...")
    trajectories, metadata = load_all_trajectories(
        npz_path=npz_path,
        hidden_dir=hidden_dir,
        n_top_components=10,
        verbose=True,
    )

    print_trajectory_stats(trajectories, metadata, layer=14)
