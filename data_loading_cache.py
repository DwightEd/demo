#!/usr/bin/env python3
"""缓存版：一次计算，永久使用

关键优化：
1. 计算结果缓存到磁盘
2. 内存映射加载（mmap）
3. 批量处理减少I/O
4. 支持断点续传
"""

import numpy as np
import pickle
import hashlib
from pathlib import Path
from tqdm import tqdm
import time
from dataclasses import dataclass
from scipy.linalg import eigvalsh  # 只计算特征值，比eigh快~30%
from multiprocessing import Pool, cpu_count

HIDDEN_LAYERS = [10, 14, 18, 22]


def compute_step_geometry_ultra_fast(H: np.ndarray, step_id: int, layer_id: int, n_top: int = 10):
    """Step几何特征计算：优化版（float32 + eigvalsh）

    优化点：
    1. 使用float32加速（内存减半，计算加速~1.5x）
    2. 使用eigvalsh而非eigh（只计算特征值，加速~1.3x）
    3. 避免不必要的数组复制

    Args:
        H: hidden states, shape (n_tokens, d)
        step_id: step索引
        layer_id: 层ID
        n_top: 保留前n个特征值

    Returns:
        dict: 几何特征字典
    """
    n_tokens, d = H.shape
    if n_tokens == 0:
        return None

    eps = 1e-12

    # 使用float32加速
    H = H.astype(np.float32, copy=False)

    # 归一化token向量
    H_norm = H / (np.linalg.norm(H, axis=1, keepdims=True) + eps)

    # 一阶矩：kappa = ||mean(û)||
    mu = H_norm.mean(axis=0)
    kappa = float(np.linalg.norm(mu))
    norm = float(H.mean())

    # Scatter matrix
    S = (H_norm.T @ H_norm) / n_tokens  # (d, d)

    # 特征值分解 - 只计算特征值（比eigh快）
    try:
        eigvals = eigvalsh(S)
    except Exception as e:
        # 如果分解失败，返回基本信息
        return {
            'step_id': step_id,
            'layer': layer_id,
            'n_tokens': n_tokens,
            'kappa': kappa,
            'norm': norm,
            'eff_rank': np.nan,
            'spectral_entropy': np.nan,
            'eigenvalues': np.array([]),
        }

    # 降序排列并归一化
    eigvals = np.sort(eigvals)[::-1]
    eigvals = eigvals / (eigvals.sum() + eps)

    # 取前n_top个特征值（使用float32）
    eigenvalues = eigvals[:n_top].astype(np.float32)

    # 计算有效秩 (使用完整特征值)
    lam = eigvals[eigvals > eps]
    if len(lam) > 0:
        eff_rank = float(np.exp(-np.sum(lam * np.log(lam + eps))))
        eff_rank = float(min(eff_rank, n_tokens))  # 不超过token数
    else:
        eff_rank = 1.0

    # 计算谱熵 (使用完整特征值)
    spectral_entropy = float(-np.sum(eigvals * np.log(eigvals + eps)))

    return {
        'step_id': step_id,
        'layer': layer_id,
        'n_tokens': n_tokens,
        'kappa': kappa,
        'eff_rank': eff_rank,
        'spectral_entropy': spectral_entropy,
        'norm': norm,
        'eigenvalues': eigenvalues,
    }


def get_cache_key(npz_path: str, hidden_dir: str, subset: str) -> str:
    """生成缓存key"""
    key_str = f"{subset}_{len(npz_path)}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def save_cached_features(cache_path: Path, chain_id: int, features: dict):
    """保存单个chain的特征"""
    cache_file = cache_path / f"chain_{chain_id}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cached_features(cache_path: Path, chain_id: int):
    """加载单个chain的特征（带错误处理）"""
    cache_file = cache_path / f"chain_{chain_id}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError, Exception) as e:
            # 缓存损坏，返回None触发重新计算
            return None
    return None


# 在模块级别定义类（可pickle）
@dataclass
class StepGeometry:
    step_id: int
    layer: int
    n_tokens: int
    kappa: float = np.nan
    eff_rank: float = np.nan
    spectral_entropy: float = np.nan
    norm: float = np.nan
    eigenvalues: np.ndarray = None
    principal_directions: np.ndarray = None  # 兼容性

    def __post_init__(self):
        if self.eigenvalues is None:
            self.eigenvalues = np.array([])
        if self.principal_directions is None:
            self.principal_directions = np.array([])


@dataclass
class ReasoningTrajectory:
    chain_id: int
    problem_id: int
    is_correct: bool
    n_steps: int
    step_ranges: list = None
    steps: dict = None

    def __init__(self, chain_id, problem_id, is_correct, n_steps, step_ranges=None, steps=None):
        self.chain_id = chain_id
        self.problem_id = problem_id
        self.is_correct = bool(is_correct)
        self.n_steps = n_steps
        if step_ranges is None:
            self.step_ranges = []
        elif isinstance(step_ranges, np.ndarray):
            self.step_ranges = step_ranges.tolist()
        elif hasattr(step_ranges, '__iter__'):
            self.step_ranges = list(step_ranges)
        else:
            self.step_ranges = []
        self.steps = steps or {}

    def has_layer(self, layer):
        return layer in self.steps and len(self.steps[layer]) > 0

    def get_geometry_sequence(self, layer):
        if not self.has_layer(layer):
            return []
        return [self.steps[layer][i] for i in sorted(self.steps[layer].keys())]


def load_all_trajectories_cached(npz_path: str,
                                hidden_dir: str,
                                force_recompute: bool = False,
                                verbose: bool = True) -> tuple:
    """使用缓存加载轨迹（优化版：float32 + eigvalsh）

    优化：
    1. float32精度（1.5x加速）
    2. eigvalsh只算特征值（1.3x加速）
    3. 缓存机制

    首次运行：计算并缓存（需要时间）
    后续运行：直接加载缓存（秒级）
    """

    # 加载NPZ
    data = np.load(npz_path, allow_pickle=True)
    problem_ids = data['problem_ids']
    is_correct_strict = data['is_correct_strict']
    step_token_ranges = data['step_token_ranges']
    subset = Path(npz_path).stem.replace('full_', '')

    # 缓存目录（统一使用hidden_dir的父目录）
    cache_dir = Path(hidden_dir).parent / "cache" / subset
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_chains = len(problem_ids)
    trajectories = []

    if verbose:
        print(f"Cache directory: {cache_dir}")
        print(f"Total chains: {n_chains}")

    # 统计
    cache_hits = 0
    cache_misses = 0
    compute_times = []

    for idx in tqdm(range(n_chains), desc="Loading trajectories"):
        # 尝试加载缓存
        cached = load_cached_features(cache_dir, idx)

        if cached is not None and not force_recompute:
            # 缓存命中
            cache_hits += 1
            traj = cached
        else:
            # 缓存未命中，需要计算
            cache_misses += 1
            start = time.time()

            # 加载hidden
            hidden_path = Path(hidden_dir) / f"{subset}-{idx}.npy"
            try:
                hidden = np.load(hidden_path, mmap_mode='r')  # 内存映射
            except:
                trajectories.append(ReasoningTrajectory(
                    idx, int(problem_ids[idx]), bool(is_correct_strict[idx] == 0), 0
                ))
                continue

            # 计算特征
            steps = {}
            for layer_idx, layer_id in enumerate(HIDDEN_LAYERS):
                layer_steps = {}
                ranges = step_token_ranges[idx]

                if ranges is None:
                    continue

                for step_id, (start, end) in enumerate(ranges):
                    if end <= start:
                        continue

                    H = hidden[start:end, layer_idx, :].copy()  # 只复制需要的部分
                    geom = compute_step_geometry_ultra_fast(H, step_id, layer_id, n_top=10)
                    if geom:
                        layer_steps[step_id] = StepGeometry(**geom)

                if layer_steps:
                    steps[layer_id] = layer_steps

            traj = ReasoningTrajectory(
                idx, int(problem_ids[idx]), bool(is_correct_strict[idx] == 0),
                len(step_token_ranges[idx]) if step_token_ranges[idx] is not None else 0,
                step_ranges=step_token_ranges[idx],
                steps=steps,
            )

            # 保存缓存
            save_cached_features(cache_dir, idx, traj)

            elapsed = time.time() - start
            compute_times.append(elapsed)

        trajectories.append(traj)

    if verbose:
        print(f"\nCache stats:")
        print(f"  Hits: {cache_hits}")
        print(f"  Misses: {cache_misses}")
        if compute_times:
            print(f"  Avg compute time: {np.mean(compute_times):.2f}s")

    n_correct = sum(1 for t in trajectories if bool(t.is_correct) == True)
    n_error = sum(1 for t in trajectories if bool(t.is_correct) == False)

    metadata = {
        'subset': subset,
        'n_chains': n_chains,
        'n_correct': n_correct,
        'n_error': n_error,
        'layers': HIDDEN_LAYERS,
        'cache_hits': cache_hits,
        'cache_misses': cache_misses,
    }

    return trajectories, metadata


def clear_cache(npz_path: str, hidden_dir: str):
    """清除缓存"""
    subset = Path(npz_path).stem.replace('full_', '')
    cache_dir = Path(hidden_dir).parent / "cache" / subset  # 修正路径：hidden_dir的父目录/cache/subset

    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        print(f"Cleared cache: {cache_dir}")
        print(f"Shell equivalent: rm -rf {cache_dir}")
    else:
        print(f"No cache to clear (directory not found): {cache_dir}")


if __name__ == "__main__":
    npz_path = "/gz-data/research/demo/data/features/full_omnimath.npz"
    hidden_dir = "/gz-data/research/demo/data/hidden/omnimath/"

    # 清除旧缓存（如果需要）
    # clear_cache(npz_path, hidden_dir)

    print("Loading with cache...")
    start = time.time()

    trajectories, metadata = load_all_trajectories_cached(
        npz_path=npz_path,
        hidden_dir=hidden_dir,
        force_recompute=False,
        verbose=True,
    )

    elapsed = time.time() - start
    print(f"\nDone! {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Next run will use cache (much faster!)")
