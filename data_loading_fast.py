#!/usr/bin/env python3
"""全优化版（CPU）：多进程 + float32 + eigvalsh

优化方法：
1. 多进程并行处理chains (4-8x)
2. float32精度 (1.5x)
3. eigvalsh代替eigh (1.3x) - 只计算特征值
4. 避免不必要的数组复制

总预期加速：8-15x (89小时 → 6-11小时)

如需GPU加速 (50-100x)，请安装CuPy: pip install cupy-cuda12x
"""

import numpy as np
import pickle
import hashlib
from pathlib import Path
from tqdm import tqdm
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from scipy.linalg import eigvalsh
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

HIDDEN_LAYERS = [10, 14, 18, 22]


def compute_step_geometry_fast(H: np.ndarray,
                               step_id: int,
                               layer_id: int,
                               n_top: int = 10) -> Optional[Dict]:
    """优化的Step几何特征计算（CPU版本）

    优化点：
    1. 使用float32而非float64
    2. 使用eigvalsh而非eigh（只计算特征值，速度快~30%）
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

    # 使用float32加速（内存减半，计算加速）
    H = H.astype(np.float32, copy=False)

    # 归一化token向量
    row_norms = np.linalg.norm(H, axis=1, keepdims=True)
    H_norm = H / (row_norms + eps)

    # 一阶矩：kappa = ||mean(û)||
    mu = H_norm.mean(axis=0)
    kappa = float(np.linalg.norm(mu))
    norm = float(H.mean())

    # Scatter matrix (使用float32)
    S = (H_norm.T @ H_norm) / n_tokens  # (d, d)

    # 特征值分解 - 只计算特征值（eigvalsh比eigh快~30%）
    try:
        eigvals = eigvalsh(S)
    except Exception as e:
        # 如果分解失败，返回None
        return None

    # 降序排列并归一化
    eigvals = np.sort(eigvals)[::-1]
    eigvals = eigvals / (eigvals.sum() + eps)

    # 取前n_top个特征值
    eigenvalues = eigvals[:n_top].astype(np.float32)

    # 计算有效秩
    lam = eigvals[eigvals > eps]
    if len(lam) > 0:
        eff_rank = float(np.exp(-np.sum(lam * np.log(lam + eps))))
        eff_rank = float(min(eff_rank, n_tokens))
    else:
        eff_rank = 1.0

    # 计算谱熵
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


@dataclass
class StepGeometry:
    step_id: int
    layer: int
    n_tokens: int
    kappa: float = np.nan
    eff_rank: float = np.nan
    spectral_entropy: float = np.nan
    norm: float = np.nan
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    principal_directions: np.ndarray = field(default_factory=lambda: np.array([]))

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
    step_ranges: List[Tuple[int, int]] = field(default_factory=list)
    steps: Dict[int, Dict[int, StepGeometry]] = field(default_factory=dict)

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


def get_cache_key(npz_path: str, hidden_dir: str, subset: str) -> str:
    """生成缓存key"""
    key_str = f"{subset}_{len(npz_path)}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def save_cached_features(cache_path: Path, chain_id: int, features):
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
        except (EOFError, pickle.UnpicklingError, Exception):
            return None
    return None


def _compute_single_chain(args):
    """计算单个chain的特征（用于多进程）"""
    idx, npz_path, hidden_dir, subset = args

    # 加载NPZ数据
    data = np.load(npz_path, allow_pickle=True)
    problem_ids = data['problem_ids']
    is_correct_strict = data['is_correct_strict']
    step_token_ranges = data['step_token_ranges']

    # 加载hidden states
    hidden_path = Path(hidden_dir) / f"{subset}-{idx}.npy"
    try:
        hidden = np.load(hidden_path, mmap_mode='r')
    except Exception as e:
        return ReasoningTrajectory(
            idx, int(problem_ids[idx]), bool(is_correct_strict[idx] == 0), 0
        )

    # 计算特征
    steps = {}
    ranges = step_token_ranges[idx]

    if ranges is None or len(ranges) == 0:
        return ReasoningTrajectory(
            idx, int(problem_ids[idx]), bool(is_correct_strict[idx] == 0), 0
        )

    for layer_idx, layer_id in enumerate(HIDDEN_LAYERS):
        layer_steps = {}

        for step_id, (start, end) in enumerate(ranges):
            if end <= start:
                continue

            H = hidden[start:end, layer_idx, :].copy()
            geom = compute_step_geometry_fast(H, step_id, layer_id, n_top=10)

            if geom:
                layer_steps[step_id] = StepGeometry(**geom)

        if layer_steps:
            steps[layer_id] = layer_steps

    traj = ReasoningTrajectory(
        idx, int(problem_ids[idx]), bool(is_correct_strict[idx] == 0),
        len(ranges),
        step_ranges=ranges,
        steps=steps,
    )

    return traj


def load_all_trajectories_fast(npz_path: str,
                                hidden_dir: str,
                                force_recompute: bool = False,
                                verbose: bool = True,
                                n_workers: int = None,
                                use_cache: bool = True) -> Tuple[List[ReasoningTrajectory], Dict]:
    """使用多进程加速加载轨迹

    Args:
        npz_path: NPZ文件路径
        hidden_dir: hidden目录路径
        force_recompute: 强制重新计算
        verbose: 显示详细信息
        n_workers: 进程数，None则自动检测
        use_cache: 是否使用缓存

    Returns:
        (trajectories, metadata)
    """

    # 加载NPZ
    data = np.load(npz_path, allow_pickle=True)
    problem_ids = data['problem_ids']
    is_correct_strict = data['is_correct_strict']
    step_token_ranges = data['step_token_ranges']
    subset = Path(npz_path).stem.replace('full_', '')

    # 缓存目录
    cache_dir = Path(hidden_dir).parent / "cache" / subset
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    n_chains = len(problem_ids)

    if verbose:
        print(f"Cache directory: {cache_dir}")
        print(f"Total chains: {n_chains}")
        print(f"Optimizations: float32 + eigvalsh + multiprocessing")

    # 确定worker数量
    if n_workers is None:
        n_workers = max(1, cpu_count() - 2)  # 留2个核心给系统

    if verbose:
        print(f"Using {n_workers} workers (CPU cores: {cpu_count()})")

    # 统计
    cache_hits = 0
    cache_misses = 0

    # 分批处理：先检查缓存
    to_compute = []
    trajectories = [None] * n_chains

    if use_cache:
        for idx in range(n_chains):
            cached = load_cached_features(cache_dir, idx)
            if cached is not None and not force_recompute:
                cache_hits += 1
                trajectories[idx] = cached
            else:
                cache_misses += 1
                to_compute.append(idx)

        if verbose:
            print(f"Cache hits: {cache_hits}, Cache misses: {cache_misses}")
    else:
        to_compute = list(range(n_chains))

    # 使用多进程计算未缓存的chains
    if to_compute:
        if verbose:
            print(f"Computing {len(to_compute)} chains with multiprocessing...")

        # 准备参数
        compute_args = [(i, npz_path, hidden_dir, subset) for i in to_compute]

        start_time = time.time()

        with Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap(_compute_single_chain, compute_args),
                total=len(compute_args),
                desc="Computing features"
            ))

        elapsed = time.time() - start_time

        # 保存缓存并填充结果
        for i, traj in enumerate(results):
            if use_cache:
                save_cached_features(cache_dir, to_compute[i], traj)
            trajectories[to_compute[i]] = traj

        if verbose:
            avg_time = elapsed / len(to_compute)
            total_estimated = avg_time * n_chains
            print(f"Average compute time per chain: {avg_time:.2f}s")
            print(f"This batch time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
            print(f"Estimated total time (no cache): {total_estimated/3600:.1f} hours")

    # 移除None值
    trajectories = [t for t in trajectories if t is not None]

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
        'n_workers': n_workers,
    }

    return trajectories, metadata


def clear_cache(npz_path: str, hidden_dir: str):
    """清除缓存"""
    subset = Path(npz_path).stem.replace('full_', '')
    cache_dir = Path(hidden_dir).parent / "cache" / subset

    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        print(f"Cleared cache: {cache_dir}")
    else:
        print(f"No cache to clear")


def benchmark_comparison():
    """对比优化前后的性能"""
    import time
    from data_loading_cache import compute_step_geometry_ultra_fast as compute_old

    print("=" * 60)
    print("性能对比测试")
    print("=" * 60)

    # 创建测试数据
    np.random.seed(42)
    n_tokens = 50
    d = 4096
    H = np.random.randn(n_tokens, d).astype(np.float32)

    methods = [
        ("旧版 (eigh, float64)", lambda: compute_old(H.astype(np.float64), 0, 14)),
        ("新版 (eigvalsh, float32)", lambda: compute_step_geometry_fast(H, 0, 14)),
    ]

    for name, func in methods:
        times = []
        for _ in range(5):
            start = time.time()
            result = func()
            elapsed = time.time() - start
            if result is not None:
                times.append(elapsed)

        if times:
            avg_time = np.mean(times)
            print(f"{name}: {avg_time*1000:.2f}ms (±{np.std(times)*1000:.2f}ms)")

    # 计算预期总时间
    print("\n预期总时间对比:")
    n_chains = 1000  # 假设1000个chains
    n_steps = 5  # 平均每个chain 5个steps
    n_layers = 4  # 4个层

    # 测量实际时间
    times_old = []
    times_new = []
    for _ in range(3):
        start = time.time()
        compute_old(H.astype(np.float64), 0, 14)
        times_old.append(time.time() - start)

        start = time.time()
        compute_step_geometry_fast(H, 0, 14)
        times_new.append(time.time() - start)

    old_time_per = np.mean(times_old)
    new_time_per = np.mean(times_new)

    old_total = n_chains * n_steps * n_layers * old_time_per / 3600
    new_total = n_chains * n_steps * n_layers * new_time_per / 3600
    parallel_total = new_total / n_workers if n_workers else new_total / 8

    print(f"  旧版 (单线程): {old_total:.1f} 小时")
    print(f"  新版 (单线程): {new_total:.1f} 小时")
    print(f"  新版 (8核并行): {parallel_total:.1f} 小时")
    print(f"  加速比: {old_total/parallel_total:.1f}x")


# 兼容旧接口
def load_all_trajectories(npz_path: str,
                         hidden_dir: str,
                         n_top_components: int = 10,
                         store_scatter: bool = False,
                         verbose: bool = True) -> Tuple[List[ReasoningTrajectory], Dict]:
    """兼容旧接口，内部调用fast版本"""
    return load_all_trajectories_fast(
        npz_path=npz_path,
        hidden_dir=hidden_dir,
        force_recompute=False,
        verbose=verbose,
        n_workers=None,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark_comparison()
    elif len(sys.argv) > 1 and sys.argv[1] == "clear":
        npz_path = "/gz-data/research/demo/data/features/full_omnimath.npz"
        hidden_dir = "/gz-data/research/demo/data/hidden/omnimath/"
        clear_cache(npz_path, hidden_dir)
    else:
        # 正常运行
        npz_path = "/gz-data/research/demo/data/features/full_omnimath.npz"
        hidden_dir = "/gz-data/research/demo/data/hidden/omnimath/"

        print("Loading with optimizations (float32 + eigvalsh + multiprocessing)...")
        start = time.time()

        trajectories, metadata = load_all_trajectories_fast(
            npz_path=npz_path,
            hidden_dir=hidden_dir,
            force_recompute=False,
            verbose=True,
            n_workers=8,
        )

        elapsed = time.time() - start
        print(f"\nDone! {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print(f"Metadata: {metadata}")
