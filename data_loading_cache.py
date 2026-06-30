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

HIDDEN_LAYERS = [10, 14, 18, 22]


def compute_step_geometry_ultra_fast(H: np.ndarray, step_id: int, layer_id: int):
    """超快速计算：只算κ，跳过其他"""
    n_tokens = H.shape[0]
    if n_tokens == 0:
        return None

    eps = 1e-12
    H_norm = H / (np.linalg.norm(H, axis=1, keepdims=True) + eps)
    mu = H_norm.mean(axis=0)
    kappa = float(np.linalg.norm(mu))
    norm = float(H.mean())

    # 用一个非常简单的rank估计
    # rank ≈ (sum of diagonal)^2 / sum of all elements squared
    S_diag = np.sum(H_norm ** 2, axis=0) / n_tokens
    if S_diag.sum() > eps:
        eff_rank = (S_diag.sum() ** 2) / np.sum(S_diag ** 2)
        eff_rank = float(min(eff_rank, n_tokens))
    else:
        eff_rank = 1.0

    return {
        'step_id': step_id,
        'layer': layer_id,
        'n_tokens': n_tokens,
        'kappa': kappa,
        'eff_rank': eff_rank,
        'norm': norm,
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
    """加载单个chain的特征"""
    cache_file = cache_path / f"chain_{chain_id}.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
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

    def __post_init__(self):
        if self.eigenvalues is None:
            self.eigenvalues = np.array([])


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
    """使用缓存加载轨迹

    首次运行：计算并缓存（需要时间）
    后续运行：直接加载缓存（秒级）
    """

    # 加载NPZ
    data = np.load(npz_path, allow_pickle=True)
    problem_ids = data['problem_ids']
    is_correct_strict = data['is_correct_strict']
    step_token_ranges = data['step_token_ranges']
    subset = Path(npz_path).stem.replace('full_', '')

    # 缓存目录
    cache_dir = Path(hidden_dir) / "../cache" / subset
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
                    geom = compute_step_geometry_ultra_fast(H, step_id, layer_id)
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
    cache_dir = Path(hidden_dir) / "../cache" / subset

    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        print(f"Cleared cache: {cache_dir}")
    else:
        print(f"No cache to clear")


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
