#!/usr/bin/env python3
"""滑动窗口计算方式

像谱几何论文那样：使用 w=10 的滑动窗口，
在每个generation step计算谱特征，得到per-token spectral trajectory

优势：
1. 更细粒度的分析（token-level而非step-level）
2. 不依赖step_token_ranges
3. 可以看到step内部的几何演化
"""

import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from scipy.sparse.linalg import eigsh
from scipy.stats import entropy as scipy_entropy
import time

HIDDEN_LAYERS = [10, 14, 18, 22]
WINDOW_SIZE = 10  # 滑动窗口大小
STRIDE = 5  # 步长


def compute_window_geometry(H_window: np.ndarray, window_id: int, layer_id: int, n_top: int = 10):
    """计算单个窗口的几何特征"""
    n_tokens, d = H_window.shape
    if n_tokens == 0:
        return None

    eps = 1e-12

    # 归一化
    H_norm = H_window / (np.linalg.norm(H_window, axis=1, keepdims=True) + eps)

    # κ（一阶矩）
    mu = H_norm.mean(axis=0)
    kappa = float(np.linalg.norm(mu))

    # 二阶矩：scatter matrix（只算前n_top个特征值）
    S = (H_norm.T @ H_norm) / n_tokens
    n_eigvals = min(n_top, d - 2, n_tokens - 1)
    n_eigvals = max(n_eigvals, 2)

    try:
        eigvals = eigsh(S, k=n_eigvals, which='LA')[0]
        eigvals = eigvals[::-1]  # 降序
    except:
        # 降级：用对角元
        eigvals = np.diag(S)
        eigvals = np.sort(eigvals)[::-1][:n_eigvals]

    # 归一化
    eigvals = eigvals / (eigvals.sum() + eps)

    # eff_rank
    lam = eigvals[eigvals > eps]
    eff_rank = float(np.exp(-np.sum(lam * np.log(lam + eps)))) if len(lam) > 0 else 1.0

    # spectral entropy
    spec_entropy = float(scipy_entropy(eigvals + eps))

    return {
        'window_id': window_id,
        'layer': layer_id,
        'n_tokens': n_tokens,
        'kappa': kappa,
        'eff_rank': eff_rank,
        'spectral_entropy': spec_entropy,
        'eigenvalues': eigvals,
    }


def compute_sliding_window_features(hidden: np.ndarray, layer_idx: int, layer_id: int):
    """使用滑动窗口计算整条链的特征

    Args:
        hidden: (R, 4, 4096) hidden states
        layer_idx: 层索引
        layer_id: 层ID

    Returns:
        list of window geometries
    """
    R = hidden.shape[0]  # 总token数
    windows = []

    # 滑动窗口
    for start in range(0, R - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        H_window = hidden[start:end, layer_idx, :]

        geom = compute_window_geometry(H_window, len(windows), layer_id)
        if geom:
            windows.append(geom)

    return windows


def load_all_trajectories_sliding(npz_path: str,
                                    hidden_dir: str,
                                    use_cache: bool = True,
                                    verbose: bool = True):
    """滑动窗口方式加载所有轨迹"""
    from dataclasses import dataclass
    from typing import Dict, List

    @dataclass
    class WindowGeometry:
        window_id: int
        token_position: int  # 窗口起始位置
        layer: int
        n_tokens: int
        kappa: float = np.nan
        eff_rank: float = np.nan
        spectral_entropy: float = np.nan
        eigenvalues: np.ndarray = None

        def __post_init__(self):
            if self.eigenvalues is None:
                self.eigenvalues = np.array([])

    @dataclass
    class SlidingTrajectory:
        chain_id: int
        problem_id: int
        is_correct: bool
        n_windows: int
        n_tokens: int
        windows: Dict[int, List[WindowGeometry]] = None  # {layer: [windows]}

        def has_layer(self, layer):
            return layer in self.windows and len(self.windows[layer]) > 0

        def get_window_sequence(self, layer):
            if not self.has_layer(layer):
                return []
            return self.windows[layer]

    # 加载NPZ
    data = np.load(npz_path, allow_pickle=True)
    problem_ids = data['problem_ids']
    is_correct_strict = data['is_correct_strict']

    subset = Path(npz_path).stem.replace('full_', '')

    # 缓存目录
    cache_dir = Path(hidden_dir) / "../cache_sliding" / subset
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_chains = len(problem_ids)
    trajectories = []

    if verbose:
        print(f"Sliding window: w={WINDOW_SIZE}, stride={STRIDE}")
        print(f"Cache directory: {cache_dir}")

    cache_hits = 0
    cache_misses = 0

    for idx in tqdm(range(n_chains), desc="Loading trajectories"):
        # 尝试加载缓存
        cache_file = cache_dir / f"chain_{idx}.pkl"
        if use_cache and cache_file.exists():
            with open(cache_file, 'rb') as f:
                traj = pickle.load(f)
            cache_hits += 1
            trajectories.append(traj)
            continue

        # 计算新的
        cache_misses += 1
        start = time.time()

        # 加载hidden
        hidden_path = Path(hidden_dir) / f"{subset}-{idx}.npy"
        try:
            hidden = np.load(hidden_path, mmap_mode='r')
        except:
            trajectories.append(SlidingTrajectory(
                idx, int(problem_ids[idx]), bool(is_correct_strict[idx] == 0), 0, 0, {}
            ))
            continue

        R = hidden.shape[0]
        windows_by_layer = {}

        # 每一层计算滑动窗口
        for layer_idx, layer_id in enumerate(HIDDEN_LAYERS):
            window_geoms = []

            for start in range(0, R - WINDOW_SIZE + 1, STRIDE):
                end = start + WINDOW_SIZE
                H_window = hidden[start:end, layer_idx, :]

                geom = compute_window_geometry(H_window, len(window_geoms), layer_id)
                if geom:
                    window_geoms.append(WindowGeometry(
                        window_id=geom['window_id'],
                        token_position=start,
                        layer=geom['layer'],
                        n_tokens=geom['n_tokens'],
                        kappa=geom['kappa'],
                        eff_rank=geom['eff_rank'],
                        spectral_entropy=geom['spectral_entropy'],
                        eigenvalues=geom['eigenvalues'],
                    ))

            if window_geoms:
                windows_by_layer[layer_id] = window_geoms

        traj = SlidingTrajectory(
            idx,
            int(problem_ids[idx]),
            bool(is_correct_strict[idx] == 0),
            n_windows=sum(len(w) for w in windows_by_layer.values()),
            n_tokens=R,
            windows=windows_by_layer,
        )

        # 保存缓存
        if use_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump(traj, f, protocol=pickle.HIGHEST_PROTOCOL)

        trajectories.append(traj)

    if verbose:
        print(f"\nCache stats: {cache_hits} hits, {cache_misses} misses")

    n_correct = sum(1 for t in trajectories if t.is_correct)
    n_error = sum(1 for t in trajectories if not t.is_correct)

    metadata = {
        'subset': subset,
        'n_chains': n_chains,
        'n_correct': n_correct,
        'n_error': n_error,
        'layers': HIDDEN_LAYERS,
        'window_size': WINDOW_SIZE,
        'stride': STRIDE,
        'cache_hits': cache_hits,
        'cache_misses': cache_misses,
    }

    return trajectories, metadata


if __name__ == "__main__":
    npz_path = "/gz-data/research/demo/data/features/full_omnimath.npz"
    hidden_dir = "/gz-data/research/demo/data/hidden/omnimath/"

    print("Sliding window loading...")
    start = time.time()

    trajectories, metadata = load_all_trajectories_sliding(
        npz_path=npz_path,
        hidden_dir=hidden_dir,
        use_cache=True,
        verbose=True,
    )

    elapsed = time.time() - start
    print(f"\nDone! {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # 打印示例
    if trajectories:
        traj = trajectories[0]
        print(f"\nExample chain {traj.chain_id}:")
        print(f"  Correct: {traj.is_correct}")
        print(f"  Total tokens: {traj.n_tokens}")
        print(f"  Total windows: {traj.n_windows}")
        for layer in HIDDEN_LAYERS:
            if traj.has_layer(layer):
                windows = traj.get_window_sequence(layer)
                print(f"  Layer {layer}: {len(windows)} windows")
                if windows:
                    print(f"    First window: pos={windows[0].token_position}, κ={windows[0].kappa:.3f}")
