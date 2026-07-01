#!/usr/bin/env python3
"""全优化版：GPU + 多进程 + float32 + eigvalsh

优化方法：
1. CuPy GPU加速 (10-50x)
2. 多进程并行处理chains (4-8x)
3. float32精度 (1.5x)
4. eigvalsh代替eigh (1.3x)
5. 批处理优化 (1.2x)

总预期加速：50-100x (89小时 → 1-2小时)
"""

import numpy as np
import pickle
import hashlib
from pathlib import Path
from tqdm import tqdm
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
import os

HIDDEN_LAYERS = [10, 14, 18, 22]


def compute_step_geometry_gpu(H: np.ndarray, step_id: int, layer_id: int, n_top: int = 10):
    """GPU加速的Step几何特征计算

    Args:
        H: hidden states, shape (n_tokens, d)
        step_id: step索引
        layer_id: 层ID
        n_top: 保留前n个特征值

    Returns:
        dict: 几何特征字典
    """
    try:
        import cupy as cp
        import cupy.linalg as cla
    except ImportError:
        print("CuPy not installed, falling back to CPU")
        return compute_step_geometry_cpu(H, step_id, layer_id, n_top)

    n_tokens, d = H.shape
    if n_tokens == 0:
        return None

    eps = 1e-12

    # 转到GPU并使用float32
    H_gpu = cp.asarray(H.astype(np.float32))

    # 归一化token向量 (GPU)
    H_norm = H_gpu / cp.linalg.norm(H_gpu, axis=1, keepdims=True) + eps

    # 一阶矩：kappa = ||mean(û)||
    mu = H_norm.mean(axis=0)
    kappa = float(cp.linalg.norm(mu))
    norm = float(H_gpu.mean())

    # Scatter matrix (GPU)
    S = (H_norm.T @ H_norm) / n_tokens  # (d, d)

    # 特征值分解 (GPU) - 只计算特征值
    try:
        eigvals = cla.eighvalsh(S)  # 只返回特征值，更快
    except Exception as e:
        print(f"GPU eigvalsh failed: {e}, falling back to CPU")
        return compute_step_geometry_cpu(H, step_id, layer_id, n_top)

    # 降序排列并归一化
    eigvals = cp.sort(eigvals)[::-1]
    eigvals = eigvals / (eigvals.sum() + eps)

    # 取前n_top个特征值
    eigenvalues = cp.asnumpy(eigvals[:n_top])  # 转回CPU

    # 计算有效秩 (使用完整特征值)
    eigvals_cpu = cp.asnumpy(eigvals)
    lam = eigvals_cpu[eigvals_cpu > eps]
    if len(lam) > 0:
        eff_rank = float(np.exp(-np.sum(lam * np.log(lam + eps))))
        eff_rank = float(min(eff_rank, n_tokens))
    else:
        eff_rank = 1.0

    # 计算谱熵
    spectral_entropy = float(-np.sum(eigvals_cpu * np.log(eigvals_cpu + eps)))

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


def compute_step_geometry_cpu(H: np.ndarray, step_id: int, layer_id: int, n_top: int = 10):
    """CPU版本（使用eigvalsh优化）"""
    from scipy.linalg import eigvalsh

    n_tokens, d = H.shape
    if n_tokens == 0:
        return None

    eps = 1e-12

    # 使用float32
    H = H.astype(np.float32)

    # 归一化token向量
    H_norm = H / (np.linalg.norm(H, axis=1, keepdims=True) + eps)

    # 一阶矩
    mu = H_norm.mean(axis=0)
    kappa = float(np.linalg.norm(mu))
    norm = float(H.mean())

    # Scatter matrix
    S = (H_norm.T @ H_norm) / n_tokens  # (d, d)

    # 特征值分解 - 只计算特征值
    try:
        eigvals = eigvalsh(S)
    except Exception as e:
        return None

    # 降序排列并归一化
    eigvals = np.sort(eigvals)[::-1]
    eigvals = eigvals / (eigvals.sum() + eps)

    # 取前n_top个特征值
    eigenvalues = eigvals[:n_top]

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
            return None
    return None


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
    principal_directions: np.ndarray = None

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


def _compute_single_chain(args):
    """计算单个chain的特征（用于多进程）"""
    idx, npz_path, hidden_dir, use_gpu = args

    # 加载全局数据（需要在主进程中传递）
    data = np.load(npz_path, allow_pickle=True)
    problem_ids = data['problem_ids']
    is_correct_strict = data['is_correct_strict']
    step_token_ranges = data['step_token_ranges']

    # 加载hidden states
    hidden_path = Path(hidden_dir) / f"{Path(npz_path).stem.replace('full_', '')}-{idx}.npy"
    try:
        hidden = np.load(hidden_path, mmap_mode='r')
    except:
        return ReasoningTrajectory(
            idx, int(problem_ids[idx]), bool(is_correct_strict[idx] == 0), 0
        )

    # 计算特征
    steps = {}
    ranges = step_token_ranges[idx]

    if ranges is None:
        return ReasoningTrajectory(
            idx, int(problem_ids[idx]), bool(is_correct_strict[idx] == 0), 0
        )

    for layer_idx, layer_id in enumerate(HIDDEN_LAYERS):
        layer_steps = {}

        for step_id, (start, end) in enumerate(ranges):
            if end <= start:
                continue

            H = hidden[start:end, layer_idx, :].copy()

            if use_gpu:
                geom = compute_step_geometry_gpu(H, step_id, layer_id, n_top=10)
            else:
                geom = compute_step_geometry_cpu(H, step_id, layer_id, n_top=10)

            if geom:
                layer_steps[step_id] = StepGeometry(**geom)

        if layer_steps:
            steps[layer_id] = layer_steps

    traj = ReasoningTrajectory(
        idx, int(problem_ids[idx]), bool(is_correct_strict[idx] == 0),
        len(ranges) if ranges is not None else 0,
        step_ranges=ranges,
        steps=steps,
    )

    return traj


def load_all_trajectories_gpu(npz_path: str,
                               hidden_dir: str,
                               force_recompute: bool = False,
                               verbose: bool = True,
                               use_gpu: bool = True,
                               n_workers: int = None) -> tuple:
    """使用GPU加速+多进程加载轨迹"""

    # 加载NPZ
    data = np.load(npz_path, allow_pickle=True)
    problem_ids = data['problem_ids']
    is_correct_strict = data['is_correct_strict']
    step_token_ranges = data['step_token_ranges']
    subset = Path(npz_path).stem.replace('full_', '')

    # 缓存目录
    cache_dir = Path(hidden_dir).parent / "cache" / subset
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_chains = len(problem_ids)
    trajectories = []

    if verbose:
        print(f"Cache directory: {cache_dir}")
        print(f"Total chains: {n_chains}")
        print(f"GPU enabled: {use_gpu}")

        if use_gpu:
            try:
                import cupy as cp
                n_gpus = cp.cuda.runtime.getDeviceCount()
                print(f"Available GPUs: {n_gpus}")
            except:
                print("CuPy not available, will use CPU")

    # 确定worker数量
    if n_workers is None:
        n_workers = max(1, cpu_count() - 2)  # 留2个核心给系统

    if verbose:
        print(f"Using {n_workers} workers for multiprocessing")

    # 统计
    cache_hits = 0
    cache_misses = 0
    compute_times = []

    # 准备参数列表
    args_list = [(i, npz_path, hidden_dir, use_gpu) for i in range(n_chains)]

    # 分批处理：先检查缓存
    to_compute = []
    for idx in range(n_chains):
        cached = load_cached_features(cache_dir, idx)
        if cached is not None and not force_recompute:
            cache_hits += 1
            trajectories.append(cached)
        else:
            cache_misses += 1
            to_compute.append(idx)

    if verbose:
        print(f"Cache hits: {cache_hits}, Cache misses: {cache_misses}")

    # 使用多进程计算未缓存的chains
    if to_compute:
        if verbose:
            print(f"Computing {len(to_compute)} chains with GPU + multiprocessing...")

        # 准备参数
        compute_args = [(i, npz_path, hidden_dir, use_gpu) for i in to_compute]

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
            save_cached_features(cache_dir, to_compute[i], traj)
            trajectories.append(traj)

        if verbose:
            avg_time = elapsed / len(to_compute)
            print(f"Average compute time per chain: {avg_time:.2f}s")
            print(f"Total compute time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # 按原始顺序排序
    trajectories.sort(key=lambda x: x.chain_id)

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
        'use_gpu': use_gpu,
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


def benchmark_methods():
    """对比GPU vs CPU性能"""
    import time

    print("=" * 60)
    print("性能对比测试")
    print("=" * 60)

    # 创建测试数据
    np.random.seed(42)
    n_tokens = 50
    d = 4096
    H = np.random.randn(n_tokens, d).astype(np.float32)

    methods = [
        ("CPU (eigvalsh)", lambda: compute_step_geometry_cpu(H, 0, 14)),
    ]

    # 测试GPU是否可用
    try:
        import cupy as cp
        methods.append(("GPU (CuPy)", lambda: compute_step_geometry_gpu(H, 0, 14)))
    except ImportError:
        print("CuPy未安装，跳过GPU测试")

    for name, func in methods:
        times = []
        for _ in range(3):  # 运行3次取平均
            start = time.time()
            result = func()
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        print(f"{name}: {avg_time*1000:.2f}ms (±{np.std(times)*1000:.2f}ms)")
        if result:
            print(f"  kappa: {result['kappa']:.4f}, eff_rank: {result['eff_rank']:.2f}")


if __name__ == "__main__":
    import sys

    # 测试模式
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark_methods()
    else:
        # 正常运行
        npz_path = "/gz-data/research/demo/data/features/full_omnimath.npz"
        hidden_dir = "/gz-data/research/demo/data/hidden/omnimath/"

        print("Loading with GPU acceleration + multiprocessing...")
        start = time.time()

        trajectories, metadata = load_all_trajectories_gpu(
            npz_path=npz_path,
            hidden_dir=hidden_dir,
            force_recompute=False,
            verbose=True,
            use_gpu=True,
            n_workers=8,
        )

        elapsed = time.time() - start
        print(f"\nDone! {elapsed:.1f}s ({elapsed/60:.1f}min)")
