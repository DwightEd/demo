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

    # 特征值分解 (GPU) - CuPy只有eigh，没有eigvalsh
    try:
        eigvals, _ = cp.linalg.eigh(S)  # 返回特征值和特征向量，只取特征值
    except Exception as e:
        print(f"GPU eigh failed: {e}, falling back to CPU")
        return compute_step_geometry_cpu(H, step_id, layer_id, n_top)

    # 降序排列并归一化
    eigvals = cp.sort(eigvals)[::-1]

    # 确保特征值非负（数值稳定性）
    eigvals = cp.maximum(eigvals, 0)

    # 归一化
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

    # 降序排列
    eigvals = np.sort(eigvals)[::-1]

    # 确保特征值非负（数值稳定性）
    eigvals = np.maximum(eigvals, 0)

    # 归一化
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


def compute_chain_geometry_gpu_batch(hidden: np.ndarray, step_ranges, n_top=10):
    """GPU批量计算：一次传输，计算所有steps和layers

    优化：减少CPU-GPU数据传输，加速2-3倍
    """
    try:
        import cupy as cp
    except ImportError:
        return None

    if step_ranges is None or len(step_ranges) == 0:
        return {}

    n_tokens_total, n_layers, d = hidden.shape
    eps = 1e-12

    # 一次性传输整个hidden states到GPU
    hidden_gpu = cp.asarray(hidden.astype(np.float32))

    # 预归一化所有token向量（GPU上）
    norms = cp.linalg.norm(hidden_gpu, axis=2, keepdims=True)
    H_norm_gpu = hidden_gpu / (norms + eps)

    # 计算所有层的所有步骤
    results = {}

    for layer_idx, layer_id in enumerate(HIDDEN_LAYERS):
        layer_results = {}

        a0 = int(step_ranges[0][0])  # 绝对闭区间 -> 分片相对半开区间 (见 data_loading.py 注释)
        for step_id, (start, end) in enumerate(step_ranges):
            lo, hi = int(start) - a0, int(end) - a0 + 1
            if hi <= lo or hi > n_tokens_total:
                continue

            # 在GPU上提取该step的数据
            H_step = H_norm_gpu[lo:hi, layer_idx, :]

            if H_step.shape[0] == 0:
                continue

            n_tokens = H_step.shape[0]

            # 一阶矩（GPU）
            mu = H_step.mean(axis=0)
            kappa = float(cp.linalg.norm(mu))
            norm_val = float(hidden_gpu[lo:hi, layer_idx, :].mean())

            # Scatter matrix（GPU）
            S = (H_step.T @ H_step) / n_tokens

            # 特征值分解（GPU）
            try:
                eigvals, _ = cp.linalg.eigh(S)
            except:
                continue

            # 降序
            eigvals = cp.sort(eigvals)[::-1]

            # 确保非负
            eigvals = cp.maximum(eigvals, 0)

            # 归一化
            eigvals = eigvals / (eigvals.sum() + eps)

            # 有效秩
            lam = eigvals[eigvals > eps]
            if len(lam) > 0:
                eff_rank = float(cp.exp(-cp.sum(lam * cp.log(lam + eps))))
                eff_rank = min(eff_rank, n_tokens)
            else:
                eff_rank = 1.0

            # 谱熵
            spectral_entropy = float(-cp.sum(eigvals * cp.log(eigvals + eps)))

            # 取前n_top特征值
            eigenvalues = cp.asnumpy(eigvals[:n_top])

            layer_results[step_id] = {
                'step_id': step_id,
                'layer': layer_id,
                'n_tokens': n_tokens,
                'kappa': kappa,
                'eff_rank': eff_rank,
                'spectral_entropy': spectral_entropy,
                'norm': norm_val,
                'eigenvalues': eigenvalues,
            }

        if layer_results:
            results[layer_id] = layer_results

    return results


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
            idx, int(problem_ids[idx]), bool(is_correct_strict[idx] == 1), 0
        )

    ranges = step_token_ranges[idx]

    if ranges is None:
        return ReasoningTrajectory(
            idx, int(problem_ids[idx]), bool(is_correct_strict[idx] == 1), 0
        )

    # 使用GPU批量计算（优化）
    if use_gpu:
        batch_results = compute_chain_geometry_gpu_batch(hidden, ranges, n_top=10)

        if batch_results is not None:
            # 转换为StepGeometry格式
            steps = {}
            for layer_id, layer_results in batch_results.items():
                layer_steps = {}
                for step_id, geom in layer_results.items():
                    layer_steps[step_id] = StepGeometry(**geom)
                if layer_steps:
                    steps[layer_id] = layer_steps

            return ReasoningTrajectory(
                idx, int(problem_ids[idx]), bool(is_correct_strict[idx] == 1),
                len(ranges),
                step_ranges=ranges,
                steps=steps,
            )
        else:
            # GPU失败，降级到CPU
            use_gpu = False

    # CPU逐个计算
    steps = {}
    for layer_idx, layer_id in enumerate(HIDDEN_LAYERS):
        layer_steps = {}

        a0 = int(ranges[0][0])  # 绝对闭区间 -> 分片相对半开区间 (见 data_loading.py 注释)
        for step_id, (start, end) in enumerate(ranges):
            if end < start:
                continue

            H = hidden[int(start) - a0:int(end) - a0 + 1, layer_idx, :].copy()
            geom = compute_step_geometry_cpu(H, step_id, layer_id, n_top=10)

            if geom:
                layer_steps[step_id] = StepGeometry(**geom)

        if layer_steps:
            steps[layer_id] = layer_steps

    return ReasoningTrajectory(
        idx, int(problem_ids[idx]), bool(is_correct_strict[idx] == 1),
        len(ranges),
        step_ranges=ranges,
        steps=steps,
    )


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
            print(f"Computing {len(to_compute)} chains...")

        # GPU用单进程（CUDA不支持多进程）
        # CPU用多进程
        if use_gpu:
            try:
                import cupy as cp
                use_gpu_for_compute = True
            except:
                use_gpu_for_compute = False

            if use_gpu_for_compute:
                if verbose:
                    print("GPU mode: using single process (CUDA limitation)")
                # 单进程计算
                start_time = time.time()
                results = []
                for idx in tqdm(to_compute, desc="Computing features (GPU)"):
                    traj = _compute_single_chain((idx, npz_path, hidden_dir, True))
                    results.append(traj)
                elapsed = time.time() - start_time
            else:
                use_gpu_for_compute = False

        if not use_gpu_for_compute:
            if verbose:
                print(f"CPU mode: using {n_workers} workers")
            # 准备参数
            compute_args = [(i, npz_path, hidden_dir, False) for i in to_compute]

            start_time = time.time()

            with Pool(n_workers) as pool:
                results = list(tqdm(
                    pool.imap(_compute_single_chain, compute_args),
                    total=len(compute_args),
                    desc="Computing features (CPU)"
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
    import argparse

    parser = argparse.ArgumentParser(description='GPU加速的几何特征计算')
    parser.add_argument('--benchmark', action='store_true', help='运行性能对比测试')
    parser.add_argument('--clear', action='store_true', help='清除缓存')
    parser.add_argument('--force', action='store_true', help='强制重新计算（忽略缓存）')
    parser.add_argument('--cpu', action='store_true', help='使用CPU而非GPU')
    parser.add_argument('--workers', type=int, default=None, help='进程数（默认自动检测）')
    parser.add_argument('--npz', default='/gz-data/research/demo/data/features/full_omnimath.npz', help='NPZ文件路径')
    parser.add_argument('--hidden', default='/gz-data/research/demo/data/hidden/omnimath/', help='Hidden目录路径')

    args = parser.parse_args()

    if args.benchmark:
        benchmark_methods()
    elif args.clear:
        clear_cache(args.npz, args.hidden)
    else:
        # 正常运行
        print("Loading with GPU acceleration + multiprocessing...")
        start = time.time()

        trajectories, metadata = load_all_trajectories_gpu(
            npz_path=args.npz,
            hidden_dir=args.hidden,
            force_recompute=args.force,
            verbose=True,
            use_gpu=not args.cpu,
            n_workers=args.workers,
        )

        elapsed = time.time() - start
        print(f"\nDone! {elapsed:.1f}s ({elapsed/60:.1f}min)")
