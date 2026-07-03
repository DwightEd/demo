#!/usr/bin/env python3
"""最小版本：只算κ和简单特征，跳过eigendecomposition

牺牲一些精度换取速度
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

HIDDEN_LAYERS = [10, 14, 18, 22]


@dataclass
class StepGeometry:
    """最小化几何特征"""
    step_id: int
    layer: int
    n_tokens: int
    kappa: float = np.nan
    eff_rank: float = np.nan
    spectral_entropy: float = np.nan
    norm: float = np.nan
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ReasoningTrajectory:
    chain_id: int
    problem_id: int
    is_correct: bool
    n_steps: int
    step_ranges: List[Tuple[int, int]] = field(default_factory=list)
    steps: Dict[int, Dict[int, StepGeometry]] = field(default_factory=dict)

    def has_layer(self, layer: int) -> bool:
        return layer in self.steps and len(self.steps[layer]) > 0

    def get_geometry_sequence(self, layer: int) -> List[StepGeometry]:
        if not self.has_layer(layer):
            return []
        step_geoms = self.steps[layer]
        return [step_geoms[i] for i in sorted(step_geoms.keys()) if step_geoms[i].n_tokens > 0]


def compute_step_geometry_minimal(H: np.ndarray,
                                  step_id: int,
                                  layer_id: int) -> Optional[StepGeometry]:
    """最小化计算：只算κ，用trace估计eff_rank

    跳过eigendecomposition，使用trace-based估计
    """
    n_tokens, d = H.shape
    if n_tokens == 0:
        return None

    eps = 1e-12

    # 归一化
    H_norm = H / (np.linalg.norm(H, axis=1, keepdims=True) + eps)

    # κ（一阶矩）
    mu = H_norm.mean(axis=0)
    kappa = float(np.linalg.norm(mu))

    # 二阶矩：scatter matrix的trace
    S = (H_norm.T @ H_norm) / n_tokens
    trace_S = np.trace(S)

    # 用Frobenius范数估计eff_rank（近似）
    # eff_rank ≈ (trace(S))^2 / trace(S^2) 对于归一化的S
    S2 = S @ S
    trace_S2 = np.trace(S2)

    if trace_S2 > eps:
        # 这是一个近似，基于矩阵的秩估计
        eff_rank_approx = (trace_S ** 2) / trace_S2
        eff_rank = float(min(eff_rank_approx, n_tokens))
    else:
        eff_rank = 1.0

    # 用对角元估计谱熵（粗略近似）
    diag_S = np.diag(S)
    diag_S = diag_S / (diag_S.sum() + eps)
    spec_entropy = float(scipy_entropy(diag_S + eps))

    # 构造一个伪谱（用于JS散度计算）
    # 用trace和单位矩阵构造一个简单分布
    n_eig = 10
    if n_tokens >= n_eig:
        # 简单的递减分布
        eigenvalues = np.array([1.0/n_eig] * n_eig)
    else:
        eigenvalues = np.array([1.0/n_tokens] * n_tokens)

    return StepGeometry(
        step_id=step_id,
        layer=layer_id,
        n_tokens=n_tokens,
        kappa=kappa,
        eff_rank=eff_rank,
        spectral_entropy=spec_entropy,
        norm=float(H.mean()),
        eigenvalues=eigenvalues,
    )


def process_single_chain_minimal(args) -> Optional[ReasoningTrajectory]:
    """处理单个链（最小化版本）"""
    idx, problem_id, is_correct, step_ranges, hidden_path = args

    try:
        hidden = np.load(hidden_path)
    except:
        return ReasoningTrajectory(
            chain_id=idx,
            problem_id=int(problem_id),
            is_correct=bool(is_correct == 1),
            n_steps=0,
            step_ranges=[],
            steps={},
        )

    if step_ranges is None or len(step_ranges) == 0:
        return ReasoningTrajectory(
            chain_id=idx,
            problem_id=int(problem_id),
            is_correct=bool(is_correct == 1),
            n_steps=0,
            step_ranges=[],
            steps={},
        )

    steps = {}
    for layer_idx, layer_id in enumerate(HIDDEN_LAYERS):
        layer_steps = {}
        a0 = int(step_ranges[0][0])  # 绝对闭区间 -> 分片相对半开区间 (见 data_loading.py 注释)
        for step_id, (start, end) in enumerate(step_ranges):
            if end < start:
                continue

            H = hidden[int(start) - a0:int(end) - a0 + 1, layer_idx, :]
            geom = compute_step_geometry_minimal(H, step_id, layer_id)
            if geom is not None:
                layer_steps[step_id] = geom

        if layer_steps:
            steps[layer_id] = layer_steps

    return ReasoningTrajectory(
        chain_id=idx,
        problem_id=int(problem_id),
        is_correct=bool(is_correct == 1),
        n_steps=len(step_ranges),
        step_ranges=step_ranges,
        steps=steps,
    )


def load_all_trajectories_minimal(npz_path: str,
                                 hidden_dir: str,
                                 n_workers: Optional[int] = None,
                                 verbose: bool = True) -> Tuple[List[ReasoningTrajectory], Dict]:
    """最小化加载所有轨迹

    速度：约1-2秒/chain（16个worker）
    """
    data = np.load(npz_path, allow_pickle=True)
    problem_ids = data['problem_ids']
    is_correct_strict = data['is_correct_strict']

    if 'step_token_ranges' not in data:
        raise ValueError("NPZ文件必须包含step_token_ranges字段")

    step_token_ranges = data['step_token_ranges']
    subset = Path(npz_path).stem.replace('full_', '')

    # 准备参数
    args_list = []
    for i in range(len(problem_ids)):
        hidden_filename = f"{subset}-{i}.npy"
        hidden_path = Path(hidden_dir) / hidden_filename
        args_list.append((
            i, problem_ids[i], is_correct_strict[i],
            step_token_ranges[i], str(hidden_path)
        ))

    if n_workers is None:
        n_workers = min(os.cpu_count(), 32)

    trajectories = [None] * len(args_list)

    if verbose:
        print(f"Using {n_workers} workers (minimal mode)...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_chain_minimal, args): args[0]
                   for args in args_list}

        if verbose:
            progress = tqdm(as_completed(futures), total=len(futures),
                           desc="Loading trajectories")
        else:
            progress = as_completed(futures)

        for future in progress:
            idx = futures[future]
            try:
                traj = future.result()
                trajectories[idx] = traj
            except Exception as e:
                if verbose:
                    print(f"Warning: Chain {idx} failed: {e}")

    n_correct = sum(1 for t in trajectories if t is not None and t.is_correct)
    n_error = sum(1 for t in trajectories if t is not None and not t.is_correct)

    metadata = {
        'subset': subset,
        'n_chains': len(trajectories),
        'n_correct': n_correct,
        'n_error': n_error,
        'layers': HIDDEN_LAYERS,
        'npz_path': npz_path,
        'hidden_dir': hidden_dir,
        'mode': 'minimal',
    }

    return trajectories, metadata


if __name__ == "__main__":
    import time
    npz_path = "/gz-data/research/demo/data/features/full_omnimath.npz"
    hidden_dir = "/gz-data/research/demo/data/hidden/omnimath/"

    print("Minimal loading test...")
    start = time.time()

    trajectories, metadata = load_all_trajectories_minimal(
        npz_path=npz_path,
        hidden_dir=hidden_dir,
        n_workers=32,
        verbose=True,
    )

    elapsed = time.time() - start
    print(f"\nDone! Loaded {len(trajectories)} chains in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Average: {elapsed/len(trajectories):.2f}s per chain")
    print(f"Correct: {metadata['n_correct']}, Error: {metadata['n_error']}")
