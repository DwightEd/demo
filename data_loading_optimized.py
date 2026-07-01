#!/usr/bin/env python3
"""优化的几何特征计算：使用稀疏特征值分解

只计算前 k 个最大特征值，避免 O(d³) 的完整分解
时间复杂度: O(k·d·iterations) << O(d³)
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.stats import entropy as scipy_entropy


def compute_step_geometry_optimized(H: np.ndarray, step_id: int, layer_id: int,
                                   n_top: int = 10, max_iter: int = 1000):
    """优化的Step几何特征计算：只计算前n_top个特征值

    Args:
        H: hidden states, shape (n_tokens, d)
        step_id: step索引
        layer_id: 层ID
        n_top: 计算前n_top个最大特征值
        max_iter: eigsh最大迭代次数

    Returns:
        dict: 几何特征字典
    """
    n_tokens, d = H.shape
    if n_tokens == 0:
        return None

    eps = 1e-12

    # 归一化token向量
    H_norm = H / (np.linalg.norm(H, axis=1, keepdims=True) + eps)

    # 一阶矩：kappa = ||mean(û)||
    mu = H_norm.mean(axis=0)
    kappa = float(np.linalg.norm(mu))
    norm = float(H.mean())

    # Scatter matrix
    S = (H_norm.T @ H_norm) / n_tokens  # (d, d)

    # 使用稀疏特征值分解：只计算前n_top个最大特征值
    # 时间复杂度: O(n_top · d · iterations)
    n_eigvals = min(n_top, d - 2, n_tokens - 1)  # 不能超过 d-2 和 n_tokens-1
    n_eigvals = max(n_eigvals, 2)  # 至少2个

    try:
        # which='LA' = Largest Algebraic (最大特征值)
        eigvals = eigsh(S, k=n_eigvals, which='LA', maxiter=max_iter)
        eigvals = eigvals[::-1]  # 降序
    except Exception as e:
        # 降级：使用对角元作为最后手段
        S_diag = np.diag(S)
        eigvals = np.sort(S_diag)[::-1][:n_eigvals]

    # 归一化
    eigvals = eigvals / (eigvals.sum() + eps)

    # 截取前n_top个
    eigenvalues = eigvals[:n_top]

    # 计算有效秩 (使用完整的n_eigvals，不只是n_top)
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


def compute_step_geometry_very_fast(H: np.ndarray, step_id: int, layer_id: int,
                                    n_top: int = 10):
    """超快速版本：降维后计算

    通过PCA降维到512维，再计算特征值，大幅加速
    """
    n_tokens, d = H.shape
    if n_tokens == 0:
        return None

    eps = 1e-12

    # 归一化
    H_norm = H / (np.linalg.norm(H, axis=1, keepdims=True) + eps)

    # 一阶矩
    mu = H_norm.mean(axis=0)
    kappa = float(np.linalg.norm(mu))
    norm = float(H.mean())

    # 降维到512维（如果原始维度太大）
    target_d = 512
    if d > target_d:
        # 使用随机投影降维
        np.random.seed(42 + step_id + layer_id)
        proj = np.random.randn(d, target_d).astype(np.float32)
        proj = proj / np.linalg.norm(proj, axis=0, keepdims=True)
        H_reduced = H_norm @ proj  # (n_tokens, 512)
    else:
        H_reduced = H_norm
        target_d = d

    # Scatter matrix (降维后)
    S_reduced = (H_reduced.T @ H_reduced) / n_tokens  # (512, 512)

    # 完整分解（在512维上很快）
    from scipy.linalg import eigh
    try:
        eigvals = eigh(S_reduced, eigvals_only=True)
        eigvals = np.sort(eigvals)[::-1]
    except:
        return None

    # 归一化
    eigvals = eigvals / (eigvals.sum() + eps)

    # 取前n_top个
    eigenvalues = eigvals[:n_top]

    # 计算特征
    lam = eigvals[eigvals > eps]
    eff_rank = float(np.exp(-np.sum(lam * np.log(lam + eps)))) if len(lam) > 0 else 1.0
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


# 性能对比测试
if __name__ == "__main__":
    import time

    print("性能对比测试")
    print("=" * 50)

    # 创建测试数据
    np.random.seed(42)
    n_tokens = 50
    d = 4096
    H = np.random.randn(n_tokens, d).astype(np.float32)

    methods = [
        ("完整eigh分解 (d=4096)", lambda: __import__('scipy.linalg').eigh(
            (H / np.linalg.norm(H, axis=1, keepdims=True)).T @
            (H / np.linalg.norm(H, axis=1, keepdims=True)) / n_tokens,
            eigvals_only=True)),
        ("稀疏eigsh (k=10)", lambda: compute_step_geometry_optimized(H, 0, 14)),
        ("降维eigh (d=512)", lambda: compute_step_geometry_very_fast(H, 0, 14)),
    ]

    for name, func in methods:
        start = time.time()
        try:
            result = func()
            if hasattr(result, '__iter__'):
                result = np.sort(result)[::-1][:10]
            elapsed = time.time() - start
            print(f"{name}: {elapsed*1000:.2f}ms")
        except Exception as e:
            print(f"{name}: 错误 - {e}")
