#!/usr/bin/env python3
"""正确的实现：从hidden states计算真实的谱特征

隐藏数据位置：/gz-data/research/demo/data/hidden/<subset>/<id>.npy
格式：(R, 4, 4096) - R=tokens, 4=layers[10,14,18,22], 4096=hidden dim

正确的几何特征计算：
1. 提取每个步骤的token hidden states
2. 归一化：û = h / ||h||
3. Scatter matrix: S = (1/n) Σ û ⊗ û
4. 特征值分解：λ_1 ≥ λ_2 ≥ ... ≥ λ_d
5. 谱特征：eff_rank, spectral entropy, 谱形状
"""

import numpy as np
from scipy.linalg import eigh
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# =============================================================================
# 真实的谱特征计算
# =============================================================================


def compute_scatter_spectrum(H: np.ndarray, eps: float = 1e-12) -> dict:
    """从hidden states计算scatter matrix的谱特征

    Args:
        H: (n, d) token hidden states

    Returns:
        dict with: kappa, eff_rank, spectrum, spectral_entropy, etc.
    """
    n, d = H.shape
    if n == 0:
        return {}

    # 归一化每个token
    H_norm = H / (np.linalg.norm(H, axis=1, keepdims=True) + eps)

    # 一阶矩：kappa = ||mean(û)||
    mu = H_norm.mean(axis=0)
    kappa = float(np.linalg.norm(mu))

    # 二阶矩：scatter matrix
    S = (H_norm.T @ H_norm) / n  # (d, d)

    # 特征值分解（只取前几个最大的）
    # 对于大矩阵，用eigh比eig更稳定
    eigvals = eigh(S, eigvals_only=True)
    eigvals = np.sort(eigvals)[::-1]  # 降序
    eigvals = eigvals / (eigvals.sum() + eps)  # 归一化

    # 有效秩
    lam = eigvals[eigvals > eps]
    if len(lam) > 0:
        eff_rank = float(np.exp(-np.sum(lam * np.log(lam + eps))))
    else:
        eff_rank = 1.0

    # 谱熵
    spectral_entropy = float(scipy_entropy(eigvals + eps))

    # 前10个特征值
    spectrum_top10 = eigvals[:10]

    return {
        'kappa': kappa,
        'eff_rank': eff_rank,
        'spectrum': spectrum_top10,
        'spectral_entropy': spectral_entropy,
        'lam1': eigvals[0] if len(eigvals) > 0 else np.nan,
        'lam2': eigvals[1] if len(eigvals) > 1 else np.nan,
        'gap': eigvals[0] - eigvals[1] if len(eigvals) > 1 else np.nan,
    }


# =============================================================================
# 数据加载
# =============================================================================


def load_data_with_hidden(npz_path: str, hidden_base: str):
    """加载NPZ数据并关联hidden shards

    Args:
        npz_path: path to full_*.npz
        hidden_base: base directory for hidden shards
    """
    data = np.load(npz_path, allow_pickle=True)

    problem_ids = data['problem_ids']
    is_correct = data['is_correct_strict']
    stepcloud = data['stepcloud']
    step_token_ranges = data.get('step_token_ranges', None)

    # Hidden layers in the shards
    hidden_layers = [10, 14, 18, 22]

    # 读取hidden文件列表
    if 'hidden_files' in data:
        hidden_files = data['hidden_files']
    else:
        # 尝试自动构建文件名
        subset = Path(npz_path).stem.replace('full_', '')
        hidden_files = np.array([f"{subset}-{i}.npy" for i in range(len(problem_ids))])

    N = len(problem_ids)

    chains = []
    for i in tqdm(range(N), desc="Loading chains"):
        # 加载hidden shard
        hidden_path = Path(hidden_base) / hidden_files[i]
        if hidden_path.exists():
            try:
                hidden_data = np.load(hidden_path)  # (R, 4, 4096)
            except:
                hidden_data = None
        else:
            hidden_data = None

        chain = {
            'id': i,
            'problem_id': int(problem_ids[i]),
            'is_correct': bool(is_correct[i] == 0),
            'hidden': hidden_data,
            'step_ranges': step_token_ranges[i] if step_token_ranges is not None else None,
            'n_steps': stepcloud[i].shape[0],
        }
        chains.append(chain)

    return chains, hidden_layers


# =============================================================================
# 计算每个步骤的真实谱特征
# =============================================================================


def compute_step_spectrums(chains: list, hidden_layers: list, layer_id: int = 14):
    """计算每个步骤的真实谱特征

    Returns:
        dict: {'chain_id': {'step_id': spectrum_features, ...}}
    """
    if layer_id not in hidden_layers:
        raise ValueError(f"Layer {layer_id} not in hidden layers {hidden_layers}")

    layer_idx = hidden_layers.index(layer_id)

    all_spectrums = {}

    for chain in tqdm(chains, desc=f"Computing spectrums (L{layer_id})"):
        if chain['hidden'] is None:
            continue

        hidden = chain['hidden']  # (R, 4, 4096)
        step_ranges = chain['step_ranges']

        if step_ranges is None:
            continue

        chain_spectrums = {}

        for step_idx in range(min(len(step_ranges), chain['n_steps'])):
            start, end = step_ranges[step_idx]
            if end <= start:
                continue

            # 提取该步骤的hidden states
            H = hidden[start:end, layer_idx, :]  # (n_tokens, 4096)

            # 计算谱特征
            features = compute_scatter_spectrum(H)
            if features:
                chain_spectrums[step_idx] = features

        if chain_spectrums:
            all_spectrums[chain['id']] = chain_spectrums

    return all_spectrums


# =============================================================================
# 轨迹分析
# =============================================================================


def analyze_trajectory_smoothness(spectrums_dict: dict, chains: list):
    """分析轨迹的smoothness：相邻步骤的谱相似度"""
    from scipy.spatial.distance import jensenshannon

    smoothness_correct = []
    smoothness_error = []

    for chain in chains:
        chain_id = chain['id']
        if chain_id not in spectrums_dict:
            continue

        steps_spectrums = spectrums_dict[chain_id]
        step_ids = sorted(steps_spectrums.keys())

        if len(step_ids) < 2:
            continue

        # 计算相邻步骤的谱相似度
        step_smoothness = []
        for i in range(len(step_ids) - 1):
            s1 = steps_spectrums[step_ids[i]]['spectrum']
            s2 = steps_spectrums[step_ids[i + 1]]['spectrum']

            # JS散度作为距离
            js_dist = jensenshannon(s1, s2)
            similarity = 1.0 - js_dist
            step_smoothness.append(similarity)

        if step_smoothness:
            avg_smoothness = np.mean(step_smoothness)
            if chain['is_correct']:
                smoothness_correct.append(avg_smoothness)
            else:
                smoothness_error.append(avg_smoothness)

    return np.array(smoothness_correct), np.array(smoothness_error)


# =============================================================================
# 主函数
# =============================================================================


def main():
    npz_path = "/gz-data/research/demo/data/features/full_omnimath.npz"
    hidden_base = "/gz-data/research/demo/data/hidden/omnimath/"

    print("=" * 80)
    print("Real Spectrum Analysis from Hidden States")
    print("=" * 80)
    print(f"NPZ: {npz_path}")
    print(f"Hidden base: {hidden_base}")

    # 加载数据
    chains, hidden_layers = load_data_with_hidden(npz_path, hidden_base)

    correct_count = sum(1 for c in chains if c['is_correct'])
    error_count = len(chains) - correct_count

    print(f"\nLoaded {len(chains)} chains")
    print(f"  Correct: {correct_count}, Error: {error_count}")
    print(f"Hidden layers: {hidden_layers}")

    # 计算L14的真实谱特征
    layer_id = 14
    spectrums = compute_step_spectrums(chains, hidden_layers, layer_id)

    print(f"\nComputed spectrums for {len(spectrums)} chains at layer {layer_id}")

    # 分析smoothness
    smooth_correct, smooth_error = analyze_trajectory_smoothness(spectrums, chains)

    print(f"\nSmoothness from real spectrums:")
    print(f"  Correct: {len(smooth_correct)} chains, mean={smooth_correct.mean():.4f}")
    print(f"  Error: {len(smooth_error)} chains, mean={smooth_error.mean():.4f}")

    # AUROC
    if len(smooth_correct) > 0 and len(smooth_error) > 0:
        y_true = np.array([0] * len(smooth_correct) + [1] * len(smooth_error))
        y_score = np.concatenate([smooth_correct, smooth_error])

        auroc = roc_auc_score(y_true, y_score)
        print(f"  AUROC (error detection): {auroc:.4f}")

    # 保存结果
    import json
    output_path = "/gz-data/research/demo/real_spectrum_results.json"

    results = {
        'layer': layer_id,
        'n_chains': len(chains),
        'smooth_correct': smooth_correct.tolist(),
        'smooth_error': smooth_error.tolist(),
        'smooth_correct_mean': float(smooth_correct.mean()) if len(smooth_correct) > 0 else None,
        'smooth_error_mean': float(smooth_error.mean()) if len(smooth_error) > 0 else None,
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    print("=" * 80)


if __name__ == "__main__":
    main()
