from __future__ import annotations

import numpy as np


def _paired_values(
    scores: np.ndarray, labels: np.ndarray, pair_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    retained: list[int] = []
    differences: list[float] = []
    wins: list[float] = []
    for pair in np.unique(pair_ids):
        rows = np.where(pair_ids == pair)[0]
        error = scores[rows[labels[rows] == 1]]
        control = scores[rows[labels[rows] == 0]]
        if error.size != 1 or control.size != 1 or not np.isfinite(error[0] + control[0]):
            continue
        difference = float(error[0] - control[0])
        retained.append(int(pair))
        differences.append(difference)
        wins.append(1.0 if difference > 0 else 0.5 if difference == 0 else 0.0)
    return np.asarray(retained), np.asarray(differences), np.asarray(wins)


def paired_summary(
    scores: np.ndarray,
    labels: np.ndarray,
    pair_ids: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 17,
) -> dict[str, float | int]:
    _, differences, wins = _paired_values(scores, labels, pair_ids)
    if differences.size == 0:
        raise ValueError("no complete finite matched pairs")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(int(n_boot), differences.size))
    boot_auc = np.mean(wins[indices], axis=1)
    boot_diff = np.mean(differences[indices], axis=1)
    signs = rng.choice(
        np.asarray([-1.0, 1.0]), size=(max(int(n_boot), 1000), differences.size)
    )
    observed = abs(float(np.mean(differences)))
    null = np.abs(np.mean(signs * differences, axis=1))
    p_value = (1.0 + np.sum(null >= observed)) / (signs.shape[0] + 1.0)
    return {
        "n_pairs": int(differences.size),
        "paired_auroc": float(np.mean(wins)),
        "paired_auroc_ci_low": float(np.quantile(boot_auc, 0.025)),
        "paired_auroc_ci_high": float(np.quantile(boot_auc, 0.975)),
        "mean_paired_difference": float(np.mean(differences)),
        "difference_ci_low": float(np.quantile(boot_diff, 0.025)),
        "difference_ci_high": float(np.quantile(boot_diff, 0.975)),
        "sign_flip_p": float(p_value),
    }

def paired_auc_difference(
    candidate: np.ndarray,
    baseline: np.ndarray,
    labels: np.ndarray,
    pair_ids: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 17,
) -> dict[str, float | int]:
    candidate_pairs, _, candidate_wins = _paired_values(candidate, labels, pair_ids)
    baseline_pairs, _, baseline_wins = _paired_values(baseline, labels, pair_ids)
    common = np.intersect1d(candidate_pairs, baseline_pairs)
    if common.size == 0:
        raise ValueError("no common finite pairs for method comparison")
    candidate_lookup = dict(zip(candidate_pairs.tolist(), candidate_wins.tolist()))
    baseline_lookup = dict(zip(baseline_pairs.tolist(), baseline_wins.tolist()))
    differences = np.asarray(
        [candidate_lookup[int(pair)] - baseline_lookup[int(pair)] for pair in common]
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, common.size, size=(int(n_boot), common.size))
    bootstrap = np.mean(differences[indices], axis=1)
    return {
        "n_pairs": int(common.size),
        "delta_paired_auroc": float(np.mean(differences)),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
    }
