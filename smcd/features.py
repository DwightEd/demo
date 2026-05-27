"""Load geometry JSONL and extract 5-dim feature vectors per step.

Feature vector f_j = (ER, SG, IV, d, alpha) where:
  ER    = step_effective_rank   (expressiveness)
  SG    = step_spectral_gap     (compression indicator)
  IV    = norm_mean             (proxy for information volume)
  d     = displacement          (step dynamics)
  alpha = inter_step_angle      (subspace rotation)
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class FeatureConfig:
    """Which geometry fields map to each feature dimension."""
    feature_keys: List[str] = field(default_factory=lambda: [
        "step_effective_rank",
        "step_spectral_gap",
        "norm_mean",
        "displacement",
        "inter_step_angle",
    ])
    fill_value: float = 0.0  # for None / missing


def _extract_features(step_metrics: dict, cfg: FeatureConfig) -> np.ndarray:
    vec = []
    for key in cfg.feature_keys:
        val = step_metrics.get(key)
        vec.append(val if val is not None else cfg.fill_value)
    return np.array(vec, dtype=np.float32)


def load_geometry(
    path: str,
    cfg: Optional[FeatureConfig] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """Load geometry JSONL → (features_list, labels_list, example_labels).

    Returns:
        features_list: list of [n_steps, feat_dim] arrays, one per example
        labels_list:   list of [n_steps] arrays (0/1 is_error per step)
        example_labels: list of int (-1 = all correct, else first error step idx)
    """
    if cfg is None:
        cfg = FeatureConfig()

    features_list = []
    labels_list = []
    example_labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            steps = rec["step_metrics"]
            if len(steps) < 2:
                continue

            feats = np.stack([_extract_features(s, cfg) for s in steps])
            labs = np.array([s.get("is_error", 0) for s in steps], dtype=np.float32)

            features_list.append(feats)
            labels_list.append(labs)
            example_labels.append(rec["label"])

    return features_list, labels_list, example_labels


def compute_global_stats(
    features_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean/std across all steps for z-score normalization."""
    all_feats = np.concatenate(features_list, axis=0)
    mu = all_feats.mean(axis=0)
    sigma = all_feats.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return mu, sigma
