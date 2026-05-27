"""Build transition representations from subspace data.

Each step j gets a transition vector:
    t_j = [v_j in R^r,  sigma_j in R^k,  delta_sigma_j in R^k]

where:
    v_j         = PCA_r(Log_{V_{j-1}}(V_j))  — projected tangent vector (subspace rotation)
    sigma_j     = top-k singular values        — spectral shape (encodes E/C/N)
    delta_sigma = sigma_j - sigma_{j-1}        — spectral dynamics

For step 0: v_0 = 0, delta_sigma_0 = 0 (no previous step).

The three CIM constraints are readable from this representation:
    E (Expressiveness):  effective rank of sigma_j (spectral entropy)
    C (Compression):     ||v_j|| bounded (subspace doesn't rotate wildly)
    N (Non-degeneracy):  ||v_j|| > 0 and delta_sigma != 0 (evolution continues)
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from .grassmann import grassmann_log, TangentPCA


def load_subspaces(path: str) -> List[Dict]:
    """Load subspace data saved by 01b_extract_subspaces.py."""
    return torch.load(path, weights_only=False)


def learn_tangent_pca(
    data: List[Dict],
    n_components: int = 16,
) -> TangentPCA:
    """Learn PCA basis from correct trajectory tangent vectors.

    Uses only correct steps: all steps from correct trajectories,
    plus steps before first error in error trajectories.
    """
    tangent_vectors = []

    for ex in data:
        label = ex["label"]
        steps = ex["steps"]
        # How many steps are correct in this trajectory
        n_correct = len(steps) if label == -1 else min(label, len(steps))

        for j in range(1, n_correct):
            V_prev = steps[j - 1]["V"].float()
            V_curr = steps[j]["V"].float()
            Delta = grassmann_log(V_prev, V_curr)
            tangent_vectors.append(Delta)

    print(f"  Learning tangent PCA from {len(tangent_vectors)} correct-step tangent vectors")

    pca = TangentPCA(n_components=n_components)
    pca.fit(tangent_vectors)
    return pca


def compute_representations(
    data: List[Dict],
    pca: TangentPCA = None,
    mode: str = "full",
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """Compute transition representations for all examples.

    Args:
        mode: "full"       = [v_j, sigma_j, delta_sigma_j]  (r + 2k dims)
              "spectral"   = [sigma_j, delta_sigma_j]        (2k dims)
              "sigma_only" = [sigma_j]                        (k dims)
              "tangent"    = [v_j]                            (r dims)

    Returns:
        features_list, labels_list, example_labels
    """
    features_list = []
    labels_list = []
    example_labels = []

    k = data[0]["steps"][0]["V"].shape[1]
    use_tangent = mode in ("full", "tangent")
    use_sigma = mode in ("full", "spectral", "sigma_only")
    use_delta_sigma = mode in ("full", "spectral")

    if use_tangent:
        assert pca is not None, "PCA required for tangent mode"
        r = pca.n_components

    for ex in data:
        steps = ex["steps"]
        T = len(steps)
        if T < 2:
            continue

        V_list = [s["V"].float() for s in steps] if use_tangent else None
        sigma_list = [s["sigma"] for s in steps]
        step_labels = [s["is_error"] for s in steps]

        t_list = []
        for j in range(T):
            parts = []

            if use_tangent:
                if j == 0:
                    parts.append(torch.zeros(r))
                else:
                    Delta = grassmann_log(V_list[j - 1], V_list[j])
                    parts.append(pca.project(Delta))

            if use_sigma:
                parts.append(sigma_list[j])

            if use_delta_sigma:
                if j == 0:
                    parts.append(torch.zeros(k))
                else:
                    parts.append(sigma_list[j] - sigma_list[j - 1])

            t_list.append(torch.cat(parts))

        features = torch.stack(t_list).numpy()
        labels = np.array(step_labels, dtype=np.float32)

        features_list.append(features)
        labels_list.append(labels)
        example_labels.append(ex["label"])

    return features_list, labels_list, example_labels


def compute_normalization(features_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Global mean/std for z-score normalization."""
    all_feats = np.concatenate(features_list, axis=0)
    mu = all_feats.mean(axis=0)
    sigma = all_feats.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return mu, sigma
