"""PyTorch Dataset for SMCD feature sequences."""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class SMCDDataset(Dataset):
    """Dataset of per-example feature sequences with labels.

    Each item: (features [T, D], labels [T], constraint_scores [T], length)
    """

    def __init__(
        self,
        features_list: List[np.ndarray],
        labels_list: List[np.ndarray],
        constraint_scores_list: List[np.ndarray],
        normalize: bool = True,
        mu: np.ndarray = None,
        sigma: np.ndarray = None,
    ):
        self.features = features_list
        self.labels = labels_list
        self.constraint_scores = constraint_scores_list

        if normalize and mu is not None and sigma is not None:
            self.features = [(f - mu) / sigma for f in self.features]

        self.mu = mu
        self.sigma = sigma

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            torch.tensor(self.constraint_scores[idx], dtype=torch.float32),
            len(self.features[idx]),
        )


def collate_sequences(batch):
    """Pad variable-length sequences to max length in batch.

    Returns:
        features: [B, T_max, D]
        labels:   [B, T_max]
        scores:   [B, T_max]
        mask:     [B, T_max]
        lengths:  [B]
    """
    features, labels, scores, lengths = zip(*batch)
    max_len = max(lengths)
    B = len(batch)
    D = features[0].shape[-1]

    feat_pad = torch.zeros(B, max_len, D)
    lab_pad = torch.zeros(B, max_len)
    score_pad = torch.zeros(B, max_len)
    mask = torch.zeros(B, max_len)

    for i in range(B):
        T = lengths[i]
        feat_pad[i, :T] = features[i]
        lab_pad[i, :T] = labels[i]
        score_pad[i, :T] = scores[i]
        mask[i, :T] = 1.0

    return feat_pad, lab_pad, score_pad, mask, torch.tensor(lengths)
