"""Transition kernel: GRU + Gaussian Cholesky head.

Learns p(f_j | f_{<j}) on correct trajectories only.
Deviation score delta_j = -log p(f_j | f_{<j}).
"""

import torch
import torch.nn as nn
import math


class GaussianCholeskyHead(nn.Module):
    """Maps hidden state to (mu, L) where Sigma = L @ L^T."""

    def __init__(self, hidden_dim: int, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.mu_proj = nn.Linear(hidden_dim, feat_dim)
        # Lower triangular has feat_dim*(feat_dim+1)//2 free parameters
        self.tril_dim = feat_dim * (feat_dim + 1) // 2
        self.L_proj = nn.Linear(hidden_dim, self.tril_dim)

    def forward(self, h: torch.Tensor):
        """
        Args:
            h: [..., hidden_dim]
        Returns:
            mu: [..., feat_dim]
            L:  [..., feat_dim, feat_dim] lower triangular with positive diagonal
        """
        mu = self.mu_proj(h)
        tril_params = self.L_proj(h)

        # Build lower triangular matrix
        batch_shape = tril_params.shape[:-1]
        L = torch.zeros(*batch_shape, self.feat_dim, self.feat_dim,
                         device=h.device, dtype=h.dtype)
        idx = torch.tril_indices(self.feat_dim, self.feat_dim)
        L[..., idx[0], idx[1]] = tril_params
        # Ensure positive diagonal via softplus
        diag_idx = torch.arange(self.feat_dim)
        L[..., diag_idx, diag_idx] = nn.functional.softplus(L[..., diag_idx, diag_idx]) + 1e-4

        return mu, L


class TransitionKernel(nn.Module):
    """GRU-based sequential model: p(f_j | f_{<j}).

    Args:
        feat_dim:   dimension of input features (default 5)
        hidden_dim: GRU hidden size (default 64)
        n_layers:   GRU layers (default 2)
        dropout:    dropout between GRU layers
    """

    def __init__(self, feat_dim: int = 5, hidden_dim: int = 64,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head = GaussianCholeskyHead(hidden_dim, feat_dim)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor = None):
        """
        Args:
            features: [B, T, feat_dim] — full sequence
            lengths:  [B] — actual sequence lengths (for packing)

        Returns:
            mu:    [B, T, feat_dim]   predicted mean for each step
            L:     [B, T, feat_dim, feat_dim] Cholesky factor
        """
        x = self.input_proj(features)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out_packed, _ = self.gru(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        else:
            h, _ = self.gru(x)

        mu, L = self.head(h)
        return mu, L

    @staticmethod
    def nll_loss(features: torch.Tensor, mu: torch.Tensor, L: torch.Tensor,
                 mask: torch.Tensor = None) -> torch.Tensor:
        """Negative log-likelihood of Gaussian with Cholesky parametrization.

        Predict step j from context <j: use shifted targets.
        features[:, 1:] are targets, mu[:, :-1] / L[:, :-1] are predictions.
        """
        target = features[:, 1:]       # [B, T-1, D]
        pred_mu = mu[:, :-1]            # [B, T-1, D]
        pred_L = L[:, :-1]              # [B, T-1, D, D]

        diff = (target - pred_mu).unsqueeze(-1)  # [B, T-1, D, 1]

        # Mahalanobis: diff^T @ Sigma^{-1} @ diff
        # Sigma = L @ L^T, so Sigma^{-1} @ diff = L^{-T} @ L^{-1} @ diff
        # More efficient: solve L @ v = diff → v = L^{-1} @ diff
        v = torch.linalg.solve_triangular(pred_L, diff, upper=False)  # [B, T-1, D, 1]
        mahal = (v.squeeze(-1) ** 2).sum(dim=-1)  # [B, T-1]

        # Log determinant: log|Sigma| = 2 * sum(log(diag(L)))
        log_det = 2.0 * pred_L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)  # [B, T-1]

        nll = 0.5 * (mahal + log_det + self.feat_dim_const(features.shape[-1]))

        if mask is not None:
            # mask: [B, T], shift to match targets
            m = mask[:, 1:]
            nll = nll * m
            return nll.sum() / m.sum().clamp(min=1)
        return nll.mean()

    @staticmethod
    def feat_dim_const(d: int) -> float:
        return d * math.log(2 * math.pi)

    def compute_deviation(self, features: torch.Tensor, lengths: torch.Tensor = None):
        """Compute per-step deviation score delta_j = -log p(f_j | f_{<j}).

        Returns:
            delta: [B, T] tensor, delta[:, 0] = 0 (no prediction for first step)
        """
        mu, L = self.forward(features, lengths)
        B, T, D = features.shape

        delta = torch.zeros(B, T, device=features.device)

        target = features[:, 1:]
        pred_mu = mu[:, :-1]
        pred_L = L[:, :-1]

        diff = (target - pred_mu).unsqueeze(-1)
        v = torch.linalg.solve_triangular(pred_L, diff, upper=False)
        mahal = (v.squeeze(-1) ** 2).sum(dim=-1)
        log_det = 2.0 * pred_L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        nll = 0.5 * (mahal + log_det + D * math.log(2 * math.pi))

        delta[:, 1:] = nll
        return delta
