"""Conditional density model p(t_j | t_{<j}).

Single autoregressive model: GRU + Gaussian Cholesky head.
Trained on correct trajectories only, using NLL loss.
Anomaly score: delta_j = -log p(t_j | t_{<j}).

This is the core of the method — it learns the statistical pattern of
correct reasoning's spectral-geometric transitions. No external constraint
scores, no separate probes. One model, one loss, one anomaly score.
"""

import math
import torch
import torch.nn as nn


class ConditionalDensity(nn.Module):
    """Autoregressive Gaussian density model for transition sequences."""

    def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )

        # Gaussian head
        self.mu_head = nn.Linear(hidden_dim, input_dim)
        tril_size = input_dim * (input_dim + 1) // 2
        self.L_head = nn.Linear(hidden_dim, tril_size)

        self._tril_indices = torch.tril_indices(input_dim, input_dim)
        self._diag_indices = torch.arange(input_dim)

    def forward(self, t_seq, lengths=None):
        """Predict Gaussian parameters for each position.

        Args:
            t_seq: (B, T, D) transition representations
            lengths: (B,) actual sequence lengths

        Returns:
            mu: (B, T, D) predicted means
            L:  (B, T, D, D) lower-triangular Cholesky factors
        """
        x = self.input_proj(t_seq)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            h, _ = self.gru(x)

        mu = self.mu_head(h)

        D = self.input_dim
        tril_params = self.L_head(h)
        batch_shape = tril_params.shape[:-1]
        L = torch.zeros(*batch_shape, D, D, device=h.device, dtype=h.dtype)
        L[..., self._tril_indices[0], self._tril_indices[1]] = tril_params
        L[..., self._diag_indices, self._diag_indices] = \
            nn.functional.softplus(L[..., self._diag_indices, self._diag_indices]) + 1e-4

        return mu, L

    def nll_loss(self, t_seq, lengths=None, mask=None):
        """Compute NLL loss (shifted: predict step j from context <j).

        Returns scalar loss.
        """
        mu, L = self.forward(t_seq, lengths)

        target = t_seq[:, 1:]       # (B, T-1, D)
        pred_mu = mu[:, :-1]        # (B, T-1, D)
        pred_L = L[:, :-1]          # (B, T-1, D, D)

        diff = (target - pred_mu).unsqueeze(-1)  # (B, T-1, D, 1)
        v = torch.linalg.solve_triangular(pred_L, diff, upper=False)
        mahal = (v.squeeze(-1) ** 2).sum(dim=-1)  # (B, T-1)
        log_det = 2.0 * pred_L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)

        D = self.input_dim
        nll = 0.5 * (mahal + log_det + D * math.log(2 * math.pi))

        if mask is not None:
            m = mask[:, 1:]  # shift mask to match targets
            return (nll * m).sum() / m.sum().clamp(min=1)
        return nll.mean()

    @torch.no_grad()
    def compute_anomaly_scores(self, t_seq, lengths=None):
        """Per-step anomaly score delta_j = -log p(t_j | t_{<j}).

        Returns:
            delta: (B, T) — delta[:, 0] = 0 (no prediction for first step)
        """
        mu, L = self.forward(t_seq, lengths)
        B, T, D = t_seq.shape
        delta = torch.zeros(B, T, device=t_seq.device)

        if T < 2:
            return delta

        target = t_seq[:, 1:]
        pred_mu = mu[:, :-1]
        pred_L = L[:, :-1]

        diff = (target - pred_mu).unsqueeze(-1)
        v = torch.linalg.solve_triangular(pred_L, diff, upper=False)
        mahal = (v.squeeze(-1) ** 2).sum(dim=-1)
        log_det = 2.0 * pred_L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)

        delta[:, 1:] = 0.5 * (mahal + log_det + D * math.log(2 * math.pi))
        return delta
