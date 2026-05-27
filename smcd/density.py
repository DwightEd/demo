"""Conditional density model p(t_j | t_{<j}).

Autoregressive model: GRU/Transformer + Gaussian head.
Trained on correct trajectories only, using NLL loss.
Anomaly score: delta_j = -log p(t_j | t_{<j}).

Supports:
  - Full Cholesky covariance (low-dim inputs, <=32)
  - Diagonal covariance (high-dim inputs, multi-layer spectral)
"""

import math
import torch
import torch.nn as nn


class ConditionalDensity(nn.Module):
    """Autoregressive Gaussian density model for transition sequences."""

    def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.1,
                 cov_type="auto"):
        """
        Args:
            cov_type: "full" (Cholesky), "diag" (diagonal), or "auto" (diag if input_dim > 32)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if cov_type == "auto":
            cov_type = "diag" if input_dim > 32 else "full"
        self.cov_type = cov_type

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Gaussian head
        self.mu_head = nn.Linear(hidden_dim, input_dim)

        if cov_type == "full":
            tril_size = input_dim * (input_dim + 1) // 2
            self.L_head = nn.Linear(hidden_dim, tril_size)
            self._tril_indices = torch.tril_indices(input_dim, input_dim)
            self._diag_indices = torch.arange(input_dim)
        else:
            self.logvar_head = nn.Linear(hidden_dim, input_dim)

    def forward(self, t_seq, lengths=None):
        """Predict Gaussian parameters for each position.

        Returns:
            mu: (B, T, D) predicted means
            cov_param: if full -> L (B, T, D, D) Cholesky
                       if diag -> logvar (B, T, D)
        """
        x = self.input_proj(t_seq)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            h, _ = self.gru(x)

        h = self.output_proj(h)
        mu = self.mu_head(h)

        if self.cov_type == "full":
            D = self.input_dim
            tril_params = self.L_head(h)
            batch_shape = tril_params.shape[:-1]
            L = torch.zeros(*batch_shape, D, D, device=h.device, dtype=h.dtype)
            L[..., self._tril_indices[0], self._tril_indices[1]] = tril_params
            L[..., self._diag_indices, self._diag_indices] = \
                nn.functional.softplus(L[..., self._diag_indices, self._diag_indices]) + 1e-4
            return mu, L
        else:
            logvar = self.logvar_head(h)
            return mu, logvar

    def _compute_nll(self, target, pred_mu, cov_param):
        """Compute per-position NLL.

        Returns: (B, T') NLL values
        """
        D = self.input_dim
        diff = target - pred_mu  # (B, T', D)

        if self.cov_type == "full":
            pred_L = cov_param
            d = diff.unsqueeze(-1)
            v = torch.linalg.solve_triangular(pred_L, d, upper=False)
            mahal = (v.squeeze(-1) ** 2).sum(dim=-1)
            log_det = 2.0 * pred_L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        else:
            logvar = cov_param
            var = torch.exp(logvar) + 1e-6
            mahal = (diff ** 2 / var).sum(dim=-1)
            log_det = logvar.sum(dim=-1)

        nll = 0.5 * (mahal + log_det + D * math.log(2 * math.pi))
        return nll

    def nll_loss(self, t_seq, lengths=None, mask=None):
        """NLL loss (shifted: predict step j from context <j)."""
        mu, cov_param = self.forward(t_seq, lengths)

        target = t_seq[:, 1:]
        pred_mu = mu[:, :-1]
        if self.cov_type == "full":
            pred_cov = cov_param[:, :-1]
        else:
            pred_cov = cov_param[:, :-1]

        nll = self._compute_nll(target, pred_mu, pred_cov)

        if mask is not None:
            m = mask[:, 1:]
            return (nll * m).sum() / m.sum().clamp(min=1)
        return nll.mean()

    @torch.no_grad()
    def compute_anomaly_scores(self, t_seq, lengths=None):
        """Per-step anomaly score delta_j = -log p(t_j | t_{<j})."""
        mu, cov_param = self.forward(t_seq, lengths)
        B, T, D = t_seq.shape
        delta = torch.zeros(B, T, device=t_seq.device)

        if T < 2:
            return delta

        target = t_seq[:, 1:]
        pred_mu = mu[:, :-1]
        if self.cov_type == "full":
            pred_cov = cov_param[:, :-1]
        else:
            pred_cov = cov_param[:, :-1]

        nll = self._compute_nll(target, pred_mu, pred_cov)
        delta[:, 1:] = nll
        return delta
