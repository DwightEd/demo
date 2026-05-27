"""SMCD v4: Learned Spectral Encoder + Trajectory Evolution Model.

Architecture:
    1. SpectralEncoder: per-step (L, k) multi-layer spectrum → latent z
       - Treats each layer's spectrum as a "token", uses self-attention over layers
       - Learns cross-layer interactions and dimensionality reduction
    2. EvolutionModel: causal Transformer on z sequence → p(z_j | z_{<j})
       - Autoregressive prediction of next latent state
       - Diagonal Gaussian output
    3. TrajectoryModel: end-to-end encoder + evolution
       - Trained on correct trajectories only (NLL loss)
       - Anomaly score = -log p(z_j | z_{<j})

The encoder discovers what aspects of the high-dimensional spectrum matter,
rather than imposing manual dimensionality reduction.
"""

import math
import torch
import torch.nn as nn


class SpectralEncoder(nn.Module):
    """Encode per-step multi-layer spectrum (L, k) → latent z.

    Each layer's k singular values are treated as a token.
    Self-attention learns cross-layer spectral interactions.
    """

    def __init__(self, n_layers, k, d_model=64, n_heads=4, enc_layers=2,
                 latent_dim=128, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.k = k

        self.input_proj = nn.Linear(k, d_model)
        self.layer_pos = nn.Embedding(n_layers, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

        self.pool_proj = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, sigma_ml):
        """
        Args:
            sigma_ml: (B, L, k) multi-layer singular values
        Returns:
            z: (B, latent_dim)
        """
        B, L, k = sigma_ml.shape
        x = self.input_proj(sigma_ml)  # (B, L, d_model)

        pos = self.layer_pos(torch.arange(L, device=sigma_ml.device))
        x = x + pos.unsqueeze(0)

        x = self.transformer(x)  # (B, L, d_model)

        # Attention-weighted pooling over layers
        z = x.mean(dim=1)  # (B, d_model)
        z = self.pool_proj(z)  # (B, latent_dim)
        return z


class EvolutionModel(nn.Module):
    """Causal Transformer for trajectory evolution p(z_j | z_{<j}).

    Autoregressive: at position j, can only attend to positions 0..j.
    Predicts diagonal Gaussian parameters for next latent state.
    """

    def __init__(self, latent_dim=128, n_heads=4, n_layers=4, dropout=0.1,
                 max_steps=64):
        super().__init__()
        self.latent_dim = latent_dim

        self.step_pos = nn.Embedding(max_steps, latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_heads,
            dim_feedforward=latent_dim * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)

    def forward(self, z_seq):
        """
        Args:
            z_seq: (B, T, latent_dim)
        Returns:
            mu: (B, T, latent_dim)
            logvar: (B, T, latent_dim)
        """
        B, T, D = z_seq.shape

        pos = self.step_pos(torch.arange(T, device=z_seq.device))
        x = z_seq + pos.unsqueeze(0)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=z_seq.device)
        h = self.transformer(x, mask=causal_mask, is_causal=True)

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


class TrajectoryModel(nn.Module):
    """End-to-end: SpectralEncoder + EvolutionModel.

    Input: sequence of multi-layer spectra (B, T, L, k)
    Output: per-step anomaly scores
    """

    def __init__(self, n_layers, k, d_model=64, enc_heads=4, enc_layers=2,
                 latent_dim=128, evo_heads=4, evo_layers=4, dropout=0.1,
                 max_steps=64):
        super().__init__()

        self.encoder = SpectralEncoder(
            n_layers=n_layers, k=k, d_model=d_model, n_heads=enc_heads,
            enc_layers=enc_layers, latent_dim=latent_dim, dropout=dropout,
        )
        self.evolution = EvolutionModel(
            latent_dim=latent_dim, n_heads=evo_heads, n_layers=evo_layers,
            dropout=dropout, max_steps=max_steps,
        )

        self.latent_dim = latent_dim
        self.n_spectral_layers = n_layers
        self.k = k

    def encode_sequence(self, sigma_ml_seq):
        """Encode a batch of spectral sequences to latent sequences.

        Args:
            sigma_ml_seq: (B, T, L, k)
        Returns:
            z_seq: (B, T, latent_dim)
        """
        B, T, L, k = sigma_ml_seq.shape
        flat = sigma_ml_seq.reshape(B * T, L, k)
        z_flat = self.encoder(flat)  # (B*T, latent_dim)
        return z_flat.reshape(B, T, -1)

    def forward(self, sigma_ml_seq):
        """Full forward: encode + predict evolution.

        Returns: mu, logvar, z_seq
        """
        z_seq = self.encode_sequence(sigma_ml_seq)
        mu, logvar = self.evolution(z_seq)
        return mu, logvar, z_seq

    def nll_loss(self, sigma_ml_seq, lengths=None, mask=None):
        """Shifted NLL: predict z_j from z_{<j}.

        Trains encoder + evolution jointly.
        """
        mu, logvar, z_seq = self.forward(sigma_ml_seq)

        target = z_seq[:, 1:]       # (B, T-1, D)
        pred_mu = mu[:, :-1]        # (B, T-1, D)
        pred_logvar = logvar[:, :-1] # (B, T-1, D)

        var = torch.exp(pred_logvar) + 1e-6
        nll = 0.5 * ((target - pred_mu) ** 2 / var + pred_logvar
                      + math.log(2 * math.pi))
        nll = nll.sum(dim=-1)  # sum over latent dims → (B, T-1)

        if mask is not None:
            m = mask[:, 1:]
            return (nll * m).sum() / m.sum().clamp(min=1)
        return nll.mean()

    @torch.no_grad()
    def compute_anomaly_scores(self, sigma_ml_seq, lengths=None):
        """Per-step anomaly score = -log p(z_j | z_{<j}).

        Returns: (B, T) — score[:, 0] = 0 (no prediction for first step)
        """
        mu, logvar, z_seq = self.forward(sigma_ml_seq)
        B, T, D = z_seq.shape
        delta = torch.zeros(B, T, device=z_seq.device)

        if T < 2:
            return delta

        target = z_seq[:, 1:]
        pred_mu = mu[:, :-1]
        pred_logvar = logvar[:, :-1]

        var = torch.exp(pred_logvar) + 1e-6
        nll = 0.5 * ((target - pred_mu) ** 2 / var + pred_logvar
                      + math.log(2 * math.pi))
        delta[:, 1:] = nll.sum(dim=-1)
        return delta
