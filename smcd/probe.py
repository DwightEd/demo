"""Constraint-aware trajectory probe.

A discriminative model that learns to detect reasoning errors from
the step × layer spectral grid. Inspired by:
  - Lyapunov Probes (BCE + stability constraint)
  - Hypergraph learning (higher-order feature interactions)
  - Our E×C×N framework (inductive bias, not hand-designed formula)

Architecture:
    1. Per-node features: info-geometric quantities at each (step, layer) pair
    2. Layer attention: learn which layers matter and in what direction
    3. Temporal GRU: capture sequential evolution patterns
    4. Classification head: per-step error probability
    5. E×C×N regularization: encourage alignment with constraint structure

Training uses BOTH correct and error trajectories (discriminative, not generative).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerAttention(nn.Module):
    """Learn layer-specific importance weights and feature transformations.

    Different layers carry different signals with different directions.
    This module learns to aggregate across layers adaptively.
    """

    def __init__(self, n_layers, feat_per_layer, d_out, n_heads=4):
        super().__init__()
        self.n_layers = n_layers
        self.feat_per_layer = feat_per_layer

        # Project each layer's features
        self.layer_proj = nn.Linear(feat_per_layer, d_out)
        # Learnable layer embeddings (capture layer identity)
        self.layer_emb = nn.Embedding(n_layers, d_out)
        # Multi-head attention over layers
        self.attn = nn.MultiheadAttention(d_out, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        """
        Args:
            x: (B, T, L, F) per-layer features at each step
        Returns:
            out: (B, T, d_out) layer-aggregated features per step
        """
        B, T, L, F = x.shape

        # Reshape to process all (B*T) step-layer grids
        x_flat = x.reshape(B * T, L, F)
        h = self.layer_proj(x_flat)  # (B*T, L, d_out)

        # Add layer position info
        pos = self.layer_emb(torch.arange(L, device=x.device))
        h = h + pos.unsqueeze(0)

        # Self-attention over layers → learns which layers interact
        h_attn, attn_weights = self.attn(h, h, h)
        h = self.norm(h + h_attn)

        # Pool over layers (attention-weighted)
        out = h.mean(dim=1)  # (B*T, d_out)
        return out.reshape(B, T, -1), attn_weights


class TrajectoryProbe(nn.Module):
    """End-to-end probe: layer attention + temporal GRU + classification.

    Learns to detect error steps from the step × layer spectral grid.
    """

    def __init__(self, n_layers, feat_per_layer, d_model=64, n_heads=4,
                 gru_hidden=128, gru_layers=2, dropout=0.15):
        super().__init__()

        self.layer_attn = LayerAttention(n_layers, feat_per_layer, d_model, n_heads)

        self.gru = nn.GRU(
            d_model, gru_hidden, num_layers=gru_layers,
            batch_first=True, dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=False,  # causal: only look at past steps
        )

        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, 1),
        )

        self.n_layers = n_layers
        self.feat_per_layer = feat_per_layer

    def forward(self, x, lengths=None):
        """
        Args:
            x: (B, T, L, F) spectral feature grid
            lengths: (B,) actual sequence lengths
        Returns:
            logits: (B, T) per-step error logits
            attn_weights: layer attention weights for interpretability
        """
        B, T, L, F = x.shape

        # Layer attention: aggregate across layers
        h, attn_w = self.layer_attn(x)  # (B, T, d_model)

        # Temporal GRU: model sequential evolution
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                h, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            h_seq, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            h_seq, _ = self.gru(h)

        # Per-step classification
        logits = self.classifier(h_seq).squeeze(-1)  # (B, T)
        return logits, attn_w

    def compute_loss(self, x, labels, lengths=None, mask=None,
                     ecn_features=None, ecn_weight=0.1):
        """BCE loss + optional E×C×N regularization.

        Args:
            x: (B, T, L, F) feature grid
            labels: (B, T) binary labels (0=correct, 1=error)
            mask: (B, T) valid positions
            ecn_features: (B, T, 3) optional [E_mean, C_mean, N_mean] for regularization
            ecn_weight: weight for E×C×N regularization term
        """
        logits, attn_w = self.forward(x, lengths)

        # BCE loss
        if mask is not None:
            bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            bce = (bce * mask).sum() / mask.sum().clamp(min=1)
        else:
            bce = F.binary_cross_entropy_with_logits(logits, labels)

        loss = bce

        # E×C×N regularization: probe output should correlate with constraint violations
        if ecn_features is not None and ecn_weight > 0:
            probs = torch.sigmoid(logits)  # predicted error probability
            E_mean = ecn_features[:, :, 0]  # higher E → less likely error
            C_mean = ecn_features[:, :, 1]  # moderate C → less likely error
            N_mean = ecn_features[:, :, 2]  # non-zero N → less likely error

            # Encourage: high error prob when E is low or N is near zero
            # This is soft: the model can override if data disagrees
            ecn_reg = (
                torch.mean(probs * E_mean * mask) +  # penalize predicting error when E is high
                torch.mean(probs * torch.abs(N_mean) * mask)  # penalize predicting error when N is active
            ) / mask.sum().clamp(min=1) * mask.sum()

            loss = loss + ecn_weight * ecn_reg

        return loss, bce.item(), logits

    @torch.no_grad()
    def predict_scores(self, x, lengths=None):
        """Get per-step error probabilities."""
        logits, attn_w = self.forward(x, lengths)
        return torch.sigmoid(logits), attn_w
