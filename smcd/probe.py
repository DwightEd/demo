"""Constraint-aware probe: g_theta with input [f_j; delta_j].

Loss = L_cls (BCE) + lambda * L_cst (constraint margin regularizer)

L_cst encourages the probe to assign higher error probability when
the constraint score S_j is low (i.e., the step violates manifold constraints).
"""

import torch
import torch.nn as nn


class ConstraintProbe(nn.Module):
    """Binary probe: P(error | f_j, delta_j).

    Input: [f_j (5-dim), delta_j (1-dim)] = 6-dim
    Output: scalar logit
    """

    def __init__(self, feat_dim: int = 5, hidden_dim: int = 32):
        super().__init__()
        input_dim = feat_dim + 1  # features + deviation score
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, feat_dim] or [B, T, feat_dim]
            delta:    [B] or [B, T]

        Returns:
            logits: [B] or [B, T]
        """
        if delta.dim() < features.dim():
            delta = delta.unsqueeze(-1)
        x = torch.cat([features, delta], dim=-1)
        return self.net(x).squeeze(-1)


def constraint_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    constraint_scores: torch.Tensor,
    lam: float = 0.1,
    margin: float = 0.5,
    mask: torch.Tensor = None,
) -> tuple:
    """Combined loss: BCE + lambda * constraint margin regularizer.

    L_cst = mean(max(0, margin - (prob_error * (1 - S_j))))

    Intuition: when S_j is low (constraint violated), the probe should output
    high error probability. The product prob_error * (1 - S_j) should exceed margin.

    Args:
        logits: [B, T] raw logits from probe
        labels: [B, T] binary labels (0=correct, 1=error)
        constraint_scores: [B, T] S_j values in [0, 1]
        lam: weight for constraint loss
        margin: margin for hinge loss
        mask: [B, T] valid step mask

    Returns:
        (total_loss, bce_loss, cst_loss)
    """
    bce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')

    prob_error = torch.sigmoid(logits)
    violation = 1.0 - constraint_scores  # high when constraints violated
    cst = torch.relu(margin - prob_error * violation)

    if mask is not None:
        bce = (bce * mask).sum() / mask.sum().clamp(min=1)
        cst = (cst * mask).sum() / mask.sum().clamp(min=1)
    else:
        bce = bce.mean()
        cst = cst.mean()

    total = bce + lam * cst
    return total, bce, cst
