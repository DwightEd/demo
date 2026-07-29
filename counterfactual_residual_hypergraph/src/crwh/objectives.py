from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .contracts import _validated_finite_real
from .model import ModelOutputs


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    supervised: torch.Tensor
    consistency: torch.Tensor
    contrastive: torch.Tensor


class SemiSupervisedObjective:
    """Masked BCE plus paraphrase invariance and context-view contrast."""

    def __init__(
        self,
        *,
        supervised_weight: float = 1.0,
        consistency_weight: float = 0.25,
        contrastive_weight: float = 0.25,
        contrastive_margin: float = 0.5,
    ) -> None:
        self.supervised_weight = _validated_finite_real(
            supervised_weight,
            name="supervised_weight",
            minimum=0.0,
        )
        self.consistency_weight = _validated_finite_real(
            consistency_weight,
            name="consistency_weight",
            minimum=0.0,
        )
        self.contrastive_weight = _validated_finite_real(
            contrastive_weight,
            name="contrastive_weight",
            minimum=0.0,
        )
        self.contrastive_margin = _validated_finite_real(
            contrastive_margin,
            name="contrastive_margin",
            minimum=0.0,
        )

    def __call__(
        self,
        outputs: ModelOutputs,
        *,
        labels: torch.Tensor,
        context_reference_mask: torch.Tensor | None = None,
    ) -> LossBreakdown:
        labels = labels.to(device=outputs.logits.device)
        if labels.shape != outputs.logits.shape:
            raise ValueError("labels must align with response-token logits")
        if labels.dtype.is_floating_point and not torch.isfinite(labels).all():
            raise ValueError("labels must be finite")
        if not torch.all((labels == -1) | (labels == 0) | (labels == 1)):
            raise ValueError("labels must use -1/0/1")
        labeled = labels >= 0
        zero = outputs.logits.sum() * 0.0
        supervised = zero
        if torch.any(labeled):
            supervised = F.binary_cross_entropy_with_logits(
                outputs.logits[labeled],
                labels[labeled].to(dtype=outputs.logits.dtype),
            )

        consistency = zero
        if outputs.paraphrase_embedding is not None:
            consistency = F.mse_loss(
                outputs.factual_embedding,
                outputs.paraphrase_embedding,
            )

        positive_distance = torch.zeros_like(outputs.logits)
        if outputs.paraphrase_embedding is not None:
            positive_distance = torch.mean(
                torch.square(
                    outputs.factual_embedding - outputs.paraphrase_embedding
                ),
                dim=1,
            )
        negative_distance = torch.mean(
            torch.square(
                outputs.factual_embedding - outputs.counterfactual_embedding
            ),
            dim=1,
        )
        contrastive_per_token = torch.relu(
            self.contrastive_margin + positive_distance - negative_distance
        )
        contrastive_mask = torch.zeros_like(labels, dtype=torch.bool)
        if context_reference_mask is not None:
            if context_reference_mask.dtype != torch.bool:
                raise ValueError("context_reference_mask must be boolean")
            context_reference_mask = context_reference_mask.to(
                device=outputs.logits.device,
            )
            if context_reference_mask.shape != labels.shape:
                raise ValueError(
                    "context_reference_mask must align with response tokens"
                )
            contrastive_mask = contrastive_mask | context_reference_mask
        contrastive = zero
        if torch.any(contrastive_mask):
            contrastive = contrastive_per_token[contrastive_mask].mean()
        total = (
            self.supervised_weight * supervised
            + self.consistency_weight * consistency
            + self.contrastive_weight * contrastive
        )
        return LossBreakdown(
            total=total,
            supervised=supervised,
            consistency=consistency,
            contrastive=contrastive,
        )
