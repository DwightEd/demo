from __future__ import annotations

from .contracts import FirstErrorLabels


class FirstErrorSurvival:
    """Discrete-time survival objective aligned to the first invalid step."""

    @staticmethod
    def loss(logits, labels: FirstErrorLabels):
        import torch.nn.functional as functional

        if logits.ndim != 1 or logits.numel() != labels.num_steps:
            raise ValueError("logits must contain one value per reasoning step")
        if labels.first_error == -1:
            return -functional.logsigmoid(-logits).sum()
        safe_prefix = -functional.logsigmoid(-logits[: labels.first_error]).sum()
        failure = -functional.logsigmoid(logits[labels.first_error])
        return safe_prefix + failure

    @staticmethod
    def response_error_tensor(logits):
        import torch

        if logits.ndim != 1:
            raise ValueError("logits must be one-dimensional")
        return 1.0 - torch.sigmoid(-logits).prod()

    @classmethod
    def response_error_probability(cls, logits) -> float:
        return float(cls.response_error_tensor(logits).detach().cpu())
