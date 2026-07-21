from __future__ import annotations

import numpy as np

from .contracts import ContributionMap, TransportInputs


class OutputEffectiveTransport:
    """Combine attention routing, OV content, and output sensitivity."""

    def compute(self, inputs: TransportInputs) -> ContributionMap:
        per_head = inputs.attention * inputs.content_effect
        prompt_mask = np.arange(inputs.attention.shape[2]) < inputs.prompt_end
        return ContributionMap(
            per_head=per_head,
            signed=per_head.sum(axis=0),
            prompt_mask=prompt_mask,
            receiver_positions=inputs.receiver_positions,
        )
