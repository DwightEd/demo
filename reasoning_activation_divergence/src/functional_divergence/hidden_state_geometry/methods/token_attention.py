from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


def _sinusoidal_positions(
    length: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Anchor the current suffix at zero so deleting old history does not move it.
    position = torch.arange(length, device=device, dtype=dtype) - (length - 1)
    position = position.unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=dtype)
        * (-math.log(10_000.0) / width)
    )
    encoded = torch.zeros(length, width, device=device, dtype=dtype)
    encoded[:, 0::2] = torch.sin(position * frequency)
    if width > 1:
        encoded[:, 1::2] = torch.cos(position * frequency[: width // 2])
    return encoded


class TokenAttentionPool(nn.Module):
    """Linear-cost learned queries with direct access to every visible token."""

    def __init__(
        self,
        *,
        input_dim: int,
        context_dim: int,
        width: int,
        heads: int,
        queries: int,
    ) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError("attention width must be divisible by the head count")
        self.input_projection = nn.Linear(input_dim, width)
        self.token_norm = nn.LayerNorm(width)
        self.queries = nn.Parameter(torch.empty(queries, width))
        nn.init.normal_(self.queries, std=width**-0.5)
        self.attention = nn.MultiheadAttention(
            width,
            heads,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(queries * width + context_dim, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def _attend(
        self,
        states: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token = self.input_projection(states)
        position = _sinusoidal_positions(
            token.shape[1],
            token.shape[2],
            device=token.device,
            dtype=token.dtype,
        )
        token = self.token_norm(token + position.unsqueeze(0))
        padding = torch.arange(token.shape[1], device=token.device).unsqueeze(0)
        padding = padding >= lengths.to(token.device).unsqueeze(1)
        query = self.queries.unsqueeze(0).expand(token.shape[0], -1, -1)
        pooled, attention = self.attention(
            query,
            token,
            token,
            key_padding_mask=padding,
            need_weights=True,
            average_attn_weights=False,
        )
        return pooled, attention

    def forward_with_attention(
        self,
        states: torch.Tensor,
        lengths: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled, attention = self._attend(states, lengths)
        combined = torch.cat([pooled.flatten(1), context], dim=1)
        return self.head(combined).squeeze(-1), attention

    def forward(
        self,
        states: torch.Tensor,
        lengths: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self.forward_with_attention(states, lengths, context)
        return logits


def history_attention_mass(
    model: TokenAttentionPool,
    sequences: Sequence[np.ndarray],
    current_lengths: np.ndarray,
    context: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    masses = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            stop = min(start + batch_size, len(sequences))
            tensors = [torch.from_numpy(sequences[index]) for index in range(start, stop)]
            lengths = torch.tensor([len(value) for value in tensors], dtype=torch.int64)
            states = pad_sequence(tensors, batch_first=True).to(device)
            features = torch.from_numpy(
                np.asarray(context[start:stop], dtype=np.float32)
            ).to(device)
            _, attention = model.forward_with_attention(states, lengths, features)
            token_mass = attention.mean(dim=(1, 2)).cpu().numpy()
            for local, length in enumerate(lengths.tolist()):
                history_length = length - int(current_lengths[start + local])
                masses.append(float(token_mass[local, :history_length].sum()))
    return np.asarray(masses, dtype=np.float64)


__all__ = ["TokenAttentionPool", "history_attention_mass"]
