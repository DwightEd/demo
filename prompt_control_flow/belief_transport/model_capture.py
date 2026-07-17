from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ModelTopology:
    backbone: object
    blocks: Sequence[object]
    embedding: object
    final_norm: object
    output_head: object

    @property
    def num_depths(self) -> int:
        return len(self.blocks) + 1


def _first_attribute(owner: object, names: Sequence[str]) -> object | None:
    for name in names:
        value = getattr(owner, name, None)
        if value is not None:
            return value
    return None


def resolve_model_topology(model) -> ModelTopology:
    """Resolve the standard decoder-only residual stack and fail closed."""

    backbone = _first_attribute(model, ("model", "transformer", "base_model"))
    if backbone is None or backbone is model:
        backbone = getattr(model, "base_model", None)
    if backbone is None or backbone is model:
        raise TypeError("could not resolve a decoder-only backbone")

    blocks = _first_attribute(backbone, ("layers", "h", "blocks"))
    if blocks is None:
        decoder = getattr(backbone, "decoder", None)
        blocks = getattr(decoder, "layers", None) if decoder is not None else None
    if blocks is None or len(blocks) < 1:
        raise TypeError("could not resolve transformer blocks")

    final_norm = _first_attribute(
        backbone,
        ("norm", "ln_f", "final_layer_norm", "layer_norm"),
    )
    if final_norm is None:
        decoder = getattr(backbone, "decoder", None)
        final_norm = (
            _first_attribute(decoder, ("final_layer_norm", "layer_norm", "norm"))
            if decoder is not None
            else None
        )
    if final_norm is None:
        raise TypeError(
            "could not resolve the final residual normalization; refusing to mix "
            "pre-norm and standard hidden-state semantics"
        )
    embedding = model.get_input_embeddings()
    output_head = model.get_output_embeddings()
    if embedding is None or output_head is None:
        raise TypeError("model must expose input and output embedding modules")
    return ModelTopology(
        backbone=backbone,
        blocks=blocks,
        embedding=embedding,
        final_norm=final_norm,
        output_head=output_head,
    )


def parse_residual_depths(specification: str, num_depths: int) -> tuple[int, ...]:
    text = str(specification).strip().lower()
    if text == "all":
        return tuple(range(int(num_depths)))
    depths = tuple(sorted({int(value.strip()) for value in text.split(",") if value.strip()}))
    if not depths:
        raise ValueError("at least one residual depth is required")
    if any(depth < 0 or depth >= int(num_depths) for depth in depths):
        raise ValueError(
            f"residual depths {depths} fall outside [0, {int(num_depths) - 1}]"
        )
    return depths


class SelectiveBoundaryCapture:
    """Capture only batch boundary states instead of full sequence histories."""

    def __init__(
        self,
        topology: ModelTopology,
        requested_depths: Sequence[int],
        last_token_indices,
    ) -> None:
        self.topology = topology
        self.requested_depths = tuple(int(value) for value in requested_depths)
        self.last_token_indices = last_token_indices
        self._values: dict[int, object] = {}
        self._handles: list[object] = []

    def _boundary(self, output):
        import torch

        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            raise TypeError("captured residual output must have shape [batch, seq, hidden]")
        batch_index = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_index, self.last_token_indices].detach()

    def _hook(self, depth: int):
        def capture(_module, _inputs, output):
            self._values[int(depth)] = self._boundary(output)

        return capture

    def __enter__(self) -> "SelectiveBoundaryCapture":
        final_depth = self.topology.num_depths - 1
        required = set(self.requested_depths) | {final_depth}
        if 0 in required:
            self._handles.append(
                self.topology.embedding.register_forward_hook(self._hook(0))
            )
        for depth in sorted(required):
            if 1 <= depth < final_depth:
                self._handles.append(
                    self.topology.blocks[depth - 1].register_forward_hook(
                        self._hook(depth)
                    )
                )
        self._handles.append(
            self.topology.final_norm.register_forward_hook(self._hook(final_depth))
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def states(self):
        import torch

        missing = [depth for depth in self.requested_depths if depth not in self._values]
        if missing:
            raise RuntimeError(f"residual capture missed depths {missing}")
        return torch.stack([self._values[depth] for depth in self.requested_depths], dim=1)

    def final_state(self):
        final_depth = self.topology.num_depths - 1
        if final_depth not in self._values:
            raise RuntimeError("final normalized residual state was not captured")
        return self._values[final_depth]
