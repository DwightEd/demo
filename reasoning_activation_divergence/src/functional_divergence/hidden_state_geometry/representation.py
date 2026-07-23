from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from sklearn.decomposition import PCA

from ..progress import NullProgress, ProgressReporter
from .tasks import TaskExample, load_visible_states, visible_output_steps


def sample_keyed_permutation(
    example: TaskExample, *, length: int, axis: str, seed: int
) -> np.ndarray:
    """Reproducible per-example null that destroys a shared axis alignment."""
    if length < 1 or axis not in {"time", "layer"}:
        raise ValueError("null permutation needs a positive length and time/layer axis")
    identity = np.arange(length, dtype=np.int64)
    if length == 1:
        return identity
    sample = example.sample
    payload = (
        f"{seed}|{axis}|{sample.dataset}|{sample.chain_id}|"
        f"{sample.manifest_row}|{length}"
    ).encode("utf-8")
    token = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    order = np.random.default_rng(token).permutation(length)
    if length > 2 and np.array_equal(order, identity):
        order = np.roll(order, 1 + token % (length - 1))
    return np.asarray(order, dtype=np.int64)


def cosine_basis(length: int, width: int) -> np.ndarray:
    if length < 1 or width < 1 or width > length:
        raise ValueError("cosine basis width must lie in [1, length]")
    position = np.arange(length, dtype=np.float64)[:, None] + 0.5
    frequency = np.arange(width, dtype=np.float64)[None, :]
    basis = np.cos(np.pi * position * frequency / length)
    basis[:, 0] /= np.sqrt(length)
    if width > 1:
        basis[:, 1:] *= np.sqrt(2.0 / length)
    return basis


class ChainBalancedPCA:
    """Fold-only PCA fitted from an equal number of rows per independent chain."""

    def __init__(self, *, dim: int, positions_per_chain: int = 32, seed: int = 17):
        if dim < 1 or positions_per_chain < 1:
            raise ValueError("dim and positions_per_chain must be positive")
        self.dim = int(dim)
        self.positions_per_chain = int(positions_per_chain)
        self.seed = int(seed)
        self.model: PCA | None = None
        self.sampled_rows_per_chain: dict[tuple[str, int], int] = {}
        self.training_rows = 0

    def fit(
        self,
        examples: tuple[TaskExample, ...],
        *,
        progress: ProgressReporter | None = None,
    ) -> "ChainBalancedPCA":
        latest: dict[tuple[str, int], TaskExample] = {}
        for example in examples:
            key = (example.sample.dataset, example.sample.chain_id)
            if key not in latest or example.visible_steps > latest[key].visible_steps:
                latest[key] = example
        if not latest:
            raise ValueError("cannot fit PCA without training chains")
        reporter = progress or NullProgress()
        tracked = reporter.track(
            latest.items(), total=len(latest), description="PCA chains"
        )
        if self.positions_per_chain < self.dim:
            raise ValueError("positions_per_chain must be at least the PCA dimension")
        count = self.positions_per_chain
        matrix: np.ndarray | None = None
        hidden_dim: int | None = None
        for chain_index, (key, example) in enumerate(tracked):
            states = load_visible_states(example)
            values = states.reshape(-1, states.shape[-1])
            if hidden_dim is None:
                hidden_dim = int(values.shape[1])
                matrix = np.empty(
                    (len(latest) * count, hidden_dim), dtype=np.float32
                )
            elif values.shape[1] != hidden_dim:
                raise ValueError("hidden dimension differs across chains")
            indices = np.linspace(0, len(values) - 1, count, dtype=np.int64)
            if matrix is None:
                raise RuntimeError("PCA sampling matrix was not initialized")
            matrix[chain_index * count : (chain_index + 1) * count] = values[indices]
            self.sampled_rows_per_chain[key] = count
        if matrix is None:
            raise RuntimeError("PCA sampling matrix was not created")
        self.model = PCA(
            n_components=self.dim,
            whiten=True,
            svd_solver="randomized",
            random_state=self.seed,
        ).fit(matrix)
        self.training_rows = int(len(matrix))
        return self

    @property
    def mean_(self) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PCA is not fitted")
        return np.asarray(self.model.mean_)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PCA is not fitted")
        return np.asarray(self.model.explained_variance_ratio_)

    def transform(self, states: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PCA is not fitted")
        values = np.asarray(states, dtype=np.float32)
        flat = values.reshape(-1, values.shape[-1])
        return self.model.transform(flat).reshape(*values.shape[:-1], self.dim)


@dataclass(frozen=True)
class FunctionalEncoder:
    projector: ChainBalancedPCA
    time_basis: int = 3
    layer_basis: int = 3
    null_seed: int = 17

    def _ordered_states(
        self,
        example: TaskExample,
        null: str | None,
        projected: np.ndarray | None,
    ) -> np.ndarray:
        if projected is None:
            states = self.projector.transform(load_visible_states(example))
        else:
            states = np.asarray(projected[: example.visible_steps], dtype=np.float64)
            if len(states) != example.visible_steps:
                raise ValueError("cached projection is shorter than the visible prefix")
        if null == "time":
            states = states[
                sample_keyed_permutation(
                    example, length=len(states), axis="time", seed=self.null_seed
                )
            ]
        elif null == "layer":
            states = states[
                :,
                sample_keyed_permutation(
                    example, length=states.shape[1], axis="layer", seed=self.null_seed
                ),
            ]
        elif null is not None:
            raise ValueError("null must be None, 'time', or 'layer'")
        return states

    def hidden_tensor(
        self,
        example: TaskExample,
        *,
        null: str | None = None,
        projected: np.ndarray | None = None,
    ) -> np.ndarray:
        values = self._ordered_states(example, null, projected)
        layer_width = min(self.layer_basis, values.shape[1])
        layer = cosine_basis(values.shape[1], layer_width)
        if example.boundary_step is None:
            time_width = min(self.time_basis, values.shape[0])
            time = cosine_basis(values.shape[0], time_width)
            encoded = np.einsum("ta,lb,tlq->abq", time, layer, values, optimize=True)
            output = np.zeros((self.time_basis, self.layer_basis, values.shape[-1]))
            output[:time_width, :layer_width] = encoded
            return output
        current = values[-1]
        remote = np.zeros_like(current) if len(values) == 1 else values[:-1].mean(axis=0)
        history = np.stack([current, remote], axis=0)
        encoded = np.einsum("rlq,lb->rbq", history, layer, optimize=True)
        output = np.zeros((2, self.layer_basis, values.shape[-1]))
        output[:, :layer_width] = encoded
        return output

    def output_features(self, example: TaskExample) -> np.ndarray:
        values = visible_output_steps(example).astype(np.float64)
        if example.boundary_step is None:
            width = min(self.time_basis, len(values))
            time = cosine_basis(len(values), width)
            encoded = np.full((self.time_basis, values.shape[1]), np.nan)
            encoded[:width] = time.T @ values
            return encoded.reshape(-1)
        remote = np.full(values.shape[1], np.nan) if len(values) == 1 else np.nanmean(
            values[:-1], axis=0
        )
        return np.concatenate([values[-1], remote])
