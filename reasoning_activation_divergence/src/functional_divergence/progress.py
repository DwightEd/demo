from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from tqdm.auto import tqdm


T = TypeVar("T")


class ProgressReporter(Protocol):
    def stage(self, name: str, detail: str = "") -> None: ...

    def track(self, values: Iterable[T], *, total: int, description: str) -> Iterable[T]: ...


class TqdmProgress:
    """Foreground progress reporter used by the production CLI."""

    def stage(self, name: str, detail: str = "") -> None:
        suffix = f": {detail}" if detail else ""
        tqdm.write(f"[{name}]{suffix}")

    def track(self, values: Iterable[T], *, total: int, description: str) -> Iterable[T]:
        return tqdm(values, total=total, desc=description, unit="item", dynamic_ncols=True)


class NullProgress:
    def stage(self, name: str, detail: str = "") -> None:
        return None

    def track(self, values: Iterable[T], *, total: int, description: str) -> Iterable[T]:
        return values


@dataclass
class RecordingProgress:
    """Deterministic reporter for contract tests."""

    events: list[tuple[str, str, object]] = field(default_factory=list)

    def stage(self, name: str, detail: str = "") -> None:
        self.events.append(("stage", name, detail))

    def track(self, values: Iterable[T], *, total: int, description: str) -> Iterator[T]:
        self.events.append(("start", description, total))
        yield from values
        self.events.append(("finish", description, total))
