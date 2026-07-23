from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class ContrastSpec:
    name: str
    baseline: str
    candidate: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.baseline or not self.candidate:
            raise ValueError("contrast name and arms cannot be empty")


@dataclass(frozen=True)
class RandomizationSpec:
    name: str
    baseline_prefix: str
    candidate: str
    minimum_visible_steps: int = 1
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.baseline_prefix or not self.candidate:
            raise ValueError("randomization name and arms cannot be empty")
        if self.minimum_visible_steps < 1:
            raise ValueError("minimum_visible_steps must be positive")


@dataclass(frozen=True)
class MethodSpec:
    name: str
    factory: type[Any]
    contrasts: tuple[ContrastSpec, ...]
    randomizations: tuple[RandomizationSpec, ...]
    arm_definitions: Mapping[str, str]
    default_config: Callable[[], object] | None


_METHODS: dict[str, MethodSpec] = {}


def register_method(
    name: str,
    *,
    contrasts: tuple[ContrastSpec, ...] = (),
    randomizations: tuple[RandomizationSpec, ...] = (),
    arm_definitions: Mapping[str, str] | None = None,
    default_config: Callable[[], object] | None = None,
) -> Callable[[type[T]], type[T]]:
    key = str(name).strip()
    if not key:
        raise ValueError("method name cannot be empty")
    contrast_names = [item.name for item in contrasts]
    if len(set(contrast_names)) != len(contrast_names):
        raise ValueError("method contrast names must be unique")

    def decorate(method: type[T]) -> type[T]:
        if key in _METHODS:
            raise ValueError(f"method {key!r} is already registered")
        _METHODS[key] = MethodSpec(
            name=key,
            factory=method,
            contrasts=tuple(contrasts),
            randomizations=tuple(randomizations),
            arm_definitions=dict(arm_definitions or {}),
            default_config=default_config,
        )
        return method

    return decorate


def method_spec(name: str) -> MethodSpec:
    try:
        return _METHODS[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown method {name!r}; available={list(available_methods())}"
        ) from exc


def resolve_method_config(name: str, config: object) -> object:
    spec = method_spec(name)
    if config is not None:
        return config
    if spec.default_config is None:
        raise ValueError(f"method {name!r} requires an explicit configuration")
    return spec.default_config()


def create_method(name: str, config: object) -> Any:
    spec = method_spec(name)
    return spec.factory(resolve_method_config(name, config))


def available_methods() -> tuple[str, ...]:
    return tuple(sorted(_METHODS))
