# nts/core/registry.py — component registry (mirrors hallucination-detection core/registry.py)
from typing import Callable, Dict, Generic, List, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str):
        self.name = name
        self._reg: Dict[str, Type[T]] = {}

    def register(self, name: str) -> Callable:
        def deco(cls):
            self._reg[name] = cls
            return cls
        return deco

    def get(self, name: str) -> Type[T]:
        if name not in self._reg:
            raise KeyError(f"{self.name}: '{name}' not registered (have {list(self._reg)})")
        return self._reg[name]

    def create(self, name: str, **kw) -> T:
        return self.get(name)(**kw)

    def list(self) -> List[str]:
        return list(self._reg)


SIGNALS = Registry("signals")
GATES = Registry("gates")
