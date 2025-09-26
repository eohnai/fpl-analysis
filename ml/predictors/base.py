from typing import Protocol, Dict, Type
import pandas as pd

class Predictor(Protocol):
    def load(self, path: str) -> None: ...
    def predict(self, X: pd.DataFrame) -> list[float]: ...

_REGISTRY: Dict[str, Type[Predictor]] = {}

def register_predictor(name: str, cls: Type[Predictor]) -> None:
    _REGISTRY[name] = cls

def get_predictor(name: str) -> Type[Predictor]:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown predictor: {name}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]