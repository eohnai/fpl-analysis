from typing import List
from sklearn.ensemble import RandomForestRegressor

def load_model() -> RandomForestRegressor:
    raise NotImplementedError("Model loading is not implemented yet")

def suggest_transfer(players: List[str]) -> List[float]:
    # Simple heuristic: longer names get slightly higher scores (placeholder)
    return [len(p) / 10.0 for p in players]