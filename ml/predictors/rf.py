from typing import List
import pandas as pd
import joblib
from ml.predictors.base import register_predictor

class RFPredictor:
    def __init__(self):
        self.model = None
        self.features: list[str] | None = None

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.features = data["features"]

    def predict(self, X: pd.DataFrame) -> List[float]:
        if self.model is None or self.features is None:
            raise RuntimeError("Model and features must be loaded before prediction")
        X = X.copy()
        for col in self.features:
            if col not in X.columns:
                X[col] = 0
        X = X[self.features].apply(pd.to_numeric, errors="coerce").fillna(0)
        return self.model.predict(X).tolist()

register_predictor("random_forest", RFPredictor)