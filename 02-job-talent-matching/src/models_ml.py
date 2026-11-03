import joblib
import numpy as np
from lightgbm import LGBMClassifier


class MLJobTalentModel:
    def __init__(self, **params):
        self.model = LGBMClassifier(**params)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        if X_valid is not None and y_valid is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="binary_logloss",
            )
        else:
            self.model.fit(X_train, y_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def save(self, path: str):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str):
        model = joblib.load(path)
        obj = cls()
        obj.model = model
        return obj
