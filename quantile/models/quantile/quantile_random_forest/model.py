from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class QuantileRandomForestRegressor:
    def __init__(
        self,
        quantile: float = 0.5,
        n_estimators: int = 300,
        max_depth: int | None = 10,
        min_samples_leaf: int = 2,
        min_samples_split: int = 2,
        max_features: str | float | int | None = "sqrt",
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.quantile = quantile
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X, y):
        feature_matrix = X.to_numpy() if hasattr(X, "to_numpy") else X
        self.model.fit(feature_matrix, y)
        return self

    def predict(self, X):
        # The forest gives one prediction per tree; taking the median across
        # trees turns the ensemble into a simple quantile estimator.
        feature_matrix = X.to_numpy() if hasattr(X, "to_numpy") else X
        tree_predictions = np.column_stack(
            [tree.predict(feature_matrix) for tree in self.model.estimators_]
        )
        return np.quantile(tree_predictions, self.quantile, axis=1)


def build_model(q=0.5, model_params=None):
    params = {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_leaf": 2,
        "min_samples_split": 2,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": 1,
    }
    params.update(model_params or {})
    return QuantileRandomForestRegressor(quantile=q, **params)
