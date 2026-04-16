from sklearn.ensemble import GradientBoostingRegressor


def build_model(q=0.5, model_params=None):
    params = {
        "loss": "quantile",
        "alpha": q,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_samples_leaf": 10,
        "max_features": None,
        "subsample": 0.9,
        "random_state": 42,
    }
    params.update(model_params or {})
    return GradientBoostingRegressor(**params)
