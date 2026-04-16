from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor


def build_model(q=0.5, model_params=None):
    params = dict(model_params or {})
    max_depth = params.get("max_depth", 6)
    min_child_samples = params.get(
        "min_child_samples",
        params.get("min_samples_leaf", 20),
    )

    try:
        from lightgbm import LGBMRegressor  # type: ignore
    except ModuleNotFoundError:
        # The roadmap expects a lightgbm folder, but the local environment does
        # not ship with LightGBM. This fallback keeps the folder runnable while
        # preserving quantile-loss behavior.
        return GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=max_depth,
            n_estimators=params.get("n_estimators", params.get("max_iter", 300)),
            min_samples_leaf=min_child_samples,
            max_features=params.get("max_features"),
            subsample=params.get("subsample", 0.9),
            random_state=params.get("random_state", 42),
        )

    lightgbm_params = {
        "objective": "quantile",
        "alpha": q,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": max_depth,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_samples": min_child_samples,
        "random_state": 42,
        "n_jobs": -1,
    }
    lightgbm_params.update(params)
    return LGBMRegressor(**lightgbm_params)
