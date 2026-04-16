from xgboost import XGBRegressor


def build_model(q=0.5, model_params=None):
    params = {
        "objective": "reg:quantileerror",
        "quantile_alpha": q,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(model_params or {})
    return XGBRegressor(**params)
