from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred, quantile: float = 0.5) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true_array, y_pred_array)
    mse = mean_squared_error(y_true_array, y_pred_array)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_array, y_pred_array)
    pinball_loss = mean_pinball_loss(y_true_array, y_pred_array, alpha=quantile)
    bias = float(np.mean(y_pred_array - y_true_array))
    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "PinballLoss": float(pinball_loss),
        "Bias": bias,
    }


def build_metrics_row(
    model_name: str,
    mode: str,
    stock_name: str,
    y_true,
    y_pred,
    quantile: float = 0.5,
) -> dict[str, float | str]:
    metrics = compute_metrics(y_true, y_pred, quantile=quantile)
    return {
        "Model": model_name,
        "Mode": mode,
        "Stock": stock_name,
        **metrics,
    }
