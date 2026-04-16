from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import ParameterGrid, ParameterSampler

from models.quantile.common.data_loader import (
    create_lagged_supervised_frame,
    list_stock_files,
    load_stock_data,
    split_by_time,
    split_features_and_target,
)
from models.quantile.common.metrics import build_metrics_row, compute_metrics
from models.quantile.common.plotting import (
    plot_metrics_overview,
    plot_mode_comparison,
    plot_stock_diagnostics,
)

DEFAULT_TUNING_CONFIGS: dict[str, dict] = {
    "quantile_regression": {
        "enabled": True,
        "selection_metric": "PinballLoss",
        "n_iter": 8,
        "random_state": 42,
        "lags": [20, 40, 60],
        "target_transforms": ["raw", "difference", "log_return"],
        "param_grid": {
            "alpha": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        },
    },
    "quantile_random_forest": {
        "enabled": True,
        "selection_metric": "PinballLoss",
        "n_iter": 8,
        "random_state": 42,
        "lags": [20, 40, 50],
        "target_transforms": ["difference", "log_return"],
        "param_grid": {
            "n_estimators": [150, 250],
            "max_depth": [8, 12],
            "min_samples_leaf": [1, 2],
            "min_samples_split": [2, 5],
            "max_features": ["sqrt", 0.8],
        },
    },
    "gbm_quantile": {
        "enabled": True,
        "selection_metric": "PinballLoss",
        "n_iter": 10,
        "random_state": 42,
        "lags": [20, 40, 50],
        "target_transforms": ["difference", "log_return"],
        "param_grid": {
            "n_estimators": [200, 350],
            "learning_rate": [0.03, 0.05, 0.08],
            "max_depth": [2, 3, 4],
            "subsample": [0.8, 1.0],
            "min_samples_leaf": [5, 15],
            "max_features": [None, 0.8],
        },
    },
    "lightgbm_quantile": {
        "enabled": True,
        "selection_metric": "PinballLoss",
        "n_iter": 10,
        "random_state": 42,
        "lags": [20, 40, 50],
        "target_transforms": ["difference", "log_return"],
        "param_grid": {
            "n_estimators": [200, 350],
            "learning_rate": [0.03, 0.05, 0.08],
            "max_depth": [3, 6],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
    },
    "xgboost_quantile": {
        "enabled": True,
        "selection_metric": "PinballLoss",
        "n_iter": 10,
        "random_state": 42,
        "lags": [20, 40, 50],
        "target_transforms": ["difference", "log_return"],
        "param_grid": {
            "n_estimators": [200, 300],
            "learning_rate": [0.03, 0.05],
            "max_depth": [3, 4, 6],
            "min_child_weight": [1, 3],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [1.0, 3.0],
        },
    },
}

METRIC_COLUMNS = ["MAE", "MSE", "RMSE", "R2", "PinballLoss", "Bias"]


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def resolve_config_path(model_name: str, mode: str, explicit_path: str | None = None) -> str:
    if explicit_path:
        return explicit_path
    return f"models/quantile/configs/{model_name}_{mode}_config.json"


def prepare_output_dirs(output_root: str | Path, model_name: str) -> dict[str, Path]:
    root = Path(output_root)
    directories = {
        "saved_models": root / "saved_models" / model_name,
        "predictions": root / "predictions" / model_name,
        "metrics": root / "metrics",
        "plots": root / "plots" / model_name,
        "logs": root / "logs" / model_name,
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _serialize_params(model_params: dict) -> str:
    return json.dumps(model_params, sort_keys=True, default=str)


def _describe_estimator(model: object) -> str:
    if hasattr(model, "named_steps"):
        return " -> ".join(type(step).__name__ for step in model.named_steps.values())
    return type(model).__name__


def _build_prediction_frame(
    model_name: str,
    mode: str,
    stock_name: str,
    lag_count: int,
    target_transform: str,
    test_frame: pd.DataFrame,
    predictions,
) -> pd.DataFrame:
    output_frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(test_frame["Date"], utc=True, errors="coerce"),
            "Stock": stock_name,
            "Model": model_name,
            "Mode": mode,
            "Lag_Count": lag_count,
            "Target_Transform": target_transform,
            "Actual": test_frame["Target_Close"].to_numpy(),
            "Predicted": np.asarray(predictions, dtype=float),
        }
    )
    output_frame["Residual"] = output_frame["Actual"] - output_frame["Predicted"]
    output_frame["Absolute_Error"] = output_frame["Residual"].abs()
    return output_frame


def _collapse_unique_value(values: pd.Series, fallback: str = "VARIED") -> str:
    cleaned = values.dropna().astype(str).unique().tolist()
    if not cleaned:
        return fallback
    return cleaned[0] if len(cleaned) == 1 else fallback


def _build_overall_metrics_rows(
    model_name: str,
    mode: str,
    metrics_rows: list[dict[str, float | str]],
    overall_predictions: pd.DataFrame,
    quantile: float,
) -> list[dict[str, float | str]]:
    stock_metrics_frame = pd.DataFrame(metrics_rows)
    metadata = {
        "Lag_Count": _collapse_unique_value(stock_metrics_frame["Lag_Count"]),
        "Target_Transform": _collapse_unique_value(stock_metrics_frame["Target_Transform"]),
        "Estimator": _collapse_unique_value(stock_metrics_frame["Estimator"]),
        "Best_Params": _collapse_unique_value(stock_metrics_frame["Best_Params"]),
    }

    macro_row = {
        "Model": model_name,
        "Mode": mode,
        "Stock": "OVERALL_MACRO",
        **stock_metrics_frame[METRIC_COLUMNS].mean(numeric_only=True).to_dict(),
        **metadata,
    }
    micro_row = {
        "Model": model_name,
        "Mode": mode,
        "Stock": "OVERALL_MICRO",
        **compute_metrics(
            overall_predictions["Actual"],
            overall_predictions["Predicted"],
            quantile=quantile,
        ),
        **metadata,
    }
    return [macro_row, micro_row]


def _maybe_plot_mode_comparison(model_name: str, output_dirs: dict[str, Path]) -> None:
    metrics_dir = output_dirs["metrics"]
    nonsentiment_path = metrics_dir / f"{model_name}_nonsentiment_metrics_summary.csv"
    sentiment_path = metrics_dir / f"{model_name}_sentiment_metrics_summary.csv"

    if nonsentiment_path.exists() and sentiment_path.exists():
        comparison_frame = pd.concat(
            [
                pd.read_csv(nonsentiment_path),
                pd.read_csv(sentiment_path),
            ],
            ignore_index=True,
        )
        plot_mode_comparison(
            metrics_frame=comparison_frame,
            model_name=model_name,
            save_path=output_dirs["plots"] / f"{model_name}_mode_comparison.png",
        )


def _merge_tuning_config(model_name: str, config: dict) -> dict:
    default_config = deepcopy(DEFAULT_TUNING_CONFIGS.get(model_name, {}))
    user_config = deepcopy(config.get("tuning", {}))

    param_grid = default_config.pop("param_grid", {})
    param_grid.update(user_config.pop("param_grid", {}))

    merged_config = {**default_config, **user_config}
    merged_config["param_grid"] = param_grid
    merged_config["lags"] = list(merged_config.get("lags") or [int(config["lags"])])
    merged_config["target_transforms"] = list(
        merged_config.get("target_transforms") or ["raw"]
    )
    merged_config["enabled"] = bool(merged_config.get("enabled", True))
    merged_config["selection_metric"] = str(
        merged_config.get("selection_metric", "PinballLoss")
    )
    merged_config["n_iter"] = int(merged_config.get("n_iter", 1))
    merged_config["random_state"] = int(merged_config.get("random_state", 42))
    return merged_config


def _grid_size(search_space: dict[str, list]) -> int:
    return math.prod(len(values) for values in search_space.values())


def _candidate_signature(candidate: dict) -> str:
    return json.dumps(
        {
            "lags": candidate["lags"],
            "target_transform": candidate["target_transform"],
            "model_params": candidate["model_params"],
        },
        sort_keys=True,
        default=str,
    )


def _build_search_candidates(model_name: str, config: dict) -> list[dict]:
    tuning_config = _merge_tuning_config(model_name=model_name, config=config)
    default_target_transform = (
        str(tuning_config["target_transforms"][0])
        if tuning_config["target_transforms"]
        else "raw"
    )
    base_candidate = {
        "lags": int(config["lags"]),
        "target_transform": str(config.get("target_transform", default_target_transform)),
        "model_params": deepcopy(config.get("model_params", {})),
    }

    if not tuning_config["enabled"]:
        return [base_candidate]

    search_space = {
        "lags": list(tuning_config["lags"]),
        "target_transform": list(tuning_config["target_transforms"]),
    }
    for param_name, values in tuning_config["param_grid"].items():
        search_space[param_name] = list(values) if isinstance(values, (list, tuple)) else [values]

    total_candidates = _grid_size(search_space)
    if total_candidates <= tuning_config["n_iter"]:
        sampled_candidates = list(ParameterGrid(search_space))
    else:
        sampled_candidates = list(
            ParameterSampler(
                search_space,
                n_iter=tuning_config["n_iter"],
                random_state=tuning_config["random_state"],
            )
        )

    candidates = [base_candidate]
    seen = {_candidate_signature(base_candidate)}

    for sampled_candidate in sampled_candidates:
        lag_count = int(sampled_candidate.pop("lags"))
        target_transform = str(sampled_candidate.pop("target_transform"))
        candidate = {
            "lags": lag_count,
            "target_transform": target_transform,
            "model_params": {
                **deepcopy(config.get("model_params", {})),
                **sampled_candidate,
            },
        }
        signature = _candidate_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(candidate)

    return candidates


def _transform_target(y_true, anchor_close, target_transform: str) -> np.ndarray:
    y_true_array = np.asarray(y_true, dtype=float)
    anchor_array = np.asarray(anchor_close, dtype=float)

    if target_transform == "raw":
        return y_true_array
    if target_transform == "difference":
        return y_true_array - anchor_array
    if target_transform == "log_return":
        safe_anchor = np.clip(anchor_array, 1e-9, None)
        safe_target = np.clip(y_true_array, 1e-9, None)
        return np.log(safe_target / safe_anchor)
    raise ValueError(f"Unsupported target transform: {target_transform}")


def _invert_target_transform(predictions, anchor_close, target_transform: str) -> np.ndarray:
    prediction_array = np.asarray(predictions, dtype=float)
    anchor_array = np.asarray(anchor_close, dtype=float)

    if target_transform == "raw":
        return prediction_array
    if target_transform == "difference":
        return prediction_array + anchor_array
    if target_transform == "log_return":
        return np.exp(prediction_array) * anchor_array
    raise ValueError(f"Unsupported target transform: {target_transform}")


def _metric_to_optimization_score(metrics: dict[str, float], metric_name: str) -> float:
    if metric_name == "R2":
        return -float(metrics["R2"])
    if metric_name == "BiasAbs":
        return abs(float(metrics["Bias"]))
    if metric_name not in metrics:
        raise KeyError(f"Selection metric {metric_name} is not available.")
    return float(metrics[metric_name])


def _get_split_bundle(
    stock_frame: pd.DataFrame,
    config: dict,
    lag_count: int,
    split_cache: dict[int, dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    if lag_count not in split_cache:
        supervised_frame = create_lagged_supervised_frame(
            df=stock_frame,
            lags=lag_count,
            use_sentiment=bool(config.get("use_sentiment", False)),
            market_features=config.get("market_features"),
            sentiment_features=config.get("sentiment_features"),
        )
        train_frame, val_frame, test_frame = split_by_time(supervised_frame)
        split_cache[lag_count] = {
            "supervised": supervised_frame,
            "train": train_frame,
            "val": val_frame,
            "test": test_frame,
        }
    return split_cache[lag_count]


def _fit_for_actual_close_predictions(
    build_model: Callable[..., object],
    quantile: float,
    model_params: dict,
    target_transform: str,
    X_train,
    y_train,
    anchor_train,
    X_eval,
    anchor_eval,
) -> tuple[object, np.ndarray]:
    model_target = _transform_target(
        y_true=y_train,
        anchor_close=anchor_train,
        target_transform=target_transform,
    )
    model = build_model(q=quantile, model_params=model_params)
    model.fit(X_train, model_target)
    raw_predictions = np.asarray(model.predict(X_eval), dtype=float)
    actual_scale_predictions = _invert_target_transform(
        predictions=raw_predictions,
        anchor_close=anchor_eval,
        target_transform=target_transform,
    )
    if not np.isfinite(actual_scale_predictions).all():
        raise ValueError("Candidate produced non-finite predictions.")
    return model, actual_scale_predictions


def _tune_stock_model(
    model_name: str,
    mode: str,
    stock_name: str,
    stock_frame: pd.DataFrame,
    config: dict,
    build_model: Callable[..., object],
    quantile: float,
) -> tuple[dict, list[dict[str, object]]]:
    tuning_config = _merge_tuning_config(model_name=model_name, config=config)
    candidates = _build_search_candidates(model_name=model_name, config=config)
    split_cache: dict[int, dict[str, pd.DataFrame]] = {}
    trial_rows: list[dict[str, object]] = []
    best_result: dict | None = None

    for trial_number, candidate in enumerate(candidates, start=1):
        lag_count = int(candidate["lags"])
        target_transform = str(candidate["target_transform"])
        model_params = deepcopy(candidate["model_params"])

        try:
            split_bundle = _get_split_bundle(
                stock_frame=stock_frame,
                config=config,
                lag_count=lag_count,
                split_cache=split_cache,
            )
            X_train, y_train = split_features_and_target(split_bundle["train"])
            X_val, y_val = split_features_and_target(split_bundle["val"])

            model, val_predictions = _fit_for_actual_close_predictions(
                build_model=build_model,
                quantile=quantile,
                model_params=model_params,
                target_transform=target_transform,
                X_train=X_train,
                y_train=y_train,
                anchor_train=split_bundle["train"]["Anchor_Close"],
                X_eval=X_val,
                anchor_eval=split_bundle["val"]["Anchor_Close"],
            )
            validation_metrics = compute_metrics(
                y_val,
                val_predictions,
                quantile=quantile,
            )
            score = _metric_to_optimization_score(
                validation_metrics,
                tuning_config["selection_metric"],
            )
            trial_row = {
                "Model": model_name,
                "Mode": mode,
                "Stock": stock_name,
                "Trial": trial_number,
                "Status": "ok",
                "Lag_Count": lag_count,
                "Target_Transform": target_transform,
                "Estimator": _describe_estimator(model),
                "Best_Params": _serialize_params(model_params),
                "Selection_Metric": tuning_config["selection_metric"],
                "Selection_Score": score,
                **{f"Val_{metric_name}": metric_value for metric_name, metric_value in validation_metrics.items()},
            }
            trial_rows.append(trial_row)

            if best_result is None or score < best_result["score"]:
                best_result = {
                    "score": score,
                    "candidate": {
                        "lags": lag_count,
                        "target_transform": target_transform,
                        "model_params": deepcopy(model_params),
                    },
                    "validation_metrics": validation_metrics,
                    "split_bundle": split_bundle,
                    "estimator": _describe_estimator(model),
                }
        except Exception as exc:
            trial_rows.append(
                {
                    "Model": model_name,
                    "Mode": mode,
                    "Stock": stock_name,
                    "Trial": trial_number,
                    "Status": "failed",
                    "Lag_Count": lag_count,
                    "Target_Transform": target_transform,
                    "Best_Params": _serialize_params(model_params),
                    "Selection_Metric": tuning_config["selection_metric"],
                    "Selection_Score": np.nan,
                    "Error": str(exc),
                }
            )

    if best_result is None:
        raise RuntimeError(f"All tuning candidates failed for {model_name} {mode} {stock_name}.")

    return best_result, trial_rows


def run_model_pipeline(
    model_name: str,
    config_path: str | Path,
    build_model: Callable[..., object],
    config_overrides: dict | None = None,
) -> Path:
    config = load_config(config_path)
    for key, value in (config_overrides or {}).items():
        if value is not None:
            config[key] = value

    mode = "sentiment" if config.get("use_sentiment", False) else "nonsentiment"
    quantile = float(config.get("quantile", 0.5))
    output_dirs = prepare_output_dirs(config["output_dir"], model_name)
    stock_files = list_stock_files(config["data_dir"])
    if not stock_files:
        raise FileNotFoundError(f"No CSV files were found in {config['data_dir']}.")

    metrics_rows: list[dict[str, float | str]] = []
    validation_rows: list[dict[str, object]] = []
    best_config_rows: list[dict[str, object]] = []
    tuning_trial_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for stock_file in stock_files:
        stock_name = stock_file.stem.upper()
        stock_frame = load_stock_data(stock_file)
        best_result, trial_rows = _tune_stock_model(
            model_name=model_name,
            mode=mode,
            stock_name=stock_name,
            stock_frame=stock_frame,
            config=config,
            build_model=build_model,
            quantile=quantile,
        )
        tuning_trial_rows.extend(trial_rows)

        best_candidate = best_result["candidate"]
        split_bundle = best_result["split_bundle"]
        train_val_frame = pd.concat(
            [split_bundle["train"], split_bundle["val"]],
            ignore_index=True,
        )
        test_frame = split_bundle["test"].copy()

        X_train_val, y_train_val = split_features_and_target(train_val_frame)
        X_test, y_test = split_features_and_target(test_frame)
        fitted_model, test_predictions = _fit_for_actual_close_predictions(
            build_model=build_model,
            quantile=quantile,
            model_params=best_candidate["model_params"],
            target_transform=best_candidate["target_transform"],
            X_train=X_train_val,
            y_train=y_train_val,
            anchor_train=train_val_frame["Anchor_Close"],
            X_eval=X_test,
            anchor_eval=test_frame["Anchor_Close"],
        )
        estimator_description = _describe_estimator(fitted_model)

        validation_row = {
            "Model": model_name,
            "Mode": mode,
            "Stock": stock_name,
            "Lag_Count": best_candidate["lags"],
            "Target_Transform": best_candidate["target_transform"],
            "Estimator": estimator_description,
            "Best_Params": _serialize_params(best_candidate["model_params"]),
            **best_result["validation_metrics"],
        }
        validation_rows.append(validation_row)

        metrics_row = build_metrics_row(
            model_name=model_name,
            mode=mode,
            stock_name=stock_name,
            y_true=y_test,
            y_pred=test_predictions,
            quantile=quantile,
        )
        metrics_row.update(
            {
                "Lag_Count": best_candidate["lags"],
                "Target_Transform": best_candidate["target_transform"],
                "Estimator": estimator_description,
                "Best_Params": _serialize_params(best_candidate["model_params"]),
            }
        )
        metrics_rows.append(metrics_row)

        best_config_rows.append(
            {
                "Model": model_name,
                "Mode": mode,
                "Stock": stock_name,
                "Lag_Count": best_candidate["lags"],
                "Target_Transform": best_candidate["target_transform"],
                "Estimator": estimator_description,
                "Best_Params": _serialize_params(best_candidate["model_params"]),
                "Selection_Metric": _merge_tuning_config(model_name=model_name, config=config)["selection_metric"],
                "Selection_Score": best_result["score"],
                **{f"Val_{metric_name}": metric_value for metric_name, metric_value in best_result["validation_metrics"].items()},
            }
        )

        prediction_frame = _build_prediction_frame(
            model_name=model_name,
            mode=mode,
            stock_name=stock_name,
            lag_count=best_candidate["lags"],
            target_transform=best_candidate["target_transform"],
            test_frame=test_frame,
            predictions=test_predictions,
        )
        prediction_frames.append(prediction_frame)

        prediction_path = output_dirs["predictions"] / f"{model_name}_{mode}_{stock_name}_predictions.csv"
        prediction_frame.to_csv(prediction_path, index=False)

        model_path = output_dirs["saved_models"] / f"{model_name}_{mode}_{stock_name}.joblib"
        dump(fitted_model, model_path)

        plot_stock_diagnostics(
            prediction_frame=prediction_frame,
            title=f"{model_name} | {mode} | {stock_name}",
            save_path=output_dirs["plots"] / f"{model_name}_{mode}_{stock_name}_diagnostics.png",
            annotation_lines=[
                f"Lags: {best_candidate['lags']} | Transform: {best_candidate['target_transform']}",
                f"Val Pinball: {best_result['validation_metrics']['PinballLoss']:.4f}",
                f"Test MAE: {metrics_row['MAE']:.4f} | Test RMSE: {metrics_row['RMSE']:.4f}",
                f"Test R2: {metrics_row['R2']:.4f} | Bias: {metrics_row['Bias']:.4f}",
            ],
        )

    overall_predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics_rows.extend(
        _build_overall_metrics_rows(
            model_name=model_name,
            mode=mode,
            metrics_rows=metrics_rows,
            overall_predictions=overall_predictions,
            quantile=quantile,
        )
    )

    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_path = output_dirs["metrics"] / f"{model_name}_{mode}_metrics_summary.csv"
    metrics_frame.to_csv(metrics_path, index=False)

    validation_path = output_dirs["logs"] / f"{model_name}_{mode}_validation_metrics.csv"
    pd.DataFrame(validation_rows).to_csv(validation_path, index=False)

    best_configs_path = output_dirs["logs"] / f"{model_name}_{mode}_best_configs.csv"
    pd.DataFrame(best_config_rows).to_csv(best_configs_path, index=False)

    tuning_trials_path = output_dirs["logs"] / f"{model_name}_{mode}_tuning_trials.csv"
    pd.DataFrame(tuning_trial_rows).to_csv(tuning_trials_path, index=False)

    plot_metrics_overview(
        metrics_frame=metrics_frame,
        model_name=model_name,
        mode=mode,
        save_path=output_dirs["plots"] / f"{model_name}_{mode}_metrics_overview.png",
    )
    _maybe_plot_mode_comparison(model_name, output_dirs)
    return metrics_path


def run_model_from_cli(model_name: str, description: str, build_model: Callable[..., object]) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--mode",
        choices=["nonsentiment", "sentiment"],
        default="nonsentiment",
        help="Choose whether to use only market features or market plus sentiment features.",
    )
    parser.add_argument("--config", help="Optional custom config path. If omitted, the standard roadmap config is used.")
    parser.add_argument("--data-dir", help="Optional override for the stock CSV directory.")
    parser.add_argument("--output-dir", help="Optional override for the quantile output directory.")
    args = parser.parse_args()

    config_path = resolve_config_path(model_name=model_name, mode=args.mode, explicit_path=args.config)
    run_model_pipeline(
        model_name=model_name,
        config_path=config_path,
        build_model=build_model,
        config_overrides={
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
        },
    )
