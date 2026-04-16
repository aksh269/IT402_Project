from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _non_overall_rows(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    stock_labels = metrics_frame["Stock"].astype(str)
    return metrics_frame[~stock_labels.str.startswith("OVERALL")].copy()


def plot_stock_diagnostics(
    prediction_frame: pd.DataFrame,
    title: str,
    save_path: str | Path,
    annotation_lines: Sequence[str] | None = None,
) -> None:
    rolling_window = max(5, min(20, len(prediction_frame) // 8 or 5))
    rolling_mae = (
        prediction_frame["Absolute_Error"]
        .rolling(window=rolling_window, min_periods=1)
        .mean()
    )

    figure = plt.figure(figsize=(15, 11))
    grid = figure.add_gridspec(3, 2, height_ratios=[1.8, 1.0, 1.0])

    main_axis = figure.add_subplot(grid[0, :])
    residual_axis = figure.add_subplot(grid[1, :], sharex=main_axis)
    rolling_axis = figure.add_subplot(grid[2, 0], sharex=main_axis)
    scatter_axis = figure.add_subplot(grid[2, 1])

    main_axis.plot(
        prediction_frame["Date"],
        prediction_frame["Actual"],
        label="Actual",
        linewidth=1.7,
        color="tab:blue",
    )
    main_axis.plot(
        prediction_frame["Date"],
        prediction_frame["Predicted"],
        label="Predicted",
        linewidth=1.4,
        color="tab:orange",
    )
    main_axis.set_title(f"{title} | Actual vs Predicted")
    main_axis.legend(loc="upper left")
    main_axis.grid(alpha=0.25)

    if annotation_lines:
        main_axis.text(
            0.015,
            0.98,
            "\n".join(annotation_lines),
            transform=main_axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
        )

    residual_axis.plot(
        prediction_frame["Date"],
        prediction_frame["Residual"],
        color="tab:red",
        linewidth=1.0,
    )
    residual_axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    residual_axis.set_title("Residual Timeline")
    residual_axis.grid(alpha=0.25)

    rolling_axis.plot(
        prediction_frame["Date"],
        rolling_mae,
        color="tab:green",
        linewidth=1.2,
    )
    rolling_axis.set_title(f"Rolling MAE ({rolling_window} periods)")
    rolling_axis.grid(alpha=0.25)

    scatter_axis.scatter(
        prediction_frame["Actual"],
        prediction_frame["Predicted"],
        alpha=0.7,
        color="tab:purple",
        s=20,
    )
    min_bound = min(
        prediction_frame["Actual"].min(),
        prediction_frame["Predicted"].min(),
    )
    max_bound = max(
        prediction_frame["Actual"].max(),
        prediction_frame["Predicted"].max(),
    )
    scatter_axis.plot(
        [min_bound, max_bound],
        [min_bound, max_bound],
        linestyle="--",
        linewidth=1.0,
        color="black",
    )
    scatter_axis.set_title("Actual vs Predicted Scatter")
    scatter_axis.set_xlabel("Actual")
    scatter_axis.set_ylabel("Predicted")
    scatter_axis.grid(alpha=0.25)

    figure.autofmt_xdate(rotation=25)
    figure.tight_layout()
    figure.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_metrics_overview(
    metrics_frame: pd.DataFrame,
    model_name: str,
    mode: str,
    save_path: str | Path,
) -> None:
    stock_frame = _non_overall_rows(metrics_frame)
    if stock_frame.empty:
        return

    figure, axes = plt.subplots(2, 2, figsize=(16, 9))
    metrics_to_plot = ["MAE", "RMSE", "PinballLoss", "R2"]
    axes = axes.ravel()

    for axis, metric_name in zip(axes, metrics_to_plot):
        if metric_name == "R2":
            colors = ["tab:green" if value >= 0 else "tab:red" for value in stock_frame[metric_name]]
            axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        else:
            colors = ["tab:green"] * len(stock_frame)

        axis.bar(stock_frame["Stock"], stock_frame[metric_name], color=colors, alpha=0.85)
        axis.set_title(metric_name)
        axis.grid(axis="y", alpha=0.25)

    figure.suptitle(f"{model_name} | {mode} | Stock Metrics")
    figure.tight_layout()
    figure.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_mode_comparison(metrics_frame: pd.DataFrame, model_name: str, save_path: str | Path) -> None:
    comparison_frame = metrics_frame[metrics_frame["Stock"] == "OVERALL_MACRO"].copy()
    if comparison_frame["Mode"].nunique() < 2:
        return

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    metric_names = ["MAE", "RMSE", "PinballLoss", "R2"]
    axes = axes.ravel()

    for axis, metric_name in zip(axes, metric_names):
        colors = ["tab:blue", "tab:orange"]
        axis.bar(comparison_frame["Mode"], comparison_frame[metric_name], color=colors[: len(comparison_frame)])
        if metric_name == "R2":
            axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(f"MACRO {metric_name}")
        axis.grid(axis="y", alpha=0.25)

    figure.suptitle(f"{model_name} | Sentiment vs Nonsentiment")
    figure.tight_layout()
    figure.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
