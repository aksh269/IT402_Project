from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from models.quantile.common.utils import resolve_config_path, run_model_pipeline
from models.quantile.gbm_quantile.model import build_model as build_gbm_quantile
from models.quantile.lightgbm_quantile.model import build_model as build_lightgbm_quantile
from models.quantile.quantile_random_forest.model import build_model as build_quantile_random_forest
from models.quantile.quantile_regression.model import build_model as build_quantile_regression
from models.quantile.xgboost_quantile.model import build_model as build_xgboost_quantile

MODEL_BUILDERS = {
    "quantile_regression": build_quantile_regression,
    "quantile_random_forest": build_quantile_random_forest,
    "gbm_quantile": build_gbm_quantile,
    "lightgbm_quantile": build_lightgbm_quantile,
    "xgboost_quantile": build_xgboost_quantile,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full quantile-model family defined in the roadmap.")
    parser.add_argument(
        "--mode",
        choices=["all", "nonsentiment", "sentiment"],
        default="all",
        help="Run both roadmap modes or only one of them.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=sorted(MODEL_BUILDERS),
        default=sorted(MODEL_BUILDERS),
        help="Optional subset of quantile models to execute.",
    )
    parser.add_argument("--data-dir", help="Optional override for the stock CSV directory.")
    parser.add_argument("--output-dir", help="Optional override for the quantile output directory.")
    args = parser.parse_args()

    modes = ["nonsentiment", "sentiment"] if args.mode == "all" else [args.mode]
    for model_name in args.models:
        for mode in modes:
            print(f"Running {model_name} in {mode} mode...")
            run_model_pipeline(
                model_name=model_name,
                config_path=resolve_config_path(model_name=model_name, mode=mode),
                build_model=MODEL_BUILDERS[model_name],
                config_overrides={
                    "data_dir": args.data_dir,
                    "output_dir": args.output_dir,
                },
            )


if __name__ == "__main__":
    main()
