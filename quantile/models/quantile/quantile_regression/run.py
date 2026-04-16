
import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from models.quantile.common.utils import run_model_from_cli
from models.quantile.quantile_regression.model import build_model

def main():
    run_model_from_cli(
        model_name="quantile_regression",
        description="Run the quantile regression baseline on all four roadmap stocks.",
        build_model=build_model,
    )


if __name__ == "__main__":
    main()
