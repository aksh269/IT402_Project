# Quantile Model Family

This folder implements the roadmap-aligned quantile teammate scope:

- `common/` contains the shared lagged-data pipeline, metrics, plotting, and runner helpers.
- `configs/` contains one config per model and mode.
- `outputs/` stores saved models, predictions, metrics, plots, and logs.
- Each model folder keeps only `model.py`, `run.py`, `README.md`, and `__init__.py`.

The full family can be executed with:

```powershell
python models/quantile/run_all_quantile_models.py
```
