# lightgbm_quantile

LightGBM-style quantile model. In this environment it automatically falls back to a histogram gradient boosting quantile model because `lightgbm` is not installed.

Run:

```powershell
python models/quantile/lightgbm_quantile/run.py --mode nonsentiment
python models/quantile/lightgbm_quantile/run.py --mode sentiment
```
