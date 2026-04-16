
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model(q=0.5, model_params=None):
    params = {
        "alpha": 1e-3,
        "solver": "highs",
    }
    params.update(model_params or {})

    # Linear quantile regression benefits from scaled inputs because the lagged
    # feature matrix can span very different value ranges across price and volume.
    return make_pipeline(
        StandardScaler(),
        QuantileRegressor(quantile=q, **params),
    )
