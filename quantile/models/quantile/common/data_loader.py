
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

DEFAULT_MARKET_FEATURES = (
    "Close",
    "Volume",
    "Open",
    "High",
    "Low",
    "Adj close",
)
DEFAULT_SENTIMENT_FEATURES = (
    "Scaled_sentiment",
    "Sentiment_gpt",
    "News_flag",
)
META_COLUMNS = ("Date", "Stock", "Target_Close", "Anchor_Close")


def list_stock_files(data_dir: str | Path) -> list[Path]:
    return sorted(Path(data_dir).glob("*.csv"))


def load_stock_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError(f"{path} must contain Date and Close columns.")

    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    numeric_columns = [column for column in df.columns if column != "Date"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["Stock"] = Path(path).stem.upper()
    return df


def _select_feature_columns(
    df: pd.DataFrame,
    use_sentiment: bool,
    market_features: Sequence[str] | None = None,
    sentiment_features: Sequence[str] | None = None,
) -> list[str]:
    selected_columns = [
        column
        for column in (market_features or DEFAULT_MARKET_FEATURES)
        if column in df.columns
    ]

    if use_sentiment:
        requested_sentiment_columns = list(sentiment_features or DEFAULT_SENTIMENT_FEATURES)
        fallback_sentiment_columns = [
            column
            for column in df.columns
            if "sentiment" in column.lower() and column not in requested_sentiment_columns
        ]
        for column in requested_sentiment_columns + fallback_sentiment_columns:
            if column in df.columns and column not in selected_columns:
                selected_columns.append(column)

    if not selected_columns:
        raise ValueError("No usable feature columns were found in the stock data.")

    return selected_columns


def create_lagged_supervised_frame(
    df: pd.DataFrame,
    lags: int = 50,
    use_sentiment: bool = False,
    market_features: Sequence[str] | None = None,
    sentiment_features: Sequence[str] | None = None,
) -> pd.DataFrame:
    feature_columns = _select_feature_columns(
        df=df,
        use_sentiment=use_sentiment,
        market_features=market_features,
        sentiment_features=sentiment_features,
    )

    lagged_blocks = []
    for lag in range(1, lags + 1):
        shifted = df[feature_columns].shift(lag).add_suffix(f"_lag_{lag}")
        lagged_blocks.append(shifted)

    supervised = pd.concat(
        [
            df[["Date", "Stock", "Close"]].rename(columns={"Close": "Target_Close"}),
            df[["Close"]].shift(1).rename(columns={"Close": "Anchor_Close"}),
            *lagged_blocks,
        ],
        axis=1,
    )
    return supervised.dropna().reset_index(drop=True)


def split_by_time(
    supervised_frame: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_count = len(supervised_frame)
    if sample_count < 10:
        raise ValueError("Not enough rows remain after lag creation to split the dataset.")

    train_end = int(sample_count * train_ratio)
    val_end = int(sample_count * (train_ratio + val_ratio))

    train_frame = supervised_frame.iloc[:train_end].copy()
    val_frame = supervised_frame.iloc[train_end:val_end].copy()
    test_frame = supervised_frame.iloc[val_end:].copy()
    return train_frame, val_frame, test_frame


def split_features_and_target(
    supervised_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    feature_frame = supervised_frame.drop(columns=list(META_COLUMNS))
    target_series = supervised_frame["Target_Close"].copy()
    return feature_frame, target_series
