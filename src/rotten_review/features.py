"""Bridge between the dbt warehouse and the ML models.

All set-based feature engineering (reviewer statistics, 7-day burst
windows, z-scores, consensus/publisher deviations, composite bias score)
lives in dbt models and is materialised as `fct_review_features`.
This module reads that mart and adds the two columns that require Python:
model-predicted sentiment (Model 1) and predicted-score residual (Model 2).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from rotten_review.config import WAREHOUSE_PATH
from rotten_review.models.score_regressor import predict_score
from rotten_review.models.sentiment import predict_sentiment
from rotten_review.preprocessing import clean_text


def read_mart(
    table: str = "fct_review_features", warehouse_path: Path = WAREHOUSE_PATH
) -> pd.DataFrame:
    con = duckdb.connect(str(warehouse_path), read_only=True)
    try:
        return con.execute(f"SELECT * FROM {table}").df()
    finally:
        con.close()


def add_model_features(df: pd.DataFrame, sentiment_model, score_model) -> pd.DataFrame:
    """Add predicted_sentiment, predicted_score, score_residual, sentiment_diff."""
    df = df.copy()
    cleaned = df["review_text"].map(clean_text)
    df["text_cleaned"] = cleaned
    df["predicted_sentiment"] = [predict_sentiment(sentiment_model, t) for t in cleaned]
    df["predicted_score"] = [predict_score(score_model, t) for t in cleaned]
    df["score_residual"] = df["score_10"] - df["predicted_score"]
    df["sentiment_diff"] = df["score_residual"].abs()
    df["bias_score"] = compute_bias_score(df)
    return df


def compute_bias_score(df: pd.DataFrame) -> pd.Series:
    """Composite bias score (1-10): weighted, standardised combination of
    z_score_critic, sentiment_diff, consensus_diff and publisher_z_score."""
    from sklearn.preprocessing import StandardScaler, minmax_scale

    from rotten_review.config import BIAS_WEIGHTS

    components = ["z_score_critic", "sentiment_diff", "consensus_diff", "publisher_z_score"]
    values = StandardScaler().fit_transform(df[components].fillna(0.0))
    raw = values @ np.array(BIAS_WEIGHTS)
    return pd.Series(minmax_scale(raw, feature_range=(1, 10)), index=df.index)


def sentiment_agreement(df: pd.DataFrame) -> float:
    """Share of reviews where predicted sentiment matches the fresh/rotten label."""
    labelled = df.dropna(subset=["review_type_encoded", "predicted_sentiment"])
    if labelled.empty:
        return float("nan")
    predicted_positive = labelled["predicted_sentiment"] == 1
    actual_positive = labelled["review_type_encoded"] == 1
    return float(np.mean(predicted_positive == actual_positive))
