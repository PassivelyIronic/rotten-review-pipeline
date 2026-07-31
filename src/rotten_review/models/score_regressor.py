"""Model 2 — review text -> numeric score (1-10 scale).

TF-IDF (uni+bigrams) + Ridge regression. The predicted score is compared
with the score the critic actually gave; the residual feeds the bias and
anomaly features downstream.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from rotten_review.config import RANDOM_STATE, SCORE_REGRESSOR, ScoreRegressorConfig


def build_pipeline(cfg: ScoreRegressorConfig = SCORE_REGRESSOR) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=cfg.ngram_range,
                    min_df=cfg.min_df,
                    max_features=cfg.max_features,
                ),
            ),
            ("ridge", Ridge(alpha=cfg.alpha, random_state=RANDOM_STATE)),
        ]
    )


def fit(
    texts: pd.Series, scores: pd.Series, cfg: ScoreRegressorConfig = SCORE_REGRESSOR
) -> Pipeline:
    """Fit on exactly the rows given — no internal splitting."""
    pipeline = build_pipeline(cfg)
    pipeline.fit(texts, scores)
    return pipeline


def evaluate(model: Pipeline, texts: pd.Series, scores: pd.Series) -> dict:
    """MAE, R2 and the share of predictions within one point of the true score."""
    y_pred = np.clip(model.predict(texts), 1.0, 10.0)
    return {
        "mae": float(mean_absolute_error(scores, y_pred)),
        "r2": float(r2_score(scores, y_pred)),
        "within_1_point": float((np.abs(scores - y_pred) <= 1.0).mean()),
        "n_eval": int(len(scores)),
    }


def train(
    texts: pd.Series,
    scores: pd.Series,
    cfg: ScoreRegressorConfig = SCORE_REGRESSOR,
) -> tuple[Pipeline, dict]:
    """Convenience wrapper: hold-out split, fit, evaluate."""
    x_train, x_test, y_train, y_test = train_test_split(
        texts, scores, test_size=cfg.test_size, random_state=RANDOM_STATE
    )
    pipeline = fit(x_train, y_train, cfg)
    metrics = evaluate(pipeline, x_test, y_test)
    metrics["n_train"] = int(len(x_train))
    return pipeline, metrics


def predict_score(model: Pipeline, text: str) -> float:
    from rotten_review.preprocessing import clean_text

    return float(np.clip(model.predict([clean_text(text)])[0], 1.0, 10.0))


def save(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: Path) -> Pipeline:
    return joblib.load(path)
