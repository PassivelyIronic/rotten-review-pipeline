"""Model 1 — sentiment classifier (positive=1 / negative=-1).

Bag-of-words (CountVectorizer, uni+bigrams) + LogisticRegression with
balanced class weights, trained on IMDB 50k merged with Rotten Tomatoes
top-critic reviews. A separate instance is trained per language (EN/PL).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from rotten_review.config import RANDOM_STATE, SENTIMENT, SentimentConfig


def build_pipeline(cfg: SentimentConfig = SENTIMENT) -> Pipeline:
    return Pipeline(
        [
            (
                "vectorizer",
                CountVectorizer(
                    max_features=cfg.max_features,
                    ngram_range=cfg.ngram_range,
                    min_df=cfg.min_df,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=cfg.C,
                    max_iter=cfg.max_iter,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def fit(texts: pd.Series, labels: pd.Series, cfg: SentimentConfig = SENTIMENT) -> Pipeline:
    """Fit on exactly the rows given — no internal splitting.

    Use this when the split is decided upstream (the `split` column produced by
    the dbt staging model), so the reviews scored later are never trained on.
    """
    pipeline = build_pipeline(cfg)
    pipeline.fit(texts, labels)
    return pipeline


def evaluate(model: Pipeline, texts: pd.Series, labels: pd.Series) -> dict:
    """Accuracy and per-class report on a held-out set."""
    y_pred = model.predict(texts)
    return {
        "accuracy": float(accuracy_score(labels, y_pred)),
        "report": classification_report(labels, y_pred, output_dict=True, zero_division=0),
        "n_eval": int(len(labels)),
    }


def train(
    texts: pd.Series,
    labels: pd.Series,
    cfg: SentimentConfig = SENTIMENT,
) -> tuple[Pipeline, dict]:
    """Convenience wrapper: stratified hold-out split, fit, evaluate."""
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=cfg.test_size,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    pipeline = fit(x_train, y_train, cfg)
    metrics = evaluate(pipeline, x_test, y_test)
    metrics["n_train"] = int(len(x_train))
    return pipeline, metrics


def predict_sentiment(model: Pipeline, text: str) -> int:
    from rotten_review.preprocessing import clean_text

    return int(model.predict([clean_text(text)])[0])


def save(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: Path) -> Pipeline:
    return joblib.load(path)
