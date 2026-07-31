"""Model 3 — hybrid fake-review detector.

Isolation Forest over a concatenation of:
  * standardised behavioural features computed by the dbt marts
    (burst activity, z-scores vs the critic's own history, deviation from
    the tomatometer consensus, publisher-level deviation, composite bias
    score, sentiment/score residuals), and
  * a dense representation of the review text.

The text channel is pluggable and **stateful**: an embedder is fitted once,
stored on the detector, and reused for scoring. That distinction matters — an
embedder that refits on every call would place the reviews being scored in a
different latent space from the one the forest was trained in, which silently
invalidates the scores.

Two implementations ship:
  * `SbertUmapEmbedder` — all-MiniLM-L6-v2 + UMAP; needs the `embeddings` extra
  * `TfidfSvdEmbedder`  — TF-IDF + truncated SVD; scikit-learn only

`TfidfSvdEmbedder` is semantically weaker but keeps the test suite and the
analysis notebook runnable in a bare environment, and makes the contribution of
the text channel measurable against the behavioural one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from rotten_review.config import ANOMALY, RANDOM_STATE, AnomalyConfig


@runtime_checkable
class Embedder(Protocol):
    """Turns review text into a fixed-width dense matrix.

    `fit_transform` learns the representation; `transform` applies an already
    learned one to new text. Implementations must return the same number of
    columns from both.
    """

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray: ...

    def transform(self, texts: Sequence[str]) -> np.ndarray: ...


class TfidfSvdEmbedder:
    """TF-IDF followed by truncated SVD (latent semantic analysis).

    No transformer weights to download, deterministic, and fast enough that the
    whole pipeline stays runnable in CI.
    """

    def __init__(self, cfg: AnomalyConfig = ANOMALY) -> None:
        self.cfg = cfg
        self.width = cfg.umap_components
        self._pipeline = None

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline

        texts = list(texts)
        probe = TfidfVectorizer(min_df=2, max_features=20_000).fit_transform(texts)
        n_components = min(self.width, max(1, min(probe.shape) - 1))
        self._pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(min_df=2, max_features=20_000)),
                ("svd", TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)),
            ]
        )
        return self._pad(self._pipeline.fit_transform(texts))

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("Embedder is not fitted — call fit_transform first")
        return self._pad(self._pipeline.transform(list(texts)))

    def _pad(self, matrix: np.ndarray) -> np.ndarray:
        """Keep the output width stable even on corpora too small for full rank."""
        if matrix.shape[1] >= self.width:
            return matrix[:, : self.width]
        pad = np.zeros((matrix.shape[0], self.width - matrix.shape[1]))
        return np.hstack([matrix, pad])


class SbertUmapEmbedder:
    """Sentence-BERT embeddings reduced with UMAP. Requires the `embeddings` extra."""

    def __init__(self, cfg: AnomalyConfig = ANOMALY) -> None:
        self.cfg = cfg
        self._model = None
        self._reducer = None

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            self._model = SentenceTransformer(self.cfg.embedding_model)
        return self._model.encode(
            list(texts), batch_size=64, convert_to_numpy=True, normalize_embeddings=True
        )

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        from umap import UMAP

        vectors = self._encode(texts)
        self._reducer = UMAP(
            n_components=self.cfg.umap_components,
            n_neighbors=self.cfg.umap_neighbors,
            min_dist=self.cfg.umap_min_dist,
            metric="cosine",
            random_state=RANDOM_STATE,
        )
        return self._reducer.fit_transform(vectors)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self._reducer is None:
            raise RuntimeError("Embedder is not fitted — call fit_transform first")
        return self._reducer.transform(self._encode(texts))


def build_embedder(kind: str = "tfidf", cfg: AnomalyConfig = ANOMALY) -> Embedder:
    """Factory used by the CLI and the notebook. `kind` is 'tfidf' or 'sbert'."""
    if kind == "tfidf":
        return TfidfSvdEmbedder(cfg)
    if kind == "sbert":
        return SbertUmapEmbedder(cfg)
    raise ValueError(f"Unknown embedder {kind!r}; expected 'tfidf' or 'sbert'")


@dataclass
class AnomalyDetector:
    """Isolation Forest over behavioural features + embedded review text.

    The fitted embedder and scaler travel with the detector, so `save()` /
    `load()` round-trips a scorer that is usable on new reviews.
    """

    cfg: AnomalyConfig = ANOMALY
    embedder: Embedder = field(default_factory=TfidfSvdEmbedder)
    scaler: StandardScaler | None = None
    forest: IsolationForest | None = None
    score_min_: float | None = None
    score_max_: float | None = None

    def _behavioural(self, features: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.cfg.behavioural_features if c not in features.columns]
        if missing:
            raise ValueError(f"Missing behavioural features: {missing}")
        return features[list(self.cfg.behavioural_features)].fillna(0.0).to_numpy(dtype=float)

    def fit(self, features: pd.DataFrame, texts: Sequence[str]) -> dict:
        numeric = self._behavioural(features)
        self.scaler = StandardScaler().fit(numeric)
        matrix = np.hstack([self.scaler.transform(numeric), self.embedder.fit_transform(texts)])

        self.forest = IsolationForest(
            contamination=self.cfg.contamination,
            n_estimators=self.cfg.n_estimators,
            max_samples=min(self.cfg.max_samples, len(matrix)),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.forest.fit(matrix)

        scores = self.forest.decision_function(matrix)
        self.score_min_, self.score_max_ = float(scores.min()), float(scores.max())
        flagged = int((self.forest.predict(matrix) == -1).sum())
        return {
            "n_reviews": int(len(matrix)),
            "n_features": int(matrix.shape[1]),
            "flagged": flagged,
            "flagged_pct": float(flagged / len(matrix) * 100),
        }

    def score(self, features: pd.DataFrame, texts: Sequence[str]) -> pd.DataFrame:
        """Anomaly score, binary flag and a 0-100 fraud probability.

        `fraud_probability` is rescaled against the range seen at fit time, so
        values from different fits are not comparable, and new reviews can fall
        outside the original range (they are clipped).
        """
        if self.forest is None or self.scaler is None:
            raise RuntimeError("Detector is not fitted")
        matrix = np.hstack(
            [self.scaler.transform(self._behavioural(features)), self.embedder.transform(texts)]
        )
        raw = self.forest.decision_function(matrix)
        span = (self.score_max_ - self.score_min_) or 1.0
        return pd.DataFrame(
            {
                "anomaly_score": raw,
                "is_fake_review": (self.forest.predict(matrix) == -1).astype(int),
                "fraud_probability": np.clip(100 * (1 - (raw - self.score_min_) / span), 0, 100),
            },
            index=features.index,
        )

    def fit_score(self, features: pd.DataFrame, texts: Sequence[str]) -> tuple[dict, pd.DataFrame]:
        """Fit and score the same rows in one pass, reusing the fitted embedder."""
        info = self.fit(features, texts)
        return info, self.score(features, texts)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> AnomalyDetector:
        return joblib.load(path)
