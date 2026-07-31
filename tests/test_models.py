import numpy as np
import pandas as pd
import pytest

from rotten_review.config import ANOMALY
from rotten_review.models import score_regressor, sentiment
from rotten_review.models.anomaly import AnomalyDetector, TfidfSvdEmbedder, build_embedder

RNG = np.random.default_rng(42)

POSITIVE = [
    "a sharp moving film with terrific performances",
    "beautifully shot and confidently directed",
    "smart writing makes this an easy recommendation",
    "an absolute delight from start to finish",
]
NEGATIVE = [
    "a tedious overlong mess of a movie",
    "flat characters and clumsy pacing sink it",
    "predictable and lifeless with awful dialogue",
    "a disappointing misfire without any footing",
]


def _corpus(n_per_class: int = 30) -> tuple[pd.Series, pd.Series]:
    texts, labels = [], []
    for i in range(n_per_class):
        texts.append(POSITIVE[i % len(POSITIVE)] + f" take {i}")
        labels.append(1)
        texts.append(NEGATIVE[i % len(NEGATIVE)] + f" take {i}")
        labels.append(-1)
    return pd.Series(texts), pd.Series(labels)


def test_sentiment_learns_separable_corpus():
    texts, labels = _corpus()
    model, metrics = sentiment.train(texts, labels)
    assert metrics["accuracy"] >= 0.9
    assert sentiment.predict_sentiment(model, "An absolute delight, terrific film!") == 1
    assert sentiment.predict_sentiment(model, "A tedious, lifeless mess.") == -1


def test_sentiment_roundtrip_save_load(tmp_path):
    texts, labels = _corpus(10)
    model, _ = sentiment.train(texts, labels)
    path = tmp_path / "m.joblib"
    sentiment.save(model, path)
    loaded = sentiment.load(path)
    assert sentiment.predict_sentiment(loaded, POSITIVE[0]) == 1


def test_score_regressor_outputs_valid_range():
    texts, labels = _corpus()
    scores = pd.Series([9.0 if lab == 1 else 2.0 for lab in labels]) + RNG.normal(
        0, 0.3, len(labels)
    )
    model, metrics = score_regressor.train(texts, scores)
    assert metrics["mae"] < 2.0
    pred = score_regressor.predict_score(model, "a tedious overlong mess")
    assert 1.0 <= pred <= 10.0


class FakeEmbedder:
    """Deterministic stand-in: fits a lookup on the training texts, reuses it after."""

    def __init__(self):
        self._space = None

    def fit_transform(self, texts):
        self._space = np.random.default_rng(7).normal(size=(len(texts), 10))
        return self._space

    def transform(self, texts):
        if self._space is None:
            raise RuntimeError("not fitted")
        return self._space[: len(texts)]


def _behavioural_frame(n: int) -> pd.DataFrame:
    df = pd.DataFrame(
        RNG.normal(size=(n, len(ANOMALY.behavioural_features))),
        columns=list(ANOMALY.behavioural_features),
    )
    # plant obvious outliers in the last 3 rows
    df.iloc[-3:] += 8.0
    return df


def test_anomaly_detector_flags_planted_outliers():
    n = 120
    features = _behavioural_frame(n)
    texts = [f"review {i}" for i in range(n)]
    detector = AnomalyDetector(embedder=FakeEmbedder())
    info = detector.fit(features, texts)
    assert info["n_features"] == len(ANOMALY.behavioural_features) + 10
    scored = detector.score(features, texts)
    assert set(scored.columns) == {"anomaly_score", "is_fake_review", "fraud_probability"}
    assert scored["fraud_probability"].between(0, 100).all()
    # planted outliers should score as more suspicious than the average review
    planted = scored["fraud_probability"].iloc[-3:].mean()
    rest = scored["fraud_probability"].iloc[:-3].mean()
    assert planted > rest


def test_anomaly_detector_rejects_missing_features():
    detector = AnomalyDetector(embedder=FakeEmbedder())
    with pytest.raises(ValueError, match="Missing behavioural features"):
        detector.fit(pd.DataFrame({"word_count": [1.0]}), ["x"])


def test_tfidf_svd_embedder_shape_and_determinism():
    texts = [POSITIVE[i % len(POSITIVE)] + f" {i}" for i in range(40)]
    first = TfidfSvdEmbedder().fit_transform(texts)
    second = TfidfSvdEmbedder().fit_transform(texts)
    assert first.shape == (40, ANOMALY.umap_components)
    assert np.allclose(first, second)


def test_embedder_transform_reuses_fitted_space():
    """transform() must project into the fitted space, not learn a new one."""
    fit_texts = [POSITIVE[i % len(POSITIVE)] + f" {i}" for i in range(40)]
    embedder = TfidfSvdEmbedder()
    fitted = embedder.fit_transform(fit_texts)
    # re-projecting the training text reproduces the training embedding
    assert np.allclose(fitted, embedder.transform(fit_texts))
    # unseen text lands in the same space, same width
    assert embedder.transform(["a completely unseen review"]).shape == (
        1,
        ANOMALY.umap_components,
    )


def test_embedder_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        TfidfSvdEmbedder().transform(["x"])


def test_build_embedder_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unknown embedder"):
        build_embedder("word2vec")


def test_detector_roundtrip_scores_new_reviews(tmp_path):
    """A saved detector must score unseen rows without refitting anything."""
    n = 80
    features = _behavioural_frame(n)
    texts = [POSITIVE[i % len(POSITIVE)] + f" take {i}" for i in range(n)]

    detector = AnomalyDetector()
    info, scored = detector.fit_score(features, texts)
    assert info["n_features"] == len(ANOMALY.behavioural_features) + ANOMALY.umap_components

    path = tmp_path / "detector.joblib"
    detector.save(path)
    reloaded = AnomalyDetector.load(path)

    # identical input through a round-tripped detector gives identical scores
    assert np.allclose(reloaded.score(features, texts)["anomaly_score"], scored["anomaly_score"])

    fresh_features = _behavioural_frame(5)
    fresh = reloaded.score(fresh_features, ["a brand new review"] * 5)
    assert len(fresh) == 5
    assert fresh["fraud_probability"].between(0, 100).all()
