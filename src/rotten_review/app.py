"""Gradio demo: paste a review, get sentiment, the score its text implies, and —
if a trained detector is present — an anomaly reading.

Run after training: `python -m rotten_review.app` (requires the `app` extra).

Model 3 needs behavioural context that a single pasted review does not carry
(how often the critic publishes, how they usually score, what the consensus is).
The inputs below let you supply that context; anything left at its default is
filled with the neutral value, and the reading is labelled accordingly.
"""

from __future__ import annotations

import pandas as pd

from rotten_review import config
from rotten_review.config import ANOMALY, BIAS_WEIGHTS
from rotten_review.models import score_regressor, sentiment
from rotten_review.models.anomaly import AnomalyDetector
from rotten_review.preprocessing import basic_text_stats, clean_text

DETECTOR_PATH = config.MODELS_DIR / "anomaly_detector.joblib"


def _behavioural_row(
    context: dict[str, float], word_count: int, score_residual: float
) -> pd.DataFrame:
    """Assemble one row with every feature the detector expects."""
    z_score = context["z_score_critic"]
    consensus_diff = context["consensus_diff"]
    publisher_z = context["publisher_z_score"]
    sentiment_diff = abs(score_residual)
    bias = (
        BIAS_WEIGHTS[0] * z_score
        + BIAS_WEIGHTS[1] * sentiment_diff
        + BIAS_WEIGHTS[2] * consensus_diff
        + BIAS_WEIGHTS[3] * publisher_z
    )
    values = {
        "reviewer_reviews_last_7d": context["reviews_last_7d"],
        "reviewer_burst_ratio_7d": context["burst_ratio"],
        "positive_ratio": context["positive_ratio"],
        "critic_std_score_10": context["critic_std"],
        "days_since_release": context["days_since_release"],
        "word_count": word_count,
        "score_residual": score_residual,
        "z_score_critic": z_score,
        "sentiment_diff": sentiment_diff,
        "consensus_diff": consensus_diff,
        "publisher_z_score": publisher_z,
        "bias_score": bias,
    }
    return pd.DataFrame(
        [[values[c] for c in ANOMALY.behavioural_features]],
        columns=list(ANOMALY.behavioural_features),
    )


def build_interface():
    import gradio as gr

    sentiment_model = sentiment.load(config.MODELS_DIR / "sentiment_en.joblib")
    score_model = score_regressor.load(config.MODELS_DIR / "score_regressor.joblib")
    detector = AnomalyDetector.load(DETECTOR_PATH) if DETECTOR_PATH.exists() else None

    def analyse(review_text, declared_score, reviews_last_7d, days_since_release, positive_ratio):
        cleaned = clean_text(review_text)
        if not cleaned:
            return "—", "—", "—"

        label = sentiment.predict_sentiment(sentiment_model, review_text)
        implied = score_regressor.predict_score(score_model, review_text)
        stats = basic_text_stats(cleaned)

        sentiment_txt = "Positive" if label == 1 else "Negative"
        score_txt = (
            f"{implied:.1f} / 10  ·  {stats['word_count']} words, "
            f"lexical diversity {stats['lexical_diversity']:.2f}"
        )

        residual = 0.0
        if declared_score:
            residual = float(declared_score) - implied
            verdict = "large text/score mismatch" if abs(residual) >= 3 else "consistent"
            score_txt += f"\ndeclared {declared_score:.1f} → residual {residual:+.1f} ({verdict})"

        if detector is None:
            return sentiment_txt, score_txt, "No trained detector found — run `make train`."

        context = {
            "reviews_last_7d": reviews_last_7d or 0,
            "burst_ratio": 0.0,
            "positive_ratio": positive_ratio if positive_ratio is not None else 0.5,
            "critic_std": 1.5,
            "days_since_release": days_since_release if days_since_release is not None else 30,
            "z_score_critic": 0.0,
            "consensus_diff": abs(residual),
            "publisher_z_score": 0.0,
        }
        row = _behavioural_row(context, stats["word_count"], residual)
        result = detector.score(row, [cleaned]).iloc[0]
        anomaly_txt = (
            f"{result['fraud_probability']:.0f} / 100 unusualness"
            f"  ·  {'flagged' if result['is_fake_review'] else 'not flagged'}\n"
            "Unsupervised ranking against the training distribution — not a fraud verdict, "
            "and unfilled context fields are neutral defaults."
        )
        return sentiment_txt, score_txt, anomaly_txt

    return gr.Interface(
        fn=analyse,
        inputs=[
            gr.Textbox(lines=6, label="Review text"),
            gr.Number(label="Declared score (1-10, optional)", value=None),
            gr.Number(label="Critic's reviews in the last 7 days", value=1),
            gr.Number(label="Days since the film's release", value=30),
            gr.Slider(0, 1, value=0.5, label="Critic's historical positive ratio"),
        ],
        outputs=[
            gr.Textbox(label="Model 1 — sentiment"),
            gr.Textbox(label="Model 2 — score implied by the text"),
            gr.Textbox(label="Model 3 — anomaly reading"),
        ],
        title="Rotten Review — review integrity check",
        description=(
            "Model 1 classifies sentiment, Model 2 predicts the score the text implies, and "
            "Model 3 ranks the review against the behavioural distribution it was fitted on. "
            "A large gap between declared and implied score is one of the signals it uses."
        ),
    )


if __name__ == "__main__":
    build_interface().launch()
