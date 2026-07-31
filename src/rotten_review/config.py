"""Central configuration: paths and model hyperparameters.

Hyperparameter values are carried over unchanged from the original
course-project notebook (notebooks/original/) so results stay comparable.
"""

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
WAREHOUSE_PATH = DATA_DIR / "warehouse.duckdb"
MODELS_DIR = REPO_ROOT / "artifacts"
REPORTS_DIR = REPO_ROOT / "reports"

RANDOM_STATE = 42


@dataclass(frozen=True)
class SentimentConfig:
    """Model 1: bag-of-words logistic regression (EN: IMDB + Rotten Tomatoes)."""

    max_features: int = 10_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    C: float = 1.0
    max_iter: int = 1000
    test_size: float = 0.2


@dataclass(frozen=True)
class ScoreRegressorConfig:
    """Model 2: TF-IDF + Ridge, text -> score on a 1-10 scale."""

    max_features: int = 30_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    alpha: float = 1.0
    test_size: float = 0.2


@dataclass(frozen=True)
class AnomalyConfig:
    """Model 3: Isolation Forest on behavioural features + text embeddings."""

    contamination: float = 0.05
    n_estimators: int = 200
    max_samples: int = 256
    embedding_model: str = "all-MiniLM-L6-v2"
    umap_components: int = 10
    umap_neighbors: int = 15
    umap_min_dist: float = 0.1
    behavioural_features: tuple[str, ...] = field(
        default=(
            "reviewer_reviews_last_7d",
            "reviewer_burst_ratio_7d",
            "positive_ratio",
            "critic_std_score_10",
            "days_since_release",
            "word_count",
            "score_residual",
            "z_score_critic",
            "sentiment_diff",
            "consensus_diff",
            "publisher_z_score",
            "bias_score",
        )
    )


SENTIMENT = SentimentConfig()
SCORE_REGRESSOR = ScoreRegressorConfig()
ANOMALY = AnomalyConfig()

# Weights for the composite bias score (z_score, sentiment, consensus, publisher).
BIAS_WEIGHTS = (0.4, 0.2, 0.3, 0.1)
