"""End-to-end check of the dbt layer: build the project against the synthetic
seeds into a temporary DuckDB file, then validate mart semantics and run the
Python feature bridge on top of it."""

import os
import shutil
import subprocess
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from rotten_review import features
from rotten_review.models import score_regressor, sentiment

REPO_ROOT = Path(__file__).resolve().parents[1]
DBT_DIR = REPO_ROOT / "dbt"

pytestmark = pytest.mark.skipif(shutil.which("dbt") is None, reason="dbt not installed")


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory) -> Path:
    db_path = tmp_path_factory.mktemp("wh") / "test_warehouse.duckdb"
    env = {**os.environ, "ROTTEN_REVIEW_DB": str(db_path)}
    result = subprocess.run(
        ["dbt", "build", "--target", "test", "--profiles-dir", str(DBT_DIR)],
        cwd=DBT_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return db_path


def _query(db_path: Path, sql: str) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def test_staging_filters_and_parses(warehouse):
    df = _query(warehouse, "SELECT * FROM stg_rt_reviews")
    assert (df["review_id"].value_counts() == 1).all()
    # the non-top-critic seed row must be gone
    assert "Random User" not in set(df["critic_name"])
    # mixed formats ("4/5", "B+", "85%", "9") all parse into 1-10
    parsed = df["score_10"].dropna()
    assert len(parsed) > 0
    assert parsed.between(1, 10).all()
    # the unparseable "two thumbs down" row survives with a NULL score
    assert df["score_10"].isna().sum() >= 1


def test_burst_detection_flags_bursty_critic(warehouse):
    df = _query(
        warehouse,
        """
        SELECT r.critic_name, max(b.reviewer_reviews_last_7d) AS max_7d
        FROM stg_rt_reviews r JOIN int_burst_activity b USING (review_id)
        GROUP BY 1
        """,
    )
    by_critic = df.set_index("critic_name")["max_7d"]
    assert by_critic["Burst Bot"] > 7
    assert by_critic.drop("Burst Bot").max() < by_critic["Burst Bot"]


def test_feature_bridge_on_mart(warehouse):
    mart = features.read_mart(warehouse_path=warehouse)
    labelled = mart.dropna(subset=["score_10", "review_type_encoded"])
    texts = labelled["review_text"]
    sent_model, _ = sentiment.train(texts, labelled["review_type_encoded"].map({1: 1, 0: -1}))
    score_model, _ = score_regressor.train(texts, labelled["score_10"].astype(float))

    enriched = features.add_model_features(labelled, sent_model, score_model)
    for col in ["predicted_sentiment", "predicted_score", "score_residual", "bias_score"]:
        assert col in enriched.columns
    assert enriched["bias_score"].between(1, 10).all()
    agreement = features.sentiment_agreement(enriched)
    assert 0.0 <= agreement <= 1.0
