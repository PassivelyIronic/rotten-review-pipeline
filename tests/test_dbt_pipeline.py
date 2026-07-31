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


def test_staging_removes_source_duplicates(warehouse):
    """The dump contains exact duplicate rows; staging must collapse them."""
    raw = _query(
        warehouse,
        """
        SELECT count(*) AS eligible
        FROM sample_rt_reviews
        WHERE top_critic AND review_content IS NOT NULL
          AND critic_name IS NOT NULL AND trim(critic_name) <> ''
        """,
    )["eligible"].iloc[0]
    staged = _query(warehouse, "SELECT count(*) AS n FROM stg_rt_reviews")["n"].iloc[0]

    assert staged < raw, "seeds contain planted duplicates that staging did not remove"
    ids = _query(warehouse, "SELECT review_id FROM stg_rt_reviews")["review_id"]
    assert ids.is_unique


def test_staging_drops_unattributed_reviews(warehouse):
    """Rows with no critic name would collapse into one pseudo-critic downstream."""
    nameless = _query(
        warehouse,
        "SELECT count(*) AS n FROM sample_rt_reviews WHERE trim(coalesce(critic_name, '')) = ''",
    )["n"].iloc[0]
    assert nameless > 0, "seed fixture no longer covers the unattributed-review case"

    staged = _query(
        warehouse,
        "SELECT count(*) AS n FROM stg_rt_reviews WHERE trim(coalesce(critic_name, '')) = ''",
    )["n"].iloc[0]
    assert staged == 0

    critics = _query(warehouse, "SELECT critic_name FROM dim_reviewers")["critic_name"]
    assert critics.notna().all()


def test_archive_backfill_does_not_register_as_burst(warehouse):
    """A back catalogue imported under one timestamp is not a publication burst.

    A RANGE window includes every row tied on the ordering value, so without
    special handling the most prolific critics score the highest burst in the
    dataset — the first full run rated Roger Ebert at 2,159 reviews in 7 days.
    """
    backfill = _query(
        warehouse,
        """
        SELECT count(*) AS rows_flagged,
               count(reviewer_reviews_last_7d) AS rows_with_burst
        FROM fct_review_features WHERE is_backfill_batch
        """,
    )
    assert backfill["rows_flagged"].iloc[0] > 0, "seed fixture no longer plants a backfill batch"
    assert backfill["rows_with_burst"].iloc[0] == 0, "backfill rows must not carry burst features"

    summary = _query(warehouse, "SELECT critic_name, max_reviews_in_7d FROM dim_reviewers")
    archive = summary[summary["critic_name"] == "Archive Critic"]["max_reviews_in_7d"]
    assert archive.empty or archive.isna().all() or archive.iloc[0] <= 8

    # the genuine bursty critic must survive the fix
    burst_bot = summary[summary["critic_name"] == "Burst Bot"]["max_reviews_in_7d"].iloc[0]
    assert burst_bot > 8


def test_sentinel_dates_are_nulled_not_trusted(warehouse):
    """Placeholder dates must not become timing features, but the text survives."""
    sentinel = _query(
        warehouse,
        "SELECT count(*) AS n FROM sample_rt_reviews WHERE review_date < '1900-01-01'",
    )["n"].iloc[0]
    assert sentinel > 0, "seed fixture no longer covers the sentinel-date case"

    staged = _query(
        warehouse,
        """
        SELECT count(*) AS kept, count(review_date) AS dated
        FROM stg_rt_reviews WHERE critic_name = 'Mordaunt Placeholder'
        """,
    )
    assert staged["kept"].iloc[0] == sentinel, "sentinel-dated reviews should be kept"
    assert staged["dated"].iloc[0] == 0, "sentinel dates should be NULL, not trusted"
