"""Data ingestion: Kaggle datasets -> data/raw parquet -> DuckDB warehouse.

Sources (public, downloaded via kagglehub — requires the `ingest` extra):
  * lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
      50k labelled reviews, used only to widen Model 1's training corpus.
  * stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset
      critic reviews + movie metadata; the backbone of Models 1-3.

The original notebook additionally pulled a second Rotten Tomatoes dump
(`andrezaza/clapper-...`) to train the sentiment model while scoring reviews
from the dataset above. Both dumps cover the same underlying reviews, so
training on one and evaluating on the other leaked test rows into training.
This pipeline uses a single RT source and enforces the split explicitly.

After ingestion, `load_warehouse()` registers the raw parquet files in a
DuckDB database that the dbt project builds on top of.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from rotten_review.config import RAW_DIR, WAREHOUSE_PATH

IMDB_DATASET = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
RT_DATASET = "stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset"
RT_REVIEWS_CSV = "rotten_tomatoes_critic_reviews.csv"
RT_MOVIES_CSV = "rotten_tomatoes_movies.csv"


def download_imdb(out_dir: Path = RAW_DIR) -> Path:
    import kagglehub

    dataset_path = Path(kagglehub.dataset_download(IMDB_DATASET))
    df = pd.read_csv(dataset_path / "IMDB Dataset.csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "imdb_reviews.parquet"
    df.to_parquet(out, index=False)
    return out


def download_rotten_tomatoes(out_dir: Path = RAW_DIR) -> tuple[Path, Path]:
    """Download RT critic reviews + movie metadata and store them as parquet."""
    import kagglehub

    dataset_path = Path(kagglehub.dataset_download(RT_DATASET))
    reviews_csv = dataset_path / RT_REVIEWS_CSV
    movies_csv = dataset_path / RT_MOVIES_CSV
    for path in (reviews_csv, movies_csv):
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} not found in {dataset_path}; the Kaggle dataset layout changed"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    reviews_out = out_dir / "rt_reviews.parquet"
    movies_out = out_dir / "rt_movies.parquet"
    pd.read_csv(reviews_csv).to_parquet(reviews_out, index=False)
    pd.read_csv(movies_csv).to_parquet(movies_out, index=False)
    return reviews_out, movies_out


def load_warehouse(raw_dir: Path = RAW_DIR, warehouse_path: Path = WAREHOUSE_PATH) -> None:
    """Create raw_* tables in the DuckDB warehouse from parquet files."""
    warehouse_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(warehouse_path))
    try:
        for name, filename in [
            ("raw_imdb_reviews", "imdb_reviews.parquet"),
            ("raw_rt_reviews", "rt_reviews.parquet"),
            ("raw_rt_movies", "rt_movies.parquet"),
        ]:
            path = raw_dir / filename
            if path.exists():
                con.execute(
                    f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_parquet(?)",
                    [str(path)],
                )
    finally:
        con.close()
