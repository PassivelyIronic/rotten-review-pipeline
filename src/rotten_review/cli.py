"""Command-line entry point.

    rotten-review ingest             # download Kaggle data, load DuckDB warehouse
    rotten-review train-sentiment    # Model 1 (EN), writes artifacts + metrics
    rotten-review train-score        # Model 2, writes artifacts + metrics
    rotten-review train-anomaly      # Model 3 on the dbt feature mart
                                     #   --embedder tfidf (default) | sbert
    rotten-review evaluate           # consolidated reports/metrics.json

dbt transformations run separately: `make dbt` (or `dbt build` in dbt/).
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from rotten_review import config
from rotten_review.preprocessing import clean_text


def _write_metrics(name: str, metrics: dict) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / "metrics.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing[name] = metrics
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"[{name}] {json.dumps({k: v for k, v in metrics.items() if k != 'report'})}")


def cmd_ingest(_: argparse.Namespace) -> None:
    from rotten_review import ingest

    print("Downloading IMDB 50k...")
    ingest.download_imdb()
    print("Downloading Rotten Tomatoes (Clapper)...")
    ingest.download_rotten_tomatoes()
    print("Loading DuckDB warehouse...")
    ingest.load_warehouse()
    print(f"Warehouse ready: {config.WAREHOUSE_PATH}")


def cmd_train_sentiment(_: argparse.Namespace) -> None:
    import duckdb

    from rotten_review.models import sentiment

    con = duckdb.connect(str(config.WAREHOUSE_PATH), read_only=True)
    df = con.execute("SELECT text, label, split FROM stg_sentiment_corpus").df()
    con.close()
    df["text"] = df["text"].map(clean_text)
    train_df = df[df["split"] == "train"]
    holdout_df = df[df["split"] == "holdout"]

    model = sentiment.fit(train_df["text"], train_df["label"])
    metrics = sentiment.evaluate(model, holdout_df["text"], holdout_df["label"])
    metrics["n_train"] = int(len(train_df))
    sentiment.save(model, config.MODELS_DIR / "sentiment_en.joblib")
    _write_metrics("sentiment_en", metrics)


def cmd_train_score(_: argparse.Namespace) -> None:
    import duckdb

    from rotten_review.models import score_regressor

    con = duckdb.connect(str(config.WAREHOUSE_PATH), read_only=True)
    df = con.execute(
        "SELECT review_text, score_10, split FROM fct_review_features WHERE score_10 IS NOT NULL"
    ).df()
    con.close()
    df["text"] = df["review_text"].map(clean_text)
    train_df = df[df["split"] == "train"]
    holdout_df = df[df["split"] == "holdout"]

    model = score_regressor.fit(train_df["text"], train_df["score_10"].astype(float))
    metrics = score_regressor.evaluate(
        model, holdout_df["text"], holdout_df["score_10"].astype(float)
    )
    metrics["n_train"] = int(len(train_df))
    score_regressor.save(model, config.MODELS_DIR / "score_regressor.joblib")
    _write_metrics("score_regressor", metrics)


def cmd_train_anomaly(args: argparse.Namespace) -> None:
    from rotten_review import features
    from rotten_review.models import score_regressor, sentiment
    from rotten_review.models.anomaly import AnomalyDetector, build_embedder

    sentiment_model = sentiment.load(config.MODELS_DIR / "sentiment_en.joblib")
    score_model = score_regressor.load(config.MODELS_DIR / "score_regressor.joblib")
    df = features.read_mart()
    df = features.add_model_features(df, sentiment_model, score_model)
    df = df.dropna(subset=["score_10"])
    df = df[df["text_cleaned"].str.strip() != ""]

    detector = AnomalyDetector(embedder=build_embedder(args.embedder))
    # one pass: the embedder is fitted once and reused for scoring, so the
    # scored rows live in the same latent space the forest was trained in
    metrics, scored = detector.fit_score(df, df["text_cleaned"].tolist())
    detector.save(config.MODELS_DIR / "anomaly_detector.joblib")

    out = pd.concat(
        [df[["review_id"]].reset_index(drop=True), scored.reset_index(drop=True)], axis=1
    )
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.PROCESSED_DIR / "anomaly_scores.parquet", index=False)

    metrics["embedder"] = args.embedder
    metrics["sentiment_vs_label_agreement"] = features.sentiment_agreement(df)
    _write_metrics("anomaly", metrics)


def cmd_evaluate(_: argparse.Namespace) -> None:
    path = config.REPORTS_DIR / "metrics.json"
    if not path.exists():
        raise SystemExit("No reports/metrics.json yet — run the train commands first.")
    print(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(prog="rotten-review")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in [
        ("ingest", cmd_ingest),
        ("train-sentiment", cmd_train_sentiment),
        ("train-score", cmd_train_score),
        ("evaluate", cmd_evaluate),
    ]:
        sub.add_parser(name).set_defaults(func=fn)

    anomaly_parser = sub.add_parser("train-anomaly")
    anomaly_parser.add_argument(
        "--embedder",
        choices=["tfidf", "sbert"],
        default="tfidf",
        help="text representation for Model 3; 'sbert' needs the `embeddings` extra",
    )
    anomaly_parser.set_defaults(func=cmd_train_anomaly)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
