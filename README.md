# Rotten Review Pipeline

**Review integrity analytics for movie reviews: sentiment, text-implied scores and fake-review detection — built as a tested dbt + Python pipeline.**

[![CI](https://github.com/PassivelyIronic/rotten-review-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/PassivelyIronic/rotten-review-pipeline/actions/workflows/ci.yml)

This project answers three questions about ~1M Rotten Tomatoes critic reviews:

1. **What does the review actually say?** — sentiment classifier (Model 1)
2. **What score does the text imply?** — text → 1-10 score regressor (Model 2)
3. **Which reviews look fake, paid or biased?** — hybrid anomaly detector combining behavioural signals with text embeddings (Model 3)

It started as a group university project living in a single 87-cell Colab notebook. This repository is my rework of that notebook into an engineered pipeline: SQL feature engineering in **dbt** (with data tests), an installable **Python package** with a CLI, **pytest** coverage including an end-to-end dbt integration test, and **CI** that runs everything — including the analysis notebook — on synthetic seed data.

The line-by-line review of the original is written up in **[AUDIT.md](AUDIT.md)**: a target leak into the score model, a sentiment evaluation contaminated by two overlapping copies of the same dataset, and a bilingual section whose behavioural features were randomly generated. Each finding names the cell it came from and what the rework did about it.

## Architecture

```
 Kaggle datasets                DuckDB warehouse (dbt)                    Python (rotten_review)
┌──────────────────┐   ingest   ┌──────────────────────────────┐        ┌──────────────────────────┐
│ IMDB 50k reviews │──────────▶│ staging                       │        │ Model 1: sentiment        │
│ RT reviews (~1M) │           │  · score parsing (3/5, B+, %) │  mart  │  CountVectorizer + LogReg │
│ RT movies        │           │  · top-critic filter          │───────▶│ Model 2: text → score     │
└──────────────────┘           │ intermediate                  │        │  TF-IDF + Ridge           │
                               │  · 7-day burst windows        │        │ Model 3: fake detection   │
                               │  · reviewer/publisher stats   │        │  IsolationForest over     │
                               │ marts                         │        │  behaviour + SBERT/UMAP   │
                               │  · fct_review_features        │        │  embeddings               │
                               │  · dim_reviewers              │        └────────────┬─────────────┘
                               └──────────────────────────────┘                      │
                                     dbt tests: unique, not_null,           Gradio demo + parquet
                                     accepted_values, range checks          anomaly scores
```

The split follows one rule: **set-based feature engineering lives in SQL, model-dependent features live in Python.** Reviewer z-scores, consensus deviations and burst detection are dbt models with data tests; sentiment predictions, score residuals and the composite `bias_score` (which needs Model 2's output) are computed in `rotten_review.features` on top of the mart.

## What changed vs the original notebook

| Notebook | This repo |
|---|---|
| 87 cells, single Colab file, state shared implicitly between cells | Installable package + dbt project with explicit interfaces |
| O(n²) pandas `groupby().apply()` loop for 7-day burst detection | Single DuckDB window function (`RANGE BETWEEN INTERVAL 7 DAY PRECEDING`) |
| Score parsing scattered across cells | One tested dbt macro (`score_to_100`) handling fractions, percentages, numerics and letter grades |
| No tests | 11 pytest tests + 15 dbt data tests + 2 singular SQL tests, all in CI |
| Embeddings hard-wired to SentenceTransformer | Pluggable stateful embedder — SBERT in production, TF-IDF+SVD when weights are unavailable, fitted once and persisted with the detector so a saved model can score new reviews |
| Sentiment model evaluated on rows it had trained on | Deterministic `train`/`holdout` fold assigned in dbt and hashed on `review_id`, so the split survives every rebuild |
| Digits kept in cleaned text, leaking `"3/5"` into the score model's features | Digits removed; the notebook measures the effect on exactly the rows it applied to |
| Duplicate and unattributed source rows carried straight into reviewer statistics | Stripped in staging, with the row cost printed as a funnel in the notebook and pinned by tests |
| Not reproducible without manual cell ordering | `make ingest` → `make dbt` → `make train` |

Writing the tests paid off immediately: they caught that DuckDB's `greatest`/`least` skip NULLs, which silently turned unparseable review scores into `1` instead of keeping them NULL. The fix and a regression test are in `stg_rt_reviews`.

## Quickstart

```bash
git clone https://github.com/PassivelyIronic/rotten-review-pipeline.git
cd rotten-review-pipeline
make install          # pip install -e ".[ingest,dbt,notebook,dev]"
                      # using uv? set UV=1 once and every target routes through it

# verify everything without downloading any data (what CI runs):
make lint test dbt-ci

# read the analysis (runs on generated seeds, no credentials needed):
make notebook

# full pipeline on real data (Kaggle download ~1 GB, needs a Kaggle API token):
make ingest           # Kaggle → parquet → DuckDB warehouse
make dbt              # build staging/intermediate/mart models + data tests
make train            # train Models 1-3, write artifacts/ and reports/metrics.json
make app              # Gradio demo on the trained artifacts
```

## Models

| # | Task | Approach | Training data |
|---|---|---|---|
| 1 | Sentiment (positive/negative) | CountVectorizer (uni+bigrams, 10k features) + LogisticRegression, balanced classes | IMDB 50k + RT top-critic reviews (`stg_sentiment_corpus`) |
| 2 | Text → score (1-10) | TF-IDF (uni+bigrams, 30k features) + Ridge | RT reviews with parseable scores |
| 3 | Fake/biased review detection | IsolationForest (contamination 5%) over 12 standardised behavioural features + all-MiniLM-L6-v2 embeddings reduced to 10 dims with UMAP | `fct_review_features` mart |

Model 3 is unsupervised — there is no ground-truth "fake" label in the data. Its behavioural inputs include burst activity (reviewer volume in trailing 7-day windows), z-score of the given score against the critic's own history, deviation from the tomatometer consensus, deviation from the publisher's typical scores, and the gap between the declared score and the score Model 2 infers from the text alone.

### Results

The table below is generated from `reports/metrics.json` by `python scripts/render_results.py`, so no number in this README is hand-typed. The original notebook was archived without saved outputs, so no historical figures are claimed; run `make train && make results` to fill it in.

<!-- results:start -->

| Model | Metric | Value | Evaluated on |
|---|---|---|---|
| Model 1 — sentiment | accuracy | 0.774 | 49594 holdout reviews |
| Model 2 — score from text | MAE | 1.364 | 31649 holdout reviews |
| Model 2 — score from text | R² | 0.336 | 31649 holdout reviews |
| Model 2 — score from text | within ±1 point | 0.436 | 31649 holdout reviews |
| Model 3 — anomaly detection | reviews flagged | 7955 (5.0%) | 159090 reviews, 22 features |

Model 3's flag rate follows directly from the `contamination` parameter and is not a measurement — see [AUDIT.md](AUDIT.md) and the notebook's limitations section.

<!-- results:end -->

## Repository layout

```
dbt/                    dbt-duckdb project (staging → intermediate → marts + tests)
  seeds/                generated sample data used by CI, tests and notebook sample mode
src/rotten_review/      package: config, preprocessing, models/, features, cli, app
tests/                  pytest, incl. end-to-end dbt build on seeds
notebooks/              01_review_integrity_analysis.ipynb — the analysis layer
  original/             the untouched original notebook, kept for provenance
scripts/                seed generator, README results renderer
AUDIT.md                what the review of the original found, cell by cell
.github/workflows/      CI: ruff + pytest + dbt build + notebook execution
```

## The analysis notebook

[`notebooks/01_review_integrity_analysis.ipynb`](notebooks/01_review_integrity_analysis.ipynb)
is the interpretation layer: data quality, score-parsing coverage, review timing, burst
behaviour, model evaluation on the dbt folds, and the anomaly profile — with the limitations
stated rather than buried.

It runs in one of two modes. With no warehouse present it builds the generated seeds into a
temporary DuckDB and runs against those, which takes under a minute and needs no credentials;
that is the mode CI executes on every push, so a broken cell fails the build. Point it at a
real `data/warehouse.duckdb` and the same cells produce the real numbers.

**The outputs committed here are from sample mode on generated data.** They show that the
pipeline runs and what the analysis looks like — they are not claims about real critics.

## Design notes

- **DuckDB over Snowflake/Spark** — the dataset (~1M rows) fits comfortably in a single-node columnar engine; the dbt project structure is warehouse-portable, so swapping the adapter is a config change, not a rewrite.
- **CI runs on generated seeds** — `scripts/generate_seeds.py` emits ~590 reviews across 21 critics and 40 films, with the edge cases planted deliberately: mixed score formats, a bursty critic, a non-top-critic row that staging must drop, unparseable scores, and reviews that restate their score in prose. Every push exercises the full dbt DAG, the Python bridge and the notebook in seconds, without Kaggle credentials.
- **No unverified claims** — hyperparameters are documented as carried over from the notebook (`config.py`); metrics only appear once generated by the pipeline itself.
- **Cross-platform by test, not by hope** — every text file is opened with an explicit encoding, and `test_no_text_io_without_explicit_encoding` walks the AST of the whole package to keep it that way. Python's implicit fallback is cp1252 on a default Windows install, so an em dash in this README is enough to crash a script that reads it. That test fails on any platform, which is the point.

## Provenance

This project originated as a group course project (Exploratory Data Analysis) at Cracow University of Technology, developed together with [Nachos-mic](https://github.com/Nachos-mic) — the original notebook lives in [Nachos-mic/Rotten_Review_Model](https://github.com/Nachos-mic/Rotten_Review_Model), where I was a collaborator and designed most of the analytical logic (feature design, bias scoring, the three-model structure). The engineering rework in this repository — dbt layer, package structure, tests, CI — is my own.

## License

MIT
