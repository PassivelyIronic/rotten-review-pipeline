# Audit of the original notebook

`notebooks/original/EDT_Rotten_Review_Model.ipynb` is the group course project this repository
was built from, kept unmodified. This file records what a line-by-line review of its 87 cells
turned up and what the rework did about each item. Cell numbers are zero-indexed positions in
the original notebook.

Every finding below was verified by reading the cell in question — nothing here is inferred
from output, because the notebook was saved with outputs on only 1 of its 87 cells.

Three rows are marked *introduced during the rework*: bugs this rewrite created, two caught by
its own tests and one reported from a Windows run. They are listed for the same reason as the rest — a review that only finds
faults in someone else's code is not a review.

## Correctness

| # | Finding | Where | Resolution |
|---|---|---|---|
| 1 | `model` is bound to the sentiment `LogisticRegression`, then rebound to a `SentenceTransformer` later in the same global namespace. `predict_sentiment` closes over that global, so any call after the Model 3 section silently uses the wrong object. | cells 13, 15, 69 | Models are objects with explicit lifetimes in `rotten_review.models`; no shared globals. |
| 2 | The training cell prints `LogisticRegression uses class_weight='balanced'`, but the estimator is constructed without `class_weight`. The stated behaviour never happened. | cell 13 | `build_pipeline` sets `class_weight="balanced"` for real. |
| 3 | A PyTorch `Dataset`, `DataLoader` and `WeightedRandomSampler` are constructed for both Model 1 and Model 2 and never used — the estimators are scikit-learn. The notebook's own print admits it. `torch` is imported solely for this. | cells 4, 13, 50 | Removed. `torch` is no longer a dependency. |
| 4 | `CHAT_WORDS` defines the key `"B4N"` twice. | cell 29 | Chat-word expansion dropped — it targets social-media slang that does not occur in professional critic copy. |
| 5 | `predict_sentiment_pl` is defined twice with different bodies; the second silently wins. | cells 75, 77 | Polish section removed (see Reproducibility #3). |
| 6 | Burst detection iterates every row of every critic's group and rescans the group for each one — quadratic in a critic's review count. | cell 54 | One SQL window function (`RANGE BETWEEN INTERVAL 7 DAY PRECEDING`) in `int_burst_activity`. |
| 7 | `greatest`/`least` ignore NULLs in DuckDB, so an unparseable score would clamp to 1 instead of staying NULL. | introduced during the rework | Caught by `test_staging_filters_and_parses`; fixed with an explicit NULL guard in `stg_rt_reviews`. |
| 9 | Text files were read and written without an explicit encoding, so any script touching this README crashed on a default Windows install (cp1252) the moment it hit an em dash or `R²`. Reported from a real Windows run. | introduced during the rework | Every text I/O call now passes `encoding="utf-8"`, and `test_no_text_io_without_explicit_encoding` walks the AST of `src/`, `scripts/` and `tests/` to keep new code honest. |
| 8 | The first version of this rework passed the embedder to `AnomalyDetector` as a plain callable that refit itself on every call, and the CLI handed `fit()` and `score()` two separate instances. Scoring therefore projected reviews into a different latent space from the one the forest was trained in — and a saved detector could not score new reviews at all. | introduced during the rework | Embedders are now stateful (`fit_transform` / `transform`), stored on the detector and persisted with it. `test_detector_roundtrip_scores_new_reviews` and `test_embedder_transform_reuses_fitted_space` pin the behaviour; the notebook scores unseen rows through the fitted detector as a live check. |

## Methodology

| # | Finding | Where | Resolution |
|---|---|---|---|
| 1 | **Target leakage into Model 2.** The cleaner strips punctuation but keeps digits, so a review reading *"Verdict: 3/5"* contributes the token `35`. The model can read the answer off its own input. | cells 29, 50 | `clean_text` removes digits. Section 4 of the analysis notebook measures the effect, isolated to the rows it can act on. |
| 2 | **Contaminated sentiment validation.** Model 1 trains on the Clapper RT dump; its agreement with `review_type` is then measured on the stefanoleone992 dump. Both are dumps of the same underlying Rotten Tomatoes reviews, so the evaluation set overlaps the training set. | cells 9, 18, 43 | Single RT source. `stg_rt_reviews` assigns a deterministic `train`/`holdout` fold hashed on `review_id`; Model 1 trains on one side and is scored on the other. |
| 3 | A markdown cell treats the predicted sentiment distribution matching the dataset's 64/36 Fresh/Rotten split as validation. Marginal agreement says nothing about per-review accuracy — a model that shuffles labels while preserving proportions scores identically. | cell 42 | Replaced with a confusion matrix and per-class precision/recall on the holdout fold. |
| 4 | **Residuals computed in-sample.** `score_residual` is computed across the whole dataset, including Model 2's own training rows, so residuals are optimistically small exactly where the model memorised. Those residuals feed `bias_score`, which feeds Model 3. | cell 51 | Fold-aware training; the limitation that remains (sentiment features are in-sample for train-fold rows) is stated in the notebook's limitations section rather than left implicit. |
| 5 | **Unit mismatch in `consensus_diff`.** A critic's 1-10 score is compared against `tomatometer_rating / 10`. The tomatometer is the *percentage of reviews that are Fresh*, not an average score — a film where every critic mildly likes it scores 100. | cell 56 | Feature kept (it carries signal) but documented as a proxy with fictional units, in both the notebook and the model docstring. |
| 6 | **Double-counted features.** `bias_score` is a weighted sum of four features, and then both the components and the sum are passed to the Isolation Forest. | cells 57, 68 | Kept for comparability with the original, with the correlation heatmap in section 5 making the redundancy visible instead of hidden. |
| 7 | **No held-out calibration for Model 3.** The forest is fit and scored on the same rows, and `fraud_probability` is min-max normalised over those same scores, so the scale is defined by the data it was fit on. | cell 71 | Documented as a limitation. Unsupervised with no ground truth, this is not fixable by splitting — it needs labels, which the notebook's closing section says plainly. |
| 8 | **Inconsistent fake definitions.** The English model flags `predictions == -1` (the contamination quantile); the Polish model flags `fraud_probability > 50` (a different, much larger set). The two sections report incomparable rates. | cells 71, 83 | One definition in `AnomalyDetector.score`. |
| 9 | The "distinctive words" word clouds exclude every token appearing in both Fresh and Rotten reviews. Since almost all meaningful vocabulary appears in both classes, what survives is the rare tail — the clouds show noise, not distinguishing language. | cell 44 | Replaced by the model's own coefficients, which is what "distinctive" should mean here. |
| 10 | Ridge coefficients on a continuous target are labelled `log_odds`. They are linear coefficients; log-odds belong to logistic regression. | cell 48 | Renamed and reframed as coefficient inspection. |

## Reproducibility

| # | Finding | Where | Resolution |
|---|---|---|---|
| 1 | Dependencies are installed by `!pip install` inside the notebook and imported in one 40-line cell; there is no lockfile, no version pin and no environment definition. | cells 3, 4 | `pyproject.toml` with pinned minimums and optional extras (`ingest`, `embeddings`, `app`, `dbt`, `dev`). |
| 2 | The pipeline requires roughly 3 GB of Kaggle downloads before a single line can be verified, and one input (`filmwebplus.csv`) is read from the working directory with no download step — it is not in the repository, so the Polish half cannot be run by anyone else. | cells 7, 9, 18, 76 | Generated seeds (`scripts/generate_seeds.py`) let the whole DAG, the tests and the analysis notebook run in seconds with no credentials. |
| 3 | **The Polish section fabricates its own metadata.** Filmweb reviews carry no critic identity or dates, so the notebook samples English critics at random, assigns them to Polish reviews, and generates review dates and release dates with `np.random.randint`. Burst rates, activity windows and "fraud probability" are then computed on that generated metadata and reported as results. | cell 79 | Removed. Behavioural fraud signals require real behavioural data; a bilingual version needs a Polish source that actually carries reviewer identity and timestamps. |
| 4 | 86 of 87 cells were saved without outputs, so no result in the notebook could be checked without re-running the full pipeline. | whole file | The analysis notebook is committed with outputs and is executed in CI on every push, so a broken cell fails the build. |
| 5 | Two different cleaning functions (`clean_text`, `clean_text_rt`) are applied to different models, and `predict_sentiment` re-cleans already-cleaned text with the other one — so Model 1 sees a text distribution at inference that it never saw in training. | cells 5, 29, 41 | One `clean_text` applied consistently at both training and inference. |

## Source-data defects the pipeline now handles

These are properties of the Kaggle dump rather than mistakes in the original notebook, but the
notebook inherited all of them silently. They surfaced the first time the dbt models were built
against the real data, because the `unique` test on `review_id` failed.

| Defect | Effect if left alone | Handling |
|---|---|---|
| **Exact duplicate rows** — same film, critic, date and review text appearing more than once. | Every reviewer-level aggregate inflates: a critic looks twice as prolific, which is exactly the burst signal Model 3 treats as suspicious. The same review text can also land on both sides of the train/holdout split, quietly reintroducing the leakage the fold assignment was added to prevent. | Collapsed in `stg_rt_reviews` with a `qualify row_number()`, preferring the copy that carries a parseable score so deduplication never discards a usable target. |
| **Reviews with no critic name.** | `int_reviewer_stats` groups on `critic_name`, so every unattributed row joins one enormous pseudo-critic. Its mean and standard deviation are a blend of hundreds of unrelated people, and every z-score and burst window computed for those rows is meaningless. | Dropped in staging — these rows cannot support reviewer-level behavioural features at all. |
| **Reviews with no date.** | Burst windows and `days_since_release` are undefined. | Excluded from `int_burst_activity` by its own filter; the mart left-joins, so the features arrive as NULL and are zero-filled at the model boundary rather than silently defaulting mid-pipeline. |

The staging funnel in section 1 of the analysis notebook prints how many rows each filter
removes, so the cost is visible rather than assumed. Both defects are planted in the generated
seeds, and `test_staging_removes_source_duplicates` and
`test_staging_drops_unattributed_reviews` fail if the handling regresses.

## Dependency change worth flagging

The original trained Model 1 on `andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews`
while Models 2 and 3 ran on
`stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset`. This rework uses only the
latter. That removes a ~1 GB download and the cross-dump contamination described in
Methodology #2, at the cost of a smaller sentiment training corpus — IMDB's 50k labelled
reviews still carry that side.

## What the rework did not change

Model choices and hyperparameters are carried over unchanged (documented in
`src/rotten_review/config.py`): bag-of-words + logistic regression, TF-IDF + Ridge, and an
Isolation Forest at 5% contamination over behavioural features concatenated with text
embeddings. The point of the exercise was to make the existing pipeline verifiable, not to
replace its modelling.
