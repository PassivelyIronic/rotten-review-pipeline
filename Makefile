.PHONY: install lint format test dbt ingest train app notebook seeds results clean

install:  ## editable install with dev + dbt extras
	pip install -e ".[dbt,dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:  ## unit + integration tests (runs dbt build on synthetic seeds)
	pytest

dbt:  ## build warehouse models on real ingested data
	cd dbt && dbt build --target dev --profiles-dir .

dbt-ci:  ## build against synthetic seeds only (what CI runs)
	cd dbt && dbt build --target ci --profiles-dir .

ingest:  ## download Kaggle datasets and load the DuckDB warehouse
	rotten-review ingest

train:  ## full training sequence (requires `make ingest` and `make dbt` first)
	rotten-review train-sentiment
	rotten-review train-score
	rotten-review train-anomaly

app:  ## launch the Gradio demo (requires trained artifacts)
	python -m rotten_review.app

notebook:  ## execute the analysis notebook in place (sample mode if no warehouse)
	jupyter nbconvert --to notebook --execute --inplace \
		notebooks/01_review_integrity_analysis.ipynb

seeds:  ## regenerate the synthetic seed data
	python scripts/generate_seeds.py

results:  ## rewrite the README results table from reports/metrics.json
	python scripts/render_results.py

clean:
	rm -rf data/warehouse.duckdb artifacts/ dbt/target dbt/logs .pytest_cache
