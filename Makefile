# Set UV=1 (or export it once) to run everything through uv instead of pip.
#   PowerShell:  $env:UV=1
#   bash/zsh:    export UV=1
ifeq ($(UV),1)
  DBT := uv run dbt
  PIP := uv pip
  PYTHON := uv run python
  JUPYTER := uv run jupyter
else
  DBT ?= dbt
  PIP ?= pip
  PYTHON ?= python
  JUPYTER ?= jupyter
endif

.PHONY: install install-ci lint format test dbt dbt-ci ingest train app notebook seeds results clean

install:  ## editable install with everything needed to run the pipeline locally
	$(PIP) install -e ".[ingest,dbt,notebook,dev]"

install-ci:  ## the slimmer set CI uses (no Kaggle client)
	$(PIP) install -e ".[dbt,notebook,dev]"

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

test:  ## unit + integration tests (runs dbt build on synthetic seeds)
	$(PYTHON) -m pytest

dbt:  ## build warehouse models on real ingested data
	cd dbt && $(DBT) build --target dev --profiles-dir .

dbt-ci:  ## build against synthetic seeds only (what CI runs)
	cd dbt && $(DBT) build --target ci --profiles-dir .

ingest:  ## download Kaggle datasets and load the DuckDB warehouse
	$(PYTHON) -m rotten_review.cli ingest

train:  ## full training sequence (requires `make ingest` and `make dbt` first)
	$(PYTHON) -m rotten_review.cli train-sentiment
	$(PYTHON) -m rotten_review.cli train-score
	$(PYTHON) -m rotten_review.cli train-anomaly

app:  ## launch the Gradio demo (requires trained artifacts)
	$(PYTHON) -m rotten_review.app

notebook:  ## execute the analysis notebook in place (sample mode if no warehouse)
	$(JUPYTER) nbconvert --to notebook --execute --inplace \
		notebooks/01_review_integrity_analysis.ipynb

seeds:  ## regenerate the synthetic seed data
	$(PYTHON) scripts/generate_seeds.py

results:  ## rewrite the README results table from reports/metrics.json
	$(PYTHON) scripts/render_results.py

clean:  ## remove build artefacts (portable: no rm, works in PowerShell)
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(d, ignore_errors=True) for d in ['artifacts', 'dbt/target', 'dbt/logs', '.pytest_cache', '.ruff_cache']]; pathlib.Path('data/warehouse.duckdb').unlink(missing_ok=True)"
