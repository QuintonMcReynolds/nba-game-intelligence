PY ?= .venv/bin/python

.PHONY: all ingest etl wp clutch recaps analyze tableau figures test lint

all: ingest etl wp clutch recaps analyze tableau figures

ingest:   ## 22 seasons of play-by-play -> Parquet lake (~230 MB)
	$(PY) scripts/ingest.py

etl:      ## PySpark: clean, parse text, infer possession, box scores, quality report (~5 min)
	$(PY) scripts/etl.py

wp:       ## win probability: logistic + boosted trees (scikit-learn) + MLP (TensorFlow)
	$(PY) scripts/train_wp.py

clutch:   ## WPA per event, shrunken clutch leaderboard, SQL era trends
	$(PY) scripts/clutch.py

recaps:   ## LLM recaps (PyTorch, local Qwen2.5-0.5B): pilot + randomized experiment (~25 min)
	$(PY) scripts/recap_experiment.py

analyze:  ## fact-check every recap, validate the checker, power analysis, A/B test
	$(PY) scripts/analyze_experiment.py

tableau:  ## Tableau-ready extracts in tableau/data
	$(PY) scripts/tableau_export.py

figures:
	$(PY) scripts/figures.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	.venv/bin/cfn-lint cloud/template.yaml
