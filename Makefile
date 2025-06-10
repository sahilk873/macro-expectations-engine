.PHONY: install test lint typecheck clean setup data

install:
	pip install -e ".[dev,yfinance,fred,bls,notebook]"

test:
	pytest tests/ -v --cov=src/macro_engine --cov-report=term-missing

lint:
	ruff check src/macro_engine/ tests/ --fix

typecheck:
	mypy src/macro_engine/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf reports/tables/* reports/figures/*

setup: install
	pip install -e ".[dev,yfinance,fred,bls,notebook]"
	cp -n .env.example .env 2>/dev/null || true
	mkdir -p data/kalshi data/macro data/prices data/manual data/output
	mkdir -p reports/tables reports/figures reports/notebooks

run-event-calendar:
	python scripts/build_event_calendar.py

run-kalshi-fetch:
	python scripts/fetch_kalshi.py

run-macro-fetch:
	python scripts/fetch_macro_data.py

run-price-fetch:
	python scripts/fetch_price_data.py

run-mapping:
	python scripts/build_market_mapping.py

run-expectations:
	python scripts/compute_implied_expectations.py

run-surprises:
	python scripts/compute_surprises.py

run-event-studies:
	python scripts/run_event_studies.py

run-regime-model:
	python scripts/build_regime_model.py

run-backtest:
	python scripts/run_backtest.py

run-robustness:
	python scripts/run_robustness_checks.py

run-report:
	python scripts/generate_report.py

run-all: run-event-calendar run-kalshi-fetch run-macro-fetch run-price-fetch \
	run-mapping run-expectations run-surprises run-event-studies \
	run-regime-model run-backtest run-robustness run-report
