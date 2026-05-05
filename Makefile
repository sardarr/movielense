.PHONY: help install sync run test lint clean data report ui ui-wandb sync-wandb

UV ?= uv
MLFLOW_PORT ?= 5000

help:
	@echo "make sync       - install dependencies via uv"
	@echo "make sync-wandb - install with W&B extra"
	@echo "make run        - run full pipeline end-to-end"
	@echo "make ui         - launch MLflow UI at http://localhost:$(MLFLOW_PORT)"
	@echo "make test       - run test suite"
	@echo "make lint       - lint with ruff"
	@echo "make clean      - remove generated artifacts"

sync:
	$(UV) sync --extra dev

sync-wandb:
	$(UV) sync --extra dev --extra wandb

run:
	$(UV) run python -m movielense.run

ui:
	$(UV) run mlflow ui --backend-store-uri sqlite:///mlruns.db --port $(MLFLOW_PORT)

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check src tests

clean:
	rm -rf artifacts mlruns mlruns.db data/raw data/processed
	find . -name "__pycache__" -type d -exec rm -rf {} +
