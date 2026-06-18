# Developer convenience targets. Run `make help` for the list.
.PHONY: help install lint format typecheck test cov check

help:
	@echo "Targets:"
	@echo "  install    Install runtime + dev dependencies"
	@echo "  lint       Run ruff lint checks"
	@echo "  format     Auto-format with ruff"
	@echo "  typecheck  Run mypy on the typed surface"
	@echo "  test       Run the test suite"
	@echo "  cov        Run tests with coverage report"
	@echo "  check      Run lint + typecheck + tests (the CI gate)"

install:
	pip install -r requirements.txt -r requirements-dev.txt

lint:
	ruff check src tests

format:
	ruff format src tests

typecheck:
	mypy

test:
	pytest -q

cov:
	pytest --cov --cov-report=term-missing

check: lint typecheck test
