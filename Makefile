.PHONY: install test lint format

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

format:
	ruff format . && ruff check . --fix
