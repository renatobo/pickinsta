PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: install-dev test lint format check pre-commit-install

install-dev:
	$(PIP) install --upgrade pip setuptools
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff format src tests

check: lint test

pre-commit-install:
	$(PYTHON) -m pre_commit install
