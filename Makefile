.DEFAULT_GOAL := help

PYTHONPATH=
SHELL=bash
VENV=.venv
PROJECT=pyproject.toml

ifeq ($(OS),Windows_NT)
	VENV_BIN=$(VENV)/Scripts
else
	VENV_BIN=$(VENV)/bin
endif

.venv:  ## Set up Python virtual environment and install requirements
	python3 -m venv $(VENV)
	$(MAKE) requirements

.PHONY: requirements
requirements: .venv  ## Install/refresh Python project requirements
	$(VENV_BIN)/python -m pip install --upgrade pip \
	&& $(VENV_BIN)/python -m pip install --upgrade uv \
	&& $(VENV_BIN)/uv pip install --upgrade --compile-bytecode --no-build \
		--all-extras -r $(PROJECT)

.PHONY: build
build: .venv  ## Install Jaxley to Python Environment
	$(VENV_BIN)/uv pip install -e "."

.PHONY: test
test: .venv build  ## Run fast unittests
	$(VENV_BIN)/pytest $(PYTEST_ARGS)

.PHONY: test-all
test-all: .venv build  ## Run all tests
	$(VENV_BIN)/pytest -m "" $(PYTEST_ARGS)

.PHONY: ruff-lint
ruff-lint: .venv  ## Run lint checks (only)
	$(VENV_BIN)/ruff check --output-format concise

.PHONY: lint
lint: .venv  ## Run lint checks (only)
	$(VENV_BIN)/ruff check
	-$(VENV_BIN)/mypy

.PHONY: fmt
fmt: .venv  ## Run autoformatting (and lint)
	$(VENV_BIN)/ruff format
	$(VENV_BIN)/ruff check
	$(VENV_BIN)/typos .
	-$(VENV_BIN)/mypy

.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@rm -rf .mypy_cache/
	@rm -rf .pytest_cache/
	@$(VENV_BIN)/ruff clean
	@find . -type f -name '*.py[co]' -delete -or -type d -name __pycache__ -exec rm -r {} +
