.DEFAULT_GOAL := help

PYTHONPATH=
SHELL=bash
VENV=.venv
PROJECT=pyproject.toml
TUTORIALS=docs/tutorials

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
ruff-lint: .venv  ## Run lint checks (only ruff)
	$(VENV_BIN)/ruff check --output-format concise

.PHONY: type-lint
type-lint: .venv  ## Run lint checks (only mypy)
	-$(VENV_BIN)/mypy

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

.PHONY: todo
todo: .venv  ## Get all TODO/FIXME lines
	$(VENV_BIN)/ruff check --select="FIX"

.PHONY: test-notebooks
test-notebooks: .venv  ## Test all jupyter notebook tutorials
	for notebook in $(TUTORIALS)/*.ipynb; do \
	  echo "Testing $$notebook"; \
	  $(VENV_BIN)/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 "$$notebook"; \
	done
	find $(TUTORIALS)/*nbconvert.ipynb -delete

.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@rm -rf .mypy_cache/
	@rm -rf .pytest_cache/
	@$(VENV_BIN)/ruff clean
	@find . -type f -name '*.py[co]' -delete -or -type d -name __pycache__ -exec rm -r {} +
