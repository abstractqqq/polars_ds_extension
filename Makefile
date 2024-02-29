SHELL=/bin/bash

VENV=.venv

ifeq ($(OS),Windows_NT)
	VENV_BIN=$(VENV)/Scripts
else
	VENV_BIN=$(VENV)/bin
endif

.venv:
	python3 -m venv $(VENV)
	$(MAKE) dev-requirements
	
requirements: .venv
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/python -m pip install --upgrade uv \
	&& $(VENV_BIN)/uv pip install --upgrade -r requirements.txt \
	&& $(VENV_BIN)/uv pip install --upgrade -r requirements.txt \
	&& $(VENV_BIN)/uv pip install --upgrade -r tests/requirements-test.txt \
	&& $(VENV_BIN)/uv pip install --upgrade -r docs/requirements-docs.txt \

dev-release: .venv
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop --release -m Cargo.toml

pre-commit: .venv
	cargo fmt
	pre-commit run --all-files
