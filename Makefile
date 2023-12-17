SHELL=/bin/bash

venv:  ## Set up virtual environment
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

install: venv
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop -m Cargo.toml

dev-release: venv
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop --release -m Cargo.toml

pre-commit: venv
	cargo fmt
	pre-commit run --all-files

# run: install
# 	source .venv/bin/activate && python run.py

# run-release: install-release
# 	source venv/bin/activate && python run.py