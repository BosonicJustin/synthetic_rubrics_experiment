PYTHON ?= python3
UV_RUN ?= uv run --extra evaluation --frozen python

.PHONY: data verify-data test test-eval test-training check notebook eval-help eval-dry-run train-help train-dry-run

data:
	$(PYTHON) scripts/prepare_math500.py

verify-data:
	$(PYTHON) scripts/prepare_math500.py --verify-only

test:
	$(UV_RUN) -m unittest discover -s tests -v

test-eval:
	$(UV_RUN) -m unittest discover -s tests/evaluation -v

test-training:
	$(UV_RUN) -m unittest discover -s tests/training -v

check: test verify-data

notebook:
	uv run --extra notebook --frozen jupyter lab notebooks/math500_explorer.ipynb

eval-help:
	$(UV_RUN) scripts/evaluate_math500.py --help

eval-dry-run:
	$(UV_RUN) scripts/evaluate_math500.py plan-raw --config configs/evals/math500_raw.example.toml --run-dir outputs/evals/unused-dry-run --dry-run

train-help:
	$(UV_RUN) scripts/train_math500.py --help

train-dry-run:
	$(UV_RUN) scripts/train_math500.py prepare --config configs/training/math500_cat_grpo.example.toml --run-dir outputs/training/unused-dry-run --dry-run
