PYTHON ?= python3

.PHONY: data verify-data test test-eval check notebook eval-help eval-dry-run

data:
	$(PYTHON) scripts/prepare_math500.py

verify-data:
	$(PYTHON) scripts/prepare_math500.py --verify-only

test:
	$(PYTHON) -m unittest discover -s tests -v

test-eval:
	$(PYTHON) -m unittest discover -s tests/evaluation -v

check: test verify-data

notebook:
	uv run --extra notebook --frozen jupyter lab notebooks/math500_explorer.ipynb

eval-help:
	$(PYTHON) scripts/evaluate_math500.py --help

eval-dry-run:
	$(PYTHON) scripts/evaluate_math500.py plan-raw --config configs/evals/math500_raw.example.toml --run-dir outputs/evals/unused-dry-run --dry-run
