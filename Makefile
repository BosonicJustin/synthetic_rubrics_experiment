PYTHON ?= python3

.PHONY: data verify-data test check notebook

data:
	$(PYTHON) scripts/prepare_math500.py

verify-data:
	$(PYTHON) scripts/prepare_math500.py --verify-only

test:
	$(PYTHON) -m unittest discover -s tests -v

check: test verify-data

notebook:
	uv run --extra notebook --frozen jupyter lab notebooks/math500_explorer.ipynb
