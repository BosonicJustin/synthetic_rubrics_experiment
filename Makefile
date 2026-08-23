PYTHON ?= python3

.PHONY: data verify-data test check

data:
	$(PYTHON) scripts/prepare_math500.py

verify-data:
	$(PYTHON) scripts/prepare_math500.py --verify-only

test:
	$(PYTHON) -m unittest discover -s tests -v

check: test verify-data
