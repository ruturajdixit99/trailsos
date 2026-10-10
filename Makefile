PY ?= .venv/Scripts/python

.PHONY: install run test bench samples

install:
	python -m venv .venv && $(PY) -m pip install -r requirements.txt

samples:
	$(PY) scripts/make_samples.py

run:
	$(PY) -m uvicorn server.app:app --port 8000

test:
	$(PY) -m pytest -q -s -p no:warnings

bench:
	$(PY) scripts/benchmark.py
