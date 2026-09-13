# setup

[← back to README](../README.md)

## requirements

- Python 3.10+
- `python3-tk` (for the interactive matplotlib `TkAgg` backend used by `. cli demo`; the
  tests themselves are headless and don't need it)

## install

    . cli setup

This installs `python3-tk` via `apt`, creates a `.venv`, and installs the Python
dependencies (`matplotlib`, `pytest`) from `requirements.txt`.

## test

    . cli test

Runs everything under `tests/` with pytest, grouped one file per module under test (see
[structure](structure.md)), plus `test_training_pipeline.py` for end-to-end coverage - it
builds the same convergence-curve and decision-boundary charts as `. cli demo`, using
matplotlib's `Agg` backend, but never opens a window — so all of it runs unattended, e.g. in
CI.

`. cli clean` removes `__pycache__`/`.pytest_cache` directories (used automatically before
`. cli test`).
