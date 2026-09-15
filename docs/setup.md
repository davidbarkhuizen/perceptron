# setup

[← back to README](../README.md)

## requirements

- Python 3.10+
- `python3-tk` (for the interactive matplotlib `TkAgg` backend used by `. cli demo`; the
  tests themselves are headless and don't need it)

## install

    . cli setup

This installs `python3-tk` via `apt`, creates a `.venv`, and installs the Python
dependencies (`matplotlib`, `pytest`, `pyarrow`) from `requirements.txt`.

## test

    . cli test

Runs everything under `tests/` with pytest, grouped one file per module under test (see
[structure](structure.md)), plus `test_training_pipeline.py` for end-to-end coverage - it
builds the same convergence-curve and decision-boundary charts as `. cli demo`, using
matplotlib's `Agg` backend, but never opens a window — so all of it runs unattended (no display
needed).

One exception: `tests/test_mnist_data.py`'s tests need the real MNIST parquet/binary files
(`data/mnist/*.parquet`, `data/mnist/*.bin` - see `mnist_data.py`). These are deliberately not
committed (large binary data) and have no scripted fetch step anywhere in this codebase - the
`.parquet` files are supplied locally by whoever set up this checkout, and
`mnist_data.convert_parquet_to_binary` (which `. cli demo`'s **MNIST ensemble recognition** demo
calls automatically the first time it runs - see [demos](demos.md)) only converts an
already-present `.parquet` file to the flat binary format training actually reads; it doesn't
fetch the `.parquet` file itself from anywhere. Every other test file runs against bundled
(`data/digits/digits.csv`) or synthetic data and needs nothing external. On a fresh checkout
without the MNIST `.parquet` files, both `test_mnist_data.py` and the MNIST demo will fail until
that data is supplied.

`. cli clean` removes `__pycache__`/`.pytest_cache` directories (used automatically before
`. cli test`).
