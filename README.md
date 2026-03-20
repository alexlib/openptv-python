# openptv-python

Unified Python/OpenPTV repository for the combined `openptv_python` core library and `pyptv` GUI.

Current repository version: 0.5.0

## Overview

This repository now exposes one shared API and one shared version stream. The main pieces are:

- `openptv_python`: the Python library layer, used for pure-Python execution, Numba-accelerated kernels, and shared parameter/data handling.
- `pyptv`: the GUI and batch-processing layer, built on top of the same Python API.
- `optv`: native bindings used where available for delegated operations such as image preprocessing and full-frame target recognition.

The current setup is intentionally structured so that the same codebase can run in three modes:

1. Pure Python for readability, debugging, and visual inspection.
2. Pure Python plus Numba for accelerated kernels where available.
3. Native `optv` delegation for selected operations when the bindings are installed.

The GUI version shown in PyPTV comes from the shared version module in `src/openptv_python/version.py`.

## Versioning

Versioning is now centralized:

- Canonical version source: [src/openptv_python/version.py](src/openptv_python/version.py)
- Package version: 0.5.0
- Release boundary: this is the first combined version after merging the older `openptv_python` and `pyptv` package streams

To bump the version, run:

```bash
python src/pyptv/bump_version.py --patch
```

Use `--minor` or `--major` for larger increments. The script updates the shared version module and `pyproject.toml` together.

## Installation

The project supports Python `>=3.11,<3.14`.

### Recommended: uv

Runtime install:

```bash
uv venv
source .venv/bin/activate
uv sync
```

GUI extras:

```bash
uv sync --extra gui
```

Developer setup:

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev
```

### Alternative: pip

```bash
conda create -n openptv-python -c conda-forge python=3.12
conda activate openptv-python
pip install .
```

GUI extras:

```bash
pip install "[gui]"
```

Developer extras:

```bash
pip install "[dev]"
```

## Usage

### Python API

```python
import openptv_python

print(openptv_python.__version__)
```

### GUI

```bash
uv run pyptv
```

The GUI is versioned from the same shared source as the library.

## Documentation

Detailed guides live under [docs/](docs/), especially the PyPTV guides in [docs/pyptv](docs/pyptv/README.md).

## Testing

Run the focused test suites with:

```bash
uv run --with pytest pytest -q tests/openptv_python tests/pyptv
```

The repository also contains test datasets in [tests/testing_folder](tests/testing_folder/).

## Repository Structure

- `src/openptv_python/`: shared Python library code and canonical version module
- `src/pyptv/`: GUI, batch tools, and compatibility helpers
- `docs/`: Sphinx documentation
- `tests/`: library, GUI, and fixture tests

## License

Apache License, Version 2.0.
