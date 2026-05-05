# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

openptv-python is a pure-Python reimplementation of the OpenPTV (Open Source Particle Tracking Velocimetry) library. It provides three execution backends behind a single API: pure Python (reference), Python + Numba JIT, and optional native `optv` C bindings.

## Common Commands

```bash
# Setup
uv venv && source .venv/bin/activate && uv sync --extra dev

# Run all checks (qa + tests + mypy)
uv run make

# Run tests
uv run pytest -vv
uv run pytest tests/test_calibration_class.py -v          # single file
uv run pytest tests/test_calibration_class.py::test_name -v  # single test

# Skip stress/performance tests
OPENPTV_SKIP_STRESS_BENCHMARKS=1 uv run pytest -vv

# Linting (ruff via pre-commit)
uv run pre-commit run --all-files

# Type checking
uv run mypy .

# Build docs
uv run make docs-build
```

## Architecture

### Source layout (`openptv_python/`)

The library mirrors the original C OpenPTV modules:

- **calibration.py** — Camera calibration data structures using NumPy structured arrays with `exterior_dtype`/`interior_dtype`/`glass_dtype`. The `Calibration` class wraps these. Rotation matrices are Numba-JIT compiled.
- **parameters.py** — All parameter dataclasses (`ControlPar`, `VolumePar`, `SequencePar`, `TargetPar`, `MultimediaPar`, etc.) inherit from a `Parameters` base with YAML serialization. Legacy `.par` file readers also live here.
- **tracking_run.py / track.py** — Tracking pipeline. `TrackingRun` orchestrates multi-frame tracking using `FrameBuf` (from `tracking_frame_buf.py`).
- **correspondences.py** — Stereo correspondence / multi-camera matching.
- **image_processing.py / segmentation.py** — Image preprocessing and target detection/recognition.
- **multimed.py** — Multimedia (multi-media, i.e. glass/water) ray tracing corrections.
- **ray_tracing.py** — Ray tracing through the optical system.
- **orientation.py** — Camera orientation / exterior orientation estimation.
- **trafo.py** — Coordinate transformations (pixel <-> metric).
- **imgcoord.py** — Image coordinate computations.
- **vec_utils.py** — Vector math utilities.
- **epi.py / find_candidate.py / sortgrid.py** — Epipolar geometry, candidate search, and calibration grid sorting.

### Native backend (`_native_compat.py`, `_native_convert.py`)

When the optional `optv` package is installed, image preprocessing and target recognition automatically delegate to native C implementations. `_native_compat.py` probes for `optv` submodules at import time and sets `HAS_OPTV`, `HAS_NATIVE_PREPROCESS`, `HAS_NATIVE_SEGMENTATION` flags. `_native_convert.py` handles object conversion between Python and native types.

### Tests (`tests/`)

Tests use pytest. Test data lives in `tests/testing_folder/` with subdirectories like `test_cavity/` containing camera calibration files, parameter files, and raw images. Some tests generate synthetic data (e.g., `gen_track_data.py`, `test_synthetic_cavity_case.py`).

## Key Conventions

- Python 3.12–3.13 only
- NumPy structured arrays (not classes) for performance-critical data like exterior/interior orientation
- Numba `@njit` for hot numerical kernels — these functions operate on NumPy arrays/structured dtypes, not Python objects
- Parameters use dataclasses with YAML serialization; legacy `.par` text format readers coexist
- Ruff for linting/formatting (line length 88, numpy docstring convention)
- Pre-commit hooks: trailing whitespace, ruff, ruff-format, blackdoc, mdformat, YAML/TOML formatting, gitleaks
- Version managed by setuptools_scm (git tags)

## CI

GitHub Actions (`on-push.yml`): pre-commit, unit tests (Python 3.12 + 3.13 via micromamba), mypy, docs build, integration tests, PyPI publish on tags.
