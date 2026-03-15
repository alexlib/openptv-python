# openptv-python

Python version of the OpenPTV library - this is *a work in progress*

## What This Repo Provides

`openptv-python` keeps the Python API as the main interface and combines three
execution modes behind that API:

- Pure Python: the reference implementation and the easiest path for reading,
  debugging, and extending the code.
- Python + Numba: several hot kernels are JIT-compiled automatically on first
  use, so the Python implementation still benefits from acceleration.
- Native `optv` bindings: selected operations reuse the native OpenPTV
  implementation when the `optv` package is available.

At the moment, automatic native delegation is implemented for image
preprocessing and full-frame target recognition. The rest of the library keeps
the same Python API and remains usable even when those native paths are not in
use.

## How this is started

This work started from the https://github.com/OpenPTV/openptv/tree/pure_python branch. It's a long-standing idea to convert all the C code to Python and now it's possible with ChatGPT to save
a lot of typing time.

This repo is created using a *cookiecutter* and the rest of the readme describes the way to work with
this structure

## Supported Python Versions

The project currently supports Python `>=3.12,<3.14`.

## Installation

### Default user install

#### Recommended: uv

Create the environment and install the runtime dependencies:

```bash
uv venv
source .venv/bin/activate
uv sync
```

This gives you the standard runtime stack: NumPy, SciPy, Numba, and YAML
support.

If you also want native `optv` delegation when bindings are available for your
platform and Python version, install the optional extra:

```bash
uv sync --extra native
```

#### Alternative: pip

```bash
conda create -n openptv-python -c conda-forge python=3.12
conda activate openptv-python
pip install .
```

Optional native bindings:

```bash
pip install ".[native]"
```

### Developer install

#### Recommended: uv

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev
```

#### Alternative: conda + pip

```bash
conda create -n openptv-python -c conda-forge python=3.12
conda activate openptv-python
pip install -e ".[dev]"
```

### What gets installed

- The default install contains the runtime dependencies only.
- The optional `native` extra adds `optv` bindings for automatic native
  delegation on supported platforms.
- The optional `dev` extra adds test, docs, typing, and pre-commit tooling for
  contributors.
- The public API stays the same regardless of which backend extras are
  installed.

## Backend Behavior

### Pure Python backend

This is the base implementation for the whole library. It is always the source
of truth for the Python API and remains the fallback behavior for code paths
that are not delegated to `optv`.

### Python + Numba backend

Numba accelerates selected computational kernels inside the Python
implementation. This is automatic; there is no separate API to enable it.
Expect the first call to a JIT-compiled function to be slower due to
compilation, with later calls running faster.

### Native `optv` backend

When `optv` imports successfully, `openptv-python` automatically reuses native
implementations for:

- image preprocessing
- full-frame target recognition / segmentation

These native paths are validated against the Python implementation by parity
tests, so results stay backend-independent.

### Backend Capability Table

| Operation | Pure Python | Python + Numba | Native `optv` |
| --- | --- | --- | --- |
| Image preprocessing | Yes | Yes | Yes, automatic delegation |
| Target recognition / segmentation | Yes | Yes | Yes, automatic delegation |
| Point reconstruction | Yes | Partial internal kernels | Not used by default |
| Correspondence search / stereo matching | Yes | Partial internal kernels | Not used by default |
| Tracking | Yes | Partial internal kernels | Not used by default |
| Sequence parameter I/O | Yes | No | Available in native bindings |

`Not used by default` means the native path exists in benchmarks or conversion
helpers, but the regular `openptv-python` runtime path still uses the Python
implementation unless that operation is explicitly integrated later.

## Getting Started

### 1. Install the project

Use one of the installation methods above.

### 2. Verify imports

```bash
uv run python - <<'PY'
import openptv_python
import numba
try:
  import optv
except ImportError:
  print("optv not installed; native delegation disabled")
else:
  print("optv ok", optv.__version__)

print("openptv_python ok")
print("numba ok", numba.__version__)
PY
```

### 3. Start using the Python API

```python
>>> import openptv_python

```

### 4. Run the test suite

```bash
uv run make
```

Stress and performance tests are part of the default suite now. If you need a
faster validation pass locally, you can skip them explicitly:

```bash
OPENPTV_SKIP_STRESS_BENCHMARKS=1 uv run make
```

### Workflow for developers/contributors

Recommended contributor workflow with `uv`:

```bash
uv venv
source .venv/bin/activate
make env-update
```

This keeps the local environment synced to the locked developer dependency set.

If you prefer the conda workflow instead:

```bash
conda create -n openptv-python -c conda-forge python=3.12
conda activate openptv-python
make conda-env-update
```

Before pushing to GitHub, use the developer install above and then run the
following commands:

1. Update the environment: `make env-update` by default, or `make conda-env-update` if you are using the conda workflow
1. If you are using pip instead of uv, install the editable developer environment: `pip install -e ".[dev]"`
1. Sync with the latest [template](https://github.com/ecmwf-projects/cookiecutter-conda-package) (optional): `make template-update`
1. Run quality assurance checks: `make qa`
1. Run tests: `make unit-tests`
1. Run the static type checker: `make type-check`
1. Build the documentation (see [Sphinx tutorial](https://www.sphinx-doc.org/en/master/tutorial/)): `make docs-build`

### License

```
Copyright 2023, OpenPTV consortium.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
