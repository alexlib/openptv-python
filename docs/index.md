# Welcome to openptv_python's documentation!

Python version of the OpenPTV library.

Start with the README for installation and backend selection details. The short
version is:

- install runtime dependencies with `uv sync` on Python 3.12 or 3.13
- install contributor tooling with `uv sync --extra dev`
- use the same Python API regardless of backend
- get pure Python behavior everywhere, Numba acceleration in selected kernels,
  and automatic native `optv` delegation for preprocessing and full-frame
  target recognition when available

```{toctree}
:caption: 'Contents:'
:maxdepth: 2

README.md
bundle_adjustment.md
native_backend_unification.md
```

# Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
