# Recommender systems project

Requires uv and Python 3.12. Run commands from the repository root.

Install the locked dependencies and download MovieLens 1M:

```bash
uv sync --locked
uv run --locked scripts/download_data.py
```

The dataset is extracted into `data/ml-1m/`.

`uv sync --locked` also installs `recsys` in editable mode, so imports work
from notebooks and source changes are available without reinstalling. Select
the project's `.venv` as your notebook kernel, and restart the kernel after
the first installation.

To use a different data location:

```bash
uv run --locked scripts/download_data.py --data-dir /path/to/datasets
```

The dataset will be at `/path/to/datasets/ml-1m/`; pass that directory to
`load_movielens()` when loading it.
