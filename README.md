# Recommender systems project

Requires uv and Python 3.12. If uv is not installed, follow the
[uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
Run commands from the repository root.

Regenerate the Part 1 and Part 2 result tables and figures.

```bash
uv run --locked scripts/run_all.py --data-dir data/ml-1m
```

This installs the dependencies pinned in `uv.lock`, downloads MovieLens 1M if
the default dataset is missing, and evaluates the saved models. It reuses the
recorded validation and sparsity results. Add `--parts 1` or `--parts 2` to run one part.

Add `--retrain` to repeat model selection, training, and the sparsity experiments.

```bash
uv run --locked scripts/run_all.py --retrain
```

Mult-VAE runs on CPU. Final BPR and Mult-VAE fits and sparsity subsamples use
seeds 0–4. BPR validation uses search seed 2026 and fit seed 0. Mult-VAE
validation and the user bootstrap use seed 42.

For an existing dataset, pass `--data-dir /path/to/ml-1m`, pointing to the folder
containing `ratings.dat`, `movies.dat` and `users.dat`. Use `--out-dir /path/to/results`
to change the output directory.

Overall results are in [results.csv](results/results.csv) and
[results-part2.csv](results/part2/results-part2.csv). Figures are saved to
`results/images/` and `results/part2/images/`, with detailed tables under `results/`.

The [project report](report/Recommender_Systems_Report.pdf) ([Word version](report/Recommender_Systems_Report.docx))
and [results spreadsheet](results/Recommender_Systems_Results.xlsm) are included in this repository.
