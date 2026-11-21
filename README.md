# Recommender systems project

Requires uv and Python 3.12. If uv is not installed, follow the
[uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
Run commands from the repository root.

Reproduce every number, table and generated figure in the report (Parts 1 and 2):

```bash
uv run --locked scripts/run_all.py --data-dir data/ml-1m
```

This installs the dependencies pinned in `uv.lock`, downloads MovieLens 1M if the
default dataset is missing, and generates the data analysis, 60-setting validation
search, all nine baseline results, and figures (Part 1), followed by the Part 2
experiments. Add `--parts 1` or `--parts 2` to run only one part. Part 2 takes
roughly 30 minutes on a laptop CPU; most of it is the BPR validation search.

### Part 2

`scripts/run_part2.py` (also called by `run_all.py`) runs every Part 2 model through
the same pipeline in `recsys/experiment.py`: the Part 1 split and protocol, a
validation search with the same budget of 60 settings per model (smaller spaces are
searched completely), five seeds (0-4) for stochastic models, user-bootstrap and seed
intervals, paired Wilcoxon tests between all models (Holm-corrected), a breakdown by
user activity, item popularity and test period, and a 50 % / 20 % sparsity ablation.
Results go to `results/part2/` (`results-part2.csv` uses the workbook columns; details
in `results/part2/intermediate/`, figures in `results/part2/images/`).

Self-implemented algorithm: BPR-MF in `recsys/models/bpr.py` (numpy only).
Correctness checks: `uv run --locked python tests/test_part2.py`.

To add a model, append a `ModelSpec` in `model_specs()` of `scripts/run_part2.py`.
The model needs `fit(history, user_ids=..., catalogue=...)` (or `fit(history)`) and
`score_users(user_ids, catalogue)` or `score(user_id, item_ids)`.

Seeds: models 0-4, BPR random search 2026, bootstrap 42.

Final results are saved to `results/results.csv`, figures to `results/images/`,
and supporting report tables to `results/intermediate/`.

For an existing dataset, pass `--data-dir /path/to/ml-1m`, pointing to the folder
containing `ratings.dat`, `movies.dat` and `users.dat`. Use `--out-dir /path/to/results`
to change the output directory.

The [project report](report/Recommender_Systems_Report.pdf) ([Word version](report/Recommender_Systems_Report.docx)) and [results spreadsheet](results/Recommender_Systems_Results.xlsm) are included in this repository.
