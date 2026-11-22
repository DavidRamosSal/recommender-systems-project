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

## Markov comparison

Tune Markov order and compare it with item-kNN and Most Popular on the Part 1
splits.

```bash
uv run --locked scripts/run_markov.py --data-dir data/ml-1m
```

Each model gets a 60-setting validation grid. Markov tries orders 1–60 and
item-kNN uses its Part 1 grid. Most Popular tries all-time counts and 29 recent
windows (weekly from 7 to 182 days, plus 30, 90, and 180 days), each with all
ratings or only ratings of at least 4. Selection uses NDCG@10, with ties
favoring the smaller Markov order. The Markov table records how many users
match a learned state at each order. Pass `--order 2` for a fixed-order comparison
or `--out-dir` to change where the CSVs are written.

Test results go to `results/markov/`. The paired table gives Markov-minus-baseline
NDCG@10 differences, 95% user-bootstrap intervals, and two-sided paired
randomization p-values at each training fraction. Most Popular includes the
validation winner and the all-time positive-rating reference.

The disaggregated table uses median pre-test rating counts to divide evaluated
users and catalogue items into two groups each, with median ties assigned to the
lower-count group. Item-group metrics count hits for that group at their original
top-10 ranks and average over users with a relevant item in the group. Item-group
coverage uses the size of that group's catalogue.

The sparsity table ranks models by test NDCG@10 after refitting the selected
settings on 100%, 50%, and 20% of pre-test ratings. The smaller samples are nested
and shared across models (seed 42). All runs use the same test users, relevant
items, catalogue, and seen-item exclusions.
