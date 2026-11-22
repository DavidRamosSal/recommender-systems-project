# Recommender systems project

Requires uv and Python 3.12. If uv is not installed, follow the
[uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
Run commands from the repository root.

Reproduce every number, table and generated figure in the current Part 1 report:

```bash
uv run --locked scripts/run_all.py --data-dir data/ml-1m
```

This installs the dependencies pinned in `uv.lock`, downloads MovieLens 1M if the
default dataset is missing, and generates the data analysis, 60-setting validation
search, all nine baseline results, and figures.

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
