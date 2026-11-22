# Recommender systems project

Requires uv and Python 3.12. If uv is not installed, follow the
[uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
Run commands from the repository root.

Evaluate the saved MovieLens models and regenerate the result tables and figures.

```bash
uv run --locked scripts/run_all.py --data-dir data/ml-1m
```

This installs the dependencies pinned in `uv.lock` and downloads MovieLens 1M if
the default dataset is missing. The default run loads the saved weights, scores
the evaluation users again, and reuses the recorded validation searches, sparsity
ablation and supplementary Markov tables. Add `--parts 1` or `--parts 2` to run
one part.

To rerun the validation searches, retrain the models and regenerate all experiments,
including the ablations, use

```bash
uv run --locked scripts/run_all.py --retrain
```

`--retrain` writes weights and results under `--out-dir` (default `results/`).
A fresh output directory uses the repository's checkpoints for evaluation.
An output directory containing its own checkpoints uses those. The runner checks
the ratings against the saved training run. For a different dataset, pass
`--retrain --data-dir /path/to/ml-1m --out-dir /path/to/new-results`.

### Where to find a model's results

Each model has a folder directly under `results/`.

| Folder | Model |
| --- | --- |
| `mostpop/` | Most Popular |
| `itemknn/` | Item-based kNN |
| `bpr/` | Bayesian personalized ranking |
| `markov/` | Markov chain |
| `multvae/` | Multinomial variational autoencoder |

Each folder contains `weights/`, `tuning.csv`, `selected.json`, `test.csv`,
`disaggregated.csv`, and `ablation.csv`. For example,
`results/bpr/ablation.csv` gives BPR's results at 100%, 50%, and 20% of training
data. BPR and Mult-VAE save one final checkpoint per seed (0–4). The two
baseline folders also contain their Part 1 validation and test fits. Both
parts use the same item-kNN checkpoint when they select the same settings.
Each NumPy checkpoint stores the fitted parameters and MovieLens IDs needed
for scoring. Mult-VAE trains and scores on CPU.

### Part 2

`scripts/run_part2.py` compares Most Popular, item-kNN, BPR-MF, Markov chain and
Mult-VAE on the Part 1 split. Markov uses orders 1, 2 and 3. Each other model
gets 60 validation settings. The popularity windows match the Markov comparison
below. Mult-VAE uses the notebook's reference configuration and 59 distinct
random configurations. Each
trial trains for up to 100 epochs, stopping after 10 epochs without improvement
in validation NDCG@10. The final fits use the winning trial's best epoch.
The CPU run matches the validation split and device used in the Mult-VAE notebook.

The pipeline refits on train + validation, using seeds 0–4 for BPR and Mult-VAE.
It reports user-bootstrap and seed intervals, paired Wilcoxon tests across
models (Holm-corrected), metrics by user activity, item popularity and test
period, and rankings with 100%, 50% and 20% of training interactions.
The five-model table is `results/part2/results-part2.csv`. Cross-model
statistics are in `results/part2/intermediate/` and comparison plots are in
`results/part2/images/`. The BPR training-loss plot is in `results/bpr/`.

Self-implemented algorithm: BPR-MF in `recsys/models/bpr.py` (numpy only).
Correctness checks: `uv run --locked python tests/test_part2.py`.

To add a model, append a `ModelSpec` in `model_specs()` of `scripts/run_part2.py`.
The model needs `fit(history, user_ids=..., catalogue=...)` (or `fit(history)`) and
`score_users(user_ids, catalogue)` or `score(user_id, item_ids)`.

BPR samples configurations with seed 2026 and fits validation trials with seed 0.
Mult-VAE uses seed 42 for both its search and validation fits. The user bootstrap
also uses seed 42.

The cumulative [results.csv](results/results.csv) contains the nine Part 1
baseline variants and the five overall Part 2 evaluations. Part 2 rows are
identified in the existing `Notes` column. Detailed Part 1
scores are in each baseline folder's `part1-results.csv`. Dataset summaries
and shared figures are in `results/intermediate/` and `results/images/`.

For an existing dataset, pass `--data-dir /path/to/ml-1m`, pointing to the folder
containing `ratings.dat`, `movies.dat` and `users.dat`. Use `--out-dir /path/to/results`
to change the output directory.

The [project report](report/Recommender_Systems_Report.pdf) ([Word version](report/Recommender_Systems_Report.docx)) and [results spreadsheet](results/Recommender_Systems_Results.xlsm) are included in this repository.

### Mult-VAE notebook and pipeline

The [original notebook](notebooks/part-2multvae.ipynb) supplies the 60 CPU
validation trials and the five final Mult-VAE checkpoints (seeds 0–4). The
winning configuration reached validation NDCG@10 of 0.1788 at epoch 21. The
pipeline stores those weights in its NumPy checkpoint format and uses the
[winning validation curve](results/multvae/validation-curve.csv) in Figure 4.
On the 777 test users, the notebook and pipeline checkpoints both reach
NDCG@10 of 0.2067 and catalogue coverage of 16.5%. To verify the scores:

```bash
uv run --locked scripts/compare_multvae_notebook.py --data-dir data/ml-1m
```

The shared pipeline uses 1,000 bootstrap draws for all models. It also refits
Mult-VAE on CPU for the five shared sparsity subsamples at each reduced
fraction. The notebook's three-draw ablation is an earlier comparison.

## Markov comparison

Tune Markov order and compare it with item-kNN and Most Popular on the Part 1
splits.

```bash
uv run --locked scripts/run_markov.py --data-dir data/ml-1m
```

Markov tries histories of one, two and three liked movies. Longer sequences
rarely recur in this dataset. Item-kNN uses its 60-setting Part 1 grid.
Most Popular tries all-time counts and 29 recent
windows (weekly from 7 to 182 days, plus 30, 90, and 180 days), each with all
ratings or only ratings of at least 4. Selection uses NDCG@10, with ties
favoring the smaller Markov order. The Markov table records how many users
match a learned state at each order. Pass `--order 2` for a fixed-order comparison
or `--out-dir` to change where the CSVs are written.

The supplementary Markov comparison writes its order search and test tables
to `results/markov/`, the item-kNN grid to `results/itemknn/part1-tuning.csv`,
and the popularity grid to `results/mostpop/window-tuning.csv`.
The paired table gives Markov-minus-baseline
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
