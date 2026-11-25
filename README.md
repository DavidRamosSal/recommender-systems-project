# Recommender systems project

Requires uv and Python 3.12. If uv is not installed, follow the
[uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
Run commands from the repository root.

Regenerate the Part 1, Part 2 and Part 3 result tables and figures.

```bash
uv run --locked scripts/run_all.py --data-dir data/ml-1m
```

This installs the dependencies pinned in `uv.lock`, downloads MovieLens 1M if
the default dataset is missing, and evaluates the saved models. It reuses the
recorded validation and sparsity results. Add `--parts 1`, `--parts 2` or `--parts 3` to run one part.

Add `--retrain` to repeat model selection, training, and the sparsity experiments.

```bash
uv run --locked scripts/run_all.py --retrain
```

Mult-VAE runs on CPU. Final BPR and Mult-VAE fits and sparsity subsamples use
seeds 0–4. BPR validation uses search seed 2026 and fit seed 0. Mult-VAE
validation and the user bootstrap use seed 42. The Part 3 validation BPR uses seed 0.

For an existing dataset, pass `--data-dir /path/to/ml-1m`, pointing to the folder
containing `ratings.dat`, `movies.dat` and `users.dat`. Use `--out-dir /path/to/results`
to change the output directory.

### Part 3 (Track A: diversity and popularity bias)

`scripts/run_part3.py` (also called by `run_all.py`) re-ranks our BPR-MF with MMR
(`recsys/rerank.py`, Lecture 5.1 slide 9). Candidates are the top 100 unseen
items of BPR, BPR scores are min-max normalised per user, and the item distance is
1 − cosine similarity of the 18-dimensional multi-hot genre vectors. λ is swept
from 1.0 (= unmodified BPR) to 0.0 in steps of 0.05. λ is chosen on validation
(BPR fitted on the training period, seed 0): the smallest λ whose NDCG@10 is at
most 5 % below λ = 1. Test results use the five Part 2 BPR fits (seeds 0–4), 95 %
user-bootstrap intervals (seed 42) and paired Wilcoxon tests against λ = 1 (Holm).
Beyond-accuracy metrics live in `recsys/diversity.py`. Results go to `results/part3/`
(`results-part3.csv` uses the Part 3A workbook columns). Takes about 2 minutes.

Correctness checks, including the worked MMR example of Lecture 5.1 (slides 8–9):
`uv run --locked python tests/test_part3.py`.

Overall results are in [results.csv](results/results.csv) and
[results-part2.csv](results/part2/results-part2.csv) and
[results-part3.csv](results/part3/results-part3.csv). Figures are saved to
`results/images/`, `results/part2/images/` and `results/part3/images/`, with detailed tables under `results/`.

The [project report](report/Recommender_Systems_Report.pdf) ([Word version](report/Recommender_Systems_Report.docx))
and [results spreadsheet](results/Recommender_Systems_Results.xlsm) are included in this repository.
