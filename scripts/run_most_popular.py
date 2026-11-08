"""Evaluate the most-popular variants on validation and test.

    uv run --locked scripts/run_most_popular.py --data-dir data/ml-1m

Protocol (same as the item-kNN baseline, see notebooks/part-1-analysis_split_knn.ipynb):
* split by date: train < 2001-01-01 <= validation < 2001-07-01 <= test
* relevant = rating >= 4; evaluated users have training history and at least
  one relevant item in the evaluation window
* candidates = all 3,706 rated movies minus the user's already rated ones
* validation: fit on train; test: refit on train + validation

Most-popular is deterministic, so there is no seed for the model. The seed
below is only used for the bootstrap confidence intervals.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data import load_movielens
from recsys.evaluation import evaluate
from recsys.models.most_popular import MostPopular
from recsys.ranking import top_k

VAL_START = "2001-01-01"
TEST_START = "2001-07-01"
POSITIVE_THRESHOLD = 4
RECENT_WINDOWS_DAYS = (30, 90, 180)
BOOTSTRAP_SEED = 42
BOOTSTRAP_SAMPLES = 1000


def to_epoch(date: str) -> int:
    return int(pd.Timestamp(date, tz="UTC").timestamp())


def split_by_date(ratings: pd.DataFrame):
    val_start, test_start = to_epoch(VAL_START), to_epoch(TEST_START)
    t = ratings["timestamp"]
    train = ratings[t < val_start]
    val = ratings[(t >= val_start) & (t < test_start)]
    test = ratings[t >= test_start]
    return train, val, test


def per_user_ndcg(labels: pd.DataFrame, predictions: pd.DataFrame) -> np.ndarray:
    """NDCG@10 per user, same definition as recsys.evaluation.evaluate."""
    discounts = 1 / np.log2(np.arange(2, 12))
    ranked = predictions.sort_values("rank").groupby("user_id")["item_id"].agg(list)
    values = []
    for user_id, rows in labels.groupby("user_id"):
        expected = rows["item_id"].unique()
        top = ranked.get(user_id, [])[:10]
        hits = np.isin(top, expected)
        values.append((hits * discounts[:len(top)]).sum()
                      / discounts[:min(10, len(expected))].sum())
    return np.array(values)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """95 % percentile bootstrap CI of the mean, resampling users."""
    idx = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run(model, history, window, catalogue, rng):
    model.fit(history)
    labels = window[window["rating"] >= POSITIVE_THRESHOLD]
    labels = labels[labels["user_id"].isin(history["user_id"])][["user_id", "item_id"]]
    predictions = top_k(model, labels["user_id"].unique(), catalogue, history, k=10)
    metrics = evaluate(labels, predictions, catalogue_size=len(catalogue))
    low, high = bootstrap_ci(per_user_ndcg(labels, predictions), rng)
    return {"users": labels["user_id"].nunique(), **metrics,
            "ndcg@10_ci_low": low, "ndcg@10_ci_high": high}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/ml-1m")
    parser.add_argument("--out", default="results/most_popular.csv")
    args = parser.parse_args()

    ratings = load_movielens(args.data_dir).ratings
    train, val, test = split_by_date(ratings)
    history_for_test = pd.concat([train, val])
    catalogue = np.sort(ratings["item_id"].unique())   # 3,706 rated movies
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    models = [MostPopular("all", "all"), MostPopular("all", "positive")]
    models += [MostPopular("recent", feedback, days)
               for days in RECENT_WINDOWS_DAYS for feedback in ("all", "positive")]

    rows = []
    for model in models:
        rows.append({"model": model.name, "split": "validation",
                     **run(model, train, val, catalogue, rng)})
        rows.append({"model": model.name, "split": "test",
                     **run(model, history_for_test, test, catalogue, rng)})

    results = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    with pd.option_context("display.width", 220, "display.precision", 4):
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
