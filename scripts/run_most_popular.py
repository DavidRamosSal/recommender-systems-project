"""Evaluate the most-popular variants on validation and test.

    uv run --locked scripts/run_most_popular.py --data-dir data/ml-1m

TODO: replace provisional_split() with the team's split once it is in
recsys/ (David). Everything else stays the same.
"""

import argparse
from pathlib import Path

import pandas as pd

from recsys.data import load_movielens
from recsys.evaluation import evaluate
from recsys.models.most_popular import MostPopular
from recsys.ranking import top_k

POSITIVE_THRESHOLD = 4
RECENT_WINDOWS_DAYS = (30, 90, 180)


def provisional_split(ratings: pd.DataFrame):
    """Global temporal 80/10/10 cut on timestamp. Placeholder only."""
    t_val, t_test = ratings["timestamp"].quantile([0.8, 0.9])
    train = ratings[ratings["timestamp"] <= t_val]
    val = ratings[(ratings["timestamp"] > t_val) & (ratings["timestamp"] <= t_test)]
    test = ratings[ratings["timestamp"] > t_test]
    return train, val, test


def run(model: MostPopular, fit_on: pd.DataFrame, target: pd.DataFrame,
        catalogue: pd.Index) -> dict[str, float]:
    model.fit(fit_on)
    labels = target[target["rating"] >= POSITIVE_THRESHOLD]
    labels = labels[labels["user_id"].isin(fit_on["user_id"])][["user_id", "item_id"]]
    predictions = top_k(model, labels["user_id"].unique(), catalogue, fit_on, k=10)
    return {"users": labels["user_id"].nunique(),
            **evaluate(labels, predictions, catalogue_size=len(catalogue))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/ml-1m")
    parser.add_argument("--out", default="results/most_popular.csv")
    args = parser.parse_args()

    ratings = load_movielens(args.data_dir).ratings
    train, val, test = provisional_split(ratings)
    train_val = pd.concat([train, val])

    models = [MostPopular("all", "all"), MostPopular("all", "positive")]
    models += [MostPopular("recent", fb, days)
               for days in RECENT_WINDOWS_DAYS for fb in ("all", "positive")]

    rows = []
    for model in models:
        # Validation: fit on train. Test: refit on train+validation, so the
        # popularity counts (and the recent window) end right before the test.
        rows.append({"model": model.name, "split": "validation",
                     **run(model, train, val, pd.Index(train["item_id"].unique()))})
        rows.append({"model": model.name, "split": "test",
                     **run(model, train_val, test, pd.Index(train_val["item_id"].unique()))})

    results = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
