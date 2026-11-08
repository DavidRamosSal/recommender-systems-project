import numpy as np
import pandas as pd


def ranking_metrics(top_items, relevant, k: int = 10) -> dict[str, np.ndarray]:
    """Per-user binary ranking metrics for dense catalogue-index predictions.

    ``relevant`` has one binary row per evaluated user and one column per
    catalogue item. Every evaluated user must have at least one relevant item.
    Retain the input dtype so notebook float32 results remain reproducible.
    """
    n_relevant = relevant.sum(axis=1)
    if k < 1 or top_items.shape[1] != k or np.any(n_relevant == 0):
        raise ValueError("Need k recommendations and positive labels for every user.")
    hits = np.take_along_axis(relevant, top_items, axis=1)
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * discount).sum(axis=1)
    ideal_dcg = np.cumsum(discount)[np.minimum(n_relevant, k).astype(int) - 1]
    return {
        "precision": hits.sum(axis=1) / k,
        "recall": hits.sum(axis=1) / n_relevant,
        "ndcg": dcg / ideal_dcg,
    }


def evaluation_users(history: pd.DataFrame, window: pd.DataFrame, *, positive_threshold: int = 4):
    """Original IDs with history and at least one relevant evaluation rating."""
    positives = window.loc[window["rating"] >= positive_threshold, "user_id"]
    return np.intersect1d(history["user_id"].unique(), positives.unique())


def per_user_ndcg(labels: pd.DataFrame, predictions: pd.DataFrame) -> np.ndarray:
    """Binary NDCG@10 per user, in the same user order as evaluate()."""
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


def bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, *, samples: int = 1000,
) -> tuple[float, float]:
    """95% percentile interval of a mean, resampling users."""
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    catalogue_size: int,
) -> dict[str, float]:
    """Return Precision@10, Recall@10, binary NDCG@10 and catalogue coverage.

    labels: DataFrame with int64 columns user_id, item_id; expected matches.
    predictions: DataFrame with int64 columns user_id, item_id, rank;
        rank 1 is best. Items must be unique per user, with ranks 1, 2, ...
    catalogue_size: Total number of available items, used only for coverage.

    Ranking metrics are averaged over users in labels. Missing predictions
    count as misses. Select positive labels and exclude seen items beforehand.
    """
    if labels.empty or catalogue_size <= 0:
        raise ValueError("Evaluation needs labels and a positive catalogue size.")

    rankings = (
        predictions[predictions["rank"] <= 10]
        .sort_values("rank")
        .groupby("user_id")["item_id"]
        .agg(list)
    )
    discounts = 1 / np.log2(np.arange(2, 12))
    per_user = []

    for user_id, rows in labels.groupby("user_id"):
        expected = rows["item_id"].unique()
        top = rankings.get(user_id, [])
        hits = np.isin(top, expected)
        dcg = np.sum(hits * discounts[:len(top)])
        ideal_dcg = discounts[:min(10, len(expected))].sum()
        per_user.append((hits.sum() / 10, hits.sum() / len(expected), dcg / ideal_dcg))

    precision, recall, ndcg = np.mean(per_user, axis=0)
    return {
        "precision@10": float(precision),
        "recall@10": float(recall),
        "ndcg@10": float(ndcg),
        "catalogue_coverage": float(
            predictions.loc[predictions["rank"] <= 10, "item_id"].nunique()
            / catalogue_size
        ),
    }
