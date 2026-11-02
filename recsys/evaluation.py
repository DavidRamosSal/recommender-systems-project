import numpy as np
import pandas as pd


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
