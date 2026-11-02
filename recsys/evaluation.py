import numpy as np
import pandas as pd
from numpy.typing import NDArray

from recsys.models import Recommender


def evaluate(
    model: Recommender,
    test_ratings: pd.DataFrame,
    train_ratings: pd.DataFrame,
    catalogue: NDArray[np.int64],
) -> dict[str, float]:
    """Evaluate top-10 rankings over the catalogue, excluding training items.

    Every row in `test_ratings` counts as an expected recommendation; rating
    values are ignored. `train_ratings` contains interactions to exclude.
    Both need user_id and item_id columns and must be disjoint subsets of the
    original MovieLens ratings. `catalogue` contains every movie ID once.
    These inputs have unique user-item pairs and at least 10 unseen items per
    user, so no deduplication or short-list handling is needed.

    Precision, recall and binary NDCG are averaged equally over users present
    in `test_ratings`. Users absent from that table are excluded. Ties favor
    smaller item IDs. Coverage is distinct recommended items / catalogue size.
    """
    if test_ratings.empty or catalogue.size == 0:
        raise ValueError("Evaluation needs test ratings and a catalogue.")

    history = train_ratings.groupby("user_id")["item_id"].agg(list).to_dict()
    discounts = 1 / np.log2(np.arange(2, 12))
    per_user = []
    recommended = set()

    for user_id, ratings in test_ratings.groupby("user_id"):
        expected = ratings["item_id"].to_numpy()
        candidates = catalogue[~np.isin(catalogue, history.get(user_id, []))]

        scores = np.asarray(model.score(int(user_id), candidates), dtype=float)
        if scores.shape != candidates.shape or not np.isfinite(scores).all():
            raise ValueError("score() must return one finite score per candidate.")

        top = candidates[np.lexsort((candidates, -scores))[:10]]
        hits = np.isin(top, expected)
        dcg = np.sum(hits * discounts)
        ideal_dcg = discounts[:min(10, len(expected))].sum()
        per_user.append((hits.sum() / 10, hits.sum() / len(expected), dcg / ideal_dcg))
        recommended.update(top.tolist())

    precision, recall, ndcg = np.mean(per_user, axis=0)
    return {
        "precision@10": float(precision),
        "recall@10": float(recall),
        "ndcg@10": float(ndcg),
        "catalogue_coverage": len(recommended) / len(catalogue),
    }
