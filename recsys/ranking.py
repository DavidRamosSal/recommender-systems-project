"""Turn any Recommender into the predictions table that evaluate() expects."""

import numpy as np
import pandas as pd

from .models import Recommender


def top_k(
    model: Recommender,
    user_ids,
    candidate_items,
    seen: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    """Rank all candidate items per user and keep the best k unseen ones.

    model: anything with score(user_id, item_ids).
    user_ids: users to produce rankings for.
    candidate_items: the catalogue to rank against (full catalogue).
    seen: DataFrame with columns user_id, item_id; these items are excluded
        for that user (normally the model's training interactions).
    Returns a DataFrame with columns user_id, item_id, rank (1 = best).
    Ties are broken by lower item_id (stable sort on sorted candidates).
    """
    candidates = np.sort(np.asarray(candidate_items, dtype=np.int64))
    seen_by_user = {
        user_id: group.to_numpy()
        for user_id, group in seen.groupby("user_id")["item_id"]
    }

    rows_user, rows_item = [], []
    for user_id in user_ids:
        scores = model.score(user_id, candidates).astype("float64", copy=True)
        user_seen = seen_by_user.get(user_id)
        if user_seen is not None:
            scores[np.isin(candidates, user_seen)] = -np.inf
        order = np.argsort(-scores, kind="stable")[:k]
        order = order[np.isfinite(scores[order])]
        rows_user.append(np.full(len(order), user_id, dtype=np.int64))
        rows_item.append(candidates[order])

    users = np.concatenate(rows_user) if rows_user else np.array([], dtype=np.int64)
    items = np.concatenate(rows_item) if rows_item else np.array([], dtype=np.int64)
    predictions = pd.DataFrame({"user_id": users, "item_id": items})
    predictions["rank"] = predictions.groupby("user_id").cumcount() + 1
    return predictions
