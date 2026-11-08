"""Validation-only item-kNN search using the same arithmetic as the notebook."""

import numpy as np
import pandas as pd

from .evaluation import evaluation_users, ranking_metrics
from .models.item_knn import (
    interaction_matrix, item_knn_scores, keep_top_k, rating_matrix, similarity_matrix,
)
from .ranking import top_k_items


SIMILARITIES = ("cosine", "jaccard", "adjusted_cosine")
FEEDBACKS = ("positives", "all_ratings")
K_GRID = (5, 10, 20, 50, 100, 200, 300, 500, 1000, 2000)


def tune_itemknn(train, validation, user_ids, catalogue, *, progress=print):
    """Evaluate all 60 settings on validation; test data is never accepted."""
    users = evaluation_users(train, validation)
    indices = pd.Index(user_ids).get_indexer(users)
    relevant = interaction_matrix(validation, user_ids, catalogue)[indices]
    seen = interaction_matrix(train, user_ids, catalogue, positives_only=False)[indices]
    explicit = rating_matrix(train, user_ids, catalogue)
    rows = []
    for kind in SIMILARITIES:
        for feedback in FEEDBACKS:
            profiles = interaction_matrix(
                train, user_ids, catalogue, positives_only=feedback == "positives",
            )
            similarity = similarity_matrix(kind, profiles, explicit)
            for k in K_GRID:
                scores = item_knn_scores(profiles[indices], keep_top_k(similarity, k))
                top = top_k_items(scores, seen)
                metrics = ranking_metrics(top, relevant)
                rows.append({
                    "similarity": kind, "feedback": feedback, "k": k,
                    "precision@10": metrics["precision"].mean(),
                    "recall@10": metrics["recall"].mean(),
                    "ndcg@10": metrics["ndcg"].mean(),
                    "coverage": np.unique(top).size / len(catalogue),
                })
            if progress is not None:
                progress(f"Tuning: {kind}, {feedback} ({len(rows)}/60 settings)", flush=True)
    return pd.DataFrame(rows)


def selected_settings(grid):
    """Choose the first maximum, preserving the notebook's search order."""
    best = grid.loc[grid["ndcg@10"].idxmax()]
    return {"similarity": best["similarity"], "feedback": best["feedback"], "k": int(best["k"])}
