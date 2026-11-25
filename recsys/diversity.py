"""Beyond-accuracy metrics for Part 3, Track A (Lecture 5.1, slide 7).

Every metric receives the recommended lists ``top`` (users x k catalogue
indices) and a ``ListContext``. Register it in one of the two dictionaries at
the bottom and scripts/run_part3.py evaluates it at every lambda, adds
intervals (per-user metrics) or seed intervals (system metrics), writes it to
the result tables and draws it in figures - no other change is needed.

* PER_USER  metric(top, ctx) -> array with one value per user (row of top);
            reported as the mean over users, e.g. intra-list diversity,
            calibration error.
* SYSTEM    metric(top, ctx) -> one float for all lists together,
            e.g. catalogue coverage, Gini.

Template column names (Part 3A tab): "ild", "coverage", "gini", "calibration".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .rerank import unit_rows


@dataclass
class ListContext:
    """What a metric may use besides the lists themselves."""
    features: np.ndarray        # catalogue x 18 multi-hot genre vectors
    catalogue: np.ndarray       # item IDs, column order of every matrix
    users: np.ndarray           # user IDs, row order of top
    history: pd.DataFrame       # ratings the base model was trained on
    seen: np.ndarray            # users x catalogue, 1 = rated in history


def intra_list_diversity(top: np.ndarray, ctx: ListContext) -> np.ndarray:
    """Mean pairwise distance (1 - cosine on genre vectors) within each list."""
    unit = unit_rows(ctx.features)[top]                     # users x k x d
    k = top.shape[1]
    similarity = np.einsum("ukd,ujd->ukj", unit, unit)
    pairs = np.triu_indices(k, 1)
    return 1.0 - similarity[:, pairs[0], pairs[1]].mean(axis=1)


def catalogue_coverage(top: np.ndarray, ctx: ListContext) -> float:
    """Share of the catalogue recommended to at least one user (Lecture 3.1, slide 9)."""
    return np.unique(top).size / len(ctx.catalogue)


PER_USER = {
    "ild": intra_list_diversity,
    # "calibration": calibration_error,   # David
}
SYSTEM = {
    "coverage": catalogue_coverage,
    # "gini": gini,                        # David
}
