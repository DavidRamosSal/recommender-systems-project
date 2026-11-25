"""Part 3, Track A: MMR re-ranking of a base recommender (Lecture 5.1, slide 9).

Pipeline per user
1. candidates  the base model's top-N unseen items (N = CANDIDATES), in score
               order; already-rated items are excluded exactly as in Parts 1-2.
2. relevance   the base scores min-max normalised to [0, 1] within the user's
               candidate list, so that relevance and similarity share a scale.
               BPR scores are unbounded logits; without this step lambda would
               mean something different for every user.
3. MMR         greedy selection of k items:
                   argmax_i  lam * rel(i) - (1 - lam) * max_{j in S} sim(i, j)
               with sim = cosine on genre vectors and S = items already chosen.
               At the first step S is empty, so the max term is 0 (as on the
               slide). Ties go to the item ranked higher by the base model.

lam = 1 reproduces the base model's top-k exactly; lam = 0 ignores relevance
after the first pick. Item representation: 18-dimensional multi-hot genre
vector from movies.dat; item distance = 1 - cosine similarity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ranking import top_k_items

CANDIDATES = 100
GENRES = (
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
)
ITEM_DISTANCE = "1 - cosine similarity of 18-dimensional multi-hot genre vectors"


def genre_matrix(movies: pd.DataFrame, catalogue) -> np.ndarray:
    """Multi-hot genre vectors, one row per catalogue item (catalogue order)."""
    genres = movies.set_index("item_id")["genres"].reindex(catalogue)
    if genres.isna().any():
        raise ValueError("Every catalogue item needs a genre entry in movies.dat.")
    matrix = np.zeros((len(catalogue), len(GENRES)))
    column = {g: n for n, g in enumerate(GENRES)}
    for row, labels in enumerate(genres):
        for label in str(labels).split("|"):
            matrix[row, column[label]] = 1.0
    return matrix


def unit_rows(features: np.ndarray) -> np.ndarray:
    """L2-normalised rows (an all-zero row stays zero), so dot product = cosine."""
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return np.divide(features, norms, out=np.zeros_like(features, dtype=float), where=norms > 0)


def candidate_lists(scores: np.ndarray, seen: np.ndarray, n: int = CANDIDATES):
    """Top-n unseen catalogue indices per user and their base scores."""
    top = top_k_items(scores, seen, k=n)
    return top, np.take_along_axis(scores, top, axis=1)


def minmax(values: np.ndarray) -> np.ndarray:
    """Scale each row to [0, 1]; a constant row becomes all ones."""
    low = values.min(axis=1, keepdims=True)
    span = values.max(axis=1, keepdims=True) - low
    return np.divide(values - low, span, out=np.ones_like(values, dtype=float), where=span > 0)


def mmr(relevance: np.ndarray, features: np.ndarray, lam: float, k: int = 10) -> np.ndarray:
    """Greedy MMR, vectorised over users.

    relevance: users x N, larger = better, columns in base-model order.
    features:  users x N x d item vectors of the candidates (any scale;
               normalised here, so their dot products are cosines).
    Returns users x k positions into the candidate lists.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must lie in [0, 1].")
    n_users, n = relevance.shape
    if k > n:
        raise ValueError("Need at least k candidates per user.")
    unit = unit_rows(features)
    rows = np.arange(n_users)
    chosen = np.empty((n_users, k), dtype=np.int64)
    max_sim = np.zeros((n_users, n))
    available = np.ones((n_users, n), dtype=bool)
    for step in range(k):
        value = lam * relevance - (1.0 - lam) * max_sim
        value[~available] = -np.inf
        pick = value.argmax(axis=1)                  # first maximum = higher base rank
        chosen[:, step] = pick
        available[rows, pick] = False
        similarity = np.einsum("und,ud->un", unit, unit[rows, pick])
        np.maximum(max_sim, similarity, out=max_sim)
    return chosen


def rerank(candidates: np.ndarray, relevance: np.ndarray, item_features: np.ndarray,
           lam: float, k: int = 10) -> np.ndarray:
    """MMR top-k as catalogue indices (users x k)."""
    positions = mmr(relevance, item_features[candidates], lam, k)
    return np.take_along_axis(candidates, positions, axis=1)
