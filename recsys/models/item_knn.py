from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray


EPS = 1e-12


def _coordinates(ratings, user_ids, item_ids):
    users = pd.Index(user_ids).get_indexer(ratings["user_id"])
    items = pd.Index(item_ids).get_indexer(ratings["item_id"])
    if (users < 0).any() or (items < 0).any():
        raise ValueError("Matrix axes must contain every user and item in ratings.")
    return users, items


def interaction_matrix(
    ratings: pd.DataFrame,
    user_ids,
    item_ids,
    *,
    positives_only: bool = True,
    positive_threshold: int = 4,
) -> NDArray[np.float32]:
    """Dense binary matrix with explicit axes; no notebook globals are needed."""
    selected = ratings[ratings["rating"] >= positive_threshold] if positives_only else ratings
    matrix = np.zeros((len(user_ids), len(item_ids)), dtype=np.float32)
    matrix[_coordinates(selected, user_ids, item_ids)] = 1.0
    return matrix


def rating_matrix(ratings: pd.DataFrame, user_ids, item_ids) -> NDArray[np.float32]:
    """Dense explicit ratings, with zero for unobserved user/item pairs."""
    matrix = np.zeros((len(user_ids), len(item_ids)), dtype=np.float32)
    matrix[_coordinates(ratings, user_ids, item_ids)] = ratings["rating"].to_numpy()
    return matrix


def similarity_matrix(
    kind: str,
    binary: NDArray[np.float32],
    explicit: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Item similarities, with zero diagonal and negative similarities clipped.

    Cosine and Jaccard use the chosen binary profile. Adjusted cosine always
    centres all explicit training ratings by each user's mean, as in the notebook.
    """
    if kind in ("cosine", "jaccard"):
        co = binary.T @ binary
        counts = np.diag(co).copy()
        if kind == "cosine":
            similarity = co / np.sqrt(np.outer(counts, counts) + EPS)
        else:
            similarity = co / (counts[:, None] + counts[None, :] - co + EPS)
    elif kind == "adjusted_cosine":
        if explicit is None:
            raise ValueError("Adjusted cosine needs explicit ratings.")
        observed = explicit > 0
        user_mean = explicit.sum(axis=1) / np.maximum(observed.sum(axis=1), 1)
        centred = np.where(observed, explicit - user_mean[:, None], 0.0).astype(np.float32)
        co = centred.T @ centred
        norm = np.sqrt(np.diag(co))
        similarity = co / (np.outer(norm, norm) + EPS)
    else:
        raise ValueError(f"Unknown similarity: {kind}")
    similarity = np.maximum(similarity, 0).astype(np.float32)
    np.fill_diagonal(similarity, 0.0)
    return similarity


def keep_top_k(similarity: NDArray[np.float32], k: int) -> NDArray[np.float32]:
    """Keep each target item's k strongest neighbours (rows, not columns)."""
    if k < 0:
        raise ValueError("k must be nonnegative.")
    if k >= similarity.shape[0] - 1:
        return similarity
    neighbours = np.argpartition(-similarity, k, axis=1)[:, :k]
    rows = np.arange(similarity.shape[0])[:, None]
    pruned = np.zeros_like(similarity)
    pruned[rows, neighbours] = similarity[rows, neighbours]
    return pruned


def item_knn_scores(profiles, neighbours) -> NDArray[np.float32]:
    """Sum sim(target, profile item) over the target item's neighbourhood."""
    return profiles @ neighbours.T


@dataclass
class ItemKNN:
    """Configurable kNN implementing the repository's Recommender protocol.

    ``fit`` accepts fixed user/catalogue axes to retain cold users/items without
    learning from held-out ratings. Unknown IDs score zero. Seen-item removal
    belongs to recsys.ranking, just as it does for MostPopular.
    """

    similarity: str = "cosine"
    k: int = 500
    feedback: str = "positives"
    positive_threshold: int = 4
    _profiles: NDArray[np.float32] | None = field(init=False, repr=False, default=None)
    _neighbours: NDArray[np.float32] | None = field(init=False, repr=False, default=None)
    _users: pd.Index | None = field(init=False, repr=False, default=None)
    _items: pd.Index | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.similarity not in ("cosine", "jaccard", "adjusted_cosine"):
            raise ValueError("similarity must be cosine, jaccard or adjusted_cosine.")
        if self.feedback not in ("positives", "all_ratings"):
            raise ValueError("feedback must be positives or all_ratings.")
        if self.k < 1:
            raise ValueError("k must be positive.")

    @property
    def name(self) -> str:
        feedback = f"pos{self.positive_threshold}" if self.feedback == "positives" else "allratings"
        return f"itemknn_{self.similarity}_{feedback}_k{self.k}"

    def fit(self, train: pd.DataFrame, *, user_ids=None, catalogue=None) -> "ItemKNN":
        self._users = pd.Index(np.sort(train["user_id"].unique()) if user_ids is None else user_ids)
        self._items = pd.Index(np.sort(train["item_id"].unique()) if catalogue is None else catalogue)
        if self._users.empty or self._items.empty:
            raise ValueError("ItemKNN needs nonempty user and item axes.")
        self._profiles = interaction_matrix(
            train, self._users, self._items,
            positives_only=self.feedback == "positives",
            positive_threshold=self.positive_threshold,
        )
        explicit = (
            rating_matrix(train, self._users, self._items)
            if self.similarity == "adjusted_cosine" else None
        )
        self._neighbours = keep_top_k(
            similarity_matrix(self.similarity, self._profiles, explicit), self.k,
        )
        return self

    def score_users(self, user_ids, item_ids=None) -> NDArray[np.float32]:
        """Score a batch, retaining the notebook's float32 matrix arithmetic."""
        if self._profiles is None:
            raise RuntimeError("Call fit() before score().")
        rows = self._users.get_indexer(user_ids)
        profiles = np.zeros((len(rows), len(self._items)), dtype=np.float32)
        known_users = rows >= 0
        profiles[known_users] = self._profiles[rows[known_users]]
        scores = item_knn_scores(profiles, self._neighbours)
        if item_ids is None:
            return scores
        columns = self._items.get_indexer(item_ids)
        result = np.zeros((len(rows), len(columns)), dtype=np.float32)
        known_items = columns >= 0
        result[:, known_items] = scores[:, columns[known_items]]
        return result

    def score(self, user_id: int, item_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        """Return scores in the requested original item-ID order."""
        return self.score_users([user_id], item_ids)[0].astype(np.float64)
