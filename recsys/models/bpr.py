"""BPR-MF, implemented from scratch (Part 2 self-implementation requirement).

Reference: Rendle et al. (2009), "BPR: Bayesian Personalized Ranking from
Implicit Feedback"; Lecture 2.1, slides 11-12. No library implementation was
used or consulted; only numpy.

Model
    x_ui  = b_i + p_u . q_i                  score of item i for user u
    x_uij = x_ui - x_uj                      i observed positive, j unobserved

Objective (maximised), summed over sampled triples (u, i, j):
    ln sigma(x_uij) - reg/2 (|p_u|^2 + |q_i|^2 + |q_j|^2) - reg_bias/2 (b_i^2 + b_j^2)

With g = sigma(-x_uij) = d ln sigma(x) / dx the gradients are
    d/dp_u = g (q_i - q_j) - reg p_u
    d/dq_i = g p_u         - reg q_i
    d/dq_j = -g p_u        - reg q_j
    d/db_i = g             - reg_bias b_i
    d/db_j = -g            - reg_bias b_j

Training is stochastic gradient ascent with the paper's bootstrap sampling:
(u, i) is drawn uniformly with replacement from all positive training pairs,
j uniformly from the items u has *not rated at all* (rejection sampling). One
epoch draws as many triples as there are positives. A rating below the
threshold is therefore neither a positive nor a negative.

Deviation from the paper (stated in the report): triples are processed in
mini-batches and their updates are summed (np.add.at), so all triples in a
batch see the same parameters. With 2,048 triples per batch and 6,040 users x
3,706 items collisions are rare; it makes an epoch take ~1 s instead of
minutes in pure Python.

Implements the repository's Recommender protocol (score(user_id, item_ids))
plus score_users() for batch scoring, like ItemKNN. Pass the full user and
catalogue axes to fit(): users without training positives then keep their
small random initial vector (their ranking is driven by the learned item
bias), and catalogue items without positives are only ever pushed down as
negatives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable logistic function."""
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class BPR:
    def __init__(
        self,
        factors: int = 64,
        lr: float = 0.05,
        reg: float = 0.01,
        reg_bias: float = 0.0,
        epochs: int = 20,
        feedback: str = "positives",
        positive_threshold: int = 4,
        batch_size: int = 2048,
        init_std: float = 0.1,
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        if feedback not in ("positives", "all_ratings"):
            raise ValueError("feedback must be positives or all_ratings.")
        self.factors = int(factors)
        self.lr = float(lr)
        self.reg = float(reg)
        self.reg_bias = float(reg_bias)
        self.epochs = int(epochs)
        self.feedback = feedback
        self.positive_threshold = int(positive_threshold)
        self.batch_size = int(batch_size)
        self.init_std = float(init_std)
        self.seed = int(seed)
        self.verbose = verbose
        self.loss_history: list[float] = []
        self._users: pd.Index | None = None
        self._items: pd.Index | None = None

    @property
    def name(self) -> str:
        fb = f"pos{self.positive_threshold}" if self.feedback == "positives" else "allratings"
        return (f"bpr_{fb}_d{self.factors}_lr{self.lr:g}_reg{self.reg:g}"
                f"_ep{self.epochs}")

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame, *, user_ids=None, catalogue=None) -> "BPR":
        """train: columns user_id, item_id, rating, timestamp (original IDs).

        user_ids / catalogue fix the matrix axes (as in ItemKNN) so users and
        items without training data still get a score.
        """
        rng = np.random.default_rng(self.seed)
        self._users = pd.Index(np.sort(train["user_id"].unique()) if user_ids is None else user_ids)
        self._items = pd.Index(np.sort(train["item_id"].unique()) if catalogue is None else catalogue)
        n_users, n_items = len(self._users), len(self._items)

        u_all = self._users.get_indexer(train["user_id"]).astype(np.int64)
        i_all = self._items.get_indexer(train["item_id"]).astype(np.int64)
        if (u_all < 0).any() or (i_all < 0).any():
            raise ValueError("user_ids / catalogue must contain every training ID.")
        positive = (train["rating"].to_numpy() >= self.positive_threshold
                    if self.feedback == "positives" else np.ones(len(train), bool))
        pos_u, pos_i = u_all[positive], i_all[positive]
        n_pos = len(pos_u)
        # sorted keys of every rated pair: O(log n) membership test for negatives
        rated = np.unique(u_all * n_items + i_all)

        self.P = rng.normal(0.0, self.init_std, (n_users, self.factors))
        self.Q = rng.normal(0.0, self.init_std, (n_items, self.factors))
        self.b = np.zeros(n_items)

        lr, reg, reg_b = self.lr, self.reg, self.reg_bias
        self.loss_history = []
        for epoch in range(self.epochs):
            total = 0.0
            for start in range(0, n_pos, self.batch_size):
                m = min(self.batch_size, n_pos - start)
                draw = rng.integers(0, n_pos, m)            # bootstrap sampling
                u, i = pos_u[draw], pos_i[draw]
                j = sample_negatives(u, rated, n_items, rng)

                Pu, Qi, Qj = self.P[u], self.Q[i], self.Q[j]
                x = self.b[i] - self.b[j] + np.einsum("bd,bd->b", Pu, Qi - Qj)
                g = sigmoid(-x)
                total += float(np.logaddexp(0.0, -x).sum())  # sum of -ln sigma(x)

                np.add.at(self.P, u, lr * (g[:, None] * (Qi - Qj) - reg * Pu))
                np.add.at(self.Q, i, lr * (g[:, None] * Pu - reg * Qi))
                np.add.at(self.Q, j, lr * (-g[:, None] * Pu - reg * Qj))
                np.add.at(self.b, i, lr * (g - reg_b * self.b[i]))
                np.add.at(self.b, j, lr * (-g - reg_b * self.b[j]))
            self.loss_history.append(total / max(n_pos, 1))
            if self.verbose:
                print(f"  epoch {epoch + 1}/{self.epochs}: loss {self.loss_history[-1]:.4f}",
                      flush=True)
        return self

    # ---------------------------------------------------------------- score
    def score_users(self, user_ids, item_ids=None) -> NDArray[np.float64]:
        """Scores for a batch of users, columns in catalogue order (or item_ids)."""
        if self._users is None:
            raise RuntimeError("Call fit() before score().")
        rows = self._users.get_indexer(user_ids)
        if (rows < 0).any():
            raise ValueError("Unknown user ID; pass user_ids= to fit().")
        scores = self.P[rows] @ self.Q.T + self.b[None, :]
        if item_ids is None:
            return scores
        columns = self._items.get_indexer(item_ids)
        result = np.full((len(rows), len(columns)), -np.inf)
        known = columns >= 0
        result[:, known] = scores[:, columns[known]]
        return result

    def score(self, user_id: int, item_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        return self.score_users([user_id], item_ids)[0]


def sample_negatives(u, rated, n_items, rng, max_rounds: int = 100):
    """Uniform j per user u with (u, j) not rated, by rejection sampling.

    rated: sorted array of keys u * n_items + i of all rated pairs.
    """
    j = rng.integers(0, n_items, len(u))
    for _ in range(max_rounds):
        keys = u * n_items + j
        pos = np.searchsorted(rated, keys)
        pos[pos == len(rated)] = 0
        clash = rated[pos] == keys
        if not clash.any():
            return j
        j[clash] = rng.integers(0, n_items, int(clash.sum()))
    raise RuntimeError("Could not sample unrated negatives (user rated almost everything?).")
