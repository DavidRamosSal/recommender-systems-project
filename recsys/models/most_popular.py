"""Most-popular baseline (Part 1).

Every user gets the same item scores: how often the item occurs in the
training data. Two choices define a variant (project brief, Part 1):

* window   -- "all":    count over the whole training period
              "recent": count only interactions in the last ``recent_days``
                        days before the newest training timestamp
* feedback -- "all":      every rating counts as one interaction
              "positive": only ratings >= ``positive_threshold`` count

Implements the ``Recommender`` protocol: ``score(user_id, item_ids)``.
Seen items are not removed here; that happens when building the ranking
(see ``recsys.ranking.top_k``), the same way as for every other model.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

SECONDS_PER_DAY = 86_400


@dataclass
class MostPopular:
    window: str = "all"
    feedback: str = "all"
    recent_days: int = 90
    positive_threshold: int = 4
    _scores: pd.Series = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.window not in ("all", "recent"):
            raise ValueError("window must be 'all' or 'recent'")
        if self.feedback not in ("all", "positive"):
            raise ValueError("feedback must be 'all' or 'positive'")

    @property
    def name(self) -> str:
        window = f"recent{self.recent_days}d" if self.window == "recent" else "alltime"
        feedback = (
            f"pos{self.positive_threshold}" if self.feedback == "positive" else "allratings"
        )
        return f"mostpop_{window}_{feedback}"

    def fit(self, train: pd.DataFrame) -> "MostPopular":
        """train: DataFrame with columns user_id, item_id, rating, timestamp."""
        counted = train
        if self.window == "recent":
            cutoff = train["timestamp"].max() - self.recent_days * SECONDS_PER_DAY
            counted = counted[counted["timestamp"] > cutoff]
        if self.feedback == "positive":
            counted = counted[counted["rating"] >= self.positive_threshold]

        primary = counted.groupby("item_id").size().astype("float64")
        # Tie-break with the all-time count in train (scaled below 1), so the
        # ranking is fully defined even when the recent window is sparse.
        # Items with equal primary and secondary counts are ordered by the
        # stable sort in top_k (lower item_id first).
        secondary = train.groupby("item_id").size().astype("float64")
        secondary = secondary / (secondary.max() + 1)
        self._scores = primary.reindex(secondary.index, fill_value=0.0) + secondary
        return self

    def score(self, user_id: int, item_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        """Same scores for every user; items never seen in training score 0."""
        if self._scores is None:
            raise RuntimeError("Call fit() before score().")
        return self._scores.reindex(item_ids, fill_value=0.0).to_numpy()


def variants(recent_days: int = 90, positive_threshold: int = 4) -> list[MostPopular]:
    """The four combinations of window x feedback."""
    return [
        MostPopular(window, feedback, recent_days, positive_threshold)
        for window in ("all", "recent")
        for feedback in ("all", "positive")
    ]
