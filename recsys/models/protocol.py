from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Recommender(Protocol):
    def score(self, user_id: int, item_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        """Return one score per item, in input order; higher means preferred.

        IDs are the original MovieLens IDs, not contiguous array indices.
        Each implementation decides how to handle users/items absent in training.
        """
        ...
