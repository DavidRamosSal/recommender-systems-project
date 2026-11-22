from collections import Counter, defaultdict
import json
from operator import index

import numpy as np


VALIDATION_ORDERS = (1, 2, 3)


class MarkovChain:
    """Predict the next liked movie from a fixed number of recent likes."""

    def __init__(self, positive_threshold=4, *, order=1):
        self.positive_threshold = positive_threshold
        self.order = index(order)
        if self.order < 1:
            raise ValueError("Order must be at least 1.")

    def fit(self, history):
        positives = history[history["rating"] >= self.positive_threshold]
        positives = positives.sort_values(["user_id", "timestamp", "item_id"])
        transition_counts = defaultdict(Counter)
        last_movies = {}
        order = self.order

        for user_id, items in positives.groupby("user_id", sort=False)["item_id"]:
            movies = tuple(items.tolist())
            last_movies[user_id] = movies[-order:]
            for position in range(order, len(movies)):
                state = movies[position - order:position]
                transition_counts[state][movies[position]] += 1

        self.transition_counts = transition_counts
        self.last_movies = last_movies
        return self

    def score(self, user_id, item_ids):
        state = self.last_movies.get(user_id)
        counts = self.transition_counts.get(state, Counter())
        scale = 1 / max(counts.total(), 1)
        return np.array([counts[movie] * scale for movie in item_ids], dtype=float)

    def weights(self):
        transitions = [(*state, item, count) for state, counts in self.transition_counts.items()
                       for item, count in counts.items()]
        # A user's last state can be shorter than the configured order.
        histories = [[int(user), list(state)] for user, state in self.last_movies.items()]
        return {"transitions": np.asarray(transitions, dtype=np.int64),
                "histories": json.dumps(histories)}

    def load_weights(self, weights):
        self.transition_counts = defaultdict(Counter)
        for *state, item, count in weights["transitions"].tolist():
            self.transition_counts[tuple(state)][item] = count
        self.last_movies = {user: tuple(state) for user, state in
                            json.loads(weights["histories"].item())}
