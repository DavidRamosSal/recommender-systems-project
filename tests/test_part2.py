"""Correctness checks for the Part 2 code (no extra dependencies):

    uv run --locked python tests/test_part2.py

1. The pipeline's metrics equal the Part 1 reference recsys.evaluation.evaluate
   (DataFrame implementation) on a random sample.
2. The analytic BPR gradient equals a numerical gradient.
3. BPR never samples a rated item as a negative.
4. BPR learns a planted preference structure (far above random).
5. Wilcoxon/Holm/Kendall helpers give known values.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys.evaluation import evaluate                                    # noqa: E402
from recsys.experiment import ModelSpec, make_window, run                 # noqa: E402
from recsys.models.bpr import BPR, sample_negatives, sigmoid               # noqa: E402
from recsys.stats import holm, kendall_tau, wilcoxon_signed_rank           # noqa: E402


class RandomScores:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def fit(self, history):
        return self

    def score_users(self, users, items):
        return self.rng.random((len(users), len(items)))


def test_metrics_match_part1_reference():
    rng = np.random.default_rng(0)
    n = 4000
    ratings = pd.DataFrame({"user_id": rng.integers(1, 201, n), "item_id": rng.integers(1, 301, n),
                            "rating": rng.integers(1, 6, n), "timestamp": rng.integers(0, 100, n)}
                           ).drop_duplicates(["user_id", "item_id"])
    history, target = ratings[ratings.timestamp < 80], ratings[ratings.timestamp >= 80]
    user_ids, catalogue = np.sort(ratings.user_id.unique()), np.sort(ratings.item_id.unique())
    window = make_window(history, target, user_ids, catalogue)
    spec = ModelSpec("random", "random", "-", lambda p, s: RandomScores(s), [{}])
    r = run(spec, {}, 7, window, user_ids, catalogue)

    predictions = pd.DataFrame({"user_id": np.repeat(window.users, 10),
                                "item_id": catalogue[r["top"]].ravel(),
                                "rank": np.tile(np.arange(1, 11), len(window.users))})
    labels = target[(target.rating >= 4) & target.user_id.isin(window.users)][["user_id", "item_id"]]
    reference = evaluate(labels, predictions, catalogue_size=len(catalogue))
    for m in ("precision", "recall", "ndcg"):
        assert math.isclose(r["per_user"][m].mean(), reference[f"{m}@10"], rel_tol=1e-6), m
    assert math.isclose(r["coverage"], reference["catalogue_coverage"])
    # seen items are never recommended
    assert not np.take_along_axis(window.seen, r["top"], axis=1).any()


def _objective(p, qi, qj, bi, bj, reg, reg_b):
    x = bi - bj + p @ (qi - qj)
    return (math.log(sigmoid(np.array(x))) - reg / 2 * (p @ p + qi @ qi + qj @ qj)
            - reg_b / 2 * (bi ** 2 + bj ** 2))


def test_bpr_gradient_matches_numeric():
    rng = np.random.default_rng(1)
    reg, reg_b = 0.03, 0.01
    v = {"p": rng.normal(size=5), "qi": rng.normal(size=5), "qj": rng.normal(size=5),
         "bi": rng.normal(size=1), "bj": rng.normal(size=1)}
    x = v["bi"][0] - v["bj"][0] + v["p"] @ (v["qi"] - v["qj"])
    g = sigmoid(np.array(-x))
    analytic = {"p": g * (v["qi"] - v["qj"]) - reg * v["p"], "qi": g * v["p"] - reg * v["qi"],
                "qj": -g * v["p"] - reg * v["qj"], "bi": g - reg_b * v["bi"],
                "bj": -g - reg_b * v["bj"]}       # the formulas used in BPR.fit

    def f(w):
        return _objective(w["p"], w["qi"], w["qj"], w["bi"][0], w["bj"][0], reg, reg_b)
    for name, value in v.items():
        for t in range(len(value)):
            up = {k: a.copy() for k, a in v.items()}
            down = {k: a.copy() for k, a in v.items()}
            up[name][t] += 1e-6
            down[name][t] -= 1e-6
            numeric = (f(up) - f(down)) / 2e-6
            assert math.isclose(np.atleast_1d(analytic[name])[t], numeric,
                                rel_tol=1e-5, abs_tol=1e-8), name


def test_negatives_are_unrated():
    rng = np.random.default_rng(2)
    n_users, n_items = 50, 40
    keys = np.unique(rng.integers(0, n_users, 900) * n_items + rng.integers(0, n_items, 900))
    users = rng.integers(0, n_users, 5000)
    negatives = sample_negatives(users, keys, n_items, rng)
    assert not np.isin(users * n_items + negatives, keys).any()


def test_bpr_learns_planted_structure():
    """Odd users like items 0-49, even users items 50-99."""
    rng = np.random.default_rng(3)
    rows = [(u, i, 5, 0) for u in range(200)
            for i in rng.choice(np.arange(50) + 50 * (u % 2), 15, replace=False)]
    train = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    model = BPR(factors=8, lr=0.05, reg=0.001, epochs=30, seed=0).fit(
        train, user_ids=np.arange(200), catalogue=np.arange(100))
    scores = model.score_users(np.arange(200))
    scores[train.user_id, train.item_id] = -np.inf
    top = np.argsort(-scores, axis=1)[:, :10]
    own_half = (top // 50) == (np.arange(200) % 2)[:, None]
    assert own_half.mean() > 0.95, own_half.mean()           # random: 0.5
    assert model.loss_history[-1] < model.loss_history[0]


def test_stats_helpers():
    # n = 10 differences 1..10, all positive: W+ = 55, z = 2.803, p = 0.00506
    assert math.isclose(wilcoxon_signed_rank(np.arange(1, 11)), 0.005062, rel_tol=1e-3)
    assert wilcoxon_signed_rank(np.zeros(5)) == 1.0
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert kendall_tau([1, 2, 3], [1, 2, 3]) == 1.0
    assert kendall_tau([1, 2, 3], [3, 2, 1]) == -1.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
