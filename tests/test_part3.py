"""Correctness checks for the Part 3 re-ranking code (no extra dependencies):

    uv run --locked python tests/test_part3.py

1. MMR reproduces the worked example of Lecture 5.1, slide 9 step by step.
2. Intra-list diversity reproduces Lecture 5.1, slide 8 (0.33 -> 0.60).
3. lambda = 1 returns exactly the base model's top-10 (via the candidate lists).
4. Re-ranked lists never repeat an item, never contain a seen item, and only
   contain candidates.
5. Genre vectors are parsed correctly from movies.dat-style strings.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys.diversity import ListContext, intra_list_diversity            # noqa: E402
from recsys.ranking import top_k_items                                    # noqa: E402
from recsys.rerank import (GENRES, candidate_lists, genre_matrix, minmax,  # noqa: E402
                           mmr, rerank)

# Lecture 5.1, slide 8: genres (Action, Drama, Comedy) and relevance for Ann.
SLIDE_ITEMS = ["i1", "i4", "i3", "i5", "i2"]
SLIDE_GENRES = np.array([[1, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1]], float)
SLIDE_RELEVANCE = np.array([[0.95, 0.90, 0.70, 0.60, 0.40]])


def context(features, n_users=1):
    return ListContext(features=features, catalogue=np.arange(len(features)),
                       users=np.arange(n_users), history=pd.DataFrame(),
                       seen=np.zeros((n_users, len(features))))


def test_mmr_matches_lecture_example():
    picks = mmr(SLIDE_RELEVANCE, SLIDE_GENRES[None], lam=0.5, k=3)[0]
    assert [SLIDE_ITEMS[p] for p in picks] == ["i1", "i2", "i3"], picks
    # i4 (second most relevant, identical to i1) is never selected
    assert 1 not in picks


def test_ild_matches_lecture_example():
    ctx = context(SLIDE_GENRES)
    by_relevance = intra_list_diversity(np.array([[0, 1, 2]]), ctx)[0]    # {i1, i4, i3}
    by_mmr = intra_list_diversity(np.array([[0, 4, 2]]), ctx)[0]          # {i1, i2, i3}
    assert round(by_relevance, 2) == 0.33, by_relevance
    assert round(by_mmr, 2) == 0.60, by_mmr
    relevance = SLIDE_RELEVANCE[0]
    assert np.isclose(relevance[[0, 1, 2]].sum(), 2.55)
    assert np.isclose(relevance[[0, 4, 2]].sum(), 2.05)


def test_lambda_one_is_the_base_ranking():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(50, 300))
    seen = rng.random((50, 300)) < 0.2
    features = (rng.random((300, 18)) < 0.15).astype(float)
    candidates, base = candidate_lists(scores, seen, n=100)
    top = rerank(candidates, minmax(base), features, lam=1.0)
    assert np.array_equal(top, top_k_items(scores, seen, k=10))


def test_lists_are_valid():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(40, 200))
    seen = rng.random((40, 200)) < 0.3
    features = (rng.random((200, 18)) < 0.2).astype(float)
    candidates, base = candidate_lists(scores, seen, n=50)
    for lam in (0.0, 0.3, 0.7):
        top = rerank(candidates, minmax(base), features, lam)
        assert all(len(set(row)) == 10 for row in top)
        assert not seen[np.arange(40)[:, None], top].any()
        assert all(set(t) <= set(c) for t, c in zip(top, candidates))
    # more weight on diversity never lowers mean ILD on this random example
    ild = [intra_list_diversity(rerank(candidates, minmax(base), features, lam),
                                context(features, 40)).mean() for lam in (1.0, 0.5, 0.0)]
    assert ild[0] <= ild[1] <= ild[2], ild


def test_genre_matrix():
    movies = pd.DataFrame({"item_id": [1, 2, 3],
                           "genres": ["Animation|Children's|Comedy", "Drama", "Film-Noir|Sci-Fi"]})
    m = genre_matrix(movies, np.array([3, 1]))
    assert m.shape == (2, len(GENRES))
    assert [GENRES[j] for j in np.flatnonzero(m[0])] == ["Film-Noir", "Sci-Fi"]
    assert [GENRES[j] for j in np.flatnonzero(m[1])] == ["Animation", "Children's", "Comedy"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} checks passed")
