"""Recompute the two supplementary slide metrics from the five saved BPR fits.

From the repository root, run:
    uv run --locked presentation/figures/exposure_calibration.py

Requires the MovieLens data in data/ml-1m and the Git objects from origin/part3-mmr.
"""

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REVISION = "4a252eae457d5e90a1adf4bd95733b4b8cbe58b9"
FRONTIER = "results/part3/intermediate/frontier-test.csv"


def exposure_gini(top, context):
    """Count list appearances at every rank, including zero counts for unseen titles."""
    counts = np.sort(np.bincount(top.ravel(), minlength=len(context.catalogue)))
    ranks = np.arange(1, len(counts) + 1)
    return np.dot(2 * ranks - len(counts) - 1, counts) / (len(counts) * counts.sum())


def entropy(distribution):
    log_prob = np.log2(np.where(distribution > 0, distribution, 1.0))
    return -(distribution * log_prob).sum(axis=1)


def recompute(reference):
    # Imports come from the same pinned code that produced the original frontier.
    import run_part3 as mmr

    dataset = mmr.load_movielens(ROOT / "data/ml-1m")
    train, validation, test = mmr.temporal_split_by_date(dataset.ratings, "2001-01-01", "2001-07-01")
    catalogue = np.sort(dataset.ratings.item_id.unique())
    users = np.sort(dataset.ratings.user_id.unique())
    window = mmr.make_window(pd.concat([train, validation]), test, users, catalogue)
    features = mmr.genre_matrix(dataset.movies, catalogue)
    genre_share = features / features.sum(axis=1, keepdims=True)

    # Each liked movie contributes total mass one, split across its genre tags.
    likes = window.history[window.history.rating >= 4]
    likes = likes[likes.user_id.isin(window.users)]
    profiles = np.zeros((len(window.users), features.shape[1]))
    user_rows = pd.Index(window.users).get_indexer(likes.user_id)
    movie_rows = pd.Index(catalogue).get_indexer(likes.item_id)
    np.add.at(profiles, user_rows, genre_share[movie_rows])
    totals = profiles.sum(axis=1)
    eligible = totals > 0
    profiles[eligible] /= totals[eligible, None]

    def calibration(top, context):
        recommended = genre_share[top].mean(axis=1)
        midpoint = (recommended + profiles) / 2
        divergence = entropy(midpoint) - (entropy(profiles) + entropy(recommended)) / 2
        divergence[~eligible] = np.nan
        return divergence

    mmr.PER_USER["calibration"] = calibration
    mmr.SYSTEM["gini"] = exposure_gini
    checkpoints = ROOT / "results/bpr"
    settings = json.loads((checkpoints / "selected.json").read_text())
    models = []
    for seed in range(5):
        model = mmr.BPR(**settings, seed=seed)
        mmr.load_weights(checkpoints / "weights" / f"seed-{seed}.npz", model, settings, seed)
        models.append(model)

    runs = mmr.sweep(models, window, features, catalogue, mmr.LAMBDAS, mmr.CANDIDATES)
    records = []
    for weight, fits in runs.items():
        published = reference.loc[weight]
        coverage = np.mean([fit["system"]["coverage"] for fit in fits])
        ild = mmr.seed_mean(fits, "ild").mean()
        ndcg = mmr.seed_mean(fits, "ndcg").mean()
        np.testing.assert_allclose(
            [coverage, ild, ndcg], published[["coverage", "ild@10", "ndcg@10"]].to_numpy(float),
            rtol=0, atol=1e-7,
        )
        errors = mmr.seed_mean(fits, "calibration").dropna()
        records.append({
            "lambda": weight,
            "gini": np.mean([fit["system"]["gini"] for fit in fits]),
            "calibration": errors.mean(),
            "calibration_users": len(errors),
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        archive = Path(directory) / "part3.zip"
        with archive.open("wb") as output:
            subprocess.run(
                ["git", "-C", str(ROOT), "archive", "--format=zip", REVISION,
                 "recsys", "scripts/run_part3.py", FRONTIER],
                stdout=output, check=True,
            )
        sys.path[:0] = [str(archive), str(archive / "scripts")]
        with ZipFile(archive) as snapshot, snapshot.open(FRONTIER) as source:
            reference = pd.read_csv(source, index_col="lambda")
        table = recompute(reference)
    table.to_csv(Path(__file__).with_name("exposure-calibration.csv"), index=False)
