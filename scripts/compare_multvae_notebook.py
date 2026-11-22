"""Score the uploaded notebook and pipeline Mult-VAE checkpoints on one test split."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recsys.checkpoints import load_weights, ratings_digest
from recsys.data import load_movielens
from recsys.experiment import make_window, per_user_frame, score_matrix
from recsys.models.item_knn import interaction_matrix
from recsys.models.multvae import MultVAE
from recsys.ranking import top_k_items
from recsys.splitting import temporal_split_by_date

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_RATINGS_DIGEST = "fc8abd2acf4ff37380414a52a4bfaf06e227313fa36037656a753063039c28fd"
NOTEBOOK_LAYERS = {"encoder": "enc_hidden", "posterior": "enc_out",
                   "decoder": "dec_hidden", "logits": "dec_out"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/ml-1m")
    args = parser.parse_args()
    ratings = load_movielens(args.data_dir).ratings
    results = ROOT / "results"
    manifest = json.loads((results / "part2/manifest.json").read_text())
    digest = ratings_digest(ratings)
    if digest != NOTEBOOK_RATINGS_DIGEST or digest != manifest["ratings_sha256"]:
        raise ValueError("The comparison requires the MovieLens ratings used by both saved runs.")

    train, validation, test = temporal_split_by_date(ratings, "2001-01-01", "2001-07-01")
    history = pd.concat([train, validation])
    users = np.sort(ratings.user_id.unique())
    catalogue = np.sort(ratings.item_id.unique())
    window = make_window(history, test, users, catalogue)
    profiles = interaction_matrix(history, pd.Index(users), pd.Index(catalogue))
    inputs = {"users": users, "items": catalogue, "profiles": profiles}
    multvae_results = results / "multvae"
    settings = json.loads((multvae_results / "selected.json").read_text())
    notebook_dir = ROOT / "notebooks/artifacts/part2"

    rows = []
    for seed in manifest["seeds"]:
        pipeline = MultVAE(**settings, seed=seed)
        load_weights(multvae_results / "weights" / f"seed-{seed}.npz", pipeline, settings, seed)

        state = torch.load(notebook_dir / f"vae_final_seed{seed}.pt",
                           map_location="cpu", weights_only=True)
        weights = {f"{layer}.{parameter}": state[f"{original}.{parameter}"].numpy()
                   for layer, original in NOTEBOOK_LAYERS.items()
                   for parameter in ("weight", "bias")}
        notebook = MultVAE(hidden_dim=800, latent_dim=64, seed=seed)
        notebook.load_weights({**inputs, **weights})

        for source, model in (("Notebook", notebook), ("Pipeline", pipeline)):
            top = top_k_items(score_matrix(model, window.users, catalogue), window.seen)
            metrics = per_user_frame(top, window).mean()
            rows.append({"run": source, **{f"{key}@10": value for key, value in metrics.items()},
                         "coverage": np.unique(top).size / len(catalogue)})

    print(f"Mean over {len(manifest['seeds'])} checkpoints per run on {len(window.users)} test users.")
    print(pd.DataFrame(rows).groupby("run").mean().to_string(float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
