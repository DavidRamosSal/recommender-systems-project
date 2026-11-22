"""Portable NumPy checkpoints for the selected full-history models."""

import hashlib
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def ratings_digest(ratings):
    columns = ["user_id", "item_id", "rating", "timestamp"]
    return hashlib.sha256(ratings[columns].to_numpy(dtype="<i8").tobytes()).hexdigest()


def save_weights(path, model, params, seed, fit_seconds):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp:
        temporary = Path(temp) / path.name
        np.savez_compressed(temporary, params=json.dumps(params), seed=seed,
                            fit_seconds=fit_seconds, **model.weights())
        temporary.replace(path)


def load_weights(path, model, params, seed):
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint {path}. Run with --retrain to regenerate it.")
    with np.load(path, allow_pickle=False) as data:
        if json.loads(data["params"].item()) != params or data["seed"].item() != seed:
            raise ValueError(f"Checkpoint settings differ for {path}. Run with --retrain.")
        model.load_weights(data)
        return float(data["fit_seconds"].item())


def prepare_model(model, fit, params, *, seed=0, checkpoint=None, restore=False):
    """Load an existing fit or time a new fit and optionally save it."""
    if restore:
        return model, load_weights(checkpoint, model, params, seed)
    start = time.perf_counter()
    model = fit() or model
    seconds = time.perf_counter() - start
    if checkpoint is not None:
        save_weights(checkpoint, model, params, seed, seconds)
    return model, seconds
