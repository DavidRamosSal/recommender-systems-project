"""Evaluate saved models and regenerate report tables and figures (Parts 1 and 2).

    uv run --locked scripts/run_all.py --data-dir data/ml-1m
    uv run --locked scripts/run_all.py --data-dir data/ml-1m --parts 2   # Part 2 only
    uv run --locked scripts/run_all.py --retrain   # repeat model selection and training

Downloads MovieLens 1M if the default dataset is missing. For an existing copy,
--data-dir must name the directory containing ratings.dat, movies.dat and users.dat.
"""

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from recsys.analysis import characterize_data, plot_tuning
from recsys.checkpoints import ratings_digest
from recsys.data import load_movielens
from recsys.outputs import OutputPaths, copy_outputs
from recsys.splitting import temporal_split_by_date
from recsys.tuning import selected_settings, tune_itemknn
from download_data import download_data
from run_baselines import run_baselines
from run_markov import run_comparison
from run_part2 import run_part2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/ml-1m"
DATA_FILES = ("ratings.dat", "movies.dat", "users.dat")
MARKOV_TABLES = ("order-tuning", "itemknn-tuning", "most-popular-tuning",
                 "disaggregated-test", "paired-test", "sparsity-ablation")


def ensure_data(data_dir):
    """Reuse complete data; automatically obtain the default dataset if necessary."""
    data_dir = Path(data_dir).resolve()
    if all((data_dir / name).is_file() for name in DATA_FILES):
        return data_dir
    if data_dir == DEFAULT_DATA:
        return download_data(DEFAULT_DATA.parent)
    raise FileNotFoundError(
        f"{data_dir} must contain {', '.join(DATA_FILES)}. "
        "Omit --data-dir to download MovieLens to the default data/ml-1m directory."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA,
                        help="Existing MovieLens directory (default: data/ml-1m; downloaded if missing).")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results",
                        help="Directory for generated tables and figures (default: results).")
    parser.add_argument("--parts", nargs="+", type=int, choices=(1, 2), default=[1, 2],
                        help="Which parts to reproduce (default: 1 2).")
    parser.add_argument("--retrain", action="store_true",
                        help="Rerun validation, refit models, and replace the generated weights and results.")
    args = parser.parse_args()
    data_dir = ensure_data(args.data_dir)
    paths = OutputPaths(args.out_dir.resolve()).create()
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "recsys-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    dataset = load_movielens(data_dir)
    grid = None
    if 1 in args.parts:
        saved = paths.root if (paths.root / "weights").is_dir() else ROOT / "results"
        grid = run_part1(dataset, data_dir, paths, saved_dir=None if args.retrain else saved)
    if 2 in args.parts:
        saved = paths.root if (paths.root / "part2/weights").is_dir() else ROOT / "results"
        run_part2(data_dir, paths.root / "part2", dataset=dataset,
                  saved_dir=None if args.retrain else saved / "part2")
        if args.retrain:
            run_comparison(dataset.ratings, out_dir=paths.root / "markov", item_knn_grid=grid)
        else:
            copy_outputs(saved / "markov", paths.root / "markov", [f"{t}.csv" for t in MARKOV_TABLES])


def run_part1(dataset, data_dir, paths, saved_dir=None):
    output_dir = paths.root
    print("Loading MovieLens and regenerating data characterization...", flush=True)
    ratings = dataset.ratings
    weights_dir = (saved_dir or output_dir) / "weights"
    marker = weights_dir / "ratings.sha256"
    digest = ratings_digest(ratings)
    if saved_dir is None:
        marker.unlink(missing_ok=True)
        user_ids = np.sort(ratings.user_id.unique())
        catalogue = np.sort(ratings.item_id.unique())
        train, validation, _ = temporal_split_by_date(ratings, "2001-01-01", "2001-07-01")
        print("Searching all 60 item-kNN settings on validation only...", flush=True)
        grid = tune_itemknn(train, validation, user_ids, catalogue)
        grid.to_csv(paths.intermediate / "itemknn-tuning.csv", index=False)
    else:
        if not marker.is_file() or marker.read_text().strip() != digest:
            raise ValueError("Part 1 checkpoints require matching ratings. Run with --retrain.")
        copy_outputs(saved_dir / "intermediate", paths.intermediate, ["itemknn-tuning.csv"])
        grid = pd.read_csv(paths.intermediate / "itemknn-tuning.csv")
    characterize_data(dataset, output_dir)
    best_by_profile = grid.loc[grid.groupby(["similarity", "feedback"], sort=False)["ndcg@10"].idxmax()]
    best_by_profile.to_csv(paths.intermediate / "itemknn-best-settings.csv", index=False)
    settings = selected_settings(grid)
    plot_tuning(grid, output_dir)
    print(f"Selected {settings}; evaluating all nine baselines...", flush=True)
    run_baselines(data_dir, output_dir, plots=True, dataset=dataset, settings=settings,
                  weights_dir=weights_dir, restore=saved_dir is not None)
    if saved_dir is None:
        marker.write_text(digest + "\n")

    print(f"Complete: tables, nine baseline test results and figures saved to {output_dir}", flush=True)
    return grid


if __name__ == "__main__":
    main()
