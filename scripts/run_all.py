"""Reproduce all implemented Part 1 report numbers, tables, results and figures.

    uv run --locked scripts/run_all.py --data-dir data/ml-1m

Downloads MovieLens 1M if the default dataset is missing. For an existing copy,
--data-dir must name the directory containing ratings.dat, movies.dat and users.dat.
"""

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np

from recsys.analysis import characterize_data, plot_tuning
from recsys.data import load_movielens
from recsys.outputs import OutputPaths
from recsys.splitting import temporal_split_by_date
from recsys.tuning import selected_settings, tune_itemknn
from download_data import download_data
from run_baselines import run_baselines


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/ml-1m"
DATA_FILES = ("ratings.dat", "movies.dat", "users.dat")


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
    args = parser.parse_args()
    data_dir = ensure_data(args.data_dir)
    paths = OutputPaths(args.out_dir.resolve()).create()
    output_dir = paths.root
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "recsys-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    print("Loading MovieLens and regenerating data characterization...", flush=True)
    dataset = load_movielens(data_dir)
    characterize_data(dataset, output_dir)
    ratings = dataset.ratings
    user_ids = np.sort(ratings.user_id.unique())
    catalogue = np.sort(ratings.item_id.unique())
    train, validation, _ = temporal_split_by_date(ratings, "2001-01-01", "2001-07-01")
    print("Searching all 60 item-kNN settings on validation only...", flush=True)
    grid = tune_itemknn(train, validation, user_ids, catalogue)
    grid.to_csv(paths.intermediate / "itemknn-tuning.csv", index=False)
    best_by_profile = grid.loc[grid.groupby(["similarity", "feedback"], sort=False)["ndcg@10"].idxmax()]
    best_by_profile.to_csv(paths.intermediate / "itemknn-best-settings.csv", index=False)
    settings = selected_settings(grid)
    plot_tuning(grid, output_dir)
    print(f"Selected {settings}; evaluating all nine baselines...", flush=True)
    run_baselines(data_dir, output_dir, plots=True, dataset=dataset, settings=settings)

    print(f"Complete: tables, nine baseline test results and figures saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
