"""Calculate all nine baselines and write the Part 1 results CSVs.

Run from the repository root:
    uv run --locked scripts/run_baselines.py
    uv run --locked scripts/run_baselines.py --plots

Loads MovieLens directly and uses the models in recsys/models. Writes the final
results.csv at the output root and optional PNG/SVG figures in images/
(default output root: results/). Validation results are included in Notes.
Uses the validation-selected item-kNN configuration without rerunning tuning.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data import load_movielens
from recsys.evaluation import (
    bootstrap_ci, evaluate, evaluation_users, per_user_ndcg, ranking_metrics,
)
from recsys.models import ItemKNN
from recsys.models.item_knn import interaction_matrix
from recsys.models.most_popular import MostPopular
from recsys.outputs import OutputPaths
from recsys.ranking import top_k, top_k_items
from recsys.splitting import temporal_split_by_date


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SEED = 42
HEADERS = [
    "Model", "Variant / settings", "Similarity", "k", "Precision@10",
    "Recall@10", "NDCG@10", "Catalogue coverage", "Seed", "Notes",
]
METRICS = {
    "Precision@10": "precision@10",
    "Recall@10": "recall@10",
    "NDCG@10": "ndcg@10",
    "Catalogue coverage": "catalogue_coverage",
}
MOST_POPULAR_VARIANTS = {
    f"mostpop_{window}_{feedback}": f"{label}, {treatment}"
    for window, label in (
        ("alltime", "all-time"),
        ("recent30d", "last 30 days"),
        ("recent90d", "last 90 days"),
        ("recent180d", "last 180 days"),
    )
    for feedback, treatment in (("pos4", "ratings >= 4"), ("allratings", "all ratings"))
}


def most_popular_results(train, validation, test, catalogue):
    """Evaluate all eight popularity variants with reproducible user-bootstrap CIs."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    models = [MostPopular("all", "all"), MostPopular("all", "positive")]
    models += [MostPopular("recent", feedback, days)
               for days in (30, 90, 180) for feedback in ("all", "positive")]
    rows = []
    for model in models:
        for split, history, window in (
            ("validation", train, validation),
            ("test", pd.concat([train, validation]), test),
        ):
            model.fit(history)
            labels = window[window["rating"] >= 4]
            labels = labels[labels["user_id"].isin(history["user_id"])][["user_id", "item_id"]]
            predictions = top_k(model, labels["user_id"].unique(), catalogue, history, k=10)
            metrics = evaluate(labels, predictions, catalogue_size=len(catalogue))
            low, high = bootstrap_ci(per_user_ndcg(labels, predictions), rng)
            rows.append({
                "model": model.name, "split": split,
                "users": labels["user_id"].nunique(), **metrics,
                "ndcg@10_ci_low": low, "ndcg@10_ci_high": high,
            })
    return pd.DataFrame(rows)


def itemknn_results(train, validation, test, user_ids, catalogue, settings=None):
    """Evaluate selected settings; the default is the report's cosine/positive/k=500."""
    rows = []
    for split, history, window in (
        ("validation", train, validation),
        ("test", pd.concat([train, validation]), test),
    ):
        users = evaluation_users(history, window)
        indices = pd.Index(user_ids).get_indexer(users)
        relevant = interaction_matrix(window, user_ids, catalogue)[indices]
        seen = interaction_matrix(history, user_ids, catalogue, positives_only=False)[indices]
        model = ItemKNN(**(settings or {})).fit(
            history, user_ids=user_ids, catalogue=catalogue,
        )
        top = top_k_items(model.score_users(users), seen)
        metrics = ranking_metrics(top, relevant)
        rows.append({
            "model": model.name, "split": split, "users": len(users),
            "precision@10": metrics["precision"].mean(),
            "recall@10": metrics["recall"].mean(),
            "ndcg@10": metrics["ndcg"].mean(),
            "catalogue_coverage": np.unique(top).size / len(catalogue),
        })
    return pd.DataFrame(rows)


def export_results(popular, comparison, output_dir, settings=None):
    """Write nine test rows in the workbook's ten-column format."""
    popular = {(row["model"], row["split"]): row for row in popular.to_dict("records")}
    comparison = {(row["model"], row["split"]): row for row in comparison.to_dict("records")}
    paths = OutputPaths(output_dir).create()
    # Match the workbook order: validation-selected popularity variant first.
    selected = max(
        MOST_POPULAR_VARIANTS,
        key=lambda model: float(popular[model, "validation"]["ndcg@10"]),
    )
    rows = []
    for model in sorted(MOST_POPULAR_VARIANTS, key=lambda name: name != selected):
        test = popular[model, "test"]
        validation = popular[model, "validation"]
        notes = (
            f"Validation NDCG@10 {validation['ndcg@10']}; "
            f"95% CI validation NDCG@10 [{validation['ndcg@10_ci_low']}, "
            f"{validation['ndcg@10_ci_high']}]; "
            f"test users {test['users']}; "
            f"95% CI test NDCG@10 [{test['ndcg@10_ci_low']}, "
            f"{test['ndcg@10_ci_high']}] "
            "(1000 user-bootstrap samples, seed 42)"
        )
        if model == selected:
            notes = "Selected by validation NDCG@10; " + notes
        rows.append({
            "Model": "Most Popular",
            "Variant / settings": MOST_POPULAR_VARIANTS[model]
            + (" (selected)" if model == selected else ""),
            "Similarity": "–",
            "k": "–",
            **{header: test[column] for header, column in METRICS.items()},
            "Seed": "deterministic",
            "Notes": notes,
        })

    model = ItemKNN(**(settings or {}))
    test = comparison[model.name, "test"]
    validation = comparison[model.name, "validation"]
    rows.append({
        "Model": "Item-based kNN",
        "Variant / settings": (
            f"liked-movie profile (ratings >= {model.positive_threshold})"
            if model.feedback == "positives" else "all-rated-movie profile"
        ) + ", top-k neighbours, no filtering",
        "Similarity": model.similarity,
        "k": str(model.k),
        **{header: test[column] for header, column in METRICS.items()},
        "Seed": "deterministic",
        "Notes": (
            f"Best validation NDCG@10 {validation['ndcg@10']} among 60 settings "
            "(3 similarities x 2 profiles x 10 values of k); "
            f"test users {test['users']}"
        ),
    })

    destination = paths.root / "results.csv"
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=HEADERS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} baseline test results to {destination}")


def plot_results(results, output_dir, settings=None):
    """Render the comparison only when --plots is requested."""
    import os
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "recsys-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    images = OutputPaths(output_dir).create().images
    itemknn = ItemKNN(**(settings or {}))
    popularity = results[
        (results.split == "validation") & results.model.str.startswith("mostpop_")
    ].sort_values("ndcg@10", ascending=False, kind="stable")
    models = [(itemknn.name, f"Item-kNN\n{itemknn.similarity}, k = {itemknn.k}")]
    for row in popularity.itertuples():
        window = MOST_POPULAR_VARIANTS[row.model].split(",")[0].replace("all-time", "all time")
        models.append((row.model, f"Most Popular\n{window}"))
    colours = {"validation": "#6baed6", "test": "#183153"}
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8,
        "text.color": "#183153", "axes.labelcolor": "#183153",
        "xtick.color": "#475569", "ytick.color": "#183153",
        "axes.edgecolor": "#94a3b8", "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8))
    positions = np.arange(len(models))
    for split_index, (split, colour) in enumerate(colours.items()):
        records = results[results.split == split].set_index("model")
        x = positions + (-.21 if split_index == 0 else .21)
        ndcg = [records.loc[model, "ndcg@10"] for model, _ in models]
        coverage = [100 * records.loc[model, "catalogue_coverage"] for model, _ in models]
        for axis, values, labels in (
            (axes[0], ndcg, [f"{value:.4f}" for value in ndcg]),
            (axes[1], coverage, [f"{value:.1f}%" for value in coverage]),
        ):
            bars = axis.bar(x, values, width=.34, color=colour, edgecolor="white",
                            linewidth=.6, label=split.title())
            axis.bar_label(bars, labels=labels, padding=3, fontsize=6.5)
    axes[0].set_ylim(0, .25)
    axes[0].set_yticks([0, .05, .10, .15, .20, .25])
    axes[0].set_ylabel("NDCG@10", fontsize=8)
    axes[0].set_title("Ranking accuracy", fontsize=9, pad=8)
    axes[1].set_ylim(0, 12)
    axes[1].set_yticks([0, 3, 6, 9, 12])
    axes[1].yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axes[1].set_ylabel("Coverage (%)", fontsize=8)
    axes[1].set_title("Catalogue coverage", fontsize=9, pad=8)
    for axis in axes:
        axis.set_xticks(positions, labels=[label for _, label in models], fontsize=7)
        axis.set_xlabel("Model", fontsize=8, labelpad=6)
        axis.tick_params(axis="x", length=0, pad=6)
        axis.grid(axis="y", color="#94a3b8", alpha=.18)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, 1.02),
               ncol=2, frameon=False, fontsize=8)
    fig.subplots_adjust(left=.085, right=.99, bottom=.25, top=.80, wspace=.37)
    for extension in ("png", "svg"):
        fig.savefig(images / f"baseline-comparison.{extension}", dpi=300,
                    bbox_inches="tight", pad_inches=.07, facecolor="white")
    plt.close(fig)


def run_baselines(
    data_dir=ROOT / "data/ml-1m", output_dir=ROOT / "results", *,
    plots=False, dataset=None, settings=None,
):
    """Compute and export baselines; run_all can supply freshly tuned settings."""
    dataset = load_movielens(data_dir) if dataset is None else dataset
    ratings = dataset.ratings
    paths = OutputPaths(output_dir).create()
    user_ids = np.sort(ratings["user_id"].unique())
    catalogue = np.sort(ratings["item_id"].unique())
    train, validation, test = temporal_split_by_date(ratings, "2001-01-01", "2001-07-01")
    popular = most_popular_results(train, validation, test, catalogue)
    columns = ["model", "split", "users", "precision@10", "recall@10", "ndcg@10", "catalogue_coverage"]
    validation_rows = popular[popular.split == "validation"]
    selected = validation_rows.loc[validation_rows["ndcg@10"].idxmax(), "model"]
    comparisons = list(dict.fromkeys([selected, "mostpop_alltime_pos4"]))
    if len(comparisons) == 1:
        comparisons.append("mostpop_recent30d_pos4")
    comparison = pd.concat([
        itemknn_results(train, validation, test, user_ids, catalogue, settings),
        popular[popular.model.isin(comparisons)][columns],
    ], ignore_index=True)

    export_results(popular, comparison, paths.root, settings)
    if plots:
        plot_results(comparison, paths.root, settings)
    return popular, comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/ml-1m")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--plots", action="store_true", help="Also write comparison PNG/SVG figures.")
    args = parser.parse_args()
    run_baselines(args.data_dir, args.out_dir, plots=args.plots)


if __name__ == "__main__":
    main()
