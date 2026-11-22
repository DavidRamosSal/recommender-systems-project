"""Part 2: tune, compare, disaggregate and ablate all models in one pipeline.

    uv run --locked scripts/run_part2.py --data-dir data/ml-1m

Same split and protocol as Part 1 (see scripts/run_baselines.py). Every model
gets the same validation budget (BUDGET settings; smaller spaces are searched
exhaustively). Seeds: models 0-4, BPR random search 2026, bootstrap 42.

Each model has its own directory under results/ for weights, trials, selected settings,
test and subgroup metrics, and training-data ablations. Cross-model comparisons
and figures are under results/part2/.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data import load_movielens
from recsys.checkpoints import invalidate_mismatched_runs, ratings_digest
from recsys.experiment import (
    METRICS, ModelSpec, Setup, disaggregate, final_runs, make_window, overall,
    pairwise_tests, search_space, seed_mean, sparsity_ablation, tune,
)
from recsys.models import BPR, ItemKNN, MarkovChain
from recsys.models.markov import VALIDATION_ORDERS
from recsys.models.most_popular import MostPopular, validation_configs as popularity_configs
from recsys.models.multvae import MultVAE, validation_configs as multvae_configs
from recsys.outputs import (
    PART2_RESULT_COLUMNS, OutputPaths, copy_outputs, update_overall_results, write_results_csv,
)
from recsys.splitting import temporal_split_by_date
from recsys.tuning import FEEDBACKS, K_GRID, SIMILARITIES

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 60            # = the 60-setting item-kNN search of Part 1
SEARCH_SEED = 2026

BPR_GRID = {
    "feedback": ["positives", "all_ratings"],
    "factors": [16, 32, 64, 128],
    "lr": [0.01, 0.025, 0.05, 0.1],
    "reg": [0.0001, 0.001, 0.005, 0.01, 0.05],
    "epochs": [10, 20, 40],
}


def model_specs(budget: int = BUDGET) -> list[ModelSpec]:
    if not 1 <= budget <= BUDGET:
        raise ValueError(f"budget must be between 1 and {BUDGET}.")
    itemknn = [{"similarity": s, "feedback": f, "k": k}
               for s in SIMILARITIES for f in FEEDBACKS for k in K_GRID]
    return [
        ModelSpec("mostpop", "Most Popular", "Popularity",
                  lambda p, seed: MostPopular(**p), popularity_configs()[:budget], deterministic=True),
        ModelSpec("itemknn", "Item-based kNN", "Neighbourhood CF",
                  lambda p, seed: ItemKNN(**p), itemknn[:budget], deterministic=True),
        ModelSpec("bpr", "BPR-MF (own implementation)", "Ranking-loss CF",
                  lambda p, seed: BPR(**p, seed=seed),
                  search_space(BPR_GRID, budget, SEARCH_SEED), self_implemented=True),
        ModelSpec("markov", "Markov chain", "Sequential",
                  lambda p, seed: MarkovChain(**p),
                  [{"order": order} for order in VALIDATION_ORDERS[:budget]], deterministic=True),
        ModelSpec("multvae", "Mult-VAE", "Neural", lambda p, seed: MultVAE(**p, seed=seed),
                  multvae_configs(budget), tuning_seed=42),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/ml-1m")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--models", nargs="*", help="subset of model keys")
    parser.add_argument("--budget", type=int, default=BUDGET,
                        help="validation settings per model (quick checks only)")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()
    run_part2(args.data_dir, args.out_dir, models=args.models, budget=args.budget,
              seeds=tuple(args.seeds), ablation=not args.skip_ablation)


def run_part2(data_dir, out_dir, *, models=None, budget=BUDGET, seeds=(0, 1, 2, 3, 4),
              ablation=True, dataset=None, saved_dir=None):
    started = time.perf_counter()
    root = Path(out_dir)
    paths = OutputPaths(root / "part2").create()
    setup = Setup(seeds=seeds)
    dataset = load_movielens(data_dir) if dataset is None else dataset
    ratings = dataset.ratings
    user_ids = np.sort(ratings["user_id"].unique())
    catalogue = np.sort(ratings["item_id"].unique())           # 3,706 rated movies
    train, validation, test = temporal_split_by_date(ratings, "2001-01-01", "2001-07-01")
    history = pd.concat([train, validation])

    specs = [s for s in model_specs(budget) if models is None or s.key in models]
    test_window = make_window(history, test, user_ids, catalogue)
    source = Path(saved_dir) if saved_dir is not None else root
    for spec in specs:
        (root / spec.key).mkdir(parents=True, exist_ok=True)
    manifest = {"ratings_sha256": ratings_digest(ratings), "seeds": list(seeds),
                "trials_per_model": {s.key: len(s.configs) for s in specs}, "ablation": ablation}
    metadata = source / "part2/manifest.json"
    if saved_dir is None:
        if any(s.key in ("mostpop", "itemknn") for s in specs):
            invalidate_mismatched_runs(root, manifest["ratings_sha256"])
        metadata.unlink(missing_ok=True)
        val_window = make_window(train, validation, user_ids, catalogue)
        print(f"Part 2: {len(specs)} models, budget {budget}, seeds {list(seeds)}; "
              f"{len(val_window.users)} validation / {len(test_window.users)} test users", flush=True)
        trials, selected = tune(specs, val_window, user_ids, catalogue, setup)
        for spec in specs:
            folder = root / spec.key
            trials[trials.model == spec.key].to_csv(folder / "tuning.csv", index=False)
            (folder / "selected.json").write_text(json.dumps(selected[spec.key], indent=2) + "\n")
    else:
        if not metadata.is_file() or json.loads(metadata.read_text()) != manifest:
            raise ValueError("Saved Part 2 run differs from the dataset or settings. Use --retrain.")
        for spec in specs:
            copy_outputs(source / spec.key, root / spec.key, ["tuning.csv", "selected.json"])
        trials = pd.concat([pd.read_csv(root / s.key / "tuning.csv") for s in specs], ignore_index=True)
        selected = {s.key: json.loads((root / s.key / "selected.json").read_text()) for s in specs}
        print(f"Evaluating saved weights on {len(test_window.users)} test users.", flush=True)

    runs = final_runs(specs, selected, test_window, user_ids, catalogue, setup,
                      results_dir=source, restore=saved_dir is not None)
    for spec in specs:
        rows = [{"seed": r["seed"] if not spec.deterministic else "deterministic",
                 "users": len(r["users"]), **{f"{m}@10": r["per_user"][m].mean() for m in METRICS},
                 "coverage": r["coverage"]} for r in runs[spec.key]]
        pd.DataFrame(rows).to_csv(root / spec.key / "test.csv", index=False)
    table = overall(specs, selected, runs, setup)
    table.to_csv(paths.intermediate / "overall.csv", index=False)
    pairwise_tests(specs, runs, setup).to_csv(paths.intermediate / "pairwise.csv", index=False)
    pd.DataFrame({s.key: seed_mean(runs[s.key], "ndcg") for s in specs}).to_csv(
        paths.intermediate / "per-user-ndcg.csv")
    disagg, thresholds = disaggregate(specs, runs, test_window, user_ids, catalogue, setup)
    for spec in specs:
        disagg[disagg.model == spec.key].to_csv(root / spec.key / "disaggregated.csv", index=False)
    (paths.intermediate / "subgroups.json").write_text(json.dumps(thresholds, indent=2))

    abl = None
    if ablation and saved_dir is not None:
        for spec in specs:
            copy_outputs(source / spec.key, root / spec.key, ["ablation.csv"])
        abl = pd.concat([pd.read_csv(root / s.key / "ablation.csv") for s in specs], ignore_index=True)
        copy_outputs(source / "part2/intermediate", paths.intermediate,
                     ["ablation-tau.csv", "ablation-info.csv"])
    elif ablation:
        abl, taus, info = sparsity_ablation(specs, selected, runs, test_window, user_ids,
                                            catalogue, setup)
        for spec in specs:
            abl[abl.model == spec.key].to_csv(root / spec.key / "ablation.csv", index=False)
        taus.to_csv(paths.intermediate / "ablation-tau.csv", index=False)
        info.to_csv(paths.intermediate / "ablation-info.csv", index=False)

    rows = export_results(table, trials, paths.root / "results-part2.csv", setup)
    update_overall_results(root, rows, part=2, partial=models is not None)
    plot_all(specs, runs, disagg, abl, paths.images, root / "bpr")
    if saved_dir is None:
        metadata.write_text(json.dumps(manifest, indent=2) + "\n")
    minutes = (time.perf_counter() - started) / 60
    cols = ["model"] + [f"{m}@10" for m in METRICS] + ["coverage"]
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(table[cols].to_string(index=False))
    print(f"Part 2 complete in {minutes:.1f} min; outputs in {paths.root}", flush=True)
    return table


# ---------------------------------------------------------------- outputs
def export_results(table, trials, destination, setup):
    """Write the Part 2 template and return rows for the cumulative results.csv."""
    part2_rows, cumulative_rows = [], []
    for r in table.to_dict("records"):
        settings = json.loads(r["settings"])
        validation_scores = trials.loc[trials.model == r["model"], "ndcg@10"]
        seeds = r["seeds"].split()
        deterministic = r["seeds"] == "deterministic"
        metrics = {"Precision@10": r["precision@10"], "Recall@10": r["recall@10"],
                   "NDCG@10": r["ndcg@10"], "Catalogue coverage": r["coverage"]}
        user_ci = {label: f"[{r[f'{m}@10_user_ci_low']:.4f}, {r[f'{m}@10_user_ci_high']:.4f}]"
                   for m, label in (("precision", "Precision@10"), ("recall", "Recall@10"),
                                    ("ndcg", "NDCG@10"))}
        seed_ci = {label: f"[{r[f'{m}@10_seed_ci_low']:.4f}, {r[f'{m}@10_seed_ci_high']:.4f}]"
                   for m, label in (("ndcg", "NDCG@10"), ("recall", "Recall@10"))}
        bootstrap = f"{setup.bootstrap_samples} samples, seed {setup.bootstrap_seed}"

        variability = f"95% user bootstrap ({bootstrap}). "
        variability += ", ".join(f"{label} {ci}" for label, ci in user_ci.items()) + "."
        if len(seeds) > 1:
            variability += f" 95% across fits, NDCG@10 {seed_ci['NDCG@10']}."
        part2_rows.append({
            "Model": r["label"].removesuffix(" (own implementation)"),
            "Paradigm": r["paradigm"],
            "Self-implemented?": "Yes" if r["self_implemented"] else "No",
            "Key hyperparameters": ", ".join(f"{k}={v}" for k, v in settings.items()),
            **metrics,
            "Variability / CI / test": variability,
            "Seed(s)": "Deterministic" if deterministic else ", ".join(seeds),
            "Notes": (f"Best validation NDCG@10 {validation_scores.max():.4f} across "
                      f"{len(validation_scores)} settings. Refit on train + validation. "
                      f"Test: {r['users']} users."),
        })

        notes = [f"Selected on validation NDCG@10 among {len(validation_scores)} settings",
                 f"test users {r['users']}"]
        for label in seed_ci:
            if not deterministic:
                notes.append(f"95% seed CI {label} {seed_ci[label]}")
            notes.append(f"95% user-bootstrap CI {label} {user_ci[label]} ({bootstrap})")
        if r["self_implemented"]:
            notes.insert(0, "Self-implemented (recsys/models/bpr.py)")
        cumulative_rows.append({
            "Model": r["label"],
            "Variant / settings": ", ".join(f"{k}={v}" for k, v in settings.items()
                                            if k not in ("similarity", "k")),
            "Similarity": settings.get("similarity", "–"),
            "k": str(settings.get("k", "–")),
            **metrics,
            "Seed": "deterministic" if deterministic else f"mean over seeds {r['seeds']}",
            "Notes": "; ".join(notes),
        })
    write_results_csv(Path(destination), part2_rows, columns=PART2_RESULT_COLUMNS)
    return cumulative_rows


def plot_all(specs, runs, disagg, abl, images, bpr_dir):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "recsys-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False})
    labels = {s.key: s.label.replace(" (own implementation)", " (own)") for s in specs}

    def save(fig, name, folder=images):
        for ext in ("png", "svg"):
            fig.savefig(folder / f"{name}.{ext}", dpi=300, bbox_inches="tight",
                        facecolor="white")
        plt.close(fig)

    if abl is not None:
        fig, (accuracy, coverage) = plt.subplots(1, 2, figsize=(7.2, 2.65), sharex=True)
        markers = ["o", "s", "^", "D", "v"]
        for i, (key, g) in enumerate(abl.groupby("model", sort=False)):
            g = g.sort_values("fraction")
            x = 100 * g["fraction"]
            style = dict(marker=markers[i % len(markers)], ms=4, lw=1.5,
                         color=f"C{i % 10}", label=labels[key].removesuffix(" (own)"))
            accuracy.plot(x, g["ndcg@10"], **style)
            intervals = g.dropna(subset=["ndcg@10_ci_low", "ndcg@10_ci_high"])
            low = intervals["ndcg@10"] - intervals["ndcg@10_ci_low"]
            high = intervals["ndcg@10_ci_high"] - intervals["ndcg@10"]
            accuracy.errorbar(100 * intervals["fraction"], intervals["ndcg@10"],
                              yerr=[low, high], fmt="none", color=style["color"],
                              elinewidth=.85, capsize=2)
            coverage.plot(x, 100 * g["coverage"], **style)
        accuracy.set_ylabel("Test NDCG@10")
        coverage.set_ylabel("Catalogue coverage (%)")
        coverage.set_ylim(bottom=0)
        accuracy.set_xlim(105, 15)
        for ax in (accuracy, coverage):
            ax.set_xticks([100, 50, 20])
            ax.set_xlabel("Training interactions retained (%)")
            ax.grid(axis="y", alpha=.22)
        handles, model_names = accuracy.get_legend_handles_labels()
        fig.legend(handles, model_names, loc="lower center", ncol=min(5, len(handles)),
                   frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, .12, 1, 1))
        save(fig, "ablation")

    nd = disagg[disagg.metric == "ndcg@10"]
    dims = list(dict.fromkeys(nd.dimension))
    fig, axes = plt.subplots(1, len(dims), figsize=(2.4 * len(dims), 2.6), sharey=False)
    for ax, dim in zip(np.atleast_1d(axes), dims):
        d = nd[nd.dimension == dim]
        groups = list(dict.fromkeys(d.group))
        width = .8 / len(specs)
        for n, spec in enumerate(specs):
            g = d[d.model == spec.key].set_index("group").loc[groups]
            x = np.arange(len(groups)) + (n - (len(specs) - 1) / 2) * width
            ax.bar(x, g["mean"], width, yerr=[g["mean"] - g["user_ci_low"],
                                              g["user_ci_high"] - g["mean"]],
                   capsize=1.5, label=labels[spec.key], error_kw={"lw": .6})
        ax.set_xticks(np.arange(len(groups)), [g.split(" (")[0] for g in groups])
        ax.set_title(dim.replace("_", " "), fontsize=8)
        ax.grid(axis="y", alpha=.25)
    np.atleast_1d(axes)[0].set_ylabel("Test NDCG@10")
    handles, names = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=len(specs), frameon=False,
               fontsize=7, bbox_to_anchor=(.5, 1.06))
    fig.tight_layout()
    save(fig, "disaggregation")

    if "bpr" in runs:
        fig, ax = plt.subplots(figsize=(3.6, 2.4))
        for r in runs["bpr"]:
            h = r["loss_history"]
            ax.plot(np.arange(1, len(h) + 1), h, lw=1, label=f"seed {r['seed']}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"Mean BPR loss $-\ln\sigma(x_{uij})$")
        ax.grid(alpha=.25)
        ax.legend(frameon=False, fontsize=6)
        save(fig, "loss", bpr_dir)


if __name__ == "__main__":
    main()
