"""Part 2: tune, compare, disaggregate and ablate all models in one pipeline.

    uv run --locked scripts/run_part2.py --data-dir data/ml-1m

Same split and protocol as Part 1 (see scripts/run_baselines.py). Every model
gets the same validation budget (BUDGET settings; smaller spaces are searched
exhaustively). Seeds: models 0-4, BPR random search 2026, bootstrap 42.

Outputs (default results/part2/):
    weights/                     selected full-history model, one file per fit
    results-part2.csv            test results in the results-workbook columns
    intermediate/tuning.csv      every validation trial
    intermediate/selected.json   selected setting per model
    intermediate/overall.csv     test metrics, seed CI and user-bootstrap CI
    intermediate/pairwise.csv    paired per-user tests, all model pairs (Holm)
    intermediate/disaggregated.csv, subgroups.json
    intermediate/ablation.csv, ablation-tau.csv, ablation-info.csv
    intermediate/per-user-ndcg.csv
    images/ablation.png|svg, disaggregation.png|svg, bpr-loss.png|svg
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data import load_movielens
from recsys.checkpoints import ratings_digest
from recsys.experiment import (
    METRICS, ModelSpec, Setup, disaggregate, final_runs, make_window, overall,
    pairwise_tests, search_space, seed_mean, sparsity_ablation, tune,
)
from recsys.models import BPR, ItemKNN, MarkovChain
from recsys.models.most_popular import MostPopular, validation_configs as popularity_configs
from recsys.models.multvae import MultVAE, validation_configs as multvae_configs
from recsys.outputs import OutputPaths, copy_outputs
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
        ModelSpec("mostpop", "Most Popular", "baseline (Part 1)",
                  lambda p, seed: MostPopular(**p), popularity_configs()[:budget], deterministic=True),
        ModelSpec("itemknn", "Item-based kNN", "baseline (Part 1)",
                  lambda p, seed: ItemKNN(**p), itemknn[:budget], deterministic=True),
        ModelSpec("bpr", "BPR-MF (own implementation)", "ranking-loss CF",
                  lambda p, seed: BPR(**p, seed=seed),
                  search_space(BPR_GRID, budget, SEARCH_SEED), self_implemented=True),
        ModelSpec("markov", "Markov chain", "sequential",
                  lambda p, seed: MarkovChain(**p),
                  [{"order": order} for order in range(1, budget + 1)], deterministic=True),
        ModelSpec("multvae", "Mult-VAE", "neural", lambda p, seed: MultVAE(**p, seed=seed),
                  multvae_configs(budget), tuning_seed=42),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/ml-1m")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/part2")
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
    paths = OutputPaths(out_dir).create()
    setup = Setup(seeds=seeds)
    dataset = load_movielens(data_dir) if dataset is None else dataset
    ratings = dataset.ratings
    user_ids = np.sort(ratings["user_id"].unique())
    catalogue = np.sort(ratings["item_id"].unique())           # 3,706 rated movies
    train, validation, test = temporal_split_by_date(ratings, "2001-01-01", "2001-07-01")
    history = pd.concat([train, validation])

    specs = [s for s in model_specs(budget) if models is None or s.key in models]
    test_window = make_window(history, test, user_ids, catalogue)
    weights = paths.root / "weights"
    manifest = {"ratings_sha256": ratings_digest(ratings), "seeds": list(seeds),
                "budget": budget, "models": [s.key for s in specs], "ablation": ablation}
    if saved_dir is None:
        weights.mkdir(parents=True, exist_ok=True)
        (weights / "manifest.json").unlink(missing_ok=True)
        val_window = make_window(train, validation, user_ids, catalogue)
        print(f"Part 2: {len(specs)} models, budget {budget}, seeds {list(seeds)}; "
              f"{len(val_window.users)} validation / {len(test_window.users)} test users", flush=True)
        trials, selected = tune(specs, val_window, user_ids, catalogue, setup)
        trials.to_csv(paths.intermediate / "tuning.csv", index=False)
        (paths.intermediate / "selected.json").write_text(json.dumps(selected, indent=2))
    else:
        saved_dir = Path(saved_dir)
        weights = saved_dir / "weights"
        metadata = weights / "manifest.json"
        if not metadata.is_file() or json.loads(metadata.read_text()) != manifest:
            raise ValueError("Saved Part 2 run differs from the dataset or settings. Use --retrain.")
        copy_outputs(saved_dir / "intermediate", paths.intermediate, ["tuning.csv", "selected.json"])
        trials = pd.read_csv(paths.intermediate / "tuning.csv")
        selected = json.loads((paths.intermediate / "selected.json").read_text())
        print(f"Evaluating saved weights on {len(test_window.users)} test users.", flush=True)

    runs = final_runs(specs, selected, test_window, user_ids, catalogue, setup,
                      weights_dir=weights, restore=saved_dir is not None)
    table = overall(specs, selected, runs, setup)
    table.to_csv(paths.intermediate / "overall.csv", index=False)
    pairwise_tests(specs, runs, setup).to_csv(paths.intermediate / "pairwise.csv", index=False)
    pd.DataFrame({s.key: seed_mean(runs[s.key], "ndcg") for s in specs}).to_csv(
        paths.intermediate / "per-user-ndcg.csv")
    disagg, thresholds = disaggregate(specs, runs, test_window, user_ids, catalogue, setup)
    disagg.to_csv(paths.intermediate / "disaggregated.csv", index=False)
    (paths.intermediate / "subgroups.json").write_text(json.dumps(thresholds, indent=2))

    abl = None
    if ablation and saved_dir is not None:
        copy_outputs(saved_dir / "intermediate", paths.intermediate,
                     ["ablation.csv", "ablation-tau.csv", "ablation-info.csv"])
        abl = pd.read_csv(paths.intermediate / "ablation.csv")
    elif ablation:
        abl, taus, info = sparsity_ablation(specs, selected, runs, test_window, user_ids,
                                            catalogue, setup)
        abl.to_csv(paths.intermediate / "ablation.csv", index=False)
        taus.to_csv(paths.intermediate / "ablation-tau.csv", index=False)
        info.to_csv(paths.intermediate / "ablation-info.csv", index=False)

    export_results(table, trials, budget, paths.root / "results-part2.csv")
    plot_all(specs, runs, disagg, abl, paths.images)
    if saved_dir is None:
        (weights / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    minutes = (time.perf_counter() - started) / 60
    cols = ["model"] + [f"{m}@10" for m in METRICS] + ["coverage"]
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(table[cols].to_string(index=False))
    print(f"Part 2 complete in {minutes:.1f} min; outputs in {paths.root}", flush=True)
    return table


# ---------------------------------------------------------------- outputs
HEADERS = ["Model", "Variant / settings", "Similarity", "k", "Precision@10", "Recall@10",
           "NDCG@10", "Catalogue coverage", "Seed", "Notes"]


def export_results(table, trials, budget, destination):
    """Test rows in the same ten columns as the Part 1 results.csv / workbook."""
    rows = []
    for r in table.to_dict("records"):
        settings = json.loads(r["settings"])
        tried = int((trials.model == r["model"]).sum())
        notes = [f"Selected on validation NDCG@10 among {tried} settings (budget {budget})",
                 f"test users {r['users']}"]
        for m, label in (("ndcg", "NDCG@10"), ("recall", "Recall@10")):
            if r["seeds"] != "deterministic":
                notes.append(f"95% seed CI {label} [{r[f'{m}@10_seed_ci_low']:.4f}, "
                             f"{r[f'{m}@10_seed_ci_high']:.4f}]")
            notes.append(f"95% user-bootstrap CI {label} [{r[f'{m}@10_user_ci_low']:.4f}, "
                         f"{r[f'{m}@10_user_ci_high']:.4f}] (1000 samples, seed 42)")
        if r["self_implemented"]:
            notes.insert(0, "Self-implemented (recsys/models/bpr.py)")
        rows.append({
            "Model": r["label"],
            "Variant / settings": ", ".join(f"{k}={v}" for k, v in settings.items()
                                            if k not in ("similarity", "k")),
            "Similarity": settings.get("similarity", "–"),
            "k": str(settings.get("k", "–")),
            "Precision@10": r["precision@10"], "Recall@10": r["recall@10"],
            "NDCG@10": r["ndcg@10"], "Catalogue coverage": r["coverage"],
            "Seed": "deterministic" if r["seeds"] == "deterministic"
            else f"mean over seeds {r['seeds']}",
            "Notes": "; ".join(notes),
        })
    with open(destination, "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=HEADERS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_all(specs, runs, disagg, abl, images):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "recsys-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False})
    labels = {s.key: s.label.replace(" (own implementation)", " (own)") for s in specs}

    def save(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(images / f"{name}.{ext}", dpi=300, bbox_inches="tight",
                        facecolor="white")
        plt.close(fig)

    if abl is not None:
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        for key, g in abl.groupby("model", sort=False):
            g = g.sort_values("fraction")
            low = (g["ndcg@10"] - g["ndcg@10_ci_low"]).fillna(0)
            high = (g["ndcg@10_ci_high"] - g["ndcg@10"]).fillna(0)
            ax.errorbar(100 * g["fraction"], g["ndcg@10"], yerr=[low, high], marker="o",
                        ms=3, capsize=2, label=labels[key])
        ax.set_xticks([20, 50, 100])
        ax.set_xlabel("Share of training interactions kept (%)")
        ax.set_ylabel("Test NDCG@10")
        ax.grid(alpha=.25)
        ax.legend(frameon=False, fontsize=7)
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
        save(fig, "bpr-loss")


if __name__ == "__main__":
    main()
