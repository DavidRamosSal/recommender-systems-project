"""Part 3, Track A: MMR re-ranking of BPR and the accuracy-diversity frontier.

    uv run --locked scripts/run_part3.py --data-dir data/ml-1m

Same split, users, candidate set and accuracy metrics as Parts 1-2. Base
recommender: our own BPR-MF with the Part 2 setting (results/bpr/selected.json).

1. validation  BPR fitted on train only (seed 0), lambda swept on the
               validation window; the operating point is chosen here.
2. test        the five Part 2 BPR fits (seeds 0-4, train + validation),
               the same lambda grid; per-user values are averaged over seeds.
3. statistics  95 % user-bootstrap intervals for every per-user metric,
               seed intervals for system metrics, and a paired Wilcoxon test
               of every lambda against lambda = 1 (the base ranking),
               Holm-corrected over the lambda grid, per metric.

Operating point rule (stated in the report): the smallest lambda (= most
diversification) whose validation NDCG@10 is at most MAX_NDCG_LOSS below the
base ranking. The test set is never used to choose lambda.

Outputs under results/part3/: results-part3.csv (Part 3A template columns),
selected.json, intermediate/*.csv, images/frontier.* and images/lambda-sweep.*.
Seeds: BPR fits 0-4 (test) and 0 (validation), bootstrap 42.
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

from recsys.checkpoints import load_weights, prepare_model
from recsys.data import load_movielens
from recsys.diversity import PER_USER, SYSTEM, ListContext
from recsys.evaluation import ranking_metrics
from recsys.experiment import METRICS, Setup, fit_model, make_window
from recsys.models import BPR
from recsys.outputs import OutputPaths, write_results_csv
from recsys.rerank import CANDIDATES, ITEM_DISTANCE, candidate_lists, genre_matrix, minmax, rerank
from recsys.splitting import temporal_split_by_date
from recsys.stats import bootstrap_interval, holm, paired_comparison, seed_interval

ROOT = Path(__file__).resolve().parents[1]
LAMBDAS = tuple(round(x, 2) for x in np.linspace(1.0, 0.0, 21))   # 1.00, 0.95, ..., 0.00
MAX_NDCG_LOSS = 0.05       # operating point: <= 5 % relative NDCG@10 loss on validation
VALIDATION_SEED = 0
PART3A_COLUMNS = [
    "Base model", "Re-ranking method", "Trade-off parameter", "Item distance",
    "Precision@10", "Recall@10", "NDCG@10", "Intra-list diversity", "Catalogue coverage",
    "Gini", "Calibration error", "Notes",
]
TEMPLATE_METRIC = {"ild": "Intra-list diversity", "coverage": "Catalogue coverage",
                   "gini": "Gini", "calibration": "Calibration error"}
LABELS = {"precision": "Precision@10", "recall": "Recall@10", "ndcg": "NDCG@10",
          "ild": "Intra-list diversity", "coverage": "Catalogue coverage", "gini": "Gini",
          "calibration": "Calibration error", "candidate_coverage": "Candidate-pool coverage"}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/ml-1m")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--candidates", type=int, default=CANDIDATES)
    parser.add_argument("--max-ndcg-loss", type=float, default=MAX_NDCG_LOSS)
    parser.add_argument("--retrain", action="store_true",
                        help="refit the validation BPR model instead of loading its weights")
    args = parser.parse_args()
    run_part3(args.data_dir, args.out_dir, n_candidates=args.candidates,
              max_loss=args.max_ndcg_loss, retrain=args.retrain)


def run_part3(data_dir, out_dir, *, dataset=None, retrain=False, lambdas=LAMBDAS,
              n_candidates=CANDIDATES, max_loss=MAX_NDCG_LOSS, seeds=(0, 1, 2, 3, 4)):
    started = time.perf_counter()
    root = Path(out_dir)
    paths = OutputPaths(root / "part3").create()
    setup = Setup(seeds=seeds)
    dataset = load_movielens(data_dir) if dataset is None else dataset
    ratings = dataset.ratings
    user_ids = np.sort(ratings["user_id"].unique())
    catalogue = np.sort(ratings["item_id"].unique())
    train, validation, test = temporal_split_by_date(ratings, "2001-01-01", "2001-07-01")
    history = pd.concat([train, validation])
    features = genre_matrix(dataset.movies, catalogue)
    if 1.0 not in lambdas:
        raise ValueError("The lambda grid must contain 1.0 (the base ranking).")

    bpr_dir = root / "bpr" if (root / "bpr/selected.json").is_file() else ROOT / "results/bpr"
    params = json.loads((bpr_dir / "selected.json").read_text())
    print(f"Part 3: MMR on BPR {params}, {n_candidates} candidates, {len(lambdas)} lambdas",
          flush=True)

    # 1. validation: BPR on train only, lambda chosen here
    val_window = make_window(train, validation, user_ids, catalogue)
    checkpoint = paths.root / "weights" / f"bpr-validation-seed-{VALIDATION_SEED}.npz"
    if not checkpoint.is_file() and (ROOT / "results/part3/weights" / checkpoint.name).is_file():
        checkpoint = ROOT / "results/part3/weights" / checkpoint.name
    model = BPR(**params, seed=VALIDATION_SEED)
    restore = checkpoint.is_file() and not retrain
    if not restore:
        checkpoint = paths.root / "weights" / checkpoint.name
        print("Fitting BPR on the training period for the validation sweep...", flush=True)
    model, _ = prepare_model(model, lambda: fit_model(model, train, user_ids, catalogue),
                             params, seed=VALIDATION_SEED, checkpoint=checkpoint, restore=restore)
    val_runs = sweep([model], val_window, features, catalogue, lambdas, n_candidates)
    val_table = summarise(val_runs, setup)
    base_ndcg = val_table.loc[val_table["lambda"] == 1.0, "ndcg@10"].item()
    allowed = val_table[val_table["ndcg@10"] >= (1 - max_loss) * base_ndcg]
    chosen = float(allowed["lambda"].min())
    selection = {
        "lambda": chosen, "rule": f"smallest lambda with validation NDCG@10 >= "
                                  f"(1 - {max_loss:g}) x NDCG@10 at lambda = 1",
        "max_relative_ndcg_loss": max_loss, "candidates": n_candidates,
        "validation_ndcg_base": base_ndcg,
        "validation_ndcg_selected": allowed.loc[allowed["lambda"] == chosen, "ndcg@10"].item(),
        "base_model": "bpr", "base_params": params, "item_distance": ITEM_DISTANCE,
    }
    (paths.root / "selected.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Validation: lambda = {chosen:g} (NDCG@10 {selection['validation_ndcg_selected']:.4f} "
          f"vs {base_ndcg:.4f} at lambda = 1)", flush=True)

    # 2. test: the five Part 2 fits on train + validation
    test_window = make_window(history, test, user_ids, catalogue)
    models = []
    for seed in seeds:
        m = BPR(**params, seed=seed)
        load_weights(bpr_dir / "weights" / f"seed-{seed}.npz", m, params, seed)
        models.append(m)
    test_runs = sweep(models, test_window, features, catalogue, lambdas, n_candidates)
    test_table = summarise(test_runs, setup)
    paired = paired_tests(test_runs, setup)

    val_table.insert(0, "split", "validation")
    test_table.insert(0, "split", "test")
    for table in (val_table, test_table):
        table["selected"] = table["lambda"] == chosen
    val_table.to_csv(paths.intermediate / "frontier-validation.csv", index=False)
    test_table.to_csv(paths.intermediate / "frontier-test.csv", index=False)
    paired.to_csv(paths.intermediate / "paired-test.csv", index=False)
    per_user_frame(test_runs).to_csv(paths.intermediate / "per-user-test.csv", index=False)
    export_results(test_table, paired, paths.root / "results-part3.csv", selection, setup, seeds)
    plot_all(val_table, test_table, chosen, paths.images)

    cols = ["lambda", "ndcg@10", "ild@10", "coverage", "candidate_coverage"]
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(test_table[[c for c in cols if c in test_table]].to_string(index=False))
    minutes = (time.perf_counter() - started) / 60
    print(f"Part 3 complete in {minutes:.1f} min; outputs in {paths.root}", flush=True)
    return test_table


# ---------------------------------------------------------------- sweep
def sweep(models, window, features, catalogue, lambdas, n_candidates):
    """{lambda: [per-seed run]} with per-user frames and system metrics."""
    runs = {lam: [] for lam in lambdas}
    ctx = ListContext(features=features, catalogue=catalogue, users=window.users,
                      history=window.history, seen=window.seen)
    for model in models:
        scores = np.asarray(model.score_users(window.users, catalogue), dtype=np.float64)
        candidates, base = candidate_lists(scores, window.seen, n_candidates)
        relevance = minmax(base)
        pool = np.unique(candidates).size / len(catalogue)
        for lam in lambdas:
            top = rerank(candidates, relevance, features, lam)
            accuracy = ranking_metrics(top, window.relevant)
            per_user = pd.DataFrame({m: np.asarray(accuracy[m], float) for m in METRICS},
                                    index=pd.Index(window.users, name="user_id"))
            for name, metric in PER_USER.items():
                per_user[name] = metric(top, ctx)
            system = {name: float(metric(top, ctx)) for name, metric in SYSTEM.items()}
            system["candidate_coverage"] = pool
            runs[lam].append({"seed": model.seed, "per_user": per_user, "system": system})
    return runs


def seed_mean(runs, metric):
    return pd.concat([r["per_user"][metric] for r in runs], axis=1).mean(axis=1)


def summarise(runs, setup):
    rows = []
    for lam, rs in runs.items():
        row = {"lambda": lam, "seeds": " ".join(str(r["seed"]) for r in rs),
               "users": len(rs[0]["per_user"])}
        for metric in rs[0]["per_user"].columns:
            mean, low, high = bootstrap_interval(seed_mean(rs, metric),
                                                 samples=setup.bootstrap_samples,
                                                 seed=setup.bootstrap_seed)
            row.update({f"{metric}@10": mean, f"{metric}@10_user_ci_low": low,
                        f"{metric}@10_user_ci_high": high})
        for metric in rs[0]["system"]:
            mean, low, high = seed_interval([r["system"][metric] for r in rs])
            row.update({metric: mean, f"{metric}_seed_ci_low": low,
                        f"{metric}_seed_ci_high": high})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(runs, setup):
    """Every lambda against lambda = 1 on the same users; Holm over the grid."""
    rows = []
    base = runs[1.0]
    for metric in base[0]["per_user"].columns:
        block = [{"metric": f"{metric}@10", "lambda": lam, "against": 1.0,
                  **paired_comparison(seed_mean(rs, metric), seed_mean(base, metric),
                                      samples=setup.bootstrap_samples,
                                      seed=setup.bootstrap_seed)}
                 for lam, rs in runs.items() if lam != 1.0]
        for r, p in zip(block, holm([r["p_wilcoxon"] for r in block])):
            r["p_wilcoxon_holm"] = p
        rows += block
    return pd.DataFrame(rows)


def per_user_frame(runs):
    """Seed-averaged per-user values at every lambda (for the disaggregation)."""
    frames = []
    for lam, rs in runs.items():
        metrics = rs[0]["per_user"].columns
        frame = pd.DataFrame({m: seed_mean(rs, m) for m in metrics})
        frames.append(frame.reset_index().assign(**{"lambda": lam}))
    return pd.concat(frames, ignore_index=True)[["lambda", "user_id", *metrics]]


# ---------------------------------------------------------------- outputs
def export_results(table, paired, destination, selection, setup, seeds):
    """One row per lambda in the Part 3A template; lambda = 1 is the base model."""
    rows = []
    bootstrap = f"{setup.bootstrap_samples} samples, seed {setup.bootstrap_seed}"
    for r in table.to_dict("records"):
        lam = r["lambda"]
        notes = [f"Test, mean over BPR fits (seeds {', '.join(map(str, seeds))}), {r['users']} users",
                 f"95% user bootstrap ({bootstrap}) NDCG@10 "
                 f"[{r['ndcg@10_user_ci_low']:.4f}, {r['ndcg@10_user_ci_high']:.4f}], ILD "
                 f"[{r['ild@10_user_ci_low']:.4f}, {r['ild@10_user_ci_high']:.4f}]"]
        if lam == 1.0:
            notes.insert(0, "Unmodified base recommender (MMR with lambda = 1 returns the BPR top 10)")
        else:
            tests = paired[paired["lambda"] == lam].set_index("metric")
            notes.append("Paired Wilcoxon vs lambda = 1 (Holm over 20 lambdas): "
                         f"NDCG@10 diff {tests.loc['ndcg@10', 'mean_difference']:+.4f}, "
                         f"p = {tests.loc['ndcg@10', 'p_wilcoxon_holm']:.2g}; ILD diff "
                         f"{tests.loc['ild@10', 'mean_difference']:+.4f}, "
                         f"p = {tests.loc['ild@10', 'p_wilcoxon_holm']:.2g}")
        if r["selected"]:
            notes.insert(0, f"Selected operating point ({selection['rule']}, on validation)")
        notes.append(f"Candidate-pool coverage {r['candidate_coverage']:.4f}")
        row = {
            "Base model": "BPR-MF (own implementation)",
            "Re-ranking method": "None (base)" if lam == 1.0
            else f"MMR over the top {selection['candidates']} BPR candidates",
            "Trade-off parameter": f"lambda = {lam:.2f}",
            "Item distance": ITEM_DISTANCE,
            "Precision@10": r["precision@10"], "Recall@10": r["recall@10"],
            "NDCG@10": r["ndcg@10"],
            "Notes": "; ".join(notes) + ".",
        }
        for key, column in TEMPLATE_METRIC.items():
            value = r.get(f"{key}@10", r.get(key))
            row[column] = "" if value is None or pd.isna(value) else value
        rows.append(row)
    write_results_csv(Path(destination), rows, columns=PART3A_COLUMNS)


def plot_all(val, test, chosen, images):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "recsys-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    marked = {1.0, 0.8, 0.6, 0.4, 0.2, 0.0, chosen}

    def save(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(images / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # accuracy-diversity frontier (Lecture 5.1, slide 13)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, table, title, color in ((axes[0], val, "Validation (lambda chosen here)", "#7a8fa6"),
                                    (axes[1], test, "Test (5 BPR fits)", "#1f5a96")):
        x, y = table["ild@10"], table["ndcg@10"]
        ax.errorbar(x, y, xerr=[x - table["ild@10_user_ci_low"], table["ild@10_user_ci_high"] - x],
                    yerr=[y - table["ndcg@10_user_ci_low"], table["ndcg@10_user_ci_high"] - y],
                    fmt="o-", ms=3, lw=1.3, color=color, ecolor=color, elinewidth=.5,
                    capsize=0, alpha=.95)
        for r in table.to_dict("records"):
            if r["lambda"] in marked:
                ax.annotate(f"λ={r['lambda']:g}", (r["ild@10"], r["ndcg@10"]),
                            textcoords="offset points", xytext=(4, 3), fontsize=6.5)
        pick = table[table["lambda"] == chosen]
        ax.scatter(pick["ild@10"], pick["ndcg@10"], s=70, facecolors="none",
                   edgecolors="#c0392b", lw=1.4, zorder=5, label=f"operating point λ={chosen:g}")
        ax.set(xlabel="Intra-list diversity@10 (1 − cosine, genres)", ylabel="NDCG@10", title=title)
        ax.grid(alpha=.22)
        ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    save(fig, "frontier")

    # every metric against lambda (more diversification to the right)
    metrics = ["ndcg@10", "ild@10"] + [c for c in ("calibration@10",) if c in test]
    system = ["coverage"] + [c for c in ("gini",) if c in test]
    panels = metrics + system
    fig, axes = plt.subplots(1, len(panels), figsize=(2.3 * len(panels), 2.4))
    for ax, column in zip(np.atleast_1d(axes), panels):
        lam, y = test["lambda"], test[column]
        if column.endswith("@10"):
            low, high = test[f"{column}_user_ci_low"], test[f"{column}_user_ci_high"]
        else:
            low, high = test[f"{column}_seed_ci_low"], test[f"{column}_seed_ci_high"]
        ax.plot(lam, y, marker="o", ms=2.5, lw=1.3, color="#1f5a96", label="test")
        ax.fill_between(lam, low, high, color="#1f5a96", alpha=.18, lw=0)
        if column == "coverage":
            ax.plot(lam, test["candidate_coverage"], ls="--", lw=1, color="#888",
                    label="candidate pool (upper bound)")
            ax.legend(frameon=False, fontsize=6.5)
        ax.axvline(chosen, color="#c0392b", ls=":", lw=1)
        ax.set_xlim(1.02, -0.02)
        ax.set(xlabel="λ (← relevance · diversity →)",
               title=LABELS[column.removesuffix("@10")])
        ax.grid(alpha=.22)
    fig.tight_layout()
    save(fig, "lambda-sweep")


if __name__ == "__main__":
    main()
