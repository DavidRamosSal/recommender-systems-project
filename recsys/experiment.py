"""Shared Part 2 pipeline: every model goes through exactly the same steps.

1. tune          validation only, same budget for every model (Lecture 3.1, slide 10)
2. final runs    refit on train + validation with the selected setting, one run
                 per seed (deterministic models: one run), evaluate on test
3. compare       seed interval + user-bootstrap interval per model, paired
                 per-user tests between all pairs of models (Holm-corrected)
4. disaggregate  low/high-activity users, popular/long-tail items, early/late test
5. ablation      training interactions subsampled to 50 % and 20 %

The evaluation protocol is the Part 1 one and reuses its code:
recsys.evaluation.evaluation_users / ranking_metrics and recsys.ranking.top_k_items
(relevant = rating >= 4, users need history and >= 1 relevant item, full
3,706-movie catalogue, every already-rated movie excluded, ties -> lower item_id).

Adding a model: a ModelSpec whose build(params, seed) returns an object with
fit(history, user_ids=..., catalogue=...) (or fit(history)) and either
score_users(user_ids) -> (users x catalogue) or score(user_id, item_ids).
"""

from __future__ import annotations

import inspect
import itertools
import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .evaluation import evaluation_users, ranking_metrics
from .checkpoints import prepare_model
from .models.item_knn import interaction_matrix
from .ranking import top_k_items
from .stats import bootstrap_interval, holm, kendall_tau, paired_comparison, seed_interval

METRICS = ("precision", "recall", "ndcg")


@dataclass
class ModelSpec:
    key: str
    label: str
    paradigm: str
    build: Callable[[dict, int], object]
    configs: list[dict]
    deterministic: bool = False
    self_implemented: bool = False
    tuning_seed: int | None = None


@dataclass
class Setup:
    seeds: tuple = (0, 1, 2, 3, 4)
    fractions: tuple = (0.5, 0.2)
    head_share: float = 0.2          # top 20 % of the catalogue by history count
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 42
    progress: Callable = field(default=print)


def search_space(grid: dict, budget: int, seed: int) -> list[dict]:
    """The full grid if it fits in the budget, else `budget` random settings."""
    keys = list(grid)
    settings = list(itertools.product(*(grid[k] for k in keys)))
    if len(settings) > budget:
        chosen = np.random.default_rng(seed).choice(len(settings), budget, replace=False)
        settings = [settings[i] for i in sorted(chosen)]
    return [dict(zip(keys, s)) for s in settings]


# ----------------------------------------------------------------- windows
@dataclass
class Window:
    """One evaluation setting: what the model sees and what it is judged on."""
    history: pd.DataFrame    # interactions given to fit()
    target: pd.DataFrame     # validation or test ratings
    users: np.ndarray        # evaluated users (original IDs)
    relevant: np.ndarray     # users x catalogue, 1 = relevant
    seen: np.ndarray         # users x catalogue, 1 = excluded from ranking


def make_window(history, target, user_ids, catalogue, *, seen_history=None, users=None):
    """seen_history / users default to `history` / evaluation_users(history, target).

    The ablation passes the full history for both, so a subsampled model is
    ranked over the same candidates and judged on the same users.
    """
    seen_history = history if seen_history is None else seen_history
    users = evaluation_users(seen_history, target) if users is None else users
    rows = pd.Index(user_ids).get_indexer(users)
    return Window(
        history=history, target=target, users=users,
        relevant=interaction_matrix(target, user_ids, catalogue)[rows],
        seen=interaction_matrix(seen_history, user_ids, catalogue, positives_only=False)[rows],
    )


def restrict(window: Window, relevant: np.ndarray) -> Window:
    """Same top-k lists, different relevant set; drops users left with none."""
    keep = relevant.sum(axis=1) > 0
    return Window(window.history, window.target, window.users[keep], relevant[keep],
                  window.seen[keep])


# ----------------------------------------------------------------- one run
def fit_model(model, history, user_ids, catalogue, validation=None):
    """Pass the fixed user/catalogue axes when fit() accepts them."""
    parameters = inspect.signature(model.fit).parameters
    kwargs = {}
    if "user_ids" in parameters and "catalogue" in parameters:
        kwargs.update(user_ids=user_ids, catalogue=catalogue)
    if validation is not None and "validation" in parameters:
        def validation_ndcg(fitted):
            top = top_k_items(fitted.score_users(validation.users), validation.seen)
            return ranking_metrics(top, validation.relevant)["ndcg"].mean()
        kwargs["validation"] = validation_ndcg
    return model.fit(history, **kwargs) or model


def score_matrix(model, users, catalogue) -> np.ndarray:
    if hasattr(model, "score_users"):
        return np.asarray(model.score_users(users, catalogue), dtype=np.float64)
    return np.vstack([np.asarray(model.score(u, catalogue), dtype=np.float64) for u in users])


def run(spec: ModelSpec, params: dict, seed: int, window: Window, user_ids, catalogue,
        *, tuning=False, checkpoint=None, restore=False) -> dict:
    model = spec.build(params, seed)
    model, fit_seconds = prepare_model(
        model, lambda: fit_model(model, window.history, user_ids, catalogue, window if tuning else None),
        params, seed=seed, checkpoint=checkpoint, restore=restore,
    )
    top = top_k_items(score_matrix(model, window.users, catalogue), window.seen)
    return {
        "seed": seed, "top": top, "users": window.users,
        "per_user": per_user_frame(top, window),
        "coverage": np.unique(top).size / len(catalogue),
        "fit_seconds": fit_seconds,
        "params": {**params, "epochs": model.best_epoch} if hasattr(model, "best_epoch") else params,
        "loss_history": list(getattr(model, "loss_history", []) or []),
    }


def per_user_frame(top, window: Window) -> pd.DataFrame:
    metrics = ranking_metrics(top, window.relevant)
    return pd.DataFrame({m: np.asarray(metrics[m], dtype=float) for m in METRICS},
                        index=pd.Index(window.users, name="user_id"))


def seed_mean(runs, metric) -> pd.Series:
    """Per-user metric averaged over seeds."""
    return pd.concat([r["per_user"][metric] for r in runs], axis=1).mean(axis=1)


# ----------------------------------------------------------------- 1. tuning
def tune(specs, validation: Window, user_ids, catalogue, setup: Setup):
    trials, selected = [], {}
    for spec in specs:
        best = -np.inf
        for n, params in enumerate(spec.configs, 1):
            seed = setup.seeds[0] if spec.tuning_seed is None else spec.tuning_seed
            r = run(spec, params, seed, validation, user_ids, catalogue, tuning=True)
            row = {"model": spec.key, "trial": n, "params": json.dumps(r["params"]),
                   **{f"{m}@10": r["per_user"][m].mean() for m in METRICS},
                   "coverage": r["coverage"], "fit_seconds": round(r["fit_seconds"], 2)}
            trials.append(row)
            if row["ndcg@10"] > best:        # first maximum wins, as in Part 1
                best, selected[spec.key] = row["ndcg@10"], r["params"]
            setup.progress(f"[tune {spec.key} {n}/{len(spec.configs)}] {params} "
                           f"NDCG@10 {row['ndcg@10']:.4f}", flush=True)
    return pd.DataFrame(trials), selected


# ----------------------------------------------------------------- 2. final
def final_runs(specs, selected, test: Window, user_ids, catalogue, setup: Setup, tag="test",
               *, results_dir=None, restore=False):
    runs = {}
    for spec in specs:
        runs[spec.key] = []
        for seed in ((0,) if spec.deterministic else setup.seeds):
            checkpoint = None
            if results_dir is not None:
                filename = f"seed-{seed}"
                if spec.deterministic:
                    model = spec.build(selected[spec.key], seed)
                    filename = f"{getattr(model, 'name', spec.key)}-test"
                checkpoint = results_dir / spec.key / "weights" / f"{filename}.npz"
            r = run(spec, selected[spec.key], seed, test, user_ids, catalogue,
                    checkpoint=checkpoint, restore=restore)
            runs[spec.key].append(r)
            setup.progress(f"[{tag} {spec.key} seed {seed}] NDCG@10 "
                           f"{r['per_user']['ndcg'].mean():.4f} ({r['fit_seconds']:.0f}s)",
                           flush=True)
    return runs


# ----------------------------------------------------------------- 3. compare
def overall(specs, selected, runs, setup: Setup) -> pd.DataFrame:
    rows = []
    for spec in specs:
        rs = runs[spec.key]
        row = {"model": spec.key, "label": spec.label, "paradigm": spec.paradigm,
               "self_implemented": spec.self_implemented,
               "settings": json.dumps(selected[spec.key]),
               "seeds": " ".join(str(r["seed"]) for r in rs) if not spec.deterministic
               else "deterministic", "users": len(rs[0]["users"])}
        for m in METRICS:
            mean, lo, hi = seed_interval([r["per_user"][m].mean() for r in rs])
            _, ulo, uhi = bootstrap_interval(seed_mean(rs, m), samples=setup.bootstrap_samples,
                                             seed=setup.bootstrap_seed)
            row.update({f"{m}@10": mean, f"{m}@10_seed_ci_low": lo, f"{m}@10_seed_ci_high": hi,
                        f"{m}@10_user_ci_low": ulo, f"{m}@10_user_ci_high": uhi})
        row["coverage"], row["coverage_seed_ci_low"], row["coverage_seed_ci_high"] = \
            seed_interval([r["coverage"] for r in rs])
        row["fit_seconds"] = float(np.mean([r["fit_seconds"] for r in rs]))
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_tests(specs, runs, setup: Setup) -> pd.DataFrame:
    """All model pairs, per-user values averaged over seeds; Holm per metric."""
    rows = []
    for m in METRICS:
        block = []
        for a, b in itertools.combinations([s.key for s in specs], 2):
            block.append({"metric": f"{m}@10", "model_a": a, "model_b": b,
                          **paired_comparison(seed_mean(runs[a], m), seed_mean(runs[b], m),
                                              samples=setup.bootstrap_samples,
                                              seed=setup.bootstrap_seed)})
        if block:
            adjusted = holm([r["p_wilcoxon"] for r in block])
            for r, p in zip(block, adjusted):
                r["p_wilcoxon_holm"] = p
        rows += block
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- 4. disaggregation
def subgroups(test: Window, user_ids, catalogue, setup: Setup):
    """{(dimension, group): Window} plus the thresholds used."""
    history = test.history
    activity = history.groupby("user_id").size().reindex(test.users, fill_value=0).to_numpy()
    cut_users = float(np.median(activity))
    popularity = history.groupby("item_id").size().reindex(catalogue, fill_value=0).to_numpy()
    n_head = int(np.ceil(setup.head_share * len(catalogue)))
    head = np.zeros(len(catalogue), bool)
    head[np.argsort(-popularity, kind="stable")[:n_head]] = True
    cut_time = float(test.target["timestamp"].median())
    rows = pd.Index(user_ids).get_indexer(test.users)

    def relevant_in(window_part):
        return interaction_matrix(window_part, user_ids, catalogue)[rows]

    groups = {
        ("user_activity", "low (<= median)"): Window(
            test.history, test.target, test.users[activity <= cut_users],
            test.relevant[activity <= cut_users], test.seen[activity <= cut_users]),
        ("user_activity", "high (> median)"): Window(
            test.history, test.target, test.users[activity > cut_users],
            test.relevant[activity > cut_users], test.seen[activity > cut_users]),
        ("item_popularity", "popular (top 20 %)"): restrict(test, test.relevant * head),
        ("item_popularity", "long tail (other 80 %)"): restrict(test, test.relevant * ~head),
        ("test_period", "early half"): restrict(
            test, relevant_in(test.target[test.target["timestamp"] <= cut_time])),
        ("test_period", "late half"): restrict(
            test, relevant_in(test.target[test.target["timestamp"] > cut_time])),
    }
    thresholds = {
        "user_activity_median_history_ratings": cut_users,
        "popular_items": n_head,
        "popular_min_history_ratings": int(popularity[head].min()),
        "popular_share_of_history_ratings": float(popularity[head].sum() / popularity.sum()),
        "test_period_split_utc": pd.Timestamp(cut_time, unit="s", tz="UTC").isoformat(),
    }
    return groups, thresholds


def disaggregate(specs, runs, test: Window, user_ids, catalogue, setup: Setup):
    groups, thresholds = subgroups(test, user_ids, catalogue, setup)
    rows = []
    for spec in specs:
        for (dimension, group), w in groups.items():
            index = pd.Index(test.users).get_indexer(w.users)
            per_seed = [per_user_frame(r["top"][index], w) for r in runs[spec.key]]
            for m in METRICS:
                values = pd.concat([p[m] for p in per_seed], axis=1).mean(axis=1)
                mean, lo, hi = bootstrap_interval(values, samples=setup.bootstrap_samples,
                                                  seed=setup.bootstrap_seed)
                rows.append({"model": spec.key, "dimension": dimension, "group": group,
                             "metric": f"{m}@10", "users": len(values), "mean": mean,
                             "user_ci_low": lo, "user_ci_high": hi,
                             "relevant_items": int(w.relevant.sum())})
    return pd.DataFrame(rows), thresholds


# ----------------------------------------------------------------- 5. ablation
def sparsity_ablation(specs, selected, full_runs, test: Window, user_ids, catalogue,
                      setup: Setup):
    """Retrain on a uniform random share of the history (one subsample per seed).

    Hyperparameters stay at the values selected on the full data. Users and the
    excluded (already rated) items are those of the full history, so only the
    information available to the model changes between fractions.
    """
    rows, info = [], []
    per_fraction = {1.0: full_runs}
    for fraction in setup.fractions:
        per_fraction[fraction] = {spec.key: [] for spec in specs}
        for seed in setup.seeds:
            sample = test.history.sample(frac=fraction, random_state=seed)
            w = make_window(sample, test.target, user_ids, catalogue,
                            seen_history=test.history, users=test.users)
            left = sample.groupby("user_id").size().reindex(test.users, fill_value=0)
            info.append({"fraction": fraction, "seed": seed, "history_ratings": len(sample),
                         "evaluated_users": len(test.users),
                         "evaluated_users_without_history": int((left == 0).sum()),
                         "catalogue_items_without_history": int(
                             len(catalogue) - sample["item_id"].nunique())})
            for spec in specs:
                r = run(spec, selected[spec.key], seed, w, user_ids, catalogue)
                per_fraction[fraction][spec.key].append(r)
                setup.progress(f"[ablation {fraction:.0%} {spec.key} seed {seed}] NDCG@10 "
                               f"{r['per_user']['ndcg'].mean():.4f}", flush=True)
    for fraction, runs in per_fraction.items():
        for spec in specs:
            rs = runs[spec.key]
            row = {"fraction": fraction, "model": spec.key, "runs": len(rs)}
            for m in METRICS:
                row[f"{m}@10"], row[f"{m}@10_ci_low"], row[f"{m}@10_ci_high"] = \
                    seed_interval([r["per_user"][m].mean() for r in rs])
            row["coverage"] = float(np.mean([r["coverage"] for r in rs]))
            rows.append(row)
    table = pd.DataFrame(rows)
    table["rank_ndcg"] = table.groupby("fraction")["ndcg@10"].rank(ascending=False,
                                                                  method="min").astype(int)
    full = table[table.fraction == 1.0].set_index("model")["ndcg@10"]
    taus = pd.DataFrame([{"fraction": f, "kendall_tau_vs_full": kendall_tau(
        full.values, g.set_index("model").loc[full.index, "ndcg@10"].values)}
        for f, g in table.groupby("fraction")])
    return table, taus, pd.DataFrame(info)
