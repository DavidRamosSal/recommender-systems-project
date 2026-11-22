"""Tune Markov order and compare it with the Part 1 baselines on MovieLens."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data import load_movielens
from recsys.evaluation import bootstrap_ci, evaluation_users, ranking_metrics
from recsys.models import ItemKNN, MarkovChain
from recsys.models.item_knn import interaction_matrix
from recsys.models.most_popular import MostPopular, validation_configs
from recsys.ranking import top_k_items
from recsys.splitting import temporal_split_by_date
from recsys.tuning import selected_settings, tune_itemknn


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results/markov"
ORDERS = range(1, 61)
SEED = 42


def evaluation_data(history, window, catalogue):
    users = evaluation_users(history, window)
    seen = interaction_matrix(
        history[history.user_id.isin(users)], users, catalogue, positives_only=False,
    )
    relevant = interaction_matrix(window[window.user_id.isin(users)], users, catalogue)
    return users, seen, relevant


def markov_top(order, history, users, catalogue, seen):
    model = MarkovChain(order=order).fit(history)
    scores = np.stack([model.score(user, catalogue) for user in users])
    matched_users = sum(model.last_movies.get(user) in model.transition_counts for user in users)
    return top_k_items(scores, seen), matched_users


def popular_top(model, history, users, catalogue, seen):
    model.fit(history)
    scores = np.broadcast_to(model.score(users[0], catalogue), seen.shape)
    return top_k_items(scores, seen)


def summarize(top, relevant, items=None):
    if items is not None:
        relevant = relevant * items
        users = relevant.any(axis=1)
        top, relevant = top[users], relevant[users]
    if len(top) == 0:
        return dict.fromkeys(
            ("precision@10", "recall@10", "ndcg@10", "coverage"), np.nan,
        )
    metrics = ranking_metrics(top, relevant)
    recommended = top if items is None else top[items[top]]
    catalogue_size = relevant.shape[1] if items is None else items.sum()
    return {
        **{f"{name}@10": values.mean() for name, values in metrics.items()},
        "coverage": np.unique(recommended).size / catalogue_size,
    }


def validation_search(train, validation, user_ids, catalogue, orders, item_knn_grid=None):
    users, seen, relevant = evaluation_data(train, validation, catalogue)
    print(f"Searching {len(orders)} Markov orders on {len(users)} validation users", flush=True)
    rows = []
    for order in orders:
        top, matched = markov_top(order, train, users, catalogue, seen)
        rows.append({
            "order": order, "users": len(users), "matched_state_users": matched,
            **summarize(top, relevant),
        })
        if len(rows) % 10 == 0:
            print(f"Markov tuning: {len(rows)}/{len(orders)} orders", flush=True)
    markov_grid = pd.DataFrame(rows)

    if item_knn_grid is None:
        item_knn_grid = tune_itemknn(train, validation, user_ids, catalogue)
    popular_models = [MostPopular(**params) for params in validation_configs()]
    print(f"Searching {len(popular_models)} Most Popular settings on validation", flush=True)
    popular_grid = pd.DataFrame([
        {"model": model.name, **summarize(
            popular_top(model, train, users, catalogue, seen), relevant,
        )}
        for model in popular_models
    ])
    return markov_grid, item_knn_grid, popular_grid, popular_models


def disaggregate(top, relevant, low_activity, few_interactions):
    yield {
        "group": "overall", "users": len(top), **summarize(top, relevant),
    }
    for group, mask in (("low-activity users", low_activity), ("high-activity users", ~low_activity)):
        yield {
            "group": group, "users": int(mask.sum()), **summarize(top[mask], relevant[mask]),
        }
    for group, mask in (
        ("few training interactions (items)", few_interactions),
        ("many training interactions (items)", ~few_interactions),
    ):
        yield {
            "group": group, "users": int(relevant[:, mask].any(axis=1).sum()),
            **summarize(top, relevant, items=mask),
        }


def paired_differences(tops, relevant, markov_name):
    rng = np.random.default_rng(SEED)
    markov_ndcg = ranking_metrics(tops[markov_name], relevant)["ndcg"]
    rows = []
    for name, top in tops.items():
        if name == markov_name:
            continue
        delta = markov_ndcg - ranking_metrics(top, relevant)["ndcg"]
        low, high = bootstrap_ci(delta, rng)
        # Swap the models within users to simulate the paired null hypothesis.
        signs = 2 * rng.integers(0, 2, size=(9_999, len(delta)), dtype=np.int8) - 1
        extremes = np.count_nonzero(np.abs(signs @ delta) >= abs(delta.sum()))
        rows.append({
            "model": markov_name, "baseline": name, "users": len(delta),
            "ndcg@10_difference": delta.mean(), "ci_low": low, "ci_high": high,
            "p_value": (1 + extremes) / (1 + len(signs)),
        })
    return pd.DataFrame(rows)


def run_comparison(ratings, *, order=None, out_dir=OUTPUT_DIR, item_knn_grid=None):
    user_ids = np.sort(ratings.user_id.unique())
    catalogue = np.sort(ratings.item_id.unique())
    train, validation, test = temporal_split_by_date(ratings, "2001-01-01", "2001-07-01")

    orders = ORDERS if order is None else (order,)
    markov_grid, item_knn_grid, popular_grid, popular_models = validation_search(
        train, validation, user_ids, catalogue, orders, item_knn_grid,
    )
    # The grid is in ascending order, so exact ties choose the shorter history.
    best_markov = markov_grid.loc[markov_grid["ndcg@10"].idxmax()]
    order = int(best_markov["order"])
    markov_name = f"markov_order{order}"
    item_knn = ItemKNN(**selected_settings(item_knn_grid))
    best_popular_name = popular_grid.loc[popular_grid["ndcg@10"].idxmax(), "model"]
    all_time_name = MostPopular("all", "positive").name
    popular_models = [
        model for model in popular_models if model.name in (best_popular_name, all_time_name)
    ]
    print(
        f"Selected {markov_name}, {item_knn.name}, and {best_popular_name} on validation.",
        flush=True,
    )
    print(
        f"Markov matched a learned state for {int(best_markov['matched_state_users'])}"
        f"/{int(best_markov['users'])} validation users.",
        flush=True,
    )

    full_history = pd.concat([train, validation], ignore_index=True)
    users, seen, relevant = evaluation_data(full_history, test, catalogue)
    user_counts = full_history.user_id.value_counts().reindex(users, fill_value=0)
    item_counts = full_history.item_id.value_counts().reindex(catalogue, fill_value=0)
    user_cutoff, item_cutoff = user_counts.median(), item_counts.median()
    low_activity = user_counts.le(user_cutoff).to_numpy()
    few_interactions = item_counts.le(item_cutoff).to_numpy()
    print(f"Test groups use pre-test medians of {user_cutoff:g} ratings per user and "
          f"{item_cutoff:g} per item. Median ties go in the lower group.", flush=True)

    # Use the same nested samples for every model and keep the test task fixed.
    shuffled = np.random.default_rng(SEED).permutation(len(full_history))
    sparsity_rows = []
    paired_by_fraction = {}
    for fraction in (1.0, 0.5, 0.2):
        count = round(fraction * len(full_history))
        history = full_history if fraction == 1 else full_history.iloc[shuffled[:count]]
        print(f"Evaluating {fraction:.0%} fit data ({count:,} ratings)", flush=True)

        markov_rank, matched = markov_top(order, history, users, catalogue, seen)
        item_knn.fit(history, user_ids=user_ids, catalogue=catalogue)
        tops = {
            markov_name: markov_rank,
            item_knn.name: top_k_items(item_knn.score_users(users), seen),
        }
        for model in popular_models:
            tops[model.name] = popular_top(model, history, users, catalogue, seen)
        for name, top in tops.items():
            sparsity_rows.append({
                "training_fraction": fraction, "training_ratings": count, "model": name,
                "users": len(users), "matched_state_users": matched if name == markov_name else None,
                **summarize(top, relevant),
            })

        paired_by_fraction[fraction] = paired_differences(tops, relevant, markov_name)
        if fraction == 1:
            groups = pd.DataFrame([
                {"model": name, **row} for name, top in tops.items()
                for row in disaggregate(top, relevant, low_activity, few_interactions)
            ])

    paired = pd.concat(paired_by_fraction, names=["training_fraction"]).reset_index(level=0)
    sparsity = pd.DataFrame(sparsity_rows)
    sparsity["rank_by_ndcg"] = (
        sparsity.groupby("training_fraction")["ndcg@10"].rank(method="min", ascending=False).astype(int)
    )
    sparsity = sparsity.sort_values(["training_fraction", "rank_by_ndcg"], ascending=[False, True])
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in (
        ("order-tuning.csv", markov_grid),
        ("itemknn-tuning.csv", item_knn_grid),
        ("most-popular-tuning.csv", popular_grid),
        ("disaggregated-test.csv", groups),
        ("paired-test.csv", paired),
        ("sparsity-ablation.csv", sparsity),
    ):
        table.to_csv(out_dir / filename, index=False)
    for title, table in (
        ("Full-data test results by group", groups),
        ("Paired test-set NDCG@10 differences (Markov minus baseline)", paired),
        ("Test results by training fraction", sparsity),
    ):
        print(f"\n{title}\n{table.to_string(index=False, float_format='%.4f')}")
    print(f"\nSaved tables to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/ml-1m")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--order", type=int, help="Use a fixed order instead of tuning orders 1–60.")
    args = parser.parse_args()
    run_comparison(load_movielens(args.data_dir).ratings, order=args.order, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
