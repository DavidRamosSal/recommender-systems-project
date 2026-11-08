"""Regenerate Part 1 report tables and figures from unfiltered MovieLens data."""

import numpy as np
import pandas as pd

from .evaluation import evaluation_users
from .outputs import OutputPaths
from .splitting import temporal_split_by_date, to_epoch


def save_figure(figure, output_dir, name):
    images = OutputPaths(output_dir).create().images
    for extension in ("png", "svg"):
        figure.savefig(images / f"{name}.{extension}", dpi=300,
                       bbox_inches="tight", pad_inches=.07, facecolor="white")


def characterize_data(dataset, output_dir):
    """Write report numbers, distributions, split comparisons and filtering counts."""
    paths = OutputPaths(output_dir).create()
    tables = paths.intermediate
    ratings, movies, users = dataset.ratings, dataset.movies, dataset.users
    timestamps = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    per_user = ratings.groupby("user_id").size().reindex(users["user_id"], fill_value=0)
    per_item = ratings.groupby("item_id").size().reindex(movies["item_id"], fill_value=0)
    interaction_summary = pd.DataFrame({
        "per_user": per_user.describe(), "per_item": per_item.describe(),
    }).T
    interaction_summary.to_csv(tables / "interaction-summary.csv", index_label="population")
    rating_counts = ratings["rating"].value_counts().sort_index()
    monthly = timestamps.dt.strftime("%Y-%m").value_counts().sort_index()
    user_time = ratings.groupby("user_id")["timestamp"].agg(["min", "max"])
    span_days = (user_time["max"] - user_time["min"]) / 86400
    summary = {
        "users": int(users["user_id"].nunique()),
        "users_have_unique_ids": bool(users["user_id"].is_unique),
        "users_missing_values": int(users.isna().sum().sum()),
        "movies": int(movies["item_id"].nunique()),
        "rated_movies": int(ratings["item_id"].nunique()),
        "unrated_movies": int((per_item == 0).sum()),
        "ratings": len(ratings),
        "first_rating_utc": timestamps.min().isoformat(),
        "last_rating_utc": timestamps.max().isoformat(),
        "density_rated_movies": len(ratings) / (len(users) * ratings["item_id"].nunique()),
        "density_all_movies": len(ratings) / (len(users) * len(movies)),
        "positive_rating_share": float((ratings["rating"] >= 4).mean()),
        "year_2000_rating_share": float((timestamps.dt.year == 2000).mean()),
        "user_activity_under_24_hours_share": float((span_days < 1).mean()),
        "user_activity_under_7_days_share": float((span_days < 7).mean()),
        "median_user_active_hours": float(span_days.median() * 24),
        "busiest_month": str(monthly.idxmax()),
        "busiest_month_rating_share": float(monthly.max() / len(ratings)),
    }
    pd.Series(summary, name="value").rename_axis("metric").to_csv(tables / "data-summary.csv")

    candidates = [
        ("A", int(ratings.timestamp.quantile(.8)), int(ratings.timestamp.quantile(.9))),
        ("B", to_epoch("2000-12-01"), to_epoch("2001-01-01")),
        ("C", to_epoch("2001-01-01"), to_epoch("2001-07-01")),
    ]
    split_options = []
    for name, val_start, test_start in candidates:
        train, validation, test = temporal_split_by_date(ratings, val_start, test_start)
        row = {
            "option": name,
            "validation_start_utc": pd.Timestamp(val_start, unit="s", tz="UTC").isoformat(),
            "test_start_utc": pd.Timestamp(test_start, unit="s", tz="UTC").isoformat(),
            "train_ratings": len(train), "validation_ratings": len(validation),
            "test_ratings": len(test),
            "train_rating_share": len(train) / len(ratings),
            "validation_rating_share": len(validation) / len(ratings),
            "test_rating_share": len(test) / len(ratings),
        }
        for split, history, window in (
            ("validation", train, validation), ("test", pd.concat([train, validation]), test),
        ):
            eligible = evaluation_users(history, window)
            history_counts = history.groupby("user_id").size()
            row[f"{split}_evaluated_users"] = len(eligible)
            row[f"{split}_no_history_rating_share"] = float(
                (~window["user_id"].isin(history["user_id"])).mean()
            )
            row[f"{split}_median_history_ratings"] = float(history_counts.reindex(eligible).median())
        split_options.append(row)
    pd.DataFrame(split_options).to_csv(tables / "split-options.csv", index=False)

    train, validation, _ = temporal_split_by_date(ratings, "2001-01-01", "2001-07-01")
    catalogue = np.sort(ratings.item_id.unique())
    item_train = train.groupby("item_id").size().reindex(catalogue, fill_value=0)
    user_train = train.groupby("user_id").size().reindex(users.user_id, fill_value=0)
    filtering = []
    for threshold in (1, 5, 10, 20, 50):
        low_items = item_train.index[item_train < threshold]
        low_users = user_train.index[user_train < threshold]
        filtering.append({
            "minimum_training_ratings": threshold,
            "items_dropped": len(low_items), "item_share_dropped": len(low_items) / len(catalogue),
            "ratings_removed_by_item_filter": int(train.item_id.isin(low_items).sum()),
            "rating_share_removed_by_item_filter": float(train.item_id.isin(low_items).mean()),
            "users_dropped": len(low_users),
            "ratings_removed_by_user_filter": int(train.user_id.isin(low_users).sum()),
        })
    pd.DataFrame(filtering).to_csv(tables / "filtering-diagnostics.csv", index=False)
    window_counts = []
    for split, history in (("validation", train), ("test", pd.concat([train, validation]))):
        for days in (30, 90, 180):
            window = history[history.timestamp > history.timestamp.max() - days * 86400]
            window_counts.append({
                "split": split, "days": days, "ratings": len(window),
                "positive_ratings": int((window.rating >= 4).sum()),
            })
    pd.DataFrame(window_counts).to_csv(tables / "popularity-window-counts.csv", index=False)
    plot_data(rating_counts, timestamps, per_user, per_item, paths.root)
    return summary


def plot_data(rating_counts, timestamps, per_user, per_item, output_dir):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10, "text.color": "#183153",
        "axes.labelcolor": "#183153", "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#94a3b8", "legend.frameon": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    rating_counts.plot.bar(ax=axes[0], color="#6baed6", rot=0,
                           title="Rating distribution", xlabel="Rating", ylabel="Count")
    months = pd.date_range(
        timestamps.min().normalize().replace(day=1),
        timestamps.max().normalize().replace(day=1) + pd.offsets.MonthBegin(1), freq="MS",
    )
    timestamps.hist(ax=axes[1], bins=mdates.date2num(months.to_pydatetime()), color="#6baed6")
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[1].set(title="Monthly rating activity", xlabel="Rating date (UTC)", ylabel="Count")
    save_figure(fig, output_dir, "rating-activity")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for row, counts, population in ((0, per_user, "user"), (1, per_item, "item")):
        hist, ecdf_axis = axes[row]
        counts.plot.hist(ax=hist, bins=40, color="#6baed6",
                         title=f"Distribution of interactions per {population}",
                         xlabel=f"Ratings per {population}", ylabel=f"Number of {population}s")
        hist.axvline(counts.mean(), color="#e6550d", linestyle="--", label=f"Mean: {counts.mean():.1f}")
        hist.axvline(counts.median(), color="#183153", linestyle=":", label=f"Median: {counts.median():g}")
        hist.legend()
        cumulative = counts.value_counts().sort_index().cumsum() / len(counts)
        # Include an origin only when the first observed count is positive.
        if cumulative.index[0] > 0:
            cumulative = pd.concat([pd.Series([0.0], index=[0]), cumulative])
        ecdf_axis.step(cumulative.index, cumulative.values, where="post", color="#3182bd")
        maximum = max(2500, counts.max() * 1.05) if population == "user" else np.ceil(counts.max() / 500) * 500
        ecdf_axis.set(title=f"Cumulative distribution of interactions per {population}",
                      xlabel=f"Ratings per {population}", ylabel=f"Cumulative percentage of {population}s",
                      ylim=(0, 1.04))
        ecdf_axis.set_yticks([0, .25, .5, .75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
        for label, cutoff in zip(("Q1", "Median", "Q3"), counts.quantile([.25, .5, .75])):
            count = int(counts.le(cutoff).sum())
            share = count / len(counts)
            ecdf_axis.vlines(cutoff, 0, share, color="#94a3b8", linestyle=":")
            ecdf_axis.hlines(share, 0, cutoff, color="#94a3b8", linestyle=":")
            ecdf_axis.scatter([cutoff], [share], color="#183153", s=20)
            ecdf_axis.annotate(f"{label}: {cutoff:g} ratings\n{count:,} {population}s ({share:.1%})",
                               (cutoff, share), xytext=(maximum * .16, share - .08),
                               textcoords="data", va="top", fontsize=8,
                               arrowprops={"arrowstyle": "-", "color": "#94a3b8"})
        for axis in (hist, ecdf_axis):
            axis.set_xlim(0, maximum)
            axis.grid(axis="y", alpha=.18)
    save_figure(fig, output_dir, "long-tail")
    plt.close(fig)


def plot_tuning(grid, output_dir):
    import matplotlib.pyplot as plt

    best = grid.loc[grid["ndcg@10"].idxmax()]
    colours = {"cosine": "#3182bd", "jaccard": "#e6550d", "adjusted_cosine": "#31a354"}
    fig, axis = plt.subplots(figsize=(10, 5), layout="constrained")
    for (kind, feedback), group in grid.groupby(["similarity", "feedback"], sort=False):
        axis.plot(group.k, group["ndcg@10"], "-" if feedback == "positives" else "--",
                  color=colours[kind], marker="o", markersize=4, label=f"{kind} / {feedback}")
    axis.scatter([best.k], [best["ndcg@10"]], s=150, facecolors="none",
                 edgecolors="#183153", label="best on validation")
    axis.set(xscale="log", title="Item-kNN validation search", xlabel="Neighbourhood size k",
             ylabel="NDCG@10 (validation)")
    axis.legend(ncol=2, fontsize=8)
    save_figure(fig, output_dir, "itemknn-tuning")
    plt.close(fig)
