"""Export the Part 3 frontier and sweep as vector PDFs for the Beamer deck.

Run with the project Python environment from any working directory. The frontier
data come from the teammate's saved experiment at the commit below.
"""

from io import StringIO
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
RUN_COMMIT = "4a252eae457d5e90a1adf4bd95733b4b8cbe58b9"
FRONTIER = "results/part3/intermediate/frontier-test.csv"
PER_USER = "results/part3/intermediate/per-user-test.csv"
INK = "#142923"
MUTED = "#47615A"
ACCENT = "#00775D"
MINT = "#CEE9DE"


def read_run_csv(path):
    csv_text = subprocess.check_output(["git", "show", f"{RUN_COMMIT}:{path}"],
                                       cwd=ROOT, text=True)
    return pd.read_csv(StringIO(csv_text))


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.major.pad": 3,
        "ytick.major.pad": 3,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": "#C6D4CF",
        "grid.linewidth": 0.45,
        "pdf.fonttype": 42,
    })


def export(fig, name):
    fig.savefig(OUTPUT / name, transparent=True, metadata={"CreationDate": None})
    plt.close(fig)


def lambda_axis(ax):
    ax.set_xlim(1.04, -0.04)
    ax.set_xticks([1, 0.75, 0.5, 0.25, 0], ["1", ".75", ".50", ".25", "0"])
    ax.set_xlabel(r"Relevance weight ($\lambda$)")


def frontier_plot(data):
    fig, ax = plt.subplots(figsize=(3.25, 1.98), layout="constrained")
    ax.plot(data["ild@10"], data["ndcg@10"], "o-", color=ACCENT, lw=1.4, ms=2.5)
    chosen = data.loc[data["selected"]].iloc[0]
    ax.scatter(chosen["ild@10"], chosen["ndcg@10"], marker="D", s=25, color=INK, zorder=3)
    labels = [(data.iloc[0], (0, 7), "left"), (chosen, (5, 4), "left"),
              (data.iloc[-1], (-5, -1), "right")]
    for row, offset, align in labels:
        ax.annotate(fr"$\lambda={row['lambda']:g}$", (row["ild@10"], row["ndcg@10"]),
                    xytext=offset, textcoords="offset points", ha=align, color=INK)
    ax.set(xlabel="Intra-list diversity", ylabel="NDCG@10", ylim=(0.164, 0.217),
           xticks=[0.70, 0.75, 0.80, 0.85, 0.90], yticks=[0.17, 0.18, 0.19, 0.20, 0.21])
    export(fig, "frontier.pdf")


def coverage_plot(data):
    fig, ax = plt.subplots(figsize=(3.25, 1.98), layout="constrained")
    ax.fill_between(data["lambda"], data["coverage_seed_ci_low"], data["coverage_seed_ci_high"],
                    color=MINT, linewidth=0, label="95% interval across fits")
    ax.plot(data["lambda"], data["coverage"], "o-", color=ACCENT, lw=1.4, ms=2.5)
    for weight, offset, align in [(1, (1, 5), "left"), (0.25, (0, -10), "center"),
                                  (0, (-1, 6), "right")]:
        row = data.loc[data["lambda"] == weight].iloc[0]
        ax.annotate(f"{row['coverage']:.2%}", (weight, row["coverage"]), xytext=offset,
                    textcoords="offset points", ha=align, color=INK)
    lambda_axis(ax)
    ax.set(ylabel="Catalogue coverage", ylim=(0.172, 0.201), yticks=[0.175, 0.185, 0.195])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    ax.legend(loc="lower left", fontsize=6.5, frameon=False, handlelength=1)
    export(fig, "exposure.pdf")


def exposure_calibration_plot(data, selected_lambda):
    fig, axes = plt.subplots(1, 2, figsize=(3.55, 1.55), layout="constrained")
    for ax, metric, label in zip(axes, ["gini", "calibration"],
                                 ["Exposure Gini", "Genre calibration error"]):
        ax.plot(data["lambda"], data[metric], "o-", color=ACCENT, lw=1.2, ms=2)
        lambda_axis(ax)
        ax.axvline(selected_lambda, color=MUTED, lw=0.65, ls="--", zorder=1)
        ax.annotate(fr"$\lambda={selected_lambda:g}$", (selected_lambda, 0.96),
                    xycoords=ax.get_xaxis_transform(), xytext=(-3, 0),
                    textcoords="offset points", ha="right", va="top", color=INK)
        ax.set_title(label, color=INK, fontsize=7.5, loc="left", pad=4)
        ax.ticklabel_format(axis="y", useOffset=False)
    export(fig, "exposure-calibration.pdf")


def ranked_cost_plot(per_user, selected_lambda):
    ndcg = per_user.pivot(index="user_id", columns="lambda", values="ndcg")
    changes = (ndcg[selected_lambda] - ndcg[1.0]).sort_values().to_numpy()
    ranks = pd.RangeIndex(1, len(changes) + 1)
    lower = (changes < 0).sum()
    tied = (changes == 0).sum()
    higher = (changes > 0).sum()
    within_band = (abs(changes) <= 0.05).sum()

    fig, ax = plt.subplots(figsize=(5.55, 1.6), layout="constrained")
    loss_color = "#A84E35"
    ax.axhspan(-0.05, 0.05, color="#E4ECE7", lw=0, zorder=0,
               label=f"{within_band} users within ±0.05")
    ax.axhline(0, color=MUTED, lw=0.6)
    for mask, color in [(changes < 0, loss_color), (changes > 0, ACCENT)]:
        ax.fill_between(ranks, 0, changes, where=mask, interpolate=True, color=color, lw=0)
        ax.plot(ranks[mask], changes[mask], color=color, lw=0.8)
    ax.plot([lower + 1, lower + tied], [0, 0], color=MUTED, lw=1.5, solid_capstyle="butt")

    centers = [(lower + 1) / 2, lower + (tied + 1) / 2, lower + tied + (higher + 1) / 2]
    ax.set_xticks(centers, [f"{lower} lower", f"{tied} tied", f"{higher} higher"])
    for label, color in zip(ax.get_xticklabels(), [loss_color, MUTED, ACCENT]):
        label.set_color(color)
    ax.tick_params(axis="x", length=0)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.set(xlim=(0, len(changes) + 1), yticks=[-0.2, -0.1, 0, 0.1, 0.2],
           ylabel="Change in NDCG@10",
           xlabel="Test users ordered from largest loss to largest gain")
    ax.legend(loc="upper left", frameon=False, handlelength=1.3)
    export(fig, "ranked-cost.pdf")


if __name__ == "__main__":
    style()
    sweep = read_run_csv(FRONTIER).sort_values("lambda", ascending=False)
    frontier_plot(sweep)
    coverage_plot(sweep)
    selected_lambda = sweep.loc[sweep["selected"], "lambda"].item()
    exposure_calibration_plot(pd.read_csv(OUTPUT / "exposure-calibration.csv"), selected_lambda)
    ranked_cost_plot(read_run_csv(PER_USER), selected_lambda)
