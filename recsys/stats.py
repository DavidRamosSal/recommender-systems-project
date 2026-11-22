"""Variability and significance for Part 2 (numpy only, no scipy needed).

Lecture 3.1, slide 10: a difference reported without an interval is not a
difference. We report two complementary intervals and one paired test:

* seed interval  - mean over training seeds +- t-interval (model randomness)
* user interval  - percentile bootstrap over evaluated users (user sampling)
* paired test    - per-user differences between two models on the same users:
                   bootstrap CI of the mean difference + Wilcoxon signed-rank
                   test (normal approximation, fine for hundreds of users),
                   Holm-corrected when several models are compared to one.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# two-sided 95 % critical values of Student's t, df = 1..30
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def seed_interval(values) -> tuple[float, float, float]:
    """Mean and 95 % t-interval over per-seed results (NaN for one seed)."""
    v = np.asarray(values, dtype=float)
    mean = float(v.mean())
    if len(v) < 2:
        return mean, math.nan, math.nan
    t = _T975[len(v) - 2] if len(v) - 1 <= 30 else 1.96
    half = t * v.std(ddof=1) / math.sqrt(len(v))
    return mean, mean - half, mean + half


def bootstrap_interval(values, *, samples: int = 1000, seed: int = 42
                       ) -> tuple[float, float, float]:
    """Mean and 95 % percentile bootstrap interval, resampling users."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(samples, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_signed_rank(d) -> float:
    """Two-sided p-value, zero differences dropped, ties get average ranks,
    normal approximation with tie correction (no continuity correction)."""
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    ranks = pd.Series(np.abs(d)).rank(method="average").to_numpy()
    w_plus = ranks[d > 0].sum()
    mean = n * (n + 1) / 4
    _, counts = np.unique(np.abs(d), return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24 - (counts ** 3 - counts).sum() / 48
    if var <= 0:
        return 1.0
    z = (w_plus - mean) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2))


def paired_comparison(a: pd.Series, b: pd.Series, *, samples: int = 1000,
                      seed: int = 42) -> dict:
    """Compare per-user values of model a and model b (indexed by user_id)."""
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).to_numpy(dtype=float)
    mean, low, high = bootstrap_interval(d, samples=samples, seed=seed)
    return {"users": len(d), "mean_difference": mean, "difference_ci_low": low,
            "difference_ci_high": high, "p_wilcoxon": wilcoxon_signed_rank(d),
            "a_better_users": int((d > 0).sum()), "b_better_users": int((d < 0).sum())}


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def kendall_tau(x, y) -> float:
    """Kendall's tau-b between two score vectors (rank agreement)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return (concordant - discordant) / denom if denom else math.nan
