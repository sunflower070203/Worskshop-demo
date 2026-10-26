"""Score-cluster simulation for the Satellite Health paper.

Question: can the observed leaderboard F1 dispersion (37 teams, std ~0.0092,
IQR ~0.0104) be reproduced from three observable components, all measured
locally and without access to the hidden test labels?

  (1) a frozen out-of-fold ranking (the submitted 75/25 spline/logistic-XGB
      blend), which is an honest held-out prediction for every training row;
  (2) ranking disagreement among real model families (a few percent of labels);
  (3) F1 decision-threshold choice on a finite 3,000-row test set.

The hidden test labels are unavailable, so every simulated "test set" is a
3,000-row sample drawn from the 12,000 labeled training rows. Because each
row's probability is out-of-fold, it was produced by a model that did not
train on that row: F1 on a sampled subset is a genuine held-out estimate.

Within one iteration all 37 simulated teams share the same 3,000-row set, so
each iteration models "one leaderboard = one test set, many teams".

Output: a JSON report and a PNG figure under runs/score_cluster_simulation/.
"""

from __future__ import annotations

import json
import platform
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr

# Repo-local copy of the competition train split (downloaded once via
# `kaggle competitions download -c satellite-health-challenge-niamt-ieee-student-branch`).
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RUNS = Path(__file__).resolve().parents[1] / "runs"
OUT = RUNS / "score_cluster_simulation"

N_TEST = 3000
R = 2000
SEED = 20261024
T_GRID = np.round(np.arange(0.25, 0.601, 0.005), 3)
DEPLOYED_THRESHOLD = 0.40

OBSERVED = {
    "n_teams": 37,
    "std": 0.009156039602547787,
    "iqr": 0.010376400000000063,
    "min": 0.7741935,
    "max": 0.8175979,
    "median": 0.8071050,
    "mean": 0.8071242864864864,
}

SCREEN_FAMILIES = [
    "logistic_original",
    "logistic_missing",
    "logistic_poly2",
    "histgb",
    "extratrees",
    "lightgbm",
    "xgboost",
    "catboost",
]


def load_rankings() -> tuple[dict[str, np.ndarray], np.ndarray]:
    screen = np.load(RUNS / "model_screen" / "oof_probabilities.npz")
    mvp = np.load(RUNS / "model_family_mvp" / "oof_probabilities.npz")
    rankings: dict[str, np.ndarray] = {}
    for name in SCREEN_FAMILIES:
        rankings[name] = screen[f"{name}_20260918"].astype(float)
    blend = mvp["current_blend_20260918"].astype(float)
    spline = mvp["logistic_spline_20260918"].astype(float)
    rankings["current_blend"] = blend
    rankings["logistic_spline"] = spline
    rankings["submitted"] = 0.75 * spline + 0.25 * blend
    y = pd.read_csv(DATA_DIR / "train.csv")["satellite_health"].astype(int).to_numpy()
    return rankings, y


def verify_submitted_formula(rankings: dict[str, np.ndarray]) -> float:
    baseline = np.load(RUNS / "spline_nested_probe" / "oof_probabilities.npz")[
        "baseline_probability_20260922"
    ].astype(float)
    return float(np.corrcoef(rankings["submitted"], baseline)[0, 1])


def ranking_agreement(rankings: dict[str, np.ndarray]) -> dict:
    names = list(rankings)
    k = len(names)
    rho, disagree = [], []
    for i in range(k):
        for j in range(i + 1, k):
            rho.append(spearmanr(rankings[names[i]], rankings[names[j]]).statistic)
            disagree.append(
                float(
                    np.mean(
                        (rankings[names[i]] >= DEPLOYED_THRESHOLD)
                        != (rankings[names[j]] >= DEPLOYED_THRESHOLD)
                    )
                )
            )
    return {
        "n_rankings": k,
        "spearman_mean": float(np.mean(rho)),
        "spearman_min": float(np.min(rho)),
        "label_disagreement_at_0_40_mean": float(np.mean(disagree)),
        "label_disagreement_at_0_40_min": float(np.min(disagree)),
        "label_disagreement_at_0_40_max": float(np.max(disagree)),
    }


def sample_test_set(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_pos = round(N_TEST * float(np.mean(y == 1)))
    n_neg = N_TEST - n_pos
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    idx = np.concatenate(
        [
            rng.choice(pos_idx, size=n_pos, replace=False),
            rng.choice(neg_idx, size=n_neg, replace=False),
        ]
    )
    rng.shuffle(idx)
    return idx


def f1_batch(y_s: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Vectorized F1 (class 1) for a batch of (prob column, threshold) pairs.

    y_s: (n,), probs: (n, m), thresholds: (m,). Returns (m,) F1.
    """
    pred = probs >= thresholds[None, :]
    y1 = (y_s == 1)[:, None]
    tp = (pred & y1).sum(0).astype(float)
    fp = (pred & ~y1).sum(0).astype(float)
    fn = (~pred & y1).sum(0).astype(float)
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / denom, 0.0)


def main() -> None:
    rankings, y = load_rankings()
    names = list(rankings)
    k = len(names)
    P = np.column_stack([rankings[n] for n in names])  # (12000, K)
    submitted_col = names.index("submitted")
    rng = np.random.default_rng(SEED)

    formula_corr = verify_submitted_formula(rankings)
    full_f1 = {
        float(t): float(f1_batch(y, P[:, [submitted_col]], np.array([t]))[0])
        for t in (0.50, 0.42, 0.40, 0.38)
    }
    agreement = ranking_agreement(rankings)

    threshold_ranges = {"wide": (0.30, 0.55), "tight": (0.36, 0.44)}
    n_teams = OBSERVED["n_teams"]

    sweep_curves = np.zeros((R, len(T_GRID)))
    full_std = {rng_name: np.zeros(R) for rng_name in threshold_ranges}
    full_iqr = {rng_name: np.zeros(R) for rng_name in threshold_ranges}
    thresh_only_std = np.zeros(R)
    rank_only_std = np.zeros(R)
    all_team_f1: list[np.ndarray] = []

    for r in range(R):
        idx = sample_test_set(y, rng)
        y_s = y[idx]
        p_sub = P[idx, submitted_col]

        sweep_curves[r] = f1_batch(y_s, np.repeat(p_sub[:, None], len(T_GRID), 1), T_GRID)

        for rng_name, (lo, hi) in threshold_ranges.items():
            team_model = rng.integers(0, k, size=n_teams)
            team_thr = rng.uniform(lo, hi, size=n_teams)
            team_f1 = f1_batch(y_s, P[idx][:, team_model], team_thr)
            full_std[rng_name][r] = np.std(team_f1, ddof=1)
            full_iqr[rng_name][r] = np.percentile(team_f1, 75) - np.percentile(team_f1, 25)
            if rng_name == "wide":
                all_team_f1.append(team_f1)

        thr_only = f1_batch(
            y_s,
            np.repeat(p_sub[:, None], n_teams, 1),
            rng.uniform(*threshold_ranges["wide"], size=n_teams),
        )
        thresh_only_std[r] = np.std(thr_only, ddof=1)

        rank_models = rng.integers(0, k, size=n_teams)
        rank_only = f1_batch(
            y_s, P[idx][:, rank_models], np.full(n_teams, DEPLOYED_THRESHOLD)
        )
        rank_only_std[r] = np.std(rank_only, ddof=1)

    pooled_team_f1 = np.concatenate(all_team_f1)

    def _dist(a: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(a)),
            "p05": float(np.percentile(a, 5)),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
        }

    report = {
        "question": "Reproduce observed leaderboard F1 dispersion from threshold, ranking, and finite-sample components",
        "observed": OBSERVED,
        "python": sys.version.split()[0],
        "python_bits": 64 if sys.maxsize > 2**32 else 32,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "seed": SEED,
        "n_test": N_TEST,
        "n_iterations": R,
        "threshold_grid": T_GRID.tolist(),
        "deployed_threshold": DEPLOYED_THRESHOLD,
        "submitted_formula_corr_with_saved_baseline": formula_corr,
        "full_data_f1_reference": full_f1,
        "ranking_agreement": agreement,
        "threshold_ranges": threshold_ranges,
        "simulation": {
            "full_threshold_plus_ranking_std": {
                name: _dist(full_std[name]) for name in threshold_ranges
            },
            "full_threshold_plus_ranking_iqr": {
                name: _dist(full_iqr[name]) for name in threshold_ranges
            },
            "threshold_only_std": _dist(thresh_only_std),
            "ranking_only_std": _dist(rank_only_std),
            "pooled_team_f1_mean": float(np.mean(pooled_team_f1)),
            "pooled_team_f1_std": float(np.std(pooled_team_f1, ddof=1)),
        },
        "threshold_sweep_mean_f1_by_threshold": {
            float(t): float(v)
            for t, v in zip(T_GRID, np.mean(sweep_curves, axis=0))
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / "score_cluster_simulation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # --- Figure ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "figure.dpi": 300,
        }
    )

    with zipfile.ZipFile(
        Path(__file__).resolve().parents[1] / "paper" / "evidence" / "leaderboard_2026-10-10.zip"
    ) as z:
        csv_name = z.namelist()[0]
        observed_scores = pd.read_csv(z.open(csv_name))["Score"].astype(float).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    C_BLUE = "#0072B2"   # Okabe-Ito blue (simulated)
    C_BLACK = "#000000"  # observed
    C_RED = "#D55E00"    # deployed threshold
    C_GRAY = "#888888"   # default threshold

    obs_q1, obs_med, obs_q3 = np.percentile(observed_scores, [25, 50, 75])

    # Panel (a): threshold sweep on the frozen ranking.
    ax = axes[0]
    mean_sweep = np.mean(sweep_curves, axis=0)
    sd_sweep = np.std(sweep_curves, axis=0)
    ax.plot(T_GRID, mean_sweep, color=C_BLUE, lw=1.8, zorder=3)
    ax.fill_between(T_GRID, mean_sweep - sd_sweep, mean_sweep + sd_sweep,
                    color=C_BLUE, alpha=0.15, zorder=2)
    ax.axhspan(obs_q1, obs_q3, color=C_BLACK, alpha=0.10, zorder=1)
    ax.axhline(obs_med, color=C_BLACK, lw=0.8, ls="--", zorder=2)
    ax.axvline(DEPLOYED_THRESHOLD, color=C_RED, ls="--", lw=1.2, zorder=3)
    ax.axvline(0.50, color=C_GRAY, ls=":", lw=1.2, zorder=3)
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("F1 (class 1)")
    ax.set_xlim(0.25, 0.60)
    ax.set_ylim(0.758, 0.820)
    ax.grid(alpha=0.25, zorder=0)
    handles = [
        Line2D([], [], color=C_BLUE, lw=1.8, label="F1(t), frozen ranking"),
        Patch(facecolor=C_BLACK, alpha=0.10, label="observed IQR (37 teams)"),
        Line2D([], [], color=C_RED, ls="--", lw=1.2, label="deployed 0.40"),
        Line2D([], [], color=C_GRAY, ls=":", lw=1.2, label="default 0.50"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=7)

    # Panel (b): observed vs simulated score distributions.
    ax = axes[1]
    ax.hist(observed_scores, bins=12, color=C_BLACK, alpha=0.55, density=True,
            label="observed (n=37)")
    ax.hist(pooled_team_f1, bins=40, color=C_BLUE, alpha=0.45, density=True,
            label="simulated (threshold + ranking)")
    ax.axvline(np.mean(observed_scores), color=C_BLACK, lw=0.8, ls="--")
    ax.axvline(np.mean(pooled_team_f1), color=C_BLUE, lw=0.8, ls="--")
    ax.set_xlabel("F1 (class 1)")
    ax.set_xlim(0.76, 0.83)
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(frameon=False, fontsize=8)

    for ax, label in zip(axes, ["(a)", "(b)"]):
        ax.text(-0.03, 1.03, label, transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="bottom", ha="right")

    fig.tight_layout()
    fig_path = OUT / "score_cluster_simulation.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({"report": str(report_path), "figure": str(fig_path)}, indent=2))
    print("formula_corr:", round(formula_corr, 6))
    print("full_data_f1:", {k: round(v, 6) for k, v in full_f1.items()})
    print("ranking_agreement:", json.dumps(agreement, indent=2))
    print("full sim std (wide):", report["simulation"]["full_threshold_plus_ranking_std"]["wide"])
    print("full sim std (tight):", report["simulation"]["full_threshold_plus_ranking_std"]["tight"])
    print("full sim iqr (wide):", report["simulation"]["full_threshold_plus_ranking_iqr"]["wide"])
    print("threshold-only std:", report["simulation"]["threshold_only_std"])
    print("ranking-only std:", report["simulation"]["ranking_only_std"])


if __name__ == "__main__":
    main()
