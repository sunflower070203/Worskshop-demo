"""Synthetic identifiability experiment for the Satellite Health paper.

Goal: demonstrate, under *known* ground truth, the paper's central boundary —
a leaderboard F1 cluster carries no information about the Bayes-optimal F1 or
about *why* the cluster exists.

We construct two "twin" data-generating processes in which diverse model
families reach the same achieved F1 (~0.80) and produce near-identical
leaderboard distributions, but whose Bayes-optimal F1 differs sharply:

  A. "headroom": clean labels, but the features are corrupted by measurement
     noise, so models are capped at ~0.80 while Bayes F1 (on clean features)
     is much higher.
  B. "noise ceiling": clean features, but 20% of labels are flipped, so the
     achievable F1 is capped by label noise at ~0.80 and Bayes F1 ≈ 0.81.

If the leaderboard distributions of A and B are indistinguishable while their
Bayes-optimal F1 differs, then the cluster does not identify a ceiling, label
noise, or the Bayes F1 — it only reflects threshold + finite-sample effects on
near-identical rankings.

Output: runs/synthetic_experiment/{synthetic_experiment.json, .png}
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

RUNS = Path(__file__).resolve().parents[1] / "runs"
OUT = RUNS / "synthetic_experiment"

N_TRAIN = 12000
N_EVAL = 200000  # large sample for exact Bayes-F1 estimation
D_FULL = 10
N_TEAMS = 37
R = 300
SEED = 20261025
DEPLOYED = 0.40


def make_models(seed: int) -> dict:
    return {
        "logistic": LogisticRegression(max_iter=2000),
        "xgboost": __import__("xgboost").XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            verbosity=0, n_jobs=1, random_state=seed,
        ),
        "lightgbm": __import__("lightgbm").LGBMClassifier(
            n_estimators=300, num_leaves=31, learning_rate=0.05,
            verbose=-1, random_state=seed,
        ),
        "histgb": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=seed,
        ),
        "extratrees": ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=1),
    }


def generate(separation: float, feature_noise: float, label_flip: float, seed: int):
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_EVAL
    beta = rng.normal(0, 1, size=D_FULL)
    beta /= np.linalg.norm(beta)
    beta *= separation
    X_clean = rng.normal(0, 1, size=(n, D_FULL))
    z = X_clean @ beta
    p_clean = 1.0 / (1.0 + np.exp(-z))
    y_clean = (rng.uniform(size=n) < p_clean).astype(int)

    if label_flip > 0:
        flip = rng.uniform(size=n) < label_flip
        y_obs = np.where(flip, 1 - y_clean, y_clean)
        p_obs = (1 - label_flip) * p_clean + label_flip * (1 - p_clean)
    else:
        y_obs = y_clean
        p_obs = p_clean

    if feature_noise > 0:
        X_obs = X_clean + rng.normal(0, feature_noise, size=X_clean.shape)
    else:
        X_obs = X_clean

    X_tr, y_tr = X_obs[:N_TRAIN], y_obs[:N_TRAIN]
    # Bayes-F1 evaluation uses the large clean sample with the true posterior.
    y_eval = y_obs[N_TRAIN:]
    p_eval = p_obs[N_TRAIN:]
    return X_tr, y_tr, y_eval, p_eval


def f1_at(y: np.ndarray, p: np.ndarray, t: float) -> float:
    pred = p >= t
    y1 = y == 1
    tp = (pred & y1).sum()
    fp = (pred & ~y1).sum()
    fn = (~pred & y1).sum()
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def bayes_f1(y: np.ndarray, p: np.ndarray) -> float:
    return max(f1_at(y, p, t) for t in np.linspace(0.001, 0.999, 401))


def f1_batch(y: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    pred = probs >= thresholds[None, :]
    y1 = (y == 1)[:, None]
    tp = (pred & y1).sum(0).astype(float)
    fp = (pred & ~y1).sum(0).astype(float)
    fn = (~pred & y1).sum(0).astype(float)
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / denom, 0.0)


def ranking_agreement(rankings: dict[str, np.ndarray]) -> dict:
    names = list(rankings)
    rho, disagree = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rho.append(spearmanr(rankings[names[i]], rankings[names[j]]).statistic)
            disagree.append(
                float(np.mean((rankings[names[i]] >= DEPLOYED) != (rankings[names[j]] >= DEPLOYED)))
            )
    return {
        "spearman_mean": float(np.mean(rho)),
        "spearman_min": float(np.min(rho)),
        "label_disagreement_mean": float(np.mean(disagree)),
    }


def scenario(label: str, separation: float, feature_noise: float, label_flip: float) -> dict:
    rng = np.random.default_rng(SEED)
    X_tr, y_tr, y_eval, p_eval = generate(separation, feature_noise, label_flip, SEED)

    models = make_models(SEED)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rankings = {
        name: cross_val_predict(m, X_tr, y_tr, cv=skf, method="predict_proba")[:, 1]
        for name, m in models.items()
    }
    names = list(rankings)
    P = np.column_stack([rankings[n] for n in names])
    k = len(names)

    # Threshold sweep on the mean ranking.
    mean_rank = P.mean(1)
    t_grid = np.round(np.arange(0.25, 0.601, 0.01), 3)
    sweep = {
        float(t): float(f1_batch(y_tr, mean_rank[:, None], np.array([t]))[0])
        for t in t_grid
    }

    # Simulated leaderboard: 37 teams = (random family, random threshold).
    pos = np.flatnonzero(y_tr == 1)
    neg = np.flatnonzero(y_tr == 0)
    n_pos = round(3000 * float(np.mean(y_tr == 1)))
    n_neg = 3000 - n_pos
    stds = np.zeros(R)
    for r in range(R):
        idx = np.concatenate(
            [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
        )
        team_model = rng.integers(0, k, size=N_TEAMS)
        team_thr = rng.uniform(0.30, 0.60, size=N_TEAMS)
        f1s = f1_batch(y_tr[idx], P[idx][:, team_model], team_thr)
        stds[r] = np.std(f1s, ddof=1)

    bayes = bayes_f1(y_eval, p_eval)
    # Achieved F1 (best threshold, mean ranking) on training labels.
    achieved = max(sweep.values())

    return {
        "label": label,
        "separation": separation,
        "feature_noise": feature_noise,
        "label_flip": label_flip,
        "bayes_f1": round(bayes, 6),
        "achieved_model_f1_best_threshold": round(achieved, 6),
        "headroom": round(bayes - achieved, 6),
        "ranking_agreement": ranking_agreement(rankings),
        "simulated_cluster_std": {
            "mean": float(np.mean(stds)),
            "p95": float(np.percentile(stds, 95)),
        },
        "threshold_sweep": sweep,
        "mean_ranking_curve": sweep,
    }


def main() -> None:
    A = scenario("A_headroom", separation=3.5, feature_noise=0.50, label_flip=0.0)
    B = scenario("B_noise_ceiling", separation=3.5, feature_noise=0.0, label_flip=0.08)

    report = {
        "question": "Do two DGPs with different Bayes-optimal F1 produce indistinguishable leaderboard clusters?",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "seed": SEED,
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
        "d_full": D_FULL,
        "scenarios": {"A": A, "B": B},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "synthetic_experiment.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # --- Figure ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({"font.family": "serif", "font.size": 10, "figure.dpi": 300})

    C_A = "#0072B2"
    C_B = "#D55E00"

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))

    ax = axes[0]
    for scen, color in [(A, C_A), (B, C_B)]:
        ts = sorted(float(t) for t in scen["threshold_sweep"])
        fs = [scen["threshold_sweep"][t] for t in ts]
        ax.plot(ts, fs, color=color, lw=1.8,
                label=f"{scen['label']} (Bayes F1 {scen['bayes_f1']:.2f})")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("F1 (class 1)")
    ax.set_xlim(0.25, 0.60)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for scen, color in [(A, C_A), (B, C_B)]:
        std_mean = scen["simulated_cluster_std"]["mean"]
        ax.bar(scen["label"], std_mean, color=color, alpha=0.7, width=0.5)
        ax.text(scen["label"], std_mean + 0.0002, f"{std_mean:.4f}", ha="center", fontsize=9)
    ax.set_ylabel("simulated leaderboard std")
    ax.set_title("Cluster dispersion (identical)\nwhile Bayes F1 differs")
    ax.set_ylim(0, max(A["simulated_cluster_std"]["mean"], B["simulated_cluster_std"]["mean"]) * 1.35)
    ax.grid(alpha=0.25, axis="y")

    for ax, label in zip(axes, ["(a)", "(b)"]):
        ax.text(-0.03, 1.03, label, transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="bottom", ha="right")

    fig.tight_layout()
    fig.savefig(OUT / "synthetic_experiment.png", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(report, indent=2))
    print(f"\n=== SUMMARY ===")
    for scen in (A, B):
        print(f"{scen['label']}: Bayes F1={scen['bayes_f1']:.4f}, "
              f"achieved={scen['achieved_model_f1_best_threshold']:.4f}, "
              f"headroom={scen['headroom']:.4f}, "
              f"cluster std={scen['simulated_cluster_std']['mean']:.4f}, "
              f"spearman={scen['ranking_agreement']['spearman_mean']:.3f}")


if __name__ == "__main__":
    main()
