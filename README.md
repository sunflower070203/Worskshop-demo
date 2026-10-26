# When leaderboard scores cluster: what is identifiable, and what is not

ICLR 2027 workshop submission (working draft). Companion repository for the
paper built from the [Satellite Health Challenge](https://www.kaggle.com/competitions/satellite-health-challenge-niamt-ieee-student-branch)
(binary classification, 7 telemetry features, F1 metric).

## The question

The final leaderboard (37 teams) is unusually tightly clustered: F1 from
0.774 to 0.818, standard deviation 0.009. The paper asks a bounded,
identifiability question rather than a "we won / we ranked X" claim:

> **When leaderboard scores are tightly concentrated, which causes can we
> identify from the visible data, and which causes are in principle
> unidentifiable?**

## The claim, in three moves

1. **The cluster is reproducible without the test labels.** A single frozen
   out-of-fold ranking, plus decision-threshold choice and finite-sample noise
   on the 3,000-row test set, reproduces the observed dispersion
   (`paper/score_cluster_section_draft.md`, Figure `runs/score_cluster_simulation/score_cluster_simulation.png`).
2. **The metric explains the flatness.** F1 is a CDF ratio whose derivative
   vanishes at the optimum, so threshold error is second-order; the empirical
   flat window is 0.0013 wide over thresholds [0.36, 0.44], and the optimal
   threshold equals F1*/2 (Lipton et al. 2014) — 0.402 vs the deployed 0.40
   (`paper/f1_threshold_analysis.md`).
3. **The cluster does not identify a ceiling or label noise.** Two synthetic
   "twin" data-generating processes — one with recoverable headroom, one with a
   label-noise ceiling — produce indistinguishable leaderboards while their
   Bayes-optimal F1 differs by 0.054 (`paper/synthetic_section_draft.md`).

## Repository layout

```
paper/                 the argument and the drafts
  research_dossier.md        evidence ledger, hypotheses, outline
  argument_blueprint.md      claim→evidence chains
  score_cluster_section_draft.md   empirical reproduction (Section 5)
  f1_threshold_analysis.md         F1(t) formula + flatness (Section 5/6)
  synthetic_section_draft.md       controlled identifiability demo (Section 6)
  evidence/                     leaderboard snapshot + shift audit
code/                  analysis scripts (reference implementations)
  score_cluster_simulation.py   reproduces the leaderboard cluster
  synthetic_experiment.py       the twin-DGP identifiability demo
  shift_audit.py                train/test distribution shift audit
data/                  inputs
  train.csv                    competition train split (Kaggle, public)
runs/                  committed data + results
  model_screen/ model_family_mvp/ spline_nested_probe/   out-of-fold rankings (.npz)
  score_cluster_simulation/     JSON report + figure
  synthetic_experiment/         JSON report + figure
```

The committed `runs/*/*.json` files are the authoritative numbers cited in the
drafts. The scripts regenerate them from the OOF `.npz` rankings.

## Reproduce

```bash
pip install numpy pandas scipy scikit-learn xgboost lightgbm catboost matplotlib

python code/score_cluster_simulation.py   # → runs/score_cluster_simulation/
python code/synthetic_experiment.py        # → runs/synthetic_experiment/
python -m pytest code/test_shift_audit.py  # shift-audit self-test
```

`score_cluster_simulation.py` reads the OOF rankings from `runs/` and the
labels from `data/train.csv`; the synthetic experiment is self-contained (it
generates its own data under a fixed seed).

## Provenance

- `data/train.csv` — the competition's public training split (Satellite Health
  Challenge, 12,000 rows × 7 features).
- `runs/*/oof_probabilities.npz` — out-of-fold predictions produced by the
  author's local training runs (the frozen ranking used throughout the paper).
- `paper/evidence/leaderboard_2026-10-10.*` — a snapshot of the public
  leaderboard on 2026-10-10.
