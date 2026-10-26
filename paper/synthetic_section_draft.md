# Controlled identifiability experiment

> Working draft (2026-10-26). Slots into the outline's Section 5/6 boundary:
> the constructive proof that a leaderboard cluster carries no information
> about the Bayes-optimal F1. Reproduces with `code/synthetic_experiment.py`;
> numbers in `results/synthetic_experiment.json`.

## Motivation

The real-competition analysis (`score_cluster_section_draft.md`) shows the
cluster is *consistent with* metric + sampling effects, but a single competition
cannot say whether the cluster is hiding recoverable headroom or hitting a
ceiling, because its test labels and data-generating process are unknown. Here
we close that gap under **known ground truth**: we construct two data-generating
processes (DGPs) whose leaderboards are observationally indistinguishable but
whose Bayes-optimal F1 differs sharply.

## Construction

Both DGPs use the same linear-logistic signal: $z = \beta^\top x$,
$x \sim \mathcal N(0, I_{10})$, $\|\beta\| = 3.5$, clean label
$y = \mathbb 1[u < \sigma(z)]$. The two scenarios differ in a single corruption
knob:

- **A ("headroom").** Labels clean; the *features* the models see are corrupted
  by measurement noise, $x_{\text{obs}} = x + \mathcal N(0, 0.5^2 I)$. Models are
  capped at ~0.80 by information loss, but the Bayes-optimal F1 (on clean
  features) remains high.
- **B ("noise ceiling").** Features clean; 8% of *labels* are flipped at random.
  The achievable F1 is capped by label noise, so Bayes-optimal F1 ≈ 0.80.

The two noise levels (0.50 feature-noise vs. 8% label-flip) were chosen so the
five model families reach the **same** achieved F1 (~0.80) in both scenarios.

For each scenario we train five diverse families (logistic, XGBoost, LightGBM,
histogram gradient boosting, ExtraTrees), obtain out-of-fold rankings on 12,000
rows, compute ranking agreement and the F1-threshold sweep, and simulate a
37-team leaderboard (family × threshold) exactly as in
`score_cluster_simulation.py`. Bayes-optimal F1 is computed exactly from the
known posteriors on a 200,000-row sample.

## Results

| Quantity | A: headroom | B: noise ceiling |
| --- | ---: | ---: |
| Bayes-optimal F1 | **0.858** | **0.804** |
| Achieved model F1 (best threshold) | 0.798 | 0.792 |
| Headroom (Bayes − achieved) | **0.060** | **0.013** |
| Simulated leaderboard std | 0.0111 | 0.0118 |
| Ranking convergence (Spearman ρ) | 0.974 | 0.968 |

The two scenarios produce **near-identical leaderboards**: achieved F1 within
0.006 (less than one cluster-std), simulated dispersion std 0.011 vs 0.012, and
the same ranking convergence (ρ ≈ 0.97). Yet their Bayes-optimal F1 differs by
0.054: in A the cluster sits 0.060 below a recoverable ceiling, in B it sits
essentially at an irrecoverable noise ceiling.

## Interpretation

This is the constructive half of the identifiability claim. Observing a
leaderboard clustered at ~0.80 with std ≈ 0.011 does **not** identify:

- the Bayes-optimal F1 (0.858 vs 0.804 are both consistent with it), nor
- the *cause* of the cluster (feature-limited models vs. label-noise ceiling).

The only thing the cluster reliably reflects is the mechanism of
`f1_threshold_analysis.md` and `score_cluster_section_draft.md`: thresholding a
shared ranking on a finite test set. Claims about "hitting a ceiling" or
"irreducible noise" from such a cluster are therefore not supported by the
leaderboard alone; they require independent information (generator code, an
organizer label protocol, or an external adjudicated dataset).
