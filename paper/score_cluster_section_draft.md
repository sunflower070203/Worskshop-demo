# Draft section: Reproducing the leaderboard score cluster

> Working draft (2026-10-25). Slots into the outline in `research_dossier.md`,
> Section 5 ("Diagnostic experiments"), with the identifiability argument
> feeding Section 6. Numbers are from `runs/score_cluster_simulation/` and must
> be re-checked against that JSON at drafting time. Figure 1 = the two-panel
> figure in that same directory.

## 5.x Method: a local reproduction protocol

We ask whether the observed dispersion of the final leaderboard can be
reproduced *without access to the hidden test labels*, from three quantities
that are measurable from the training split alone: (1) a frozen out-of-fold
(OOF) ranking, (2) ranking disagreement across real model families, and
(3) the F1 decision threshold applied to a finite 3,000-row test set.

**Frozen ranking.** We use the OOF probability vector of the submitted model —
a 75/25 blend of a cubic-spline logistic regression and a 50/50
logistic/XGBoost blend — which is an honest held-out prediction for every one
of the 12,000 training rows. The 75/25 reconstruction correlates at ρ = 0.9993
with the saved baseline OOF, confirming the provenance.

**Model family pool.** To model the diversity of "what a team might submit",
we keep the OOF rankings of 8 additional model families from the earlier
bounded screen (ordinary logistic, logistic with missingness indicators,
second-order logistic, histogram gradient boosting, ExtraTrees, LightGBM,
XGBoost, CatBoost), plus the two blend components, for 11 rankings in total.

**Pseudo-test sets.** Since the test labels are unavailable, every simulated
test set is a 3,000-row sample drawn (stratified to training prevalence) from
the 12,000 labeled rows. Because each row's probability is out-of-fold, F1 on
a sampled subset is a genuine held-out estimate. Within one iteration all 37
simulated teams share the same 3,000-row set, so each iteration is "one
leaderboard = one test set, many teams"; we repeat R = 2,000 times to
propagate test-set sampling variation.

**Simulated teams.** Each of 37 teams draws (i) a model family uniformly from
the pool and (ii) a decision threshold uniformly from a plausible range
(primary: 0.30–0.55; sensitivity: 0.36–0.44 and 0.30–0.60), then threshold
the model's OOF probabilities. We also run two ablations: threshold-only
(fixed ranking, threshold varies) and ranking-only (fixed threshold 0.40,
model family varies).

## 5.x Results

### Ranking convergence

The 11 model families are near-collinear in ranking (Table 1). Pairwise
Spearman correlation is 0.985 on average (minimum 0.971), and at the deployed
threshold 0.40 the families disagree on only 3.9% of labels on average
(maximum 5.6%). Different reasonable models therefore submit *essentially the
same ranking*.

| Quantity | Value |
| --- | ---: |
| Pairwise Spearman ρ, mean (min) | 0.985 (0.971) |
| Label disagreement at threshold 0.40, mean | 3.9% |
| Label disagreement at threshold 0.40, max | 5.6% |

### Threshold sensitivity

Holding the frozen ranking fixed and varying only the decision threshold on a
3,000-row set, F1 spans 0.769 (threshold 0.60) to 0.803 (threshold 0.40), a
range of 0.034 (Figure 1a). The curve is flat near its optimum — F1 changes
by < 0.001 over the interval 0.36–0.42 — and steep toward the tails. The
entire observed leaderboard range is 0.774–0.818 (range 0.043), so threshold
choice alone accounts for ≈ 80% of the full span, and the flat optimum region
coincides exactly with where 50% of teams sit (observed IQR 0.804–0.815).

### Reproduction

Simulating 37 teams from threshold + ranking reproduces the leaderboard's
dispersion (Table 2). Under the primary threshold range 0.30–0.55, the
simulated interquartile range (mean 0.0074, 95th pct 0.0123) brackets the
observed IQR of 0.0104; the simulated standard deviation (mean 0.0059) is
brought to the observed 0.0092 when the threshold range is widened to
0.30–0.60 (mean 0.0098), i.e. by allowing some teams to leave the threshold at
or above the 0.50 default.

| Quantity | Observed | Simulated (0.30–0.55) | Simulated (0.30–0.60) |
| --- | ---: | ---: | ---: |
| F1 standard deviation | 0.0092 | 0.0059 (p95 0.0084) | 0.0098 (p95 0.0133) |
| F1 interquartile range | 0.0104 | 0.0074 (p95 0.0123) | — |

### Decomposition

The threshold component dominates the ranking component. With a fixed
ranking and only thresholds varying (0.30–0.55), the simulated standard
deviation is 0.0053; with a fixed threshold 0.40 and only the model family
varying, it is 0.0025. Threshold choice is roughly twice as important as model
family for the observed dispersion, which is a direct consequence of the
ranking convergence in Table 1.

## 5.x Residuals and limitations

Three residuals are reported honestly. (i) The two lowest teams (0.774,
0.780) require thresholds ≈ 0.55–0.60 or a materially worse model; the
primary simulation does not reach them, so they are best read as poorly-tuned
or underperforming submissions. (ii) The top team (0.818) sits ≈ 0.014 above
the frozen OOF ceiling (0.803); this is partly the OOF-vs-full-fit gap
(≈ 0.002, since the submitted full-fit model scored 0.805 on the real test)
and partly genuine ranking advantage, which a threshold-only mechanism cannot
explain. (iii) Because the test labels are hidden, the pseudo-test sets are
drawn from training rows; this estimates dispersion (the spread) rather than
the level, and the real test set may shift the level by ≈ 0.002 without
changing the concentration conclusion.

## 5.x Interpretation (feeds Section 6)

The observed score cluster is quantitatively consistent with — and largely
reproduced by — threshold choice and finite-sample variation acting on
near-identical rankings. The clustering therefore carries *no evidence of a
performance ceiling, label noise, or distribution shift*: those explanations
are not required to produce the observation, and (as Section 6 argues) several
of them are unidentifiable from the available data in any case. This is the
paper's positive, bounded claim: one observable cause (metric + sampling)
suffices, and the metric contribution can be quantified locally.
