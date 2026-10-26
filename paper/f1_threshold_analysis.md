# Why F1 scores cluster: a threshold-sensitivity analysis

> Working draft (2026-10-26). Companion to `score_cluster_section_draft.md`
> (the empirical reproduction). Numbers below are computed on the frozen OOF
> ranking over all 12,000 training rows; see `code/score_cluster_simulation.py`.

## Setup and exact F1(t)

Consider a binary test set of $n$ rows, with $n_1$ positives, $n_0$ negatives,
and class proportions $\pi_1 = n_1/n$, $\pi_0 = n_0/n$. A model assigns a score
$s\in\mathbb{R}$ to each row and predicts positive iff $s \ge t$. Let
$F_0(s)$ and $F_1(s)$ be the (empirical) score CDFs in classes 0 and 1. The
confusion-matrix entries are then

$$
TP(t)=n_1(1-F_1(t)),\quad FP(t)=n_0(1-F_0(t)),\quad FN(t)=n_1F_1(t),
$$

and the F1 score is

$$
F1(t)=\frac{2\,TP}{2\,TP+FP+FN}
=\frac{2\pi_1\,(1-F_1(t))}{(1+\pi_1)-\pi_1 F_1(t)-\pi_0 F_0(t)}. \tag{1}
$$

Equation (1) is the key observation: **F1(t) is determined entirely by the two
class-conditional score distributions.** No competitor, no test label, no
training detail is required — the entire threshold curve is computable from a
single model's held-out scores.

## The optimal threshold and why the curve is flat

Let $t^\*=\arg\max_t F1(t)$. Since $t^\*$ is an interior maximum,

$$
F1'(t^\*) = 0. \tag{2}
$$

Lipton, Elkan & Narayanaswamy (2014) showed that, for calibrated scores, the
F1-optimal threshold equals **half the optimal F1**:

$$
t^\* = F1(t^\*)/2. \tag{3}
$$

*Empirical check.* On the frozen ranking (full 12,000 OOF rows), the empirical
optimum is $t^\* = 0.406$ with $F1(t^\*)=0.8031$, so $F1(t^\*)/2 = 0.4016$ —
agreement to within 0.005, and both coincide with the deployed threshold 0.40.

Expanding around $t^\*$ and using (2),

$$
F1(t) = F1(t^\*) + \tfrac12 F1''(t^\*)\,(t-t^\*)^2 + O((t-t^\*)^3). \tag{4}
$$

The linear term vanishes, so **threshold error is second order**: small
threshold perturbations near the optimum move F1 by an amount quadratic in the
perturbation. This is the mathematical reason a leaderboard clusters tightly.

## Empirical flatness

On the frozen ranking over all 12,000 OOF rows:

| threshold | F1 |
| --- | ---: |
| 0.36 | 0.8014 |
| 0.38 | 0.8017 |
| 0.40 | 0.8027 |
| 0.42 | 0.8024 |
| 0.44 | 0.8016 |
| 0.50 (default) | 0.7960 |

Over the flat window $[0.36, 0.44]$ (width 0.08) F1 varies by only **0.0013**,
while the untuned default 0.50 drops by 0.007. The flat region is exactly where
the middle 50% of teams sit (observed IQR 0.804–0.815, after a ≈ 0.002
full-fit-vs-OOF offset).

## Implication: the cluster is metric + sampling

Equation (4) predicts that teams whose thresholds fall in the flat region obtain
F1 values within ~0.001 of one another *on the same test set*. The observed
dispersion (std 0.009) therefore decomposes as:

1. **finite-sample noise** on the 3,000 test rows — the companion simulation
   (`score_cluster_section_draft.md`) puts this at std ≈ 0.009, the dominant
   term;
2. **threshold excursions outside the flat region** — teams using the 0.50
   default or worse (std ≈ 0.005 over $[0.30, 0.55]$);
3. **ranking differences** across model families (std ≈ 0.002), which are small
   because rankings converge (Spearman ρ ≈ 0.97–0.99).

The three terms together reproduce the observed std without invoking a
performance ceiling, label noise, or distribution shift — and (1)–(3) are all
measurable from a single model's validation scores.
