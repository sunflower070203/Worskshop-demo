# Argument blueprint — F1 score clustering in the Satellite Health Challenge

Date: 2026-10-10. This implements the author-approved diagnostic-audit framing and the outline in `research_dossier.md`. It is a claim/evidence plan, not manuscript Results. Numerical statements below refer to saved records and must be rechecked at drafting time.

## Central thesis

The final leaderboard exhibits a measurable score cluster (33 of 37 teams in [0.80, 0.82)), but that cluster alone does not identify an imposed noise rate or an irreducible performance ceiling. A controlled audit can determine which parts are attributable to the F1 decision rule, observed feature-limited model behavior, sampling and evaluation design, while identifying which explanations remain untestable without independent labels or a data-generation account.

The paper's positive contribution is the **diagnostic method and bounded findings**. It is not a claim that a new classifier wins the competition. Use “score clustering” rather than “hard ceiling”; the highest observed F1 is 0.8175979, and two teams scored below 0.79.

## Argument sequence and claim–evidence chains

| Claim | Direct evidence | Inference allowed | Disallowed leap |
| --- | --- | --- | --- |
| C1. Final scores cluster near 0.81. | Kaggle's downloaded 2026-10-10 leaderboard ZIP, SHA-256 and counts in `evidence/leaderboard_2026-10-10.md`: n=37, median 0.8071050, 33 in [0.80,0.82), top 0.8175979. | The **displayed final team scores** have a concentrated distribution. | “Every participant scored 0.81”; “all teams used similar models”; “the population ceiling is 0.81.” |
| C2. The score metric contributes to apparent convergence. | Historical logistic OOF F1 0.792061 at threshold 0.50 and nested threshold-selected F1 0.799282 for one outer seed; public F1 rose from 0.7981220 to 0.8016555 after changing threshold. | An F1 decision threshold can change score without changing the underlying ranking model. The forthcoming controlled curve must quantify how much of the cluster is plausibly metric-related. | Treating one submission gain as a causal estimate for all teams or equating Kaggle and local data. |
| C3. Several tested model families show modest separation under the seven provided features. | Sixteen-pipeline screen; spline AUC 0.878413 vs XGBoost 0.876006; negative nested spline and residual-interaction promotion tests. | The attempted capacity/feature changes did not produce a large, robust local gain under the recorded protocol. | Concluding that no model can do better or that Bayes F1 is near 0.81. |
| C4. Repeated errors warrant investigation but do not diagnose their cause. | 2,450 of 12,000 rows were misclassified in all four previous OOF partitions; 211 changed correctness. | Some errors are stable under these fold changes; inspect confidence, neighboring feature vectors, missingness and subgroups. | Calling the 2,450 cases mislabeled, irreducible, or a proven noise rate. |
| C5. The final scoreboard cannot independently validate public-score-guided selection. | The Kaggle [leaderboard](https://www.kaggle.com/competitions/satellite-health-challenge-niamt-ieee-student-branch/leaderboard) explicitly states that the public and private leaderboard use the same rows. The [overview](https://www.kaggle.com/competitions/satellite-health-challenge-niamt-ieee-student-branch/overview) reports 826 submissions. | Repeated access creates an adaptive-selection concern, and this competition supplies no separate private-row check. | Claiming that the leaderboard was actually overfit, or that each team made 826 submissions. |
| C6. Dataset documentation has a provenance inconsistency. | [Data page](https://www.kaggle.com/competitions/satellite-health-challenge-niamt-ieee-student-branch/data) says 7,200/1,800; hashed downloaded CSVs contain 12,000/3,000. | Reproducibility requires identifying the actual files and recording the discrepancy. | Assuming the cause was a deliberate data swap or that either count is erroneous without an organizer/version record. |
| C7. An obvious shift in the seven observed features is not detected. | `shift_audit.py` and `evidence/shift_audit_2026-10-10.json`: five-fold OOF domain AUC 0.51035 for logistic and 0.50074 for histogram gradient boosting; maximum missingness-rate difference 0.00375. | These procedures do not distinguish train from test rows materially better than chance. | Claiming that no shift of any kind exists, that test F1 is known, or that the same conditional label mechanism holds. |

## Testable versus unidentifiable explanations

**Metric and sampling:** compare ranking metrics with F1 at frozen and nested-selected thresholds, alongside paired uncertainty on the same held-out observations. A threshold effect is identifiable locally; hidden test uncertainty is only approximated because labels are unavailable.

**Feature/model limitation:** run fixed-budget learning curves and a small, predeclared family set using one nested outer evaluation. If all curves flatten and paired gains are small, report a plateau *conditional on seven features, 12,000 labeled rows, and the tested procedures*.

**Label ambiguity or imposed noise:** cross-fitted label-quality rankings and near-neighbor conflicts can nominate cases. The same observed conditional label rate can result from label flips or intrinsically stochastic outcomes; these explanations are observationally equivalent without repeated/authoritative labels, generator code, or organizer documentation. Do not report a noise percentage from model disagreement alone.

**Train/test shift:** the initial adversarial validation and missingness comparison showed little detectable covariate difference. Labeled subgroup diagnostics can still be run locally, but unlabeled test rows cannot reveal their F1 effect. Make that boundary explicit.

**Leaderboard adaptation:** the same public/private rows remove the usual independent scoreboard comparison. Present this as an evaluation-design limitation. Local nested CV can assess our own selection procedure but cannot retroactively prove the behavior of other teams.

## Objections a reviewer is likely to raise

1. **“Only one small competition; no general insight.”** Answer with a reusable, prespecified diagnostic protocol and at least one controlled comparison where noise/split conditions are known, or a second accessible tabular dataset. If neither can be completed, target a workshop that explicitly welcomes short case-study reanalyses and narrow the claim to a case study.
2. **“The top score is above 0.81, so there is no ceiling.”** Agree. We study concentration, not a hard bound. Show the entire 37-team distribution and the predeclared interval.
3. **“Model families are not independent evidence for all teams.”** Separate our local 16-pipeline screen from the leaderboard; do not assign methods to teams without their own published code/writeups.
4. **“The best baseline was selected on these labels.”** Mark earlier searches retrospective. Evaluate *new* diagnostic comparisons with predeclared candidates and nested folds; treat repeated seeds as stability checks on one dataset.
5. **“Suspicious labels are just hard examples.”** Use “candidates for review” and require independent adjudication to claim mislabeling. Do not infer organizer intent from fitted probabilities.
6. **“Public leaderboard gives an external validation.”** Explain that the public and private scores use the same rows here, so it is one benchmark feedback channel, not an independent replication.

## Logical flow

Observed final cluster → why cluster has multiple possible causes → audited data and evaluation setup → threshold/model/residual/shift tests → evidence-supported explanations → explicit non-identifiability of noise versus stochastic labels and organizer intent → implications for interpreting small Kaggle score differences.

## Promotion rule for scientific claims

Every numerical Result needs a stored code/configuration, input hashes, prediction-level archive, fixed split definition, paired uncertainty estimate where applicable, and a precise statement of what data were used for selection. Every leaderboard claim needs the dated raw ZIP and counting rule. A claim of label error needs independent adjudication; a claim of deliberate noise needs generator code or an organizer statement. If those gates fail, keep the corresponding explanation as a hypothesis rather than a conclusion.

## Writing and venue boundary

Do not draft a results-bearing abstract until diagnostic experiments finish. The [ICLR 2027 workshop guide](https://iclr.cc/Conferences/2027/WorkshopGuidelines) welcomes short reanalysis or a fresh perspective, but the actual accepted workshop and its contribution CFP determine fit and format. Its tiny/short-paper guidance permits AI assistance but requires a primarily human-authored paper; preserve an accurate AI-use record and have the human author own the interpretation and final text.
