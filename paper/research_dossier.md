# Satellite Health score clustering: research dossier (2026-10-06)

Status: Phase 1 source review and author-approved Phase 2 outline; Phase 3 argument construction is underway. This is a research protocol, not a claim that a universal F1 ceiling or organizer intervention has been demonstrated. The Kaggle final leaderboard was independently captured on 2026-10-10; see `evidence/leaderboard_2026-10-10.md`.

## Paper configuration

- Working form: English empirical workshop short paper, about 4–6 pages of main text plus references and appendix, subject to the selected workshop CFP.
- First target: a relevant ICLR 2027 workshop. Select the actual workshop only after its CFP appears; otherwise consider a relevant ICML/NeurIPS 2027 workshop when those calls become available.
- Research question: Why do diverse approaches to this seven-feature binary classification benchmark return similar F1 values, and what evidence can distinguish metric/sample effects, model limitations, distribution shift, and label ambiguity?
- Claim ceiling: a single competition cannot identify organizer intent or the Bayes-optimal F1. The reported public Kaggle score is an external observation, not an independent estimate of a population-level upper bound.

## Evidence ledger

| Item | Current evidence | Status and limitation |
| --- | --- | --- |
| Data | Official cached CSV hashes and audit in `../runs/first_round/data_audit.json`; 12,000 labeled training rows, 3,000 unlabeled test rows, seven numeric telemetry features, missing values in every feature; class counts 6,122/5,878 | Locally verified for downloaded files. The competition Data page reportedly says 7,200/1,800; reconcile from an archived or organizer source before publication. |
| Model diversity | Logistic, CatBoost, XGBoost, LightGBM, ExtraTrees, RBF-SVM, spline logistic, and other pipelines screened; `../model_family_mvp_summary.md` | Models share the same training set and feature set. This tests some model-family variation, not all possible predictors. |
| Threshold effect | Logistic threshold 0.50 OOF F1 0.792061; nested threshold selection reached 0.799282 on seed 20260918; threshold-0.42 public F1 0.8016555 | Existing experiment suggests decision rule matters. Later choices were informed by the same dataset and public feedback. |
| Best public score | User-reported/recorded Kaggle public F1 0.8053649 for fixed 75% spline + 25% logistic/XGBoost blend at threshold 0.40; `../spline_nested_probe_summary.md` | One submitted model and one competition test split. The score should be rechecked against a dated submission record for paper provenance. |
| Additional tuning | Nested spline gain -0.000979 mean across two outer seeds; residual-interaction gain -0.001357 mean across two outer seeds | Both reuse the same 12,000 labels. The negative results weaken the case for these particular enhancements; they cannot establish a universal ceiling. |
| Stable errors | Of 12,000 rows, 2,450 were misclassified under all four prior OOF partitions, 9,339 correct under all, and 211 varied; `../residual_interaction_probe_summary.md` | Stability is consistent with difficult examples, systematic model bias, ambiguous labels, or omitted features. It does not identify which. |
| Leaderboard cluster | Official final leaderboard ZIP downloaded 2026-10-10: 37 teams; 33 scored in [0.80, 0.82), 35 in [0.79, 0.82); median 0.8071050; maximum 0.8175979; our team ranked 24 at 0.8053649. Raw file, SHA-256 and exact counting rules are in `evidence/leaderboard_2026-10-10.md`. | The cluster is real for this final leaderboard but “all competitors” is false: two teams scored below 0.79. Team algorithms are unknown, so score convergence across different teams does not itself show model-family convergence. |
| Public/private split | The completed [leaderboard page](https://www.kaggle.com/competitions/satellite-health-challenge-niamt-ieee-student-branch/leaderboard) states that private and public leaderboards are calculated over the same rows. | There is no independent private set of rows for judging public leaderboard adaptation in this competition. The final score is not an independent replication of the public score. |
| Competition timing | The [overview](https://www.kaggle.com/competitions/satellite-health-challenge-niamt-ieee-student-branch/overview) shows a 2026-09-22 close date; the final leaderboard says the competition is complete. | New local experiments after closure are retrospective research. A later Kaggle score, if available through late submission, would reuse the benchmark and should not be treated as an independent test. |
| Observable train/test shift | `shift_audit.py` and `evidence/shift_audit_2026-10-10.json`: five-fold OOF domain AUC 0.51035 for median-imputed logistic and 0.50074 for histogram gradient boosting; the largest per-feature missing-rate difference was 0.00375 (test minus train). | No strong covariate shift is detectable by these two procedures on the seven supplied features. Hidden test labels are unavailable, so conditional or label shift and any effect on test F1 remain unmeasured. |
| Organizer design | No generator code, label protocol, or organizer statement currently in the evidence set | Organizer intent is not inferable from scores. Ask organizers only if this becomes necessary and user authorizes contact. |

## Source search and annotated corpus

Search scope (2026-10-06, updated 2026-10-10): official conference/journal papers on F1 thresholds, leaderboard reuse, finite-sample selection, dataset labels, and tabular-model comparisons; official ICLR 2027 dates/policy pages. Exclude blog explanations and unsourced claims as primary evidence. On 2026-10-10 the logged-in Kaggle page became readable: the [overview](https://www.kaggle.com/competitions/satellite-health-challenge-niamt-ieee-student-branch/overview) names F1 as the metric and shows 826 total submissions; the [data page](https://www.kaggle.com/competitions/satellite-health-challenge-niamt-ieee-student-branch/data) still lists 7,200/1,800 rows although the supplied CSVs contain 12,000/3,000; the final [leaderboard](https://www.kaggle.com/competitions/satellite-health-challenge-niamt-ieee-student-branch/leaderboard) is archived locally.

| Source | What it supports | What it does not support |
| --- | --- | --- |
| Lipton, Elkan & Narayanaswamy (2014), [Optimal Thresholding of Classifiers to Maximize F1 Measure](https://doi.org/10.1007/978-3-662-44851-9_15) | F1-optimal decision thresholds can differ from 0.5; under calibrated probabilities, the theoretical optimum is half the optimum F1. | It does not predict this competition's best threshold without calibration and validation. |
| Cawley & Talbot (2010), [On Over-fitting in Model Selection](https://www.jmlr.org/papers/v11/cawley10a.html) | Selecting among many CV configurations can bias the reported performance; model-selection procedure must be inside evaluation. | It does not prove that our observed plateau is caused by selection bias. |
| Blum & Hardt (2015), [The Ladder](https://proceedings.mlr.press/v37/blum15.html) | Repeated adaptive leaderboard evaluation can overfit a holdout; release mechanisms affect leaderboard reliability. | It does not establish that this particular leaderboard is overfit or implements Ladder. |
| Roelofs et al. (2019), [A Meta-Analysis of Overfitting in Machine Learning](https://proceedings.neurips.cc/paper_files/paper/2019/hash/ee39e503b6bedf0c98c388b7e8589aca-Abstract.html) | Across more than 100 Kaggle contests, the authors found little evidence of substantial public/private overfitting. This is counterevidence against assuming adaptive overfit is universal. | Their aggregate finding does not settle this contest, especially without its private scores. |
| Northcutt, Jiang & Chuang (2021), [Confident Learning](https://doi.org/10.1613/JAIR.1.12125) | Out-of-sample predicted probabilities can prioritize possible label errors for review. | Algorithmic flags are not confirmed label errors without independent adjudication. |
| Grinsztajn, Oyallon & Varoquaux (2022), [Why do tree-based models still outperform deep learning on typical tabular data?](https://proceedings.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf) | An evidence-based reason to include strong simple/tabular baselines before interpreting a deep model's result. | It does not say trees must win on this dataset. |

Research gap: the literature describes mechanisms that could produce apparent score convergence, but the existing experiment log has not separated those mechanisms in this competition. A publishable contribution would be a reproducible diagnostic audit with explicit identifiability limits, not a new high-scoring classifier.

## Competing explanations and discriminating tests

| Hypothesis | Predicted signature | Minimum test | Interpretation gate |
| --- | --- | --- | --- |
| H1: Metric/threshold, finite test sample, and public test reuse | Meaningful F1 changes from threshold at nearly fixed ranking; tiny score differences compatible with test-set variation; repeated public-score access can influence selection | Freeze OOF probabilities and use nested threshold selection; report AUC, PR-AUC, F1, precision/recall; paired bootstrap CIs over held-out predictions and a test-size sensitivity simulation; document the number of leaderboard submissions | Explain observed score variation only. The page explicitly says public and private scores use the same rows, so no independent private-row check of leaderboard adaptation exists here. |
| H2: Model/representation limit on seven features | Learning curves saturate; sufficiently distinct models share residuals; added interactions/strong models fail under nested evaluation | Pre-specify a small model set and equal tuning budget; nested outer CV; paired error overlap and learning curves at 25/50/75/100% of training rows | A plateau under this protocol is conditional on the given features, sample and models, not a Bayes bound. |
| H3: Label ambiguity, noise, or omitted variables | High-confidence cross-fitted label disagreement, conflicting near-neighbors, or errors concentrated in feature regions | Cross-fitted label-quality ranking and nearest-neighbor conflict audit; blind review of a preselected sample only if a domain expert or organizer can provide authoritative labels | Without adjudicated truth, report “suspected ambiguity” or “unexplained residual”; do not estimate a true mislabel rate. |
| H4: Train/test shift | A train-vs-test classifier distinguishes splits, or feature/missingness summaries differ | The first five-fold adversarial validation and missingness comparison were completed 2026-10-10; both logistic and gradient boosting were near chance. Prespecified labeled subgroup diagnostics remain. | Current evidence weakens an obvious observable covariate-shift explanation. Unlabeled test rows cannot establish test F1 or rule out conditional shift. |
| H5: Organizer-imposed design ceiling | Explicit data-generation rule, label protocol, or organizer statement specifying a target difficulty/cap | Seek official source or code; record exact quotation and date if found | Scores alone cannot test intent. Absence of source leaves this unanswered. |

Primary local outcome: paired outer-fold F1 difference against a frozen spline-blend baseline, evaluated with one outer partition plan and all selection inside training folds. Secondary outcomes: ranking quality, calibration, error overlap, learning-curve slope, subgroup performance, and shift diagnostics. Keep the local holdout protocol fixed before testing new hypotheses. Repeated seeds on the same rows are a robustness check, never a sample-size multiplier.

## Proposed paper outline (Phase 2 review copy)

Working title: **A Diagnostic Audit of an F1 Score Cluster in a Satellite Health Benchmark**

1. **Abstract (150–200 words).** State the observed local and public scores, four possible explanations, diagnostic method, actual results only after completion, and what remains unidentifiable.
2. **Introduction (500–650 words).** Explain the seven-feature competition and the observed model-score convergence; motivate why a leaderboard cluster can have several causes. RQ: which mechanisms can the available data discriminate? Contribution: a prospectively specified audit of the remaining tests and a bounded case study. Historical model searches are labeled retrospective.
3. **Related work (350–450 words).** F1 thresholding; adaptive leaderboard evaluation and its empirical counterevidence; model-selection bias; label-quality auditing; tabular model baselines.
4. **Data and evaluation protocol (550–700 words).** Provenance, CSV hashes, label prevalence, missingness, fixed features, positive-class F1 definition, prior submissions, nested evaluation, paired uncertainty, and leakage prevention. Figure 1: audit flow; Table 1: dataset and historical evidence.
5. **Diagnostic experiments (700–900 words).** Pre-specified H1–H4 tests. Table 2: model/threshold results with uncertainty; Figure 2: F1 vs threshold/learning curve; Figure 3: residual overlap/label-quality and shift slices. Include negative results.
6. **Discussion and identifiability (450–600 words).** Separate evidence for a conditional seven-feature plateau from evidence for label ambiguity or split artifacts. State that organizer intent and Bayes F1 require information not currently available.
7. **Limitations, ethics, reproducibility and AI-use disclosure (250–350 words).** One competition, unlabeled test set, adaptive historical exploration, no independent label adjudication; code/env/seeds/hashes, competition data-use rules, and the exact AI assistance used.

The narrative depends on the diagnostic experiments. If their confidence intervals are broad or the mechanisms remain indistinguishable, report a negative audit; do not force a “ceiling explained” conclusion. If a workshop requires a different length or format, adapt after its own CFP is published.

## Work plan and stop gates

1. **October 2026 — provenance gate:** the dated leaderboard distribution and competition overview/data/rules pages were captured or inspected on 2026-10-10. Preserve the leaderboard ZIP and source hashes. Resolve the remaining data-source discrepancy and avoid inferring team algorithms from scores alone.
2. **October–November — audit gate:** run the remaining bounded diagnostic blocks using the specified Python 3.13.6 64-bit environment. The observable train/test-shift block completed on 2026-10-10. Freeze all remaining splits/configuration and save predictions and JSON reports. If there is no robust difference, emphasize the negative result and identifiability analysis.
3. **December — writing gate:** choose an actually accepted workshop and check its CFP, page limit, originality, anonymity, data/code and AI-use policies. Draft only claims supported by saved outputs; perform independent citation and reproducibility checks.
4. **Early January 2027 — submission-readiness gate:** produce an English short paper, figures, code/metadata package, and a claim-to-evidence checklist. Formal upload requires separate user authorization and must meet the chosen workshop's actual deadline.

ICLR 2027's workshop-organizer call suggests 1 February 2027 for workshop contributions, but each accepted workshop sets its own author deadline. Its 9 October 2026 date concerns workshop *proposals*, not paper submissions. The ICLR 2027 main paper deadline was 25 September 2026. The main-conference author AI policy requires disclosure of significant AI assistance in the paper and submission form; the workshop guidance recommends following that policy while allowing individual workshops to set their own rules.

## Immediate unresolved evidence

- Generator or source provenance for the competition CSVs; resolve the displayed 7,200/1,800 versus actual 12,000/3,000 discrepancy. The overview only states F1 Score, without a class-specific computation formula.
- Independent test labels/rows or external dataset needed for a genuine generalization replication; the final public/private leaderboard rows are the same according to the competition page.
- Independent adjudication source for label-error claims, if such claims are to be made.
- The selected ICLR 2027 workshop's actual CFP after workshop acceptance decisions (scheduled 29 November 2026).
