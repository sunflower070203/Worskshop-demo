# Kaggle final leaderboard snapshot — 2026-10-10

Source: [Satellite Health Challenge leaderboard](https://www.kaggle.com/competitions/satellite-health-challenge-niamt-ieee-student-branch/leaderboard), viewed in the logged-in Kaggle web page on 2026-10-10 (Asia/Shanghai). The page said: “The private leaderboard is calculated over the same rows as the public leaderboard in this competition.” It also stated that the competition was complete and the displayed list reflected final standings.

Raw evidence: `leaderboard_2026-10-10.zip`, copied without modification from Kaggle's “Download as CSV” control. The ZIP contains one CSV named `satellite-health-challenge-niamt-ieee-student-branch-privateleaderboard-2026-10-10T00:51:17.csv` with columns `Rank,TeamId,TeamName,LastSubmissionDate,Score,SubmissionCount,TeamMemberUserNames`. ZIP SHA-256: `8d558e4f8c0c338e36dc700b669463e67744a5b8d251ee6f6925f2a60a65c2bf`. The raw file includes public team identifiers; use aggregate statistics in paper text.

All 37 CSV rows have a parseable `Score`. Score bins use half-open intervals and each team is counted once:

| Statistic | Value |
| --- | ---: |
| Teams | 37 |
| Minimum | 0.7741935 |
| First quartile (sorted index 9 of 37) | 0.8044623 |
| Median | 0.8071050 |
| Third quartile (sorted index 27 of 37) | 0.8148387 |
| Maximum | 0.8175979 |
| Arithmetic mean | 0.8071242865 |
| Sample standard deviation | 0.0091560396 |
| 0.80 ≤ F1 < 0.82 | 33/37 = 89.2% |
| 0.79 ≤ F1 < 0.82 | 35/37 = 94.6% |
| F1 < 0.79 | 2/37 |
| SunHflower | rank 24, F1 0.8053649, 4 entries |

The [overview](https://www.kaggle.com/competitions/satellite-health-challenge-niamt-ieee-student-branch/overview) displayed 37 teams and 826 submissions. The score distribution establishes a descriptive cluster. It does not identify other teams' model families, a population-level performance limit, label corruption, or organizer intent. Because the leaderboard page says public and private scores are computed from the same rows, it cannot serve as an independent final holdout after adaptive public-score use.

The [data page](https://www.kaggle.com/competitions/satellite-health-challenge-niamt-ieee-student-branch/data) displays a CC BY-SA 4.0 license and describes 7,200 training and 1,800 test rows. The actual downloaded files have 12,000 and 3,000 rows, respectively (hashes in `../../runs/first_round/data_audit.json`). Treat the mismatch as an unresolved version/provenance issue; neither number should silently replace the other in the paper.
