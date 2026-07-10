# Champion Model Report

## Weighted Probabilistic Objective
- Score = 0.35*(MAE/base) + 0.30*(CRPS/base) + 0.20*(Dist NLL/base) + 0.15*(Brier/base)
- Baseline: team_strength
- Selection policy: compare the score leader against every candidate with week-block bootstrap and Holm adjustment, then prefer the simplest model in the indistinguishable set (`blocked_holm_equivalence_prefer_simpler`).

## Candidate Ranking
| Model | Weighted Score | MAE (±fold std) | CRPS | Dist NLL | Brier (>6.5) |
|---|---:|---:|---:|---:|---:|
| team_strength | 1.0000 | 1.8475 | 1.2855 | 2.2483 | 0.2503 |
| double_poisson | 1.0007 | 1.8493 | 1.2864 | 2.2490 | 0.2505 |
| xgb_tuned | 1.0012 | 1.8493 | 1.2874 | 2.2497 | 0.2508 |
| xgb_current | 1.0017 | 1.8487 | 1.2886 | 2.2505 | 0.2512 |
| poisson_glm | 1.0074 | 1.8560 | 1.2978 | 2.2563 | 0.2541 |

## Statistical Significance (score leader vs runner-up: team_strength vs double_poisson)
- Week-block bootstrap over 1281 shared games and 26 weeks; Holm-adjusted across all candidates.
- Raw p=0.033; adjusted p=0.100.

## Champion
- Selection policy: `blocked_holm_equivalence_prefer_simpler`
- Champion: **team_strength**
- Weighted-score leader: **team_strength**
- Indistinguishable set: **team_strength, double_poisson, xgb_tuned, xgb_current**
- Rationale: Lowest weighted score (1.0000) vs double_poisson (1.0007); tie-breakers MAE then CRPS. Week-block bootstrap comparisons against every candidate used Holm adjustment; the indistinguishable set was team_strength, double_poisson, xgb_tuned, xgb_current. Preferring the simpler candidate selects `team_strength`.