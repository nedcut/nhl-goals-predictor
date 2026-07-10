# Model Card

## Intended use

- Probabilistic NHL regular-season total-goals forecasting for analytical use.
- Pregame mean, full discrete PMF, over/under probabilities, and intervals.
- Not a betting recommendation; decision diagnostics require explicit references.

## Authoritative release

- Release: `benchmark-v1`
- Protocol: `nhl-total-goals-v1`
- Champion: `team_strength`
- Artifact: `benchmark-v1-team_strength-e6a09face2c5`
- Data fingerprint: `e6a09face2c5bf823c9c75a35ea19c1ce21dbedbbefab13ac55110cb42101e15`
- Feature-schema hash: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`

## Evaluation design

- Training/tuning seasons: 20222023, 20232024, 20242025.
- Untouched holdout: 20252026.
- Primary cohort: game types R; preseason and playoffs are excluded.
- Every candidate is evaluated on one common complete-game cohort.
- Distribution calibration uses only the final historical calibration slice.
- Uncertainty uses paired ISO-week block bootstrap with Holm adjustment.
- The simplest model in the full statistically indistinguishable set is selected.

## Holdout performance

| Model | Weighted score | MAE | CRPS | Dist NLL | Brier (>6.5) |
|---|---:|---:|---:|---:|---:|
| team_strength | 1.0000 | 1.8475 | 1.2855 | 2.2483 | 0.2503 |
| double_poisson | 1.0007 | 1.8493 | 1.2864 | 2.2490 | 0.2505 |
| xgb_tuned | 1.0012 | 1.8493 | 1.2874 | 2.2497 | 0.2508 |
| xgb_current | 1.0017 | 1.8487 | 1.2886 | 2.2505 | 0.2512 |
| poisson_glm | 1.0074 | 1.8560 | 1.2978 | 2.2563 | 0.2541 |

## Feature and data scope

- Common feature count: 79.
- Core-v1 intentionally excludes goalie and xG features until historical coverage and source freshness pass the same release checks.
- Source scores must satisfy unique game IDs, valid dates/types, and home + away = total consistency.

## Monitoring

- The API loads only a promoted release-grade registry artifact.
- SQLite monitoring deduplicates by game, artifact, and forecast kind.
- Realized monitoring includes MAE, RMSE, bias, CRPS, NLL, and mid-PIT diagnostics.
- Drift is reported only against references saved in the artifact; unavailable feature drift is labeled unavailable.

## Limitations

- The holdout establishes predictive comparison, not causal effects or profit.
- Week blocking reduces but does not eliminate team/schedule dependence.
- Lineup and confirmed-goalie information are outside the core-v1 release.
- A new season requires a new locked release rather than silently updating claims.
