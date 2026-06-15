# Model Card

## Intended Use
- NHL total-goals forecasting for analytical/educational use.
- Supports pregame probabilities and live in-game updates.

## Data Sources
- NHL Web API schedule/game data.
- MoneyPuck historical xG (cached to `data/xg/{season}.csv`).

## Feature Families
- Rolling team offense/defense trends.
- Rest/back-to-back and temporal features.
- Matchup interactions and optional goalie context.
- Rolling xG features and xG-vs-goals differentials.

## Evaluation Protocol
- Expanding-window time-series CV (5 folds for final evaluation).
- Proper scoring rules: MAE, CRPS, distributional NLL, Brier for over 6.5.

## Champion Formula
- `score = 0.35*(mae/base_mae) + 0.30*(crps/base_crps) + 0.20*(dist_nll/base_dist_nll) + 0.15*(over_brier/base_brier)`
- Baseline model for normalization: `team_strength`.
- Current champion: `xgb_tuned`.
- Rationale: Lowest weighted score (0.9967) vs xgb_current (0.9978); tie-breakers MAE then CRPS.
- Champion margin over runner-up is within noise (95% CI [-0.0024, +0.0001], p=0.081); treat the two models as statistically indistinguishable.

## Known Failure Modes
- Sparse recent form early season.
- Sudden lineup or goalie changes not visible in historical aggregates.
- Extreme game states and overtime tails.

## Monitoring Plan
- Re-run CV and champion report weekly.
- Track rolling calibration and segment MAE (month/back-to-back/confidence decile).
- Alert when weighted score regresses >2% vs prior champion.

## Build Context
- Seasons: 20222023, 20232024, 20242025
