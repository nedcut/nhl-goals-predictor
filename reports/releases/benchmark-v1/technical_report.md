# NHL Total-Goals Benchmark v1

## Technical summary

The locked 20252026 regular-season holdout selects **team_strength**. The raw score leader was **team_strength**, but the full Holm-adjusted week-block comparison set was **team_strength, double_poisson, xgb_tuned, xgb_current**; the protocol therefore chooses the simplest indistinguishable candidate.

## Candidate performance on the untouched holdout

| Model | Weighted score | MAE | CRPS | Dist NLL | Brier (>6.5) |
|---|---:|---:|---:|---:|---:|
| team_strength | 1.0000 | 1.8475 | 1.2855 | 2.2483 | 0.2503 |
| double_poisson | 1.0007 | 1.8493 | 1.2864 | 2.2490 | 0.2505 |
| xgb_tuned | 1.0012 | 1.8493 | 1.2874 | 2.2497 | 0.2508 |
| xgb_current | 1.0017 | 1.8487 | 1.2886 | 2.2505 | 0.2512 |
| poisson_glm | 1.0074 | 1.8560 | 1.2978 | 2.2563 | 0.2541 |

These metrics all use the same games and the same team-strength normalization. Lower is better. Small score differences are not treated as model improvements unless the blocked comparison supports that conclusion.

## The simpler choice survives multiplicity-adjusted uncertainty

| Candidate vs score leader | Mean score difference | 95% interval | Holm p |
|---|---:|---:|---:|
| double_poisson | -0.00073 | [-0.00134, -0.00007] | 0.100 |
| xgb_tuned | -0.00120 | [-0.00321, +0.00077] | 0.470 |
| xgb_current | -0.00167 | [-0.00480, +0.00160] | 0.470 |
| poisson_glm | -0.00745 | [-0.01278, -0.00193] | 0.032 |

The bootstrap resamples ISO weeks, preserving within-week schedule dependence. Holm adjustment controls the family of leader-versus-candidate comparisons.

## Scope, data, and metric definitions

- Training/tuning: 20222023, 20232024, 20242025.
- Holdout: 20252026.
- Cohort: regular season only; 3407 training rows and 1281 holdout rows after a common complete-feature filter.
- Data as of: 2026-06-14; fingerprint `e6a09face2c5bf823c9c75a35ea19c1ce21dbedbbefab13ac55110cb42101e15`.
- Weighted score: 35% normalized MAE, 30% CRPS, 20% distribution NLL, 15% Brier at 6.5.

## Validation and data quality

The release source contains 5799 rows, including 5248 primary-cohort rows. Duplicate game IDs: 0; score mismatches: 0; invalid dates: 0; invalid game types: 0. Release readiness: **True**.

## Methodology

Hyperparameters are tuned only on historical seasons using expanding-window CV. The selected parameters are then frozen and every candidate is scored exactly once on the later holdout. Rolling features use prior games only. Distribution parameters are fit on a historical calibration slice, never holdout outcomes.

## Limitations and robustness

- Statistical indistinguishability is failure to detect a difference, not proof that models are mathematically equivalent.
- Week blocking addresses short-range dependence but not every shared-team effect.
- Core-v1 excludes goalie and xG inputs because their release-grade longitudinal coverage has not yet been established.
- The benchmark supports forecast-quality claims, not sportsbook profitability.

## Recommended next steps

1. Serve only the promoted release artifact and accumulate a deduplicated ledger.
2. Re-run the locked protocol for each completed season before changing champion.
3. Admit enriched goalie/xG candidates only after coverage and freshness gates pass.
4. Evaluate real market lines separately; never mix synthetic ROI with championing.

## Further questions

- Does a team-cluster or moving-block bootstrap change the equivalence set?
- Do enriched features improve an untouched season once source coverage is complete?
- How stable are CRPS and calibration by month and rest/fatigue segment?
