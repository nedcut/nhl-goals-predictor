# NHL Total Goals Forecasting (Probabilistic)

A production-minded pipeline for **probabilistic forecasting** of NHL total goals.
Instead of only predicting a point estimate, it can produce a full distribution over goal totals (e.g., **P(over 6.5)**), evaluate with proper scoring rules, and ship an API + dashboard for nightly inference.

## Results

| Metric | Value |
|--------|-------|
| Source-of-truth protocol | Expanding-window time-series CV (5 folds) + NB2 distribution calibration |
| Data window | Seasons 20222023, 20232024, 20242025 |
| Champion model | `xgb_tuned` (weighted probabilistic objective) |
| Champion MAE | **1.8688** |
| Champion CRPS | **1.2937** |
| Champion Brier (`P(total_goals > 6.5)`) | **0.2463** |

The project uses weighted probabilistic CV protocol as the primary model-selection criterion.
Current rankings and rationale are generated in `reports/champion_model_report.{json,md}`.

Portfolio artifacts are generated under `reports/` and documented in [MODEL_CARD.md](MODEL_CARD.md).

## Features

22+ engineered features including:
- Rolling goals for/against (20-game window)
- Win percentage and streaks
- Rest days and back-to-back indicators
- Goalie save percentage and GAA
- **Head-to-head historical matchup stats** (new)
- **Venue-specific scoring trends** (new)

## Probabilistic Forecasting

Given a point forecast `mu = E[total_goals]`, the project can emit a discrete distribution over totals:
- Poisson
- Negative Binomial (NB2; overdispersed vs Poisson)
- Poisson mixture (extra dispersion / heavy tails)

From the distribution you get:
- `P(total_goals > 6.5)`, `P(total_goals > 7.5)`, etc.
- Predictive intervals (distribution quantiles)
- Conformal intervals (distribution-free uncertainty around the point forecast)

## Quick Start

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/eda.ipynb
```

The notebook will:
1. Download 4 seasons of NHL data (2021-2025) with caching
2. Fetch starting goalie statistics for all games
3. Train an optimized XGBoost model
4. Show feature importance and predictions

Run tests:

```bash
pytest -q
```

Run the full portfolio pipeline:

```bash
python -m src.portfolio --seasons 20222023 20232024 20242025 --tune-trials 150
```

## Prediction CLI

Make predictions for upcoming games:

```bash
# Predict games for the next 7 days
python -m src.predict --model models/xgboost_v1 --days 7

# Save predictions to CSV
python -m src.predict --output predictions.csv

# Specify historical data seasons
python -m src.predict --seasons 20242025 20252026
```

Probabilistic forecasts (over/under probabilities + intervals):

```bash
python -m src.predict --probabilistic --dist-model nb2 --thresholds 5.5 6.5 7.5
```

## Honest Evaluation (Time-Series CV)

Run expanding-window CV with proper scoring rules and calibration diagnostics:

```bash
python -m src.evaluate --seasons 20222023 20232024 20242025 --point-model xgb --dist-model nb2
```

Outputs (saved under `reports/`):
- Fold metrics + aggregates (`.json`)
- Reliability curve for `P(total_goals > 6.5)` (`.png`)
- Randomized PIT histogram for distribution calibration (`.png`)
- Champion summary report (`champion_model_report.md`)

## REST API

Run the prediction API server:

```bash
# Start the server
uvicorn src.api:app --reload

# Visit http://localhost:8000/docs for interactive documentation
```

**Endpoints:**
- `GET /predict?days_ahead=7` - Get predictions for upcoming games
- `GET /predict/probabilistic?days_ahead=7` - Distribution forecast + P(over/under)
- `GET /predict/live?days_ahead=1` - Pregame + live-adjusted probabilities
- `GET /dashboard` - Tonight’s slate (simple HTML table)
- `GET /dashboard/live` - Auto-refreshing live slate (20s cadence)
- `GET /model/info` - Get model metadata and performance metrics
- `GET /monitoring/summary` - Realized accuracy + prediction-drift status from logged predictions
- `GET /health` - Health check

By default, inference loads the previous and active NHL seasons and refreshes
the active-season cache every six hours. Override runtime paths with
`NHL_MODEL_PATH` and `NHL_HISTORICAL_SEASONS` (comma-separated).

## Live Dashboard

The live dashboard uses NHL game state (score, period, clock) plus a residual-goals NB2 updater:

- `remaining_frac = remaining_minutes / 60`
- `pace_mult = 1 + 0.04 * abs(goal_diff)`
- `mu_remaining = max(0.05, mu_pregame * remaining_frac * pace_mult)`
- Remaining goals are modeled with NB2 and shifted by current goals.

## Monitoring & Drift Detection

A live forecaster needs to answer two questions *after* games are played: are we
still accurate, and has the input distribution shifted? `src/monitoring.py`
provides both.

Enable prediction logging on the API (off by default so nothing writes
unexpectedly):

```bash
NHL_MONITORING_LOG=1 uvicorn src.api:app
```

Every `/predict` and `/predict/probabilistic` call then appends to an append-only
JSON Lines log (`data/monitoring/predictions_log.jsonl`). Once games finish, get
a report of realized accuracy and prediction drift:

```bash
# Via API
curl localhost:8000/monitoring/summary

# Or offline, reconciling against cached results
python -m src.monitoring --seasons 20242025 20252026 --output reports/monitoring.json
```

The report includes:
- **Realized accuracy** — rolling MAE / RMSE / bias and over/under Brier on
  predictions whose games have now been played (the actual field performance,
  not the CV estimate).
- **Drift** — **Population Stability Index (PSI)** comparing recent inputs (or
  predictions) to a reference window. `feature_drift()` scores every feature and
  `assess_overall_drift()` rolls them into a `stable` / `moderate` / `significant`
  verdict (any feature at PSI ≥ 0.25, or ≥ 2 features ≥ 0.10).

## Explainability + Stability

Feature stability across seasons (and optional SHAP summary plots if `shap` is installed):

```bash
python -m src.explain --seasons 20222023 20232024 20242025 --outdir reports/explain
```

## Project Structure

```
├── src/
│   ├── config.py       # Centralized configuration
│   ├── data.py         # NHL API data fetching with caching
│   ├── features.py     # Feature engineering (rolling stats, H2H, venue)
│   ├── goalies.py      # Goalie data fetching from boxscores
│   ├── model.py        # XGBoost training, CV, Optuna optimization
│   ├── predict.py      # CLI for predictions
│   ├── api.py          # FastAPI REST API
│   ├── probabilistic.py # PMFs, log score, CRPS, calibration helpers
│   ├── evaluation.py   # Time-series CV with proper scoring rules
│   ├── evaluate.py     # Evaluation CLI
│   ├── explainability.py # SHAP (optional) + stability across seasons
│   ├── explain.py      # Explainability CLI
│   ├── conformal.py    # Split-conformal intervals
│   ├── team_strength.py # Shrinkage baseline on team IDs
│   ├── artifacts.py    # Model persistence with metadata
│   ├── registry.py     # Model versioning and management
│   ├── validation.py   # Input validation utilities
│   └── logging_config.py # Structured logging
├── notebooks/
│   └── eda.ipynb       # Full pipeline demonstration
├── data/               # Cached data (not in git)
└── models/             # Trained models (not in git)
```

## Resume Bullets (Template)

- Built an NHL total-goals **probabilistic forecaster** producing calibrated over/under probabilities (e.g., `P(total_goals > 6.5)`), evaluated with **expanding-window time-series CV** and proper scoring rules (**MAE 1.876**, **CRPS 1.298**, Brier **0.247** at 6.5), and shipped a **FastAPI** service with a “Tonight’s slate” dashboard for nightly inference.

## Hyperparameter Optimization

Use Optuna for Bayesian hyperparameter optimization:

```python
from src import build_dataset, add_features, optimize_hyperparameters

df = build_dataset(['20232024', '20242025'])
df = add_features(df)

# Run 100 trials of optimization
best_params = optimize_hyperparameters(df, n_trials=100)
print(best_params)
```

## Model Versioning

Track and manage model versions:

```python
from src import get_registry, ModelArtifact

# Get the registry
registry = get_registry()

# Register a new model
artifact = ModelArtifact.from_training_result(result, seasons=['20232024'])
version = registry.register(artifact, name="xgboost", promote_to_production=True)

# Load production model
prod_model = registry.get_production_model()

# List all models
for model in registry.list_models():
    print(f"{model['version']}: MAE={model['mae']:.4f}")
```

## Key Findings

1. **Time-series probabilistic CV is the decision authority** - this avoids holdout-only selection bias.
2. **Championing is weighted and probabilistic** - MAE, CRPS, dist-NLL, and Brier are normalized to the team-strength baseline.
3. **Team-strength baseline is competitive** - useful as a robust sanity check for overfitting.
4. **Model gains are modest by nature** - NHL total-goals prediction is high-noise, so consistency and calibration matter as much as point error.

## XGBoost Params

```python
# Current tuned champion (from weighted probabilistic objective)
{
    'max_depth': 4,
    'learning_rate': 0.011896873680695898,
    'n_estimators': 65,
    'reg_alpha': 1.7530910973690677,
    'reg_lambda': 2.6740596452335668,
    'subsample': 0.8393574607886646,
    'colsample_bytree': 0.5556469177410432,
    'min_child_weight': 7,
}

# Previous stable config
{
    'max_depth': 2,
    'learning_rate': 0.01,
    'n_estimators': 150,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 7,
}
```

## Configuration

All parameters are centralized in `src/config.py`:

```python
from src import config

# Feature settings
config.features.rolling_window  # 20 (games for rolling stats)
config.features.goalie_window   # 10 (games for goalie stats)
config.features.min_games       # 3 (minimum games before features valid)

# Model settings
config.model.test_size          # 0.2
config.model.cv_folds           # 5
config.model.xgb_params         # Optimized XGBoost parameters
```

## Dependencies

Core:
- pandas, numpy, scikit-learn, xgboost
- requests, tqdm (data fetching)
- matplotlib (visualization)

Optional:
- optuna (hyperparameter optimization)
- fastapi, uvicorn, pydantic (REST API)

## Data Sources

Game data is fetched from the NHL Web API (`api-web.nhle.com`). First run caches data locally for fast subsequent runs.

## License

Educational purposes. Game data from NHL's public API.
