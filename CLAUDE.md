# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NHL Total Goals Predictor - a machine learning pipeline that predicts total goals in NHL games using XGBoost. The model uses rolling team statistics, goaltender performance metrics, and head-to-head history to beat the naive baseline (predicting historical mean).

**Current Performance:** Test MAE 1.882 vs Baseline MAE 1.894 (+0.6% improvement)

## Commands

### Run Tests
```bash
pytest tests/ -v                           # Run all tests
pytest tests/test_pipeline.py -v           # Run specific test file
pytest tests/ -v -k "test_end_to_end"      # Run tests matching pattern
```

### Start REST API
```bash
uvicorn src.api:app --reload               # Development server at http://localhost:8000
```

### Make Predictions via CLI
```bash
python -m src.predict --model models/xgboost_v1 --days 7
python -m src.predict --output predictions.csv
```

### Download Game Data
```bash
python -m src.data --seasons 20232024 20242025 --out data/raw/games.csv
```

### Train Model (via Python)
```python
from src import build_dataset, add_features, train_xgboost, save_model
df = build_dataset(['20232024', '20242025'])
df = add_features(df)
result = train_xgboost(df)
save_model(result, 'models/xgboost_v1', seasons=['20232024', '20242025'])
```

## Architecture

### Data Flow
1. **data.py** - Fetches game schedules from NHL Web API (`api-web.nhle.com`), caches to `data/raw/{season}.csv`
2. **goalies.py** - Fetches starting goalie stats from boxscores, caches to `data/goalies/goalie_stats.csv`
3. **features.py** - Computes rolling features (20-game window by default) for teams and goalies
4. **model.py** - Trains XGBoost/RF models with time-series cross-validation

### Key Design Decisions

**Rolling Stats with shift(1):** All rolling features use `shift(1)` before computing to prevent data leakage - features only use data available before each game.

**Vectorized Operations:** Goalie rolling stats in `goalies.py:161-234` use `groupby().transform()` instead of iterrows for 10-50x speedup.

**Centralized Config:** All parameters live in `config.py` - rolling windows, API delays, XGBoost hyperparameters. Import via `from src.config import config`.

**Model Artifacts:** Models are saved with full metadata (MAE, features, config snapshot, git commit) via `artifacts.py`. Registry in `registry.py` handles versioning.

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `config.py` | Single source of truth for all configurable parameters |
| `data.py` | NHL API data fetching with per-season caching |
| `features.py` | Rolling team stats, H2H history, venue trends |
| `goalies.py` | Goalie data fetching and rolling save%/GAA |
| `model.py` | Training, CV, Optuna optimization |
| `artifacts.py` | Model + metadata persistence |
| `registry.py` | Model versioning and promotion |
| `predict.py` | CLI for predictions |
| `api.py` | FastAPI REST endpoints |
| `validation.py` | DataFrame validation utilities |

### Feature Categories (22 total)

- **Team Rolling Stats:** `home_avg_GF`, `away_avg_GA`, `home_win_pct`, etc. (20-game window)
- **Rest/Fatigue:** `home_rest_days`, `away_is_back_to_back`
- **Goalie:** `home_goalie_sv_pct`, `away_goalie_gaa` (10-game window)
- **Head-to-Head:** `h2h_avg_goals` - rolling average in matchups between same teams
- **Venue:** `venue_avg_goals` - rolling average goals at home arena

## API Endpoints

- `GET /predict?days_ahead=7` - Predictions for upcoming games
- `GET /model/info` - Model metadata and performance metrics
- `GET /health` - Health check

## Important Patterns

**Season Format:** Always use 8-digit format like `"20232024"` (start year + end year).

**Data Caching:** First run downloads from NHL API (~20s/season), subsequent runs load from `data/raw/`.

**Time-Series Split:** Training uses chronological splits (not random) to prevent future data leakage.

**Heavy Regularization:** XGBoost uses `max_depth=2`, high L1/L2 penalties because NHL scoring is inherently noisy.
