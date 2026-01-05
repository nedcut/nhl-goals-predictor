# NHL Total Goals Prediction

A machine learning pipeline for predicting total goals in NHL games. The model uses team rolling statistics, goaltender performance metrics, and head-to-head history to beat the naive baseline.

## Results

| Metric | Value |
|--------|-------|
| Test MAE | 1.882 |
| Baseline MAE | 1.894 |
| **Improvement** | **+0.6%** |

The model consistently outperforms predicting the historical mean across 5-fold time-series cross-validation.

## Features

22+ engineered features including:
- Rolling goals for/against (20-game window)
- Win percentage and streaks
- Rest days and back-to-back indicators
- Goalie save percentage and GAA
- **Head-to-head historical matchup stats** (new)
- **Venue-specific scoring trends** (new)

## Quick Start

```bash
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

## Prediction CLI

Make predictions for upcoming games:

```bash
# Predict games for the next 7 days
python -m src.predict --model models/xgboost_v1 --days 7

# Save predictions to CSV
python -m src.predict --output predictions.csv

# Specify historical data seasons
python -m src.predict --seasons 20232024 20242025
```

## REST API

Run the prediction API server:

```bash
# Start the server
uvicorn src.api:app --reload

# Visit http://localhost:8000/docs for interactive documentation
```

**Endpoints:**
- `GET /predict?days_ahead=7` - Get predictions for upcoming games
- `GET /model/info` - Get model metadata and performance metrics
- `GET /health` - Health check

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
│   ├── artifacts.py    # Model persistence with metadata
│   ├── registry.py     # Model versioning and management
│   ├── validation.py   # Input validation utilities
│   └── logging_config.py # Structured logging
├── notebooks/
│   └── eda.ipynb       # Full pipeline demonstration
├── data/               # Cached data (not in git)
└── models/             # Trained models (not in git)
```

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

1. **Heavy regularization is critical** - max_depth=2, high L1/L2 penalties
2. **Recent data works best** - 4 seasons (2021-2025) outperforms 11 seasons
3. **Goalie features matter** - GAA is in top 4 most important features
4. **Baseline is hard to beat** - NHL scoring is inherently noisy

## Optimized Parameters

```python
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
