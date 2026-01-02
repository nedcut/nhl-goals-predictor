# NHL Total Goals Prediction

A machine learning pipeline for predicting total goals in NHL games. The model uses team rolling statistics and **goaltender performance metrics** to beat the naive baseline.

## Results

| Metric | Value |
|--------|-------|
| Test MAE | 1.882 |
| Baseline MAE | 1.894 |
| **Improvement** | **+0.6%** |

The model consistently outperforms predicting the historical mean across 5-fold time-series cross-validation.

## Features

20 engineered features including:
- Rolling goals for/against (20-game window)
- Win percentage and streaks
- Rest days and back-to-back indicators
- **Goalie save percentage and GAA**

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

## Project Structure

```
├── src/
│   ├── data.py      # NHL API data fetching with caching
│   ├── features.py  # Feature engineering (rolling stats, rest days)
│   ├── goalies.py   # Goalie data fetching from boxscores
│   └── model.py     # XGBoost training, CV, evaluation
├── notebooks/
│   └── eda.ipynb    # Full pipeline demonstration
├── data/            # Cached data (not in git)
└── models/          # Trained models (not in git)
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

## Data Sources

Game data is fetched from the NHL Web API (`api-web.nhle.com`). First run caches data locally for fast subsequent runs.

## License

Educational purposes. Game data from NHL's public API.
