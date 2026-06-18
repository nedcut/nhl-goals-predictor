# NHL Goals Predictor: Improvement Analysis

## Current State Summary

Your pipeline is well-architected with clean separation of concerns, proper data leakage prevention via `shift(1)`, and solid engineering practices. However, the **+0.6% improvement over baseline** (MAE 1.882 vs 1.894) indicates significant room for improvement.

> **Note:** NHL scoring prediction is notoriously difficult—research suggests a ~62% accuracy ceiling for win prediction due to inherent randomness. However, your current performance suggests the model isn't yet capturing available signal. Similar projects achieve 5-10% improvement over baseline.

---

## Priority 1: Model Performance Improvements

### 1.1 Add a Poisson Regression Baseline

Your current approach predicts total goals directly. Research shows **Double Poisson regression** (modeling home/away goals separately) often outperforms direct total prediction because it captures the generative process of hockey scoring.

**Why this helps:** Goals follow a Poisson-like distribution. Modeling each team's scoring rate separately, then combining, captures offensive/defensive matchup dynamics better.

```python
# Concept: Predict lambda_home, lambda_away separately
# Expected total = lambda_home + lambda_away
# Distribution of total goals = sum of two Poisson distributions
```

**Implementation:** Add `train_poisson()` to `model.py` using `statsmodels.discrete.count_model.Poisson` or scikit-learn's `PoissonRegressor`.

### 1.2 Multi-Window Feature Engineering

Your current approach uses a single 20-game window. Research from [Evolving Hockey](https://evolving-hockey.com/blog/a-new-expected-goals-model-for-predicting-goals-in-the-nhl/) shows **multiple time horizons capture different signals**:

| Window | Signal Captured |
|--------|----------------|
| 5 games | Recent form/hot streaks |
| 10 games | Short-term trends |
| 20 games | Medium-term baseline |
| 40+ games | Season-level ability |

**Implementation:** Modify `features.py:70-162` to compute rolling stats for multiple windows (5, 10, 20, 40) and let the model learn which matters.

### 1.3 Add Interaction Features

Your features are independent. Hockey scoring is about **matchups**—a high-scoring team vs. weak defense is different from two balanced teams.

**Key interactions to add:**
- `home_avg_GF * away_avg_GA` — scoring opportunity
- `away_avg_GF * home_avg_GA` — opponent threat
- `home_goalie_sv_pct - away_avg_GF` — goalie vs. offense mismatch
- `rest_days_diff = home_rest_days - away_rest_days` — relative fatigue

### 1.4 Add Temporal/Seasonal Features

NHL scoring patterns vary throughout the season:

| Feature | Rationale |
|---------|-----------|
| `month` (one-hot) | Early season = more goals (rust), late season = tighter |
| `days_into_season` | Continuous seasonality |
| `is_divisional` | Division games tend lower-scoring |
| `is_playoff_race` | Late season intensity |

### 1.5 Calibration-Focused Training

Research from [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S266682702400015X) shows **calibration matters more than accuracy** for sports prediction. A well-calibrated model predicting 5.5 goals should be right ~50% of the time for over/under 5.5.

**Implementation:** Add calibration metrics and potentially use quantile regression:
```python
from sklearn.calibration import calibration_curve
# Or use XGBoost's quantile objective for uncertainty estimation
```

---

## Priority 2: Feature Engineering Additions

### 2.1 Expected Goals (xG) Integration

The most impactful improvement would be incorporating **expected goals** data, which measures shot quality rather than just outcomes. Sources:

- [MoneyPuck](https://moneypuck.com) provides free xG data
- [Natural Stat Trick](https://naturalstattrick.com) has team-level xG
- [Harry Shomer's xG model](https://github.com/HarryShomer/xG-Model) is open-source

**New features:**
- `home_xGF` / `away_xGF` — expected goals for
- `home_xGA` / `away_xGA` — expected goals against
- `home_xG_diff` — xGF - actual GF (luck indicator)

> **Insight:** xG measures shot quality by considering shot location, type, and game state. A team with high xGF but low actual GF is "unlucky" and likely to regress upward—a strong predictive signal the current model misses.

### 2.2 Score-Adjusted Statistics

Games with lopsided scores play differently (leading team plays conservative). **Score-adjusted** or **score-close** metrics (only counting stats when game is within 1-2 goals) are more predictive.

### 2.3 Special Teams Features

Power play and penalty kill significantly impact scoring:
- `home_pp_pct` — power play percentage
- `home_pk_pct` — penalty kill percentage
- `home_penalties_per_game` — discipline

These are available from the NHL API's team stats endpoints.

---

## Priority 3: Model Architecture Improvements

### 3.1 Ensemble Stacking

Combine multiple model types for robustness:

```python
# Level 1: Train diverse base models
models = [
    PoissonRegressor(),      # Captures count nature
    XGBRegressor(),          # Non-linear patterns
    Ridge(),                 # Linear baseline
]

# Level 2: Stack predictions
meta_model = XGBRegressor()  # Learn optimal combination
```

Research shows [ensemble methods achieve 90%+ accuracy](https://www.sciencedirect.com/science/article/abs/pii/S0957417419302556) for hockey win prediction.

### 3.2 Neural Network for Temporal Patterns

An LSTM or Transformer could capture team momentum patterns that rolling averages miss:

```python
# Sequence of last N games → predict next game
# Captures: "team is heating up" patterns
```

Consider using [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/) or Facebook's Prophet for quick experimentation.

### 3.3 Bayesian Approaches

For small sample sizes (early season), Bayesian models with informative priors can help:
- Prior: Team's previous season performance
- Likelihood: Current season games
- Posterior: Updated estimate

---

## Priority 4: Code Quality Improvements

### 4.1 Add Feature Importance Analysis Pipeline

Currently `plot_feature_importance()` exists but isn't integrated. Add:

```python
def analyze_feature_contribution(result: TrainingResult) -> pd.DataFrame:
    """Return ranked features with SHAP values for interpretability."""
    import shap
    explainer = shap.TreeExplainer(result.model)
    shap_values = explainer.shap_values(X_test)
    # Return feature importance with confidence intervals
```

### 4.2 Add Model Monitoring

Track prediction accuracy over time to detect drift:

```python
# In api.py or new monitoring.py
def log_prediction(game_id, predicted, actual=None):
    """Log predictions for later backtesting."""

def compute_recent_accuracy(days=30):
    """Compare predictions to outcomes for drift detection."""
```

### 4.3 Improve Error Handling in Data Fetching

`data.py` and `goalies.py` could benefit from retry logic:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_with_retry(url):
    ...
```

---

## Priority 5: Quick Wins

These require minimal effort but could provide immediate gains:

| Improvement | Effort | Expected Impact |
|-------------|--------|-----------------|
| Add `month` feature | 5 lines | +0.5-1% |
| Add `rest_days_diff` | 3 lines | +0.2-0.5% |
| Try LightGBM | Config change | Potentially faster + better |
| Add SHAP analysis | ~50 lines | Interpretability |
| Multiple rolling windows | ~30 lines | +1-2% |

---

## Recommended Implementation Order

1. **Multi-window features** (features.py) — highest ROI, minimal risk
2. **Poisson regression** (model.py) — new baseline comparison
3. **Interaction features** (features.py) — capture matchup dynamics
4. **xG data integration** (new data source) — major feature upgrade
5. **Ensemble stacking** (model.py) — combine model strengths
6. **Monitoring infrastructure** (new module) — production readiness

---

## Sources

- [Harvard Research: Using ML to Predict NHL Points](https://dash.harvard.edu/server/api/core/bitstreams/63ee1990-9df4-4a80-82fb-442e5561da96/content)
- [Evolving Hockey: xG Model](https://evolving-hockey.com/blog/a-new-expected-goals-model-for-predicting-goals-in-the-nhl/)
- [Sports Betting ML Systematic Review](https://arxiv.org/html/2410.21484v1)
- [Calibration vs Accuracy in Sports Betting](https://www.sciencedirect.com/science/article/pii/S266682702400015X)
- [Poisson Regression for Football](https://www.mdpi.com/2076-3417/14/16/7230)
- [LSE Over/Under Model](https://eprints.lse.ac.uk/103712/1/Predict_total_goals_LSE.pdf)
- [Harry Shomer's NHL xG Model](https://github.com/HarryShomer/xG-Model)
