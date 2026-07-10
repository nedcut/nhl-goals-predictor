"""Evaluate the project's models on all 2026 NHL playoff games (season 20252026).

The saved models/xgboost_v1.joblib lost its base_score intercept across an
XGBoost version boundary (pickled on an older version, unusable in 2.1.4), so we
retrain fresh on a strict temporal cutoff and test on the held-out playoffs.

Split (no leakage):
  TRAIN = every completed game with date < first 2026 playoff game
  TEST  = the 82 games of the 2026 Stanley Cup Playoffs

Models compared:
  - xgb_tuned      : XGBoost with config.model.xgb_params on 16 base features
  - team_strength  : regularized Poisson on team IDs (project baseline)
  - mean_baseline  : constant = training mean total goals

Probabilistic scoring uses NB2; the dispersion alpha is fit on the *training*
residuals (held out from the playoff test set), then applied to the playoffs.
"""

from __future__ import annotations

import numpy as np
from xgboost import XGBRegressor

from src import add_features, build_dataset
from src.config import config
from src.probabilistic import crps_from_pmf, fit_nb2_alpha, nb2_pmf_matrix, prob_over_from_pmf
from src.significance import paired_bootstrap
from src.team_strength import TeamStrengthConfig, TeamStrengthPoissonModel

SEASONS = ["20212022", "20222023", "20232024", "20242025", "20252026"]
THRESHOLDS = [4.5, 5.5, 6.5, 7.5]
BASE_FEATURES = [
    "home_avg_GF",
    "home_avg_GA",
    "home_avg_total",
    "home_win_pct",
    "home_rest_days",
    "home_is_back_to_back",
    "home_win_streak",
    "home_games_played",
    "away_avg_GF",
    "away_avg_GA",
    "away_avg_total",
    "away_win_pct",
    "away_rest_days",
    "away_is_back_to_back",
    "away_win_streak",
    "away_games_played",
]


def brier(prob, outcome):
    return float(np.mean((prob - outcome) ** 2))


def point_metrics(y, pred):
    err = pred - y
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "r": float(np.corrcoef(pred, y)[0, 1]),
    }


def prob_block(name, mu_test, y_test, alpha):
    pmf = nb2_pmf_matrix(mu_test, alpha=alpha, max_goals=20)
    crps = crps_from_pmf(pmf, y_test.astype(int))
    briers = {}
    for t in THRESHOLDS:
        briers[t] = brier(prob_over_from_pmf(pmf, t), (y_test > t).astype(float))
    return crps, briers


def main():
    raw = build_dataset(SEASONS)
    feat = add_features(
        raw,
        include_goalies=False,
        include_xg=False,
        include_multi_window=False,
        include_interactions=False,
        include_temporal=False,
    )
    feat = feat.dropna(subset=BASE_FEATURES + ["totalGoals"]).copy()

    # Identify the 2026 playoff games and the temporal cutoff.
    po_mask = (feat["season"].astype(str) == "20252026") & (feat["gameType"] == "P")
    cutoff = feat.loc[po_mask, "date"].min()
    test = feat[po_mask].sort_values("date")
    train = feat[feat["date"] < cutoff]

    print(f"Cutoff (first playoff game): {cutoff}")
    print(f"TRAIN games: {len(train)}  |  TEST (2026 playoffs): {len(test)}")
    print(
        f"Playoff actual total goals: mean={test['totalGoals'].mean():.3f} "
        f"std={test['totalGoals'].std():.3f}  "
        f"(train mean={train['totalGoals'].mean():.3f})"
    )

    y_train = train["totalGoals"].to_numpy(float)
    y_test = test["totalGoals"].to_numpy(float)

    # --- Model 1: tuned XGBoost ---
    xgb = XGBRegressor(**config.model.xgb_params, random_state=42, n_jobs=-1)
    xgb.fit(train[BASE_FEATURES], y_train)
    mu_xgb = xgb.predict(test[BASE_FEATURES])
    alpha_xgb = fit_nb2_alpha(y_train, xgb.predict(train[BASE_FEATURES]))

    # --- Model 2: team_strength baseline ---
    ts = TeamStrengthPoissonModel(TeamStrengthConfig(alpha=1.0)).fit(train)
    mu_ts = ts.predict_mu(test)
    alpha_ts = fit_nb2_alpha(y_train, ts.predict_mu(train))

    # --- Model 3: naive mean ---
    mu_mean = np.full_like(y_test, y_train.mean())
    alpha_mean = fit_nb2_alpha(y_train, np.full_like(y_train, y_train.mean()))

    models = {
        "xgb_tuned": (mu_xgb, alpha_xgb),
        "team_strength": (mu_ts, alpha_ts),
        "mean_baseline": (mu_mean, alpha_mean),
    }

    print("\n=== POINT ACCURACY — 2026 PLAYOFFS (n=%d) ===" % len(test))
    print(f"{'model':<16}{'MAE':>8}{'RMSE':>8}{'bias':>8}{'r':>8}{'mean_pred':>11}")
    for name, (mu, _) in models.items():
        m = point_metrics(y_test, mu)
        print(
            f"{name:<16}{m['MAE']:>8.3f}{m['RMSE']:>8.3f}{m['bias']:>8.3f}"
            f"{m['r']:>8.3f}{mu.mean():>11.3f}"
        )

    print("\n=== PROBABILISTIC (NB2) — 2026 PLAYOFFS ===")
    hdr = f"{'model':<16}{'alpha':>7}{'CRPS':>8}"
    for t in THRESHOLDS:
        hdr += f"{'Br>' + str(t):>9}"
    print(hdr)
    for name, (mu, alpha) in models.items():
        crps, briers = prob_block(name, mu, y_test, alpha)
        row = f"{name:<16}{alpha:>7.3f}{crps:>8.3f}"
        for t in THRESHOLDS:
            row += f"{briers[t]:>9.4f}"
        print(row)

    print("\n=== OVER/UNDER BASE RATES (actual) ===")
    for t in THRESHOLDS:
        print(f"  P(total > {t}) actual = {(y_test > t).mean():.3f}")

    print("\n=== SIGNIFICANCE: per-game |error| vs mean_baseline (paired bootstrap) ===")
    ae = {n: np.abs(mu - y_test) for n, (mu, _) in models.items()}
    for name in ("xgb_tuned", "team_strength"):
        cmp = paired_bootstrap(ae[name], ae["mean_baseline"], name_a=name, name_b="mean_baseline")
        print(
            f"  {name} vs mean_baseline: mean_diff={cmp.mean_diff:+.4f} "
            f"95% CI=[{cmp.ci_low:+.4f}, {cmp.ci_high:+.4f}] "
            f"p={cmp.p_value:.3f} -> {cmp.verdict}"
        )


if __name__ == "__main__":
    main()
