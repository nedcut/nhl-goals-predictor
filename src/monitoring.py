"""
Model monitoring: prediction logging, outcome reconciliation, and drift.

A live forecasting service is only trustworthy if you can answer two questions
*after* the games are played:

1. **Are we still accurate?** Realized MAE / Brier on logged predictions once the
   true totals are known — not the CV estimate, the actual field performance.
2. **Has the world shifted under the model?** Feature and prediction *drift*: is
   the distribution of inputs (or outputs) today materially different from what
   the model was trained and calibrated on?

This module provides the plumbing for both:

- :func:`log_predictions` / :func:`load_prediction_log` — an append-only JSON
  Lines store of every served prediction.
- :func:`reconcile_outcomes` — join logged predictions to realized totals.
- :func:`realized_metrics` — rolling MAE / RMSE / bias / directional accuracy.
- :func:`population_stability_index` + :func:`feature_drift` — PSI-based drift.
- :func:`assess_overall_drift` — roll per-feature PSI into one health verdict.
- :func:`monitoring_summary` — one dict combining the above, for API / CLI.

Nothing here touches the network; it operates purely on logged predictions and
already-cached game results.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import config
from .logging_config import get_logger

logger = get_logger(__name__)

# Smoothing floor so an empty histogram bin does not produce log(0) / div-by-0.
_PSI_EPS = 1e-6

# Columns we never treat as drift-able model inputs even though they are numeric.
_NON_FEATURE_COLUMNS = frozenset(
    {
        "gamePk",
        "homeScore",
        "awayScore",
        "totalGoals",
        "predicted_total_goals",
        "actual_total_goals",
        "prob_over",
        "threshold",
    }
)


# ---------------------------------------------------------------------------
# Prediction logging
# ---------------------------------------------------------------------------


@dataclass
class PredictionRecord:
    """One served prediction, persisted for later reconciliation."""

    predicted_total_goals: float
    gamePk: Optional[int] = None
    date: Optional[str] = None
    homeTeam: Optional[str] = None
    awayTeam: Optional[str] = None
    prob_over: Optional[float] = None
    threshold: Optional[float] = None
    model_version: str = "unknown"
    logged_at: str = field(default_factory=lambda: datetime.now().isoformat())


def _resolve_log_path(path: Optional[Path]) -> Path:
    return Path(path) if path is not None else config.monitoring.log_path


def log_predictions(
    predictions: pd.DataFrame,
    *,
    path: Optional[Path] = None,
    model_version: str = "unknown",
    logged_at: Optional[str] = None,
    threshold: Optional[float] = None,
) -> int:
    """Append predictions to the JSONL log. Returns the number of rows written.

    ``predictions`` must contain ``predicted_total_goals`` plus enough identity
    to later join against results — either ``gamePk`` or the
    ``(date, homeTeam, awayTeam)`` triple. Optional ``prob_over`` is recorded so
    the Brier score can be computed during reconciliation.

    Logging is intentionally best-effort at the call sites (a monitoring write
    must never break a user-facing prediction), but this function itself raises
    on a malformed frame so misuse is caught in tests.
    """
    if "predicted_total_goals" not in predictions.columns:
        raise ValueError("predictions must include a 'predicted_total_goals' column")

    target = _resolve_log_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    stamp = logged_at or datetime.now().isoformat()

    written = 0
    with target.open("a", encoding="utf-8") as handle:
        for _, row in predictions.iterrows():
            record = PredictionRecord(
                predicted_total_goals=float(row["predicted_total_goals"]),
                gamePk=int(row["gamePk"]) if "gamePk" in row and pd.notna(row["gamePk"]) else None,
                date=str(row["date"]) if "date" in row and pd.notna(row["date"]) else None,
                homeTeam=str(row["homeTeam"]) if "homeTeam" in row and pd.notna(row["homeTeam"]) else None,
                awayTeam=str(row["awayTeam"]) if "awayTeam" in row and pd.notna(row["awayTeam"]) else None,
                prob_over=float(row["prob_over"]) if "prob_over" in row and pd.notna(row["prob_over"]) else None,
                threshold=float(row["threshold"]) if "threshold" in row and pd.notna(row["threshold"]) else threshold,
                model_version=model_version,
                logged_at=stamp,
            )
            handle.write(json.dumps(asdict(record)) + "\n")
            written += 1

    logger.info("Logged %d predictions to %s", written, target)
    return written


def load_prediction_log(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the prediction log into a DataFrame (empty frame if absent)."""
    target = _resolve_log_path(path)
    if not target.exists():
        return pd.DataFrame()

    records: List[dict] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed prediction-log line in %s", target)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Outcome reconciliation + realized accuracy
# ---------------------------------------------------------------------------


def reconcile_outcomes(log_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """Attach realized ``actual_total_goals`` to logged predictions.

    Joins on ``gamePk`` when both frames carry it (the reliable key), otherwise
    falls back to the ``(date, homeTeam, awayTeam)`` triple. Predictions for
    games that have not been played yet keep ``actual_total_goals`` as NaN.
    """
    if log_df.empty:
        return log_df.copy()

    reconciled = log_df.copy()

    have_gamepk = (
        "gamePk" in reconciled.columns
        and "gamePk" in results_df.columns
        and reconciled["gamePk"].notna().any()
    )
    if have_gamepk:
        outcomes = (
            results_df[["gamePk", "totalGoals"]]
            .dropna(subset=["gamePk"])
            .drop_duplicates(subset=["gamePk"], keep="last")
        )
        outcomes = outcomes.rename(columns={"totalGoals": "actual_total_goals"})
        merged = reconciled.merge(outcomes, on="gamePk", how="left")
    else:
        keys = ["date", "homeTeam", "awayTeam"]
        if not all(k in reconciled.columns and k in results_df.columns for k in keys):
            reconciled["actual_total_goals"] = np.nan
            return reconciled
        outcomes = (
            results_df[keys + ["totalGoals"]]
            .drop_duplicates(subset=keys, keep="last")
            .rename(columns={"totalGoals": "actual_total_goals"})
        )
        merged = reconciled.merge(outcomes, on=keys, how="left")

    return merged


def realized_metrics(
    reconciled: pd.DataFrame,
    *,
    thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Compute realized accuracy over reconciled (outcome-known) predictions.

    Returns MAE / RMSE / bias plus, per over/under threshold, the point
    forecast's directional accuracy and (where probabilistic predictions were
    logged at that threshold) the Brier score.
    """
    thresholds = list(thresholds) if thresholds is not None else list(
        config.monitoring.brier_thresholds
    )
    empty = {"n": 0, "mae": None, "rmse": None, "bias": None,
             "mean_predicted": None, "mean_actual": None, "thresholds": {}}
    if reconciled.empty or "actual_total_goals" not in reconciled.columns:
        return empty

    scored = reconciled.dropna(subset=["actual_total_goals", "predicted_total_goals"])
    if scored.empty:
        return empty

    pred = scored["predicted_total_goals"].to_numpy(dtype=float)
    actual = scored["actual_total_goals"].to_numpy(dtype=float)
    errors = pred - actual

    per_threshold: Dict[str, Any] = {}
    for thr in thresholds:
        actual_over = actual > thr
        pred_over = pred > thr
        entry = {
            "point_accuracy": float(np.mean(pred_over == actual_over)),
            "actual_over_rate": float(np.mean(actual_over)),
            "predicted_over_rate": float(np.mean(pred_over)),
            "brier": None,
        }
        if "prob_over" in scored.columns and "threshold" in scored.columns:
            mask = scored["prob_over"].notna() & np.isclose(
                scored["threshold"].astype(float), thr
            )
            if mask.any():
                probs = scored.loc[mask, "prob_over"].to_numpy(dtype=float)
                outcomes = (
                    scored.loc[mask, "actual_total_goals"].to_numpy(dtype=float) > thr
                ).astype(float)
                entry["brier"] = float(np.mean((probs - outcomes) ** 2))
        per_threshold[str(thr)] = entry

    return {
        "n": int(len(scored)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "bias": float(np.mean(errors)),  # positive => over-predicting
        "mean_predicted": float(np.mean(pred)),
        "mean_actual": float(np.mean(actual)),
        "thresholds": per_threshold,
    }


# ---------------------------------------------------------------------------
# Drift detection (Population Stability Index)
# ---------------------------------------------------------------------------


def _psi_bin_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    """Quantile-based bin edges from the reference, with open outer edges.

    Quantile bins adapt to skewed distributions (hockey totals are right-
    skewed), and the open ``±inf`` outer edges ensure recent values outside the
    reference's observed range still land in a bin rather than being dropped.
    """
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:  # reference is (near-)constant
        center = float(edges[0]) if edges.size else 0.0
        edges = np.array([center - 0.5, center + 0.5])
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def population_stability_index(
    reference: Iterable[float],
    recent: Iterable[float],
    *,
    bins: Optional[int] = None,
    edges: Optional[np.ndarray] = None,
) -> float:
    """Population Stability Index between a reference and a recent sample.

    PSI = sum over bins of ``(recent% - ref%) * ln(recent% / ref%)``. Returns
    NaN when either sample is empty. ``0`` means identical distributions; larger
    is more drift.
    """
    bins = bins if bins is not None else config.monitoring.drift_bins
    ref = np.asarray(list(reference), dtype=float)
    rec = np.asarray(list(recent), dtype=float)
    ref = ref[~np.isnan(ref)]
    rec = rec[~np.isnan(rec)]
    if ref.size == 0 or rec.size == 0:
        return float("nan")

    if edges is None:
        edges = _psi_bin_edges(ref, bins)

    ref_counts, _ = np.histogram(ref, bins=edges)
    rec_counts, _ = np.histogram(rec, bins=edges)
    ref_frac = np.clip(ref_counts / ref_counts.sum(), _PSI_EPS, None)
    rec_frac = np.clip(rec_counts / rec_counts.sum(), _PSI_EPS, None)
    return float(np.sum((rec_frac - ref_frac) * np.log(rec_frac / ref_frac)))


def drift_status(psi: float) -> str:
    """Classify a PSI value as ``stable`` / ``moderate`` / ``significant``."""
    if psi is None or not np.isfinite(psi):
        return "unknown"
    if psi >= config.monitoring.psi_significant:
        return "significant"
    if psi >= config.monitoring.psi_moderate:
        return "moderate"
    return "stable"


def feature_drift(
    reference_df: pd.DataFrame,
    recent_df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    *,
    bins: Optional[int] = None,
) -> pd.DataFrame:
    """Per-feature PSI between a reference and a recent batch of inputs.

    Without ``columns``, scores every numeric column present in both frames
    (minus identity/outcome columns). Returns one row per feature sorted by PSI
    descending, with a ``status`` label from :func:`drift_status`.
    """
    if columns is None:
        shared = [c for c in reference_df.columns if c in recent_df.columns]
        columns = [
            c
            for c in shared
            if c not in _NON_FEATURE_COLUMNS
            and pd.api.types.is_numeric_dtype(reference_df[c])
            and pd.api.types.is_numeric_dtype(recent_df[c])
        ]

    rows = []
    for col in columns:
        ref_values = reference_df[col].to_numpy(dtype=float)
        rec_values = recent_df[col].to_numpy(dtype=float)
        psi = population_stability_index(ref_values, rec_values, bins=bins)
        rows.append(
            {
                "feature": col,
                "psi": psi,
                "status": drift_status(psi),
                "n_reference": int(np.sum(~np.isnan(ref_values))),
                "n_recent": int(np.sum(~np.isnan(rec_values))),
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("psi", ascending=False, na_position="last")
        result = result.reset_index(drop=True)
    return result


def assess_overall_drift(
    drift_df: pd.DataFrame, *, min_moderate: int = 2
) -> Dict[str, Any]:
    """Roll per-feature PSI into a single model-health verdict.

    Alerting policy (a product / risk-tolerance decision — see the module note)
    is deliberately **tier-dependent**, because the two PSI bands behave
    differently under many features:

    - **significant** if *any single* feature crosses ``psi_significant``
      (>= 0.25). That threshold is high enough that random noise rarely reaches
      it, so we stay maximally sensitive here — one badly-drifted input (e.g. a
      goalie metric going stale) can quietly poison forecasts, and missing it is
      worse than a rare false alarm.
    - **moderate** only if at least ``min_moderate`` features land in the
      moderate-or-worse band (>= 0.10). A single feature wandering into 0.10-0.25
      is expected noise across ~20 features, so we require corroboration before
      raising a moderate flag.
    - **stable** otherwise.

    The names of all drifted (moderate+) features are returned so an operator can
    triage quickly.
    """
    if drift_df.empty:
        return {"status": "unknown", "n_features": 0, "max_psi": None, "drifted_features": []}

    finite = drift_df[np.isfinite(drift_df["psi"])]
    max_psi = float(finite["psi"].max()) if not finite.empty else float("nan")

    n_significant = int((drift_df["status"] == "significant").sum())
    n_moderate_plus = int(drift_df["status"].isin(["moderate", "significant"]).sum())

    if n_significant >= 1:
        status = "significant"
    elif n_moderate_plus >= min_moderate:
        status = "moderate"
    elif (drift_df["status"] == "stable").any():
        status = "stable"
    else:
        status = "unknown"

    drifted = drift_df.loc[
        drift_df["status"].isin(["moderate", "significant"]), "feature"
    ].tolist()

    return {
        "status": status,
        "n_features": int(len(drift_df)),
        "max_psi": None if not np.isfinite(max_psi) else max_psi,
        "drifted_features": drifted,
    }


# ---------------------------------------------------------------------------
# Combined summary
# ---------------------------------------------------------------------------


def monitoring_summary(
    log_df: pd.DataFrame,
    results_df: pd.DataFrame,
    *,
    reference_predictions: Optional[Iterable[float]] = None,
    recent_window: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a JSON-serializable monitoring report.

    Combines realized accuracy on the most recent reconciled games with a
    prediction-drift check on the logged point forecasts. When
    ``reference_predictions`` is supplied (e.g. the training-time predicted
    totals) it is the drift reference; otherwise the older half of the log
    serves as the reference for the newer half.
    """
    recent_window = recent_window or config.monitoring.recent_window_games

    if log_df.empty:
        return {
            "n_logged": 0,
            "realized": realized_metrics(pd.DataFrame()),
            "prediction_drift": {"psi": None, "status": "unknown"},
        }

    reconciled = reconcile_outcomes(log_df, results_df)

    # Order by logged time so "recent" means most-recently served.
    if "logged_at" in reconciled.columns:
        reconciled = reconciled.sort_values("logged_at").reset_index(drop=True)
    recent = reconciled.tail(recent_window)

    realized = realized_metrics(recent)

    preds = log_df["predicted_total_goals"].to_numpy(dtype=float)
    if reference_predictions is not None:
        ref_preds = np.asarray(list(reference_predictions), dtype=float)
        rec_preds = preds
    else:
        split = len(preds) // 2
        ref_preds = preds[:split]
        rec_preds = preds[split:]

    if ref_preds.size and rec_preds.size:
        psi = population_stability_index(ref_preds, rec_preds)
        pred_drift = {"psi": psi, "status": drift_status(psi)}
    else:
        pred_drift = {"psi": None, "status": "unknown"}

    return {
        "n_logged": int(len(log_df)),
        "n_reconciled": int(reconciled["actual_total_goals"].notna().sum())
        if "actual_total_goals" in reconciled.columns
        else 0,
        "realized": realized,
        "prediction_drift": pred_drift,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize prediction-log accuracy and drift."
    )
    parser.add_argument(
        "--log", type=Path, default=None, help="Path to the prediction log (JSONL)."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Seasons of realized results to reconcile against (default: recent 2).",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional path to write the JSON summary."
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point: print (and optionally save) a monitoring summary."""
    from .data import build_dataset, recent_seasons

    args = _parse_args(argv)
    log_df = load_prediction_log(args.log)
    if log_df.empty:
        print("No predictions logged yet.")
        return

    seasons = args.seasons or recent_seasons(2)
    results = build_dataset(seasons, use_cache=True)
    summary = monitoring_summary(log_df, results)

    text = json.dumps(summary, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()
