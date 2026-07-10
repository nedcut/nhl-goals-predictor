"""Durable prediction ledger, realized probabilistic scoring, and honest drift checks."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd

from .config import config
from .probabilistic import crps_per_game_from_pmf

if TYPE_CHECKING:
    from .artifacts import ModelArtifact


def histogram_reference(values: Iterable[float], bins: int = 10) -> dict[str, Any]:
    """Create a compact quantile-bin reference distribution for PSI."""
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return {"available": False, "reason": "fewer than two finite reference values"}
    edges = np.unique(np.quantile(array, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        center = float(array[0])
        edges = np.array([center - 0.5, center + 0.5])
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    counts, _ = np.histogram(array, bins=edges)
    return {
        "available": True,
        "edges": [None if not np.isfinite(value) else float(value) for value in edges],
        "fractions": (counts / counts.sum()).astype(float).tolist(),
        "n": int(array.size),
    }


def psi_against_reference(reference: dict[str, Any], recent: Iterable[float]) -> dict[str, Any]:
    """Score recent values against a saved histogram reference."""
    if not reference.get("available"):
        return {"available": False, "status": "unavailable", "reason": reference.get("reason")}
    values = np.asarray(list(recent), dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return {"available": False, "status": "unavailable", "reason": "insufficient recent values"}
    edges = np.asarray(
        [
            -np.inf if value is None and index == 0 else np.inf if value is None else value
            for index, value in enumerate(reference["edges"])
        ],
        dtype=float,
    )
    recent_counts, _ = np.histogram(values, bins=edges)
    recent_fraction = recent_counts / recent_counts.sum()
    reference_fraction = np.asarray(reference["fractions"], dtype=float)
    epsilon = 1e-6
    ref = np.clip(reference_fraction, epsilon, None)
    rec = np.clip(recent_fraction, epsilon, None)
    psi = float(np.sum((rec - ref) * np.log(rec / ref)))
    if psi >= config.monitoring.psi_significant:
        status = "significant"
    elif psi >= config.monitoring.psi_moderate:
        status = "moderate"
    else:
        status = "stable"
    return {"available": True, "status": status, "psi": psi, "n_recent": int(values.size)}


class PredictionLedger:
    """SQLite prediction ledger with transactional deduplication."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or config.monitoring.db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    game_pk INTEGER NOT NULL,
                    artifact_id TEXT NOT NULL,
                    forecast_kind TEXT NOT NULL DEFAULT 'pregame',
                    predicted_at TEXT NOT NULL,
                    game_date TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    mu REAL NOT NULL,
                    pmf_json TEXT,
                    thresholds_json TEXT,
                    features_json TEXT,
                    feature_schema_hash TEXT,
                    data_fingerprint TEXT,
                    PRIMARY KEY (game_pk, artifact_id, forecast_kind)
                )
                """
            )

    def upsert_predictions(
        self,
        predictions: pd.DataFrame,
        artifact: "ModelArtifact",
        *,
        forecast_kind: str = "pregame",
        predicted_at: str | None = None,
    ) -> int:
        """Insert or replace one canonical forecast per game/artifact/kind."""
        if predictions.empty:
            return 0
        required = {"gamePk", "date", "homeTeam", "awayTeam", "mu"}
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(f"predictions missing required ledger columns: {sorted(missing)}")
        artifact_id = artifact.metadata.artifact_id
        if not artifact_id:
            raise ValueError("release artifact_id is required for monitoring")
        stamp = predicted_at or datetime.now(timezone.utc).isoformat()
        rows = []
        for _, row in predictions.iterrows():
            pmf = row.get("pmf")
            thresholds = row.get("over_probs")
            features = row.get("feature_values")
            rows.append(
                (
                    int(row["gamePk"]),
                    artifact_id,
                    forecast_kind,
                    stamp,
                    str(row["date"]),
                    str(row["homeTeam"]),
                    str(row["awayTeam"]),
                    float(row["mu"]),
                    json.dumps(pmf) if pmf is not None else None,
                    json.dumps(thresholds) if thresholds is not None else None,
                    json.dumps(features) if features is not None else None,
                    artifact.metadata.feature_schema_hash,
                    artifact.metadata.data_fingerprint,
                )
            )
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_pk, artifact_id, forecast_kind) DO UPDATE SET
                    predicted_at=excluded.predicted_at,
                    game_date=excluded.game_date,
                    home_team=excluded.home_team,
                    away_team=excluded.away_team,
                    mu=excluded.mu,
                    pmf_json=COALESCE(excluded.pmf_json, predictions.pmf_json),
                    thresholds_json=COALESCE(excluded.thresholds_json, predictions.thresholds_json),
                    features_json=COALESCE(excluded.features_json, predictions.features_json),
                    feature_schema_hash=excluded.feature_schema_hash,
                    data_fingerprint=excluded.data_fingerprint
                """,
                rows,
            )
        return len(rows)

    def load(self, *, artifact_id: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM predictions"
        params: tuple[Any, ...] = ()
        if artifact_id:
            query += " WHERE artifact_id = ?"
            params = (artifact_id,)
        query += " ORDER BY predicted_at, game_pk"
        with self._connect() as connection:
            rows = [dict(row) for row in connection.execute(query, params).fetchall()]
        frame = pd.DataFrame(rows)
        for column in ("pmf_json", "thresholds_json", "features_json"):
            if column in frame:
                frame[column.removesuffix("_json")] = frame[column].map(
                    lambda value: json.loads(value) if value else None
                )
        return frame


def reconcile_outcomes(predictions: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Attach outcomes, using gamePk first and row-level date/team fallback."""
    if predictions.empty:
        return predictions.copy()
    output = predictions.copy()
    output["actual_total_goals"] = np.nan
    by_id = (
        results.dropna(subset=["gamePk"]).drop_duplicates("gamePk", keep="last")
        if {"gamePk", "totalGoals"}.issubset(results.columns)
        else pd.DataFrame(columns=["gamePk", "totalGoals"])
    )
    id_map = by_id.set_index("gamePk")["totalGoals"] if not by_id.empty else pd.Series(dtype=float)
    if "game_pk" in output:
        output["actual_total_goals"] = output["game_pk"].map(id_map)

    missing = output["actual_total_goals"].isna()
    fallback_keys = ["date", "homeTeam", "awayTeam"]
    if missing.any() and all(column in results for column in fallback_keys):
        fallback = results[fallback_keys + ["totalGoals"]].drop_duplicates(
            fallback_keys, keep="last"
        )
        fallback_map = {
            (str(row.date), str(row.homeTeam), str(row.awayTeam)): float(row.totalGoals)
            for row in fallback.itertuples(index=False)
        }
        for index in output.index[missing]:
            key = (
                str(output.at[index, "game_date"]),
                str(output.at[index, "home_team"]),
                str(output.at[index, "away_team"]),
            )
            if key in fallback_map:
                output.at[index, "actual_total_goals"] = fallback_map[key]
    return output


def realized_metrics(reconciled: pd.DataFrame) -> dict[str, Any]:
    """Compute point and full-distribution scores for completed logged games."""
    scored = reconciled.dropna(subset=["actual_total_goals", "mu"]).copy()
    if scored.empty:
        return {"n": 0, "mae": None, "rmse": None, "bias": None, "probabilistic": None}
    actual = scored["actual_total_goals"].to_numpy(dtype=float)
    predicted = scored["mu"].to_numpy(dtype=float)
    errors = predicted - actual
    probabilistic_rows = scored[scored["pmf"].notna()] if "pmf" in scored else scored.iloc[0:0]
    probabilistic = None
    if not probabilistic_rows.empty:
        pmfs = np.asarray(probabilistic_rows["pmf"].tolist(), dtype=float)
        y = probabilistic_rows["actual_total_goals"].to_numpy(dtype=int)
        y_index = y.clip(0, pmfs.shape[1] - 1)
        probability = np.take_along_axis(pmfs, y_index[:, None], axis=1).squeeze(1)
        lower_cdf = np.array([pmf[:index].sum() for pmf, index in zip(pmfs, y_index)], dtype=float)
        mid_pit = lower_cdf + 0.5 * probability
        probabilistic = {
            "n": int(len(pmfs)),
            "crps": float(np.mean(crps_per_game_from_pmf(pmfs, y_index))),
            "nll": float(np.mean(-np.log(np.clip(probability, 1e-12, 1.0)))),
            "mid_pit_mean": float(np.mean(mid_pit)),
            "mid_pit_tail_rate": float(np.mean((mid_pit < 0.1) | (mid_pit > 0.9))),
        }
    return {
        "n": int(len(scored)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)),
        "probabilistic": probabilistic,
    }


def monitoring_summary(
    ledger: PredictionLedger,
    results: pd.DataFrame,
    artifact: "ModelArtifact",
) -> dict[str, Any]:
    """Return realized scores and only drift checks supported by saved references."""
    logged = ledger.load(artifact_id=artifact.metadata.artifact_id)
    if logged.empty:
        return {"n_logged": 0, "realized": realized_metrics(pd.DataFrame()), "drift": {}}
    reconciled = reconcile_outcomes(logged, results)
    recent = reconciled.tail(config.monitoring.recent_window_games)
    reference = artifact.metadata.monitoring_reference or {}
    drift: dict[str, Any] = {
        "predicted_total_goals": psi_against_reference(
            reference.get(
                "predicted_total_goals",
                {"available": False, "reason": "artifact has no prediction reference"},
            ),
            recent["mu"],
        )
    }
    feature_references = reference.get("features", {})
    feature_rows = recent[recent["features"].notna()] if "features" in recent else recent.iloc[0:0]
    if not feature_references or feature_rows.empty:
        drift["features"] = {
            "available": False,
            "status": "unavailable",
            "reason": "artifact or served ledger has no numeric feature reference",
        }
    else:
        feature_values = pd.DataFrame(feature_rows["features"].tolist())
        checks = {
            name: psi_against_reference(saved, feature_values[name])
            for name, saved in feature_references.items()
            if name in feature_values
        }
        drift["features"] = {"available": bool(checks), "checks": checks}
    return {
        "artifact_id": artifact.metadata.artifact_id,
        "n_logged": int(len(logged)),
        "n_reconciled": int(reconciled["actual_total_goals"].notna().sum()),
        "realized": realized_metrics(recent),
        "drift": drift,
    }
