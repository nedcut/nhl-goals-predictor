"""Locked, reproducible benchmark protocol and release-manifest generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ModelArtifact,
    ModelMetadata,
    feature_schema_hash,
)
from .champion import write_champion_reports
from .config import config
from .data import build_dataset
from .evaluation import CVForecastResult, _fit_point_model, evaluate_holdout_forecast
from .features import add_features
from .logging_config import setup_logging
from .model import get_feature_columns, optimize_hyperparameters
from .monitoring import histogram_reference
from .registry import ModelRegistry
from .release_docs import write_release_documents


@dataclass(frozen=True)
class BenchmarkProtocol:
    """Versioned benchmark definition used for all release claims."""

    version: str = "nhl-total-goals-v1"
    training_seasons: tuple[str, ...] = ("20222023", "20232024", "20242025")
    holdout_season: str = "20252026"
    game_types: tuple[str, ...] = ("R",)
    dist_model: str = "nb2"
    primary_threshold: float = 6.5
    thresholds: tuple[float, ...] = (5.5, 6.5, 7.5)
    cal_fraction: float = 0.2
    max_goals: int = 20
    include_goalies: bool = False
    include_xg: bool = False
    include_multi_window: bool = True
    include_interactions: bool = True
    include_temporal: bool = True
    bootstrap_unit: str = "iso_week"
    multiple_testing: str = "holm-bonferroni"
    notes: tuple[str, ...] = (
        "Primary cohort is regular-season games only.",
        "The 2025-26 season is untouched by hyperparameter tuning.",
        "Core-v1 excludes goalie and xG features until their historical coverage is release-grade.",
    )

    @property
    def seasons(self) -> tuple[str, ...]:
        return (*self.training_seasons, self.holdout_season)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataQualityReport:
    """Compact benchmark-source quality assessment."""

    ready: bool
    row_count: int
    primary_row_count: int
    duplicate_game_ids: int
    exact_duplicate_rows: int
    score_mismatches: int
    invalid_dates: int
    invalid_game_types: int
    missing_required: dict[str, int]
    season_game_type_counts: dict[str, dict[str, int]]
    date_min: str | None
    date_max: str | None
    data_fingerprint: str
    issues: tuple[dict[str, str], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_REQUIRED_COLUMNS = (
    "gamePk",
    "season",
    "gameType",
    "date",
    "homeTeam",
    "awayTeam",
    "homeScore",
    "awayScore",
    "totalGoals",
)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _frame_fingerprint(df: pd.DataFrame) -> str:
    columns = [column for column in _REQUIRED_COLUMNS if column in df.columns]
    canonical = df[columns].copy().sort_values(["date", "gamePk"]).reset_index(drop=True)
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assess_benchmark_data(
    raw_df: pd.DataFrame,
    protocol: BenchmarkProtocol,
) -> DataQualityReport:
    """Validate grain, scores, dates, cohort coverage, and source fingerprint."""
    missing_columns = [column for column in _REQUIRED_COLUMNS if column not in raw_df.columns]
    if missing_columns:
        raise ValueError(f"benchmark data missing required columns: {missing_columns}")

    frame = raw_df.copy()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    numeric = frame[["homeScore", "awayScore", "totalGoals"]].apply(pd.to_numeric, errors="coerce")
    missing_required = {column: int(frame[column].isna().sum()) for column in _REQUIRED_COLUMNS}
    duplicate_game_ids = int(frame["gamePk"].duplicated(keep=False).sum())
    exact_duplicates = int(frame.duplicated().sum())
    valid_scores = numeric.notna().all(axis=1)
    score_mismatches = int(
        (
            valid_scores & (numeric["homeScore"] + numeric["awayScore"] != numeric["totalGoals"])
        ).sum()
    )
    invalid_dates = int(dates.isna().sum())
    invalid_game_types = int((~frame["gameType"].isin(["R", "P", "O"])).sum())
    primary = frame[
        frame["season"].astype(str).isin(protocol.seasons)
        & frame["gameType"].isin(protocol.game_types)
    ].copy()

    count_table = (
        frame.groupby([frame["season"].astype(str), "gameType"]).size().unstack(fill_value=0)
    )
    counts = {
        str(season): {str(game_type): int(value) for game_type, value in row.items()}
        for season, row in count_table.iterrows()
    }
    issues: list[dict[str, str]] = []
    if duplicate_game_ids:
        issues.append(
            {
                "severity": "critical",
                "code": "duplicate_game_id",
                "message": f"{duplicate_game_ids} rows have duplicated gamePk values.",
            }
        )
    if score_mismatches:
        issues.append(
            {
                "severity": "critical",
                "code": "score_mismatch",
                "message": f"{score_mismatches} rows disagree on component and total scores.",
            }
        )
    if invalid_dates or invalid_game_types:
        issues.append(
            {
                "severity": "critical",
                "code": "invalid_domain_value",
                "message": (
                    f"invalid_dates={invalid_dates}, invalid_game_types={invalid_game_types}."
                ),
            }
        )
    missing_seasons = [
        season for season in protocol.seasons if season not in set(primary["season"].astype(str))
    ]
    if missing_seasons:
        issues.append(
            {
                "severity": "critical",
                "code": "missing_season",
                "message": f"Missing primary-cohort seasons: {missing_seasons}.",
            }
        )
    if any(missing_required.values()):
        issues.append(
            {
                "severity": "high",
                "code": "missing_required_values",
                "message": f"Required-column null counts: {missing_required}.",
            }
        )

    return DataQualityReport(
        ready=not any(issue["severity"] == "critical" for issue in issues),
        row_count=int(len(frame)),
        primary_row_count=int(len(primary)),
        duplicate_game_ids=duplicate_game_ids,
        exact_duplicate_rows=exact_duplicates,
        score_mismatches=score_mismatches,
        invalid_dates=invalid_dates,
        invalid_game_types=invalid_game_types,
        missing_required=missing_required,
        season_game_type_counts=counts,
        date_min=dates.min().date().isoformat() if dates.notna().any() else None,
        date_max=dates.max().date().isoformat() if dates.notna().any() else None,
        data_fingerprint=_frame_fingerprint(primary),
        issues=tuple(issues),
    )


def prepare_benchmark_frame(
    raw_df: pd.DataFrame,
    protocol: BenchmarkProtocol,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the locked primary cohort and one common complete feature frame."""
    primary = raw_df[
        raw_df["season"].astype(str).isin(protocol.seasons)
        & raw_df["gameType"].isin(protocol.game_types)
    ].copy()
    primary = primary.sort_values(["date", "gamePk"]).reset_index(drop=True)
    featured = add_features(
        primary,
        include_goalies=protocol.include_goalies,
        include_xg=protocol.include_xg,
        require_xg=protocol.include_xg,
        include_multi_window=protocol.include_multi_window,
        include_interactions=protocol.include_interactions,
        include_temporal=protocol.include_temporal,
    )
    feature_columns = get_feature_columns(featured)
    complete = featured.dropna(subset=feature_columns + ["totalGoals"]).copy()
    complete = complete.sort_values(["date", "gamePk"]).reset_index(drop=True)
    if complete.empty:
        raise ValueError("No complete rows remain in the locked benchmark cohort.")
    return complete, feature_columns


def _serialize_result(result: CVForecastResult) -> dict[str, Any]:
    return {
        "point_model": result.point_model,
        "dist_model": result.dist_model,
        "metrics": result.metrics_mean,
        "threshold_metrics": result.threshold_metrics or {},
        "n_test": int(sum(fold.n_test for fold in result.folds)),
        "n_train": int(sum(fold.n_train for fold in result.folds)),
        "n_cal": int(sum(fold.n_cal for fold in result.folds)),
    }


def build_release_artifact(
    *,
    manifest: dict[str, Any],
    frame: pd.DataFrame,
    feature_columns: list[str],
    protocol: BenchmarkProtocol,
    models_dir: Path = Path("models"),
) -> dict[str, Any]:
    """Fit the selected model on all release data and promote it in the registry."""
    champion_name = manifest["selection"]["champion"]["model"]
    point_model_by_name = {
        "xgb_current": "xgb",
        "xgb_tuned": "xgb",
        "team_strength": "team_strength",
        "double_poisson": "double_poisson",
        "poisson_glm": "poisson_glm",
    }
    point_model = point_model_by_name[champion_name]
    params = manifest["tuning"]["best_params"] if champion_name == "xgb_tuned" else None
    model, scaler, used_features = _fit_point_model(
        frame,
        point_model=point_model,  # type: ignore[arg-type]
        feature_cols=(
            feature_columns if point_model not in ("team_strength", "double_poisson") else None
        ),
        xgb_params=params,
    )
    if scaler is not None:
        setattr(model, "_scaler", scaler)

    metrics = manifest["models"][champion_name]["metrics"]
    baseline = manifest["models"]["team_strength"]["metrics"]
    interface = (
        "game_frame" if point_model in ("team_strength", "double_poisson") else "feature_matrix"
    )
    serving_features = [] if interface == "game_frame" else used_features
    if interface == "game_frame":
        reference_predictions = model.predict_mu(frame)
    else:
        matrix = frame[serving_features].to_numpy(dtype=float)
        if scaler is not None:
            matrix = scaler.transform(matrix)
        reference_predictions = model.predict(matrix)
    monitoring_reference = {
        "predicted_total_goals": histogram_reference(reference_predictions),
        "features": {
            name: histogram_reference(frame[name].to_numpy(dtype=float))
            for name in serving_features
        },
    }
    fingerprint = manifest["data_quality"]["data_fingerprint"]
    artifact_id = f"benchmark-v1-{champion_name}-{fingerprint[:12]}"
    metadata = ModelMetadata(
        model_type=champion_name,
        feature_names=serving_features,
        mae=float(metrics["mae"]),
        rmse=float(metrics["rmse"]),
        baseline_mae=float(baseline["mae"]),
        improvement_pct=float((1.0 - metrics["mae"] / baseline["mae"]) * 100.0),
        training_date=datetime.now(timezone.utc).isoformat(),
        n_training_samples=int(len(frame)),
        n_test_samples=int(manifest["cohort"]["holdout_rows"]),
        config_snapshot={
            "protocol": protocol.to_dict(),
            "model": {
                "point_model": point_model,
                "xgb_params": params or config.model.xgb_params,
            },
        },
        data_seasons=list(protocol.seasons),
        git_commit=_git_commit(),
        schema_version=ARTIFACT_SCHEMA_VERSION,
        benchmark_release="benchmark-v1",
        data_fingerprint=fingerprint,
        feature_schema_hash=feature_schema_hash(serving_features),
        prediction_interface=interface,
        training_cohort={
            "seasons": list(protocol.seasons),
            "game_types": list(protocol.game_types),
            "rows": int(len(frame)),
        },
        holdout_metrics={key: float(value) for key, value in metrics.items()},
        monitoring_reference=monitoring_reference,
        artifact_id=artifact_id,
    )
    artifact = ModelArtifact(model=model, metadata=metadata)
    artifact.validate_for_serving()
    registry = ModelRegistry(models_dir)
    version = registry.register(
        artifact,
        name=champion_name,
        description=(
            f"benchmark-v1 champion; holdout={protocol.holdout_season}; "
            f"cohort={','.join(protocol.game_types)}"
        ),
        promote_to_production=True,
    )
    return {
        "artifact_id": artifact_id,
        "registry_version": version,
        "registry_path": str(models_dir / "registry.json"),
        "prediction_interface": interface,
        "feature_schema_hash": metadata.feature_schema_hash,
        "data_fingerprint": fingerprint,
    }


def run_locked_benchmark(
    *,
    protocol: BenchmarkProtocol | None = None,
    tune_trials: int = 40,
    reports_dir: Path = Path("reports/releases/benchmark-v1"),
    models_dir: Path = Path("models"),
    build_artifact: bool = True,
) -> dict[str, Any]:
    """Run tuning on historical seasons and score the untouched holdout once."""
    protocol = protocol or BenchmarkProtocol()
    raw = build_dataset(protocol.seasons, use_cache=True)
    quality = assess_benchmark_data(raw, protocol)
    if not quality.ready:
        raise ValueError(f"Benchmark data failed release checks: {quality.issues}")
    frame, feature_columns = prepare_benchmark_frame(raw, protocol)
    train = frame[frame["season"].astype(str).isin(protocol.training_seasons)].copy()
    holdout = frame[frame["season"].astype(str) == protocol.holdout_season].copy()
    if train.empty or holdout.empty:
        raise ValueError("Locked training or holdout cohort is empty after feature filtering.")
    if pd.to_datetime(train["date"]).max() >= pd.to_datetime(holdout["date"]).min():
        raise ValueError("Training and holdout periods overlap.")

    tuned_params = dict()
    if tune_trials > 0:
        tuned_params = optimize_hyperparameters(
            train,
            n_trials=tune_trials,
            objective_metric="weighted_prob",
            dist_model=protocol.dist_model,  # type: ignore[arg-type]
            threshold=protocol.primary_threshold,
            tune_splits=3,
            show_progress=True,
        )

    def evaluate(name: str, point_model: str, params: dict[str, Any] | None = None):
        return name, evaluate_holdout_forecast(
            train,
            holdout,
            point_model=point_model,  # type: ignore[arg-type]
            dist_model=protocol.dist_model,  # type: ignore[arg-type]
            threshold=protocol.primary_threshold,
            thresholds=list(protocol.thresholds),
            cal_fraction=protocol.cal_fraction,
            max_goals=protocol.max_goals,
            feature_cols=feature_columns
            if point_model not in ("team_strength", "double_poisson")
            else None,
            xgb_params=params,
        )

    evaluations = dict(
        [
            evaluate("xgb_current", "xgb"),
            evaluate("xgb_tuned", "xgb", tuned_params or None),
            evaluate("team_strength", "team_strength"),
            evaluate("double_poisson", "double_poisson"),
            evaluate("poisson_glm", "poisson_glm"),
        ]
    )
    candidates = {name: result.metrics_mean for name, result in evaluations.items()}
    per_game = {
        name: result.per_game for name, result in evaluations.items() if result.per_game is not None
    }
    context = {
        "protocol": protocol.to_dict(),
        "training_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "feature_count": int(len(feature_columns)),
        "feature_names": feature_columns,
        "tune_trials": int(tune_trials),
        "tuned_params": tuned_params,
        "data_fingerprint": quality.data_fingerprint,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    champion = write_champion_reports(
        candidates=candidates,
        output_dir=reports_dir,
        context=context,
        per_game_map=per_game,
        baseline_name="team_strength",
    )
    manifest = {
        "release_schema_version": 1,
        "release_id": "benchmark-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "protocol": protocol.to_dict(),
        "data_quality": quality.to_dict(),
        "cohort": {
            "training_rows": int(len(train)),
            "holdout_rows": int(len(holdout)),
            "feature_count": int(len(feature_columns)),
            "feature_names": feature_columns,
        },
        "tuning": {"trials": int(tune_trials), "best_params": tuned_params},
        "models": {name: _serialize_result(result) for name, result in evaluations.items()},
        "selection": champion,
    }
    if build_artifact:
        manifest["production_artifact"] = build_release_artifact(
            manifest=manifest,
            frame=frame,
            feature_columns=feature_columns,
            protocol=protocol,
            models_dir=models_dir,
        )
    (reports_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_release_documents(manifest, release_dir=reports_dir)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the locked benchmark-v1 release protocol")
    parser.add_argument("--tune-trials", type=int, default=40)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/releases/benchmark-v1"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--no-artifact", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    manifest = run_locked_benchmark(
        tune_trials=args.tune_trials,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        build_artifact=not args.no_artifact,
    )
    champion = manifest["selection"]["champion"]["model"]
    print(f"benchmark-v1 complete; champion={champion}; {args.reports_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
