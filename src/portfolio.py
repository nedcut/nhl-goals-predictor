"""Compatibility entrypoint for the authoritative locked benchmark.

Historically this module tuned and evaluated candidates on the same expanding-
window CV history.  It is intentionally retained only as a thin wrapper so old
automation cannot publish a second, conflicting model card or champion report.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkProtocol, run_locked_benchmark
from .logging_config import setup_logging


def run_portfolio_pipeline(
    *,
    seasons: list[str] | None = None,
    tune_trials: int = 40,
    dist_model: str = "nb2",
    threshold: float = 6.5,
    reports_dir: Path = Path("reports/releases/benchmark-v1"),
    include_xg: bool = False,
) -> dict[str, Any]:
    """Run benchmark-v1 while rejecting settings that would change its contract.

    The arguments remain for API compatibility. Release protocol changes must be
    made by defining and reviewing a new ``BenchmarkProtocol`` version instead
    of silently changing an existing release through CLI flags.
    """
    protocol = BenchmarkProtocol()
    requested_seasons = tuple(seasons) if seasons is not None else protocol.seasons
    if requested_seasons != protocol.seasons:
        raise ValueError(
            "src.portfolio now delegates to the locked benchmark; seasons must be "
            f"{list(protocol.seasons)}. Define a new BenchmarkProtocol for a new cohort."
        )
    if dist_model != protocol.dist_model or threshold != protocol.primary_threshold:
        raise ValueError(
            "Distribution and threshold are locked by BenchmarkProtocol; define a new "
            "protocol version to change them."
        )
    if include_xg != protocol.include_xg:
        raise ValueError("xG is excluded from benchmark-v1 until its coverage is release-grade.")
    return run_locked_benchmark(
        protocol=protocol,
        tune_trials=tune_trials,
        reports_dir=reports_dir,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper for the locked benchmark-v1 release"
    )
    parser.add_argument("--tune-trials", type=int, default=40)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    run_portfolio_pipeline(tune_trials=int(args.tune_trials))


if __name__ == "__main__":
    main()
