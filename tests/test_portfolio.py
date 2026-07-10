from __future__ import annotations

import pytest

from src.benchmark import BenchmarkProtocol
from src.portfolio import run_portfolio_pipeline


def test_portfolio_rejects_unversioned_cohort_changes():
    with pytest.raises(ValueError, match="new BenchmarkProtocol"):
        run_portfolio_pipeline(seasons=["20242025", "20252026"])


def test_portfolio_delegates_to_locked_protocol(monkeypatch):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"release_id": "benchmark-v1"}

    monkeypatch.setattr("src.portfolio.run_locked_benchmark", fake_run)
    result = run_portfolio_pipeline(tune_trials=3)
    assert result == {"release_id": "benchmark-v1"}
    assert captured["protocol"] == BenchmarkProtocol()
    assert captured["tune_trials"] == 3
