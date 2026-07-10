"""Tests for src/metrics.py — the in-process request metrics registry."""

from __future__ import annotations

from datetime import datetime, timedelta

from src.metrics import MetricsRegistry, _percentile


def test_record_request_tallies_count_and_latency():
    reg = MetricsRegistry()
    for latency in (10.0, 20.0, 30.0, 40.0, 50.0):
        reg.record_request("/predict", 200, latency)

    snap = reg.snapshot()
    assert snap["total_requests"] == 5
    assert snap["routes"]["/predict"]["count"] == 5
    assert snap["routes"]["/predict"]["avg_latency_ms"] == 30.0
    assert snap["total_errors"] == 0
    assert snap["error_rate"] == 0.0


def test_server_errors_counted_client_errors_not():
    reg = MetricsRegistry()
    reg.record_request("/predict", 200, 5.0)
    reg.record_request("/predict", 404, 5.0)  # client error -> not an "error"
    reg.record_request("/predict", 503, 5.0)  # server error -> counted

    snap = reg.snapshot()
    assert snap["total_requests"] == 3
    assert snap["total_errors"] == 1
    assert snap["error_rate"] == round(1 / 3, 4)


def test_latency_percentiles_are_deterministic():
    reg = MetricsRegistry()
    for latency in (10.0, 20.0, 30.0, 40.0, 50.0):
        reg.record_request("/x", 200, latency)

    lat = reg.snapshot()["latency_ms"]
    assert lat["samples"] == 5
    assert lat["p50"] == 30.0   # nearest-rank: ceil(0.50*5)=3 -> 30
    assert lat["p95"] == 50.0   # ceil(0.95*5)=5 -> 50


def test_percentile_helper_edges():
    assert _percentile([], 50) == 0.0
    assert _percentile([7.0], 99) == 7.0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 100) == 4.0


def test_counters_increment():
    reg = MetricsRegistry()
    reg.increment("predictions_served", 3)
    reg.increment("predictions_served", 2)
    reg.increment("cache_hits")  # default by=1
    snap = reg.snapshot()
    assert snap["counters"]["predictions_served"] == 5
    assert snap["counters"]["cache_hits"] == 1


def test_uptime_uses_injected_clock():
    start = datetime(2025, 1, 1, 0, 0, 0)
    reg = MetricsRegistry(started_at=start)
    snap = reg.snapshot(now=start + timedelta(seconds=42))
    assert snap["uptime_seconds"] == 42.0


def test_max_samples_bounds_latency_window():
    reg = MetricsRegistry(max_samples=3)
    for latency in (1.0, 2.0, 3.0, 4.0, 5.0):
        reg.record_request("/x", 200, latency)
    # Only the last 3 latencies are retained for percentile computation...
    assert reg.snapshot()["latency_ms"]["samples"] == 3
    # ...but the lifetime request count is unaffected.
    assert reg.snapshot()["total_requests"] == 5


def test_reset_clears_state():
    reg = MetricsRegistry()
    reg.record_request("/x", 200, 5.0)
    reg.increment("predictions_served", 9)
    reg.reset()
    snap = reg.snapshot()
    assert snap["total_requests"] == 0
    assert snap["counters"] == {}
    assert snap["routes"] == {}
