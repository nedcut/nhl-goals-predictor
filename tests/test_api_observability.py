"""Tests for the API observability layer: middleware, /metrics, error handler.

Drives the endpoint/middleware coroutines directly via asyncio.run (the same
pattern as test_api_live.py) so no httpx/TestClient is required.
"""

from __future__ import annotations

import asyncio

import pytest

import src.api as api
from src.metrics import registry as metrics_registry


class _FakeState:
    pass


class _FakeURL:
    def __init__(self, path):
        self.path = path


class _FakeRequest:
    def __init__(self, path="/predict", method="GET"):
        self.url = _FakeURL(path)
        self.method = method
        self.state = _FakeState()


class _FakeResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code
        self.headers = {}


@pytest.fixture(autouse=True)
def _clean_registry():
    metrics_registry.reset()
    yield
    metrics_registry.reset()


def test_middleware_records_request_and_stamps_headers():
    async def call_next(request):
        return _FakeResponse(200)

    resp = asyncio.run(api.metrics_middleware(_FakeRequest("/predict"), call_next))

    assert "X-Request-ID" in resp.headers
    assert "X-Response-Time-ms" in resp.headers
    snap = metrics_registry.snapshot()
    assert snap["total_requests"] == 1
    assert snap["routes"]["/predict"]["count"] == 1


def test_middleware_records_500_and_reraises_on_unhandled_error():
    async def call_next(request):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        asyncio.run(api.metrics_middleware(_FakeRequest("/predict"), call_next))

    snap = metrics_registry.snapshot()
    assert snap["total_requests"] == 1
    assert snap["total_errors"] == 1  # recorded as a server error before re-raising


def test_metrics_endpoint_returns_snapshot():
    metrics_registry.record_request("/health", 200, 2.0)
    metrics_registry.increment("predictions_served", 4)

    snap = asyncio.run(api.get_metrics())

    assert snap["total_requests"] == 1
    assert snap["counters"]["predictions_served"] == 4
    assert "latency_ms" in snap


def test_unhandled_exception_handler_returns_safe_envelope():
    resp = asyncio.run(
        api.unhandled_exception_handler(_FakeRequest("/predict", "GET"), RuntimeError("kaboom"))
    )
    assert resp.status_code == 500
    body = resp.body.decode("utf-8")
    assert "Internal server error" in body
    assert "request_id" in body
    # The raw exception message must not leak to the client.
    assert "kaboom" not in body


def test_exception_handler_reuses_middleware_request_id():
    """Header X-Request-ID and body.request_id must be the same ID."""
    captured = {}

    async def call_next(request):
        # Simulate FastAPI: exception handler runs inside call_next and
        # returns a JSONResponse (middleware then stamps the same ID).
        captured["id"] = request.state.request_id
        return await api.unhandled_exception_handler(request, RuntimeError("kaboom"))

    resp = asyncio.run(api.metrics_middleware(_FakeRequest("/predict"), call_next))
    assert resp.status_code == 500
    rid = captured["id"]
    assert rid
    assert resp.headers["X-Request-ID"] == rid
    body = resp.body.decode("utf-8")
    assert f'"request_id":"{rid}"' in body.replace(" ", "")
    assert "kaboom" not in body
