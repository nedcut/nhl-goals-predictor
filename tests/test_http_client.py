"""Tests for the resilient HTTP layer (src/http_client.py).

These exercise the retry/backoff machinery deterministically by injecting a fake
session and a no-op sleep, so the suite never touches the network or actually
waits.
"""

from __future__ import annotations

from typing import List

import pytest
import requests

from src import http_client


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code: int, *, json_data=None, text="", headers=None):
        self.status_code = status_code
        self._json = json_data
        self.text = text
        self.headers = headers or {}
        self.closed = False

    def json(self):
        return self._json

    def close(self):
        self.closed = True

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}", response=self)


class FakeSession:
    """Replays a scripted sequence of responses/exceptions for ``request``."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls: List[dict] = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def recorded_sleeps(monkeypatch):
    """Capture sleep durations instead of waiting, for any code path that uses
    the module-level default sleep."""
    sleeps: List[float] = []
    monkeypatch.setattr(http_client.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def _no_sleep():
    sleeps: List[float] = []
    return sleeps, lambda s: sleeps.append(s)


def test_succeeds_first_try_makes_one_call():
    session = FakeSession([FakeResponse(200, json_data={"ok": True})])
    sleeps, sleep = _no_sleep()

    resp = http_client.request("GET", "http://x", session=session, sleep=sleep)

    assert resp.status_code == 200
    assert len(session.calls) == 1
    assert sleeps == []  # no backoff when first attempt succeeds


def test_retries_on_retryable_status_then_succeeds():
    session = FakeSession(
        [
            FakeResponse(503),
            FakeResponse(503),
            FakeResponse(200, json_data={"ok": True}),
        ]
    )
    sleeps, sleep = _no_sleep()

    resp = http_client.request("GET", "http://x", session=session, sleep=sleep, max_retries=3)

    assert resp.status_code == 200
    assert len(session.calls) == 3  # 2 failures + 1 success
    assert len(sleeps) == 2  # one backoff before each retry


def test_retries_on_connection_error_then_succeeds():
    session = FakeSession(
        [
            requests.ConnectionError("boom"),
            FakeResponse(200, json_data={"ok": True}),
        ]
    )
    sleeps, sleep = _no_sleep()

    resp = http_client.request("GET", "http://x", session=session, sleep=sleep, max_retries=2)

    assert resp.status_code == 200
    assert len(session.calls) == 2
    assert len(sleeps) == 1


def test_exhausts_retries_and_returns_last_bad_response():
    """After exhausting retries on a retryable status, the final (non-2xx)
    response is returned so the caller can raise_for_status()."""
    session = FakeSession([FakeResponse(500) for _ in range(4)])
    sleeps, sleep = _no_sleep()

    resp = http_client.request("GET", "http://x", session=session, sleep=sleep, max_retries=3)

    assert resp.status_code == 500
    assert len(session.calls) == 4  # initial + 3 retries
    assert len(sleeps) == 3
    with pytest.raises(requests.HTTPError):
        resp.raise_for_status()


def test_exhausts_retries_on_connection_error_and_raises():
    session = FakeSession([requests.ConnectionError("boom")] * 4)
    sleeps, sleep = _no_sleep()

    with pytest.raises(requests.ConnectionError):
        http_client.request("GET", "http://x", session=session, sleep=sleep, max_retries=3)
    assert len(session.calls) == 4


def test_non_retryable_4xx_is_not_retried():
    session = FakeSession([FakeResponse(404)])
    sleeps, sleep = _no_sleep()

    resp = http_client.request("GET", "http://x", session=session, sleep=sleep, max_retries=3)

    assert resp.status_code == 404
    assert len(session.calls) == 1  # 404 is a client error; no retry
    assert sleeps == []


def test_non_idempotent_method_is_not_retried():
    session = FakeSession([FakeResponse(503), FakeResponse(200)])
    sleeps, sleep = _no_sleep()

    resp = http_client.request("POST", "http://x", session=session, sleep=sleep, max_retries=3)

    # POST is not in RETRYABLE_METHODS, so the 503 is returned without retry.
    assert resp.status_code == 503
    assert len(session.calls) == 1


def test_retry_after_header_overrides_backoff():
    session = FakeSession(
        [
            FakeResponse(429, headers={"Retry-After": "7"}),
            FakeResponse(200),
        ]
    )
    sleeps, sleep = _no_sleep()

    http_client.request("GET", "http://x", session=session, sleep=sleep, max_retries=2)

    assert sleeps == [7.0]  # honored the server's explicit Retry-After


def test_retry_after_is_capped_at_max_backoff(monkeypatch):
    monkeypatch.setattr(http_client.config.data, "max_backoff_seconds", 5.0)
    session = FakeSession(
        [
            FakeResponse(429, headers={"Retry-After": "9999"}),
            FakeResponse(200),
        ]
    )
    sleeps, sleep = _no_sleep()

    http_client.request("GET", "http://x", session=session, sleep=sleep, max_retries=2)

    assert sleeps == [5.0]


def test_negative_retry_after_is_clamped_to_zero(monkeypatch):
    """A malicious/broken Retry-After must not raise ValueError in sleep()."""
    monkeypatch.setattr(http_client.config.data, "max_backoff_seconds", 5.0)
    session = FakeSession(
        [
            FakeResponse(429, headers={"Retry-After": "-3"}),
            FakeResponse(200),
        ]
    )
    sleeps, sleep = _no_sleep()

    http_client.request("GET", "http://x", session=session, sleep=sleep, max_retries=2)

    assert sleeps == [0.0]


def test_backoff_is_bounded_and_grows(monkeypatch):
    """Full-jitter backoff stays within [0, factor * 2**attempt], capped."""
    monkeypatch.setattr(http_client.config.data, "max_backoff_seconds", 100.0)
    # Make jitter deterministic: always return the ceiling.
    monkeypatch.setattr(http_client.random, "uniform", lambda lo, hi: hi)
    session = FakeSession([FakeResponse(503) for _ in range(4)])
    sleeps, sleep = _no_sleep()

    http_client.request(
        "GET",
        "http://x",
        session=session,
        sleep=sleep,
        max_retries=3,
        backoff_factor=1.0,
    )

    # ceilings: 1*2^0, 1*2^1, 1*2^2 = 1, 2, 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_get_json_parses_and_raises_on_error():
    session = FakeSession([FakeResponse(200, json_data={"a": 1})])
    sleeps, sleep = _no_sleep()
    assert http_client.get_json("http://x", session=session, sleep=sleep) == {"a": 1}

    session = FakeSession([FakeResponse(404)])
    with pytest.raises(requests.HTTPError):
        http_client.get_json("http://x", session=session, sleep=sleep)


def test_get_text_returns_body():
    session = FakeSession([FakeResponse(200, text="col1,col2\n1,2")])
    sleeps, sleep = _no_sleep()
    assert http_client.get_text("http://x", session=session, sleep=sleep) == "col1,col2\n1,2"


def test_session_is_shared_and_resettable():
    http_client.reset_session()
    s1 = http_client.get_session()
    s2 = http_client.get_session()
    assert s1 is s2  # cached
    assert "User-Agent" in s1.headers
    http_client.reset_session()
    s3 = http_client.get_session()
    assert s3 is not s1  # new instance after reset
    http_client.reset_session()
