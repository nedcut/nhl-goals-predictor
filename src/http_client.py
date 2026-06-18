"""
Resilient HTTP layer shared by all outbound data fetching.

Every NHL Web API and MoneyPuck request in the pipeline used to be a bare
``requests.get(url).raise_for_status()`` — a single attempt with no recovery
from transient failures (rate limits, gateway hiccups, connection resets). A
nightly data pull that hit one 503 would silently drop a week of games.

This module centralizes outbound HTTP into one tested helper that adds:

- A shared :class:`requests.Session` for connection pooling and a stable
  ``User-Agent`` (polite API citizenship).
- Bounded retries on transient status codes (429 + 5xx) and network errors.
- Exponential backoff with **full jitter** to avoid synchronized retry storms.
- Respect for a server-provided ``Retry-After`` header when present.

Usage::

    from src.http_client import get_json, get_text

    payload = get_json(f"{API_BASE}/schedule/2024-01-01")
    csv_text = get_text(moneypuck_url)

All knobs live in :class:`~src.config.DataConfig` (``max_retries``,
``backoff_factor``, ``max_backoff_seconds``, ``request_timeout``,
``user_agent``).
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any, Callable, FrozenSet, Optional

import requests

from .config import config
from .logging_config import get_logger

logger = get_logger(__name__)

# Status codes worth retrying: rate limiting (429) plus transient server-side
# failures. 4xx codes other than 429 indicate a client problem that a retry
# will not fix, so they are intentionally excluded.
DEFAULT_RETRY_STATUSES: FrozenSet[int] = frozenset({429, 500, 502, 503, 504})

# Only retry idempotent methods by default. Retrying a non-idempotent POST after
# a timeout risks double-submitting; this pipeline only ever issues GETs.
RETRYABLE_METHODS: FrozenSet[str] = frozenset({"GET", "HEAD", "OPTIONS"})

_session: Optional[requests.Session] = None
_session_lock = threading.Lock()


def build_session() -> requests.Session:
    """Construct a fresh session with the pipeline's default headers."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": config.data.user_agent,
            "Accept": "application/json, text/csv, */*",
        }
    )
    return session


def get_session() -> requests.Session:
    """Return the process-wide shared session, creating it on first use.

    Thread-safe double-checked init so concurrent feature/goalie fetches share
    one connection pool rather than each opening fresh sockets.
    """
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = build_session()
    return _session


def reset_session() -> None:
    """Close and discard the shared session.

    Useful in tests and after a fork (where an inherited session's sockets are
    unsafe to reuse).
    """
    global _session
    with _session_lock:
        if _session is not None:
            _session.close()
        _session = None


def _compute_backoff(
    response: Optional[requests.Response],
    attempt: int,
    backoff_factor: float,
    max_backoff: float,
) -> float:
    """Seconds to wait before the next attempt.

    Prefers a numeric ``Retry-After`` header (the server telling us exactly how
    long to wait); otherwise falls back to exponential backoff with full jitter:
    a uniform draw from ``[0, backoff_factor * 2**attempt]``, capped at
    ``max_backoff``. Full jitter de-correlates many clients' retries so they do
    not resynchronize into a thundering herd.
    """
    if response is not None:
        header = response.headers.get("Retry-After")
        if header:
            try:
                # Clamp to [0, max_backoff]: negative Retry-After would raise in sleep().
                return min(max(float(header), 0.0), max_backoff)
            except (TypeError, ValueError):
                # Retry-After may be an HTTP-date; fall through to computed backoff.
                pass

    ceiling = min(max_backoff, backoff_factor * (2 ** attempt))
    return random.uniform(0, ceiling)


def request(
    method: str,
    url: str,
    *,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    backoff_factor: Optional[float] = None,
    retry_statuses: FrozenSet[int] = DEFAULT_RETRY_STATUSES,
    session: Optional[requests.Session] = None,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> requests.Response:
    """Issue an HTTP request with bounded retries and backoff.

    Retries on connection errors and on any status in ``retry_statuses`` (for
    idempotent methods only). After retries are exhausted, the final response is
    returned as-is so the caller can decide whether to ``raise_for_status()`` —
    we never swallow a non-2xx silently.

    Parameters mirror ``requests``; the extras (``max_retries``,
    ``backoff_factor``, ``retry_statuses``, ``session``, ``sleep``) default from
    config and exist mainly so tests can inject deterministic behavior.
    """
    sess = session or get_session()
    timeout = timeout if timeout is not None else config.data.request_timeout
    max_retries = max_retries if max_retries is not None else config.data.max_retries
    backoff_factor = (
        backoff_factor if backoff_factor is not None else config.data.backoff_factor
    )
    max_backoff = config.data.max_backoff_seconds
    retries_allowed = method.upper() in RETRYABLE_METHODS

    response: Optional[requests.Response] = None
    for attempt in range(max_retries + 1):
        try:
            response = sess.request(method, url, timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            if not retries_allowed or attempt >= max_retries:
                logger.warning(
                    "%s %s failed after %d attempt(s): %s",
                    method, url, attempt + 1, exc,
                )
                raise
            delay = _compute_backoff(None, attempt, backoff_factor, max_backoff)
            logger.info(
                "%s %s errored (%s); retry %d/%d in %.2fs",
                method, url, exc, attempt + 1, max_retries, delay,
            )
            sleep(delay)
            continue

        retryable_status = response.status_code in retry_statuses
        if retryable_status and retries_allowed and attempt < max_retries:
            delay = _compute_backoff(response, attempt, backoff_factor, max_backoff)
            logger.info(
                "%s %s returned %d; retry %d/%d in %.2fs",
                method, url, response.status_code, attempt + 1, max_retries, delay,
            )
            response.close()
            sleep(delay)
            continue

        return response

    # Unreachable in practice: the loop always returns or raises on the final
    # attempt. Guard anyway so the type checker sees a concrete return.
    assert response is not None  # noqa: S101 - defensive, loop guarantees this
    return response


def get(url: str, **kwargs: Any) -> requests.Response:
    """GET ``url`` with retries. Caller handles status/parsing."""
    return request("GET", url, **kwargs)


def get_json(url: str, **kwargs: Any) -> Any:
    """GET ``url``, raise on non-2xx, and return parsed JSON."""
    response = get(url, **kwargs)
    response.raise_for_status()
    return response.json()


def get_text(url: str, **kwargs: Any) -> str:
    """GET ``url``, raise on non-2xx, and return the response body as text."""
    response = get(url, **kwargs)
    response.raise_for_status()
    return response.text
