"""
In-process request metrics for the prediction API.

Deliberately dependency-free: no Prometheus client, no external collector. The
goal is a single self-contained registry that the API middleware feeds and the
``GET /metrics`` endpoint reads, so the *logic* (counting, latency percentiles,
snapshotting) is trivially unit-testable in isolation from the ASGI stack.

For a multi-process deployment (e.g. several uvicorn workers) these counters are
per-process; aggregate at the scrape layer or swap in a shared backend. For the
single-process nightly-inference use case here, in-process is the right weight.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, Optional


@dataclass
class RouteStats:
    """Per-route request tallies."""

    count: int = 0
    errors: int = 0  # responses with status >= 500
    total_latency_ms: float = 0.0


def _percentile(sorted_values, pct: float) -> float:
    """Nearest-rank percentile of an already-sorted list (0.0 if empty)."""
    if not sorted_values:
        return 0.0
    rank = math.ceil((pct / 100.0) * len(sorted_values))
    rank = max(1, min(rank, len(sorted_values)))
    return round(sorted_values[rank - 1], 2)


class MetricsRegistry:
    """Thread-safe in-process registry of request counts, latencies, counters."""

    def __init__(self, *, max_samples: int = 1024, started_at: Optional[datetime] = None):
        self._lock = threading.Lock()
        self._routes: Dict[str, RouteStats] = {}
        self._latencies: Deque[float] = deque(maxlen=max_samples)
        self._counters: Dict[str, int] = {}
        self._started_at = started_at or datetime.now()

    def record_request(self, route: str, status_code: int, latency_ms: float) -> None:
        """Record one completed request."""
        with self._lock:
            stats = self._routes.setdefault(route, RouteStats())
            stats.count += 1
            stats.total_latency_ms += latency_ms
            if status_code >= 500:
                stats.errors += 1
            self._latencies.append(latency_ms)

    def increment(self, name: str, by: int = 1) -> None:
        """Bump a named business counter (e.g. ``predictions_served``)."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + by

    def snapshot(self, *, now: Optional[datetime] = None) -> dict:
        """Return a JSON-serializable view of all metrics."""
        with self._lock:
            total = sum(s.count for s in self._routes.values())
            errors = sum(s.errors for s in self._routes.values())
            routes = {
                route: {
                    "count": s.count,
                    "errors": s.errors,
                    "avg_latency_ms": round(s.total_latency_ms / s.count, 2) if s.count else 0.0,
                }
                for route, s in sorted(self._routes.items())
            }
            latencies = sorted(self._latencies)
            current = now or datetime.now()
            return {
                "uptime_seconds": round((current - self._started_at).total_seconds(), 1),
                "total_requests": total,
                "total_errors": errors,
                "error_rate": round(errors / total, 4) if total else 0.0,
                "latency_ms": {
                    "p50": _percentile(latencies, 50),
                    "p95": _percentile(latencies, 95),
                    "p99": _percentile(latencies, 99),
                    "samples": len(latencies),
                },
                "routes": routes,
                "counters": dict(sorted(self._counters.items())),
            }

    def reset(self) -> None:
        """Clear all recorded state (used in tests)."""
        with self._lock:
            self._routes.clear()
            self._latencies.clear()
            self._counters.clear()


# Process-wide registry shared by the API middleware and the /metrics endpoint.
registry = MetricsRegistry()
