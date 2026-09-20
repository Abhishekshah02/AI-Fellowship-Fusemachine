"""Rate limiting, response caching and latency metrics.

Retries live in llm.py (tenacity) because only that layer knows which errors are
transient. Everything here is process-local: one API replica, one bucket. Scale
out and you want Redis instead -- see the note in README.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import deque
from typing import Any

from cachetools import TTLCache

from .config import settings


class RateLimiter:
    """Token bucket, refilled continuously. Async-safe."""

    def __init__(self, per_minute: int):
        self.capacity = float(per_minute)
        self.tokens = float(per_minute)
        self.rate = per_minute / 60.0
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def take(self) -> bool:
        async with self._lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            if self.tokens < 1.0:
                return False
            self.tokens -= 1.0
            return True

    def retry_after(self) -> int:
        return max(1, int((1.0 - self.tokens) / self.rate) + 1)


def cache_key(message: str, history: list[dict[str, str]]) -> str:
    blob = json.dumps({"m": message.strip().lower(), "h": history}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class Metrics:
    """Counters plus a rolling latency window for /metrics."""

    def __init__(self, window: int = 500):
        self.counters: dict[str, int] = {}
        self.latencies: deque[float] = deque(maxlen=window)

    def bump(self, name: str, n: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + n

    def observe(self, ms: float) -> None:
        self.latencies.append(ms)

    def snapshot(self) -> dict[str, Any]:
        lat = sorted(self.latencies)
        def pct(p: float) -> float:
            if not lat:
                return 0.0
            return round(lat[min(len(lat) - 1, int(p * len(lat)))], 1)
        return {
            **self.counters,
            "latency_ms_p50": pct(0.50),
            "latency_ms_p95": pct(0.95),
            "samples": len(lat),
        }


_s = settings()
limiter = RateLimiter(_s.rate_limit_per_minute)
response_cache: TTLCache = TTLCache(maxsize=_s.cache_max_entries, ttl=_s.cache_ttl_seconds)
metrics = Metrics()
