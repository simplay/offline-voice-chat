"""Per-turn latency measurements using a monotonic clock."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class TurnMetrics:
    def __init__(
        self,
        *,
        submitted_at: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._started = clock() if submitted_at is None else submitted_at
        self._lock = threading.Lock()
        self._marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        with self._lock:
            if name not in self._marks:
                self._marks[name] = self._clock() - self._started

    def summary(self) -> str:
        with self._lock:
            return ", ".join(
                f"{name}={seconds * 1000:.0f} ms"
                for name, seconds in self._marks.items()
            )
