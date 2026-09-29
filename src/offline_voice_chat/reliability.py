"""Time limits for the active phases of a voice turn."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class DeadlineSignal:
    kind: str
    phase: str
    elapsed: float


class ActiveTurnDeadline:
    """
    With no phase armed there is no limit, so idle listening never times out.
    Callers poll from their watchdog thread, so tests can pass the time.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._phase: str | None = None
        self._started_at = 0.0
        self._last_activity = 0.0
        self._timeout = 0.0
        self._progress_interval: float | None = None
        self._next_progress: float | None = None

    @property
    def phase(self) -> str | None:
        with self._lock:
            return self._phase

    def arm(
        self,
        phase: str,
        timeout: float,
        *,
        progress_interval: float | None = None,
    ) -> None:
        now = self._clock()
        with self._lock:
            self._phase = phase
            self._started_at = now
            self._last_activity = now
            self._timeout = timeout
            self._progress_interval = progress_interval
            self._next_progress = None

            if progress_interval is not None:
                self._next_progress = now + progress_interval

    def touch(self, phase: str | None = None) -> bool:
        """
        Ignored when another phase is active.
        """

        now = self._clock()
        with self._lock:
            if self._phase is None:
                return False

            if phase is not None and self._phase != phase:
                return False

            self._last_activity = now
            return True

    def clear(self, phase: str | None = None) -> bool:
        """
        With a phase given, only that phase is cleared.
        """

        with self._lock:
            if self._phase is None:
                return False

            if phase is not None and self._phase != phase:
                return False

            self._clear_locked()
            return True

    def poll(self, now: float | None = None) -> DeadlineSignal | None:
        """
        A timeout also clears the deadline.
        """

        checked_at = self._clock() if now is None else now
        with self._lock:
            phase = self._phase
            if phase is None:
                return None

            elapsed = max(0.0, checked_at - self._started_at)

            if checked_at - self._last_activity >= self._timeout:
                self._clear_locked()
                return DeadlineSignal("timeout", phase, elapsed)

            interval = self._progress_interval
            next_progress = self._next_progress

            # interval and next_progress are set and cleared together.
            if interval is not None and checked_at >= next_progress:
                intervals_passed = int((checked_at - next_progress) // interval)
                self._next_progress = next_progress + (intervals_passed + 1) * interval
                return DeadlineSignal("progress", phase, elapsed)

            return None

    def _clear_locked(self) -> None:
        self._phase = None
        self._progress_interval = None
        self._next_progress = None
