"""Pace source timestamps without speeding up after slow processing."""
from math import isfinite
from time import perf_counter, sleep


class RealtimePlaybackClock:
    """Release frames no faster than their source intervals; stop waits promptly."""

    def __init__(self, *, clock=perf_counter, sleeper=sleep):
        self._clock = clock
        self._sleep = sleeper
        self._last_elapsed = 0.0
        self._last_wall = clock()

    def wait(self, elapsed, should_stop=None):
        elapsed = float(elapsed)
        if not isfinite(elapsed) or elapsed < self._last_elapsed:
            raise ValueError("Playback time must be finite and nondecreasing.")
        deadline = self._last_wall + elapsed - self._last_elapsed
        while True:
            if should_stop is not None and should_stop():
                return False
            remaining = deadline - self._clock()
            if remaining <= 0:
                break
            self._sleep(min(remaining, .05))
        # Anchor each next interval to the actual release time. A late detector
        # must not create a backlog that is then played faster than real time.
        self._last_elapsed = elapsed
        self._last_wall = self._clock()
        return True
