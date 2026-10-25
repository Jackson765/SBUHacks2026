"""Three non-overlapping five-minute windows, measured in elapsed video time."""
from dataclasses import dataclass
import math


WINDOW_SECONDS = 5 * 60
WINDOW_COUNT = 3
COLLECTION_SECONDS = WINDOW_SECONDS * WINDOW_COUNT


@dataclass(frozen=True)
class SpeedWindow:
    start_seconds: float
    end_seconds: float
    average_speed: float | None
    vehicle_count: int


class FiveMinuteAverages:
    """Give each tracked car equal weight within a window.

    First average valid speed estimates for each ID, then average across IDs.
    Empty windows stay None; measured stopped cars contribute a genuine zero.
    """
    def __init__(self):
        self.windows = []
        self._vehicles = {}  # track ID -> [sum of speed estimates, sample count]
        self._last_time = 0.0

    @property
    def complete(self):
        return len(self.windows) == WINDOW_COUNT

    @property
    def averages(self):
        return [window.average_speed for window in self.windows]

    def advance(self, elapsed_seconds):
        """Close bins BEFORE assigning a boundary-frame measurement to the next bin."""
        now = float(elapsed_seconds)
        if not math.isfinite(now) or now < self._last_time:
            raise ValueError("Elapsed video time must be finite, nonnegative, and nondecreasing.")
        self._last_time = now
        completed = []
        while not self.complete and now + 1e-9 >= (len(self.windows) + 1) * WINDOW_SECONDS:
            index = len(self.windows)
            car_averages = [total / count for total, count in self._vehicles.values()]
            average = math.fsum(car_averages) / len(car_averages) if car_averages else None
            window = SpeedWindow(index * WINDOW_SECONDS, (index + 1) * WINDOW_SECONDS,
                                 average, len(car_averages))
            self.windows.append(window)
            completed.append(window)
            self._vehicles.clear()
        return completed

    def add_measurements(self, elapsed_seconds, measurements):
        """Add (track ID, speed mph or None) for one processed frame."""
        completed = self.advance(elapsed_seconds)
        if self.complete:
            return completed
        for track_id, speed in measurements:
            if speed is None:
                continue
            speed = float(speed)
            if not math.isfinite(speed) or speed < 0:
                raise ValueError("Vehicle speed measurements must be finite and nonnegative.")
            total, count = self._vehicles.get(track_id, (0.0, 0))
            self._vehicles[track_id] = [total + speed, count + 1]
        return completed