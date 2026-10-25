"""Count unique measured vehicles within a fixed calibrated road area."""
from math import fsum, isfinite

import cv2
import numpy as np

from video_settings import validate_base_speed_limit


class AreaVehicleCounter:
    """Keep run totals while reporting only the current in-area detections.

    A bounding box's bottom center is its road-contact estimate, matching the
    speed tracker. Each track ID contributes once to each cumulative total.
    """

    def __init__(self, calibration, base_speed_limit):
        self._polygon = np.asarray(calibration.points, dtype=np.float32).copy()
        self._threshold = validate_base_speed_limit(base_speed_limit) + 10
        self._detected_ids = set()
        self._speeding_ids = set()
        self._visible_ids = set()
        self._current_average = None

    def update(self, vehicles):
        """Filter detections and update totals; None means no new inference."""
        if vehicles is None:
            return None
        filtered = []
        visible_ids = set()
        current_speeds = {}
        for box, track_id, speed in vehicles:
            try:
                x1, y1, x2, y2 = map(float, box)
            except (TypeError, ValueError, OverflowError):
                continue
            if not all(isfinite(value) for value in (x1, y1, x2, y2)):
                continue
            if cv2.pointPolygonTest(self._polygon, ((x1 + x2) / 2, y2), False) < 0:
                continue
            filtered.append((box, track_id, speed))
            visible_ids.add(track_id)
            self._detected_ids.add(track_id)
            try:
                measured = float(speed)
            except (TypeError, ValueError, OverflowError):
                continue
            if not isfinite(measured) or measured < 0:
                continue
            # Duplicate boxes must not give this ID extra weight in the mean.
            # If a duplicate carries no speed, retain its other valid sample.
            current_speeds[track_id] = measured
            if measured >= self._threshold:
                self._speeding_ids.add(track_id)
        self._visible_ids = visible_ids
        self._current_average = (
            fsum(value / len(current_speeds) for value in current_speeds.values())
            if current_speeds else None
        )
        return filtered

    def snapshot(self):
        return dict(detected_vehicle_count=len(self._detected_ids),
                    speeding_vehicle_count=len(self._speeding_ids),
                    visible_vehicles=len(self._visible_ids),
                    current_average_mph=self._current_average)
