"""Estimate vehicle velocity from timestamped positions on a measured road plane."""
from collections import deque

import cv2
import numpy as np

from calibration import RoadCalibration
from config import (
    MAX_INACTIVE_SECONDS,
    MAX_REASONABLE_SPEED_MPH,
    MAX_TRACK_GAP_SECONDS,
    MIN_TRACK_SAMPLES,
    MIN_TRACK_SECONDS,
    POSITION_JUMP_TOLERANCE_METERS,
    TRACK_WINDOW_SECONDS,
)

MPS_TO_MPH = 2.2369362921


class SpeedTracker:
    def __init__(self, src_points, road_width_m: float, road_length_m: float):
        self.track_history = {}  # ID -> deque[(video seconds, x meters, y meters)]
        self.pixel_history = {}  # Accepted raw positions, for live recalibration.
        self.last_seen = {}      # Timestamp of last ACCEPTED position.
        self.recalibrate(src_points, road_width_m, road_length_m)

    def recalibrate(self, src_points, road_width_m, road_length_m):
        """Remap recent raw positions so edits update speeds even while paused."""
        calibration = RoadCalibration(src_points, road_width_m, road_length_m)
        matrix = calibration.perspective_matrix()  # Validate before changing state.
        self.src = np.asarray(calibration.points, dtype=np.float32)
        self.M = matrix
        for track_id, pixels in list(self.pixel_history.items()):
            retained, mapped = deque(), deque()
            for timestamp, x, y in pixels:
                position = self.transform_point(x, y)
                if not self.point_in_road(x, y) or not np.isfinite(position).all():
                    retained.clear()
                    mapped.clear()
                    continue
                retained.append((timestamp, x, y))
                mapped.append((timestamp, *position))
            if mapped:
                self.pixel_history[track_id] = retained
                self.track_history[track_id] = mapped
                self.last_seen[track_id] = mapped[-1][0]
            else:
                self.forget(track_id)

    def point_in_road(self, x: float, y: float) -> bool:
        if not np.isfinite([x, y]).all():
            return False
        return cv2.pointPolygonTest(self.src, (float(x), float(y)), False) >= 0

    def transform_point(self, x: float, y: float) -> tuple:
        point = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.M)[0, 0]
        return float(transformed[0]), float(transformed[1])

    def forget(self, track_id: int):
        self.track_history.pop(track_id, None)
        self.pixel_history.pop(track_id, None)
        self.last_seen.pop(track_id, None)

    def remove_inactive_tracks(self, timestamp_seconds: float):
        for track_id, last in list(self.last_seen.items()):
            if timestamp_seconds - last > MAX_INACTIVE_SECONDS:
                self.forget(track_id)

    def update_vehicle_speed(self, track_id: int, bbox, timestamp_seconds: float):
        """Return mph, including valid zero, or None until a track is measurable.

        A least-squares line through recent metric positions estimates x/y
        velocity. No position EMA or pixel threshold suppresses slow motion.
        The bounding-box bottom center approximates road contact.
        """
        now = float(timestamp_seconds)
        if not np.isfinite(now) or now < 0:
            raise ValueError("Video timestamp must be finite and nonnegative.")
        x1, _, x2, y2 = map(float, bbox)
        cx, cy = (x1 + x2) / 2.0, y2
        if not self.point_in_road(cx, cy):
            self.forget(track_id)
            return None
        x_m, y_m = self.transform_point(cx, cy)
        if not np.isfinite([x_m, y_m]).all():
            self.forget(track_id)
            return None

        history = self.track_history.get(track_id)
        if history:
            previous_time, previous_x, previous_y = history[-1]
            dt = now - previous_time
            # Long detection gaps / reused IDs must warm up again.
            if dt <= 0 or dt > MAX_TRACK_GAP_SECONDS:
                self.forget(track_id)
                history = None
            else:
                distance_m = np.hypot(x_m - previous_x, y_m - previous_y)
                max_distance = (MAX_REASONABLE_SPEED_MPH / MPS_TO_MPH * dt
                                + POSITION_JUMP_TOLERANCE_METERS)
                if distance_m > max_distance:
                    # Do not let a rejected jump contaminate later estimates.
                    return None

        if history is None:
            history = deque()
            self.track_history[track_id] = history
            self.pixel_history[track_id] = deque()
        history.append((now, x_m, y_m))
        self.pixel_history[track_id].append((now, cx, cy))
        self.last_seen[track_id] = now
        while len(history) > 1 and now - history[0][0] > TRACK_WINDOW_SECONDS + 1e-9:
            history.popleft()
            self.pixel_history[track_id].popleft()
        return self.current_speed(track_id)

    def current_speed(self, track_id, timestamp_seconds=None):
        """Read a speed without adding another sample for a paused frame."""
        history = self.track_history.get(track_id)
        if not history:
            return None
        now = history[-1][0]
        if timestamp_seconds is not None and timestamp_seconds - now > MAX_TRACK_GAP_SECONDS:
            return None
        if (len(history) < MIN_TRACK_SAMPLES
                or now - history[0][0] + 1e-9 < MIN_TRACK_SECONDS):
            return None

        samples = np.asarray(history, dtype=float)
        times = samples[:, 0] - now  # Avoid precision loss with large timestamps.
        times -= times.mean()
        positions = samples[:, 1:]
        positions = positions - positions.mean(axis=0)
        velocity = (times @ positions) / (times @ times)
        speed = float(np.linalg.norm(velocity) * MPS_TO_MPH)
        if not np.isfinite(speed) or speed > MAX_REASONABLE_SPEED_MPH:
            return None
        # Numerical roundoff should not prevent a stationary track reading zero.
        return 0.0 if speed < 1e-6 else speed
