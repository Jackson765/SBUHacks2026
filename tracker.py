# tracker.py
from collections import deque
import cv2
import numpy as np
from config import (
    ALPHA,
    DST_POINTS,
    MIN_PIXEL_SHIFT,
    PIXELS_PER_METER,
    SPEED_QUEUE_DEPTH,
)


class SpeedTracker:
    def __init__(self, src_points: list):
        # Build Perspective Transformation Matrix
        src_np = np.float32(src_points)
        dst_np = np.float32(DST_POINTS)
        self.M = cv2.getPerspectiveTransform(src_np, dst_np)

        self.track_history = {}
        self.speed_queues = {}

    def transform_point(self, x: float, y: float) -> tuple:
        """Transforms 2D screen coordinate to flattened bird's-eye view plane."""
        pt = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.M)
        return transformed[0][0][0], transformed[0][0][1]

    def update_vehicle_speed(
        self, track_id: int, bbox: tuple, frame_count: int, frame_time: float
    ) -> float:
        """Calculates smoothed velocity for a vehicle given its bounding box."""
        x1, _, x2, y2 = bbox

        # Bottom-center point (where tires touch road surface)
        cx = int((x1 + x2) / 2)
        cy = int(y2)

        raw_flat_cx, raw_flat_cy = self.transform_point(cx, cy)
        raw_speed_mph = 0.0

        if track_id in self.track_history:
            prev_flat_cx, prev_flat_cy, prev_frame_num = self.track_history[track_id]

            # Exponential Moving Average (EMA) position smoothing
            flat_cx = ALPHA * raw_flat_cx + (1 - ALPHA) * prev_flat_cx
            flat_cy = ALPHA * raw_flat_cy + (1 - ALPHA) * prev_flat_cy

            effective_dt = (frame_count - prev_frame_num) * frame_time
            pixel_distance = (
                (flat_cx - prev_flat_cx) ** 2 + (flat_cy - prev_flat_cy) ** 2
            ) ** 0.5

            # Noise gate: filter out sub-pixel jitter
            if pixel_distance >= MIN_PIXEL_SHIFT and effective_dt > 0:
                meters = pixel_distance / PIXELS_PER_METER
                meters_per_sec = meters / effective_dt
                raw_speed_mph = meters_per_sec * 2.23694
        else:
            flat_cx, flat_cy = raw_flat_cx, raw_flat_cy

        self.track_history[track_id] = (flat_cx, flat_cy, frame_count)

        if track_id not in self.speed_queues:
            self.speed_queues[track_id] = deque(maxlen=SPEED_QUEUE_DEPTH)

        if raw_speed_mph > 0:
            self.speed_queues[track_id].append(raw_speed_mph)

        # Return moving average speed
        if len(self.speed_queues[track_id]) > 0:
            return sum(self.speed_queues[track_id]) / len(
                self.speed_queues[track_id]
            )
        return 0.0