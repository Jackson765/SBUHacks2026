"""Less inference work for offline simulation; retain source time and road coordinates."""
from dataclasses import dataclass
import math

import numpy as np

from config import MAX_TRACK_GAP_SECONDS


@dataclass(frozen=True)
class SimulationPlan:
    frame_step: int
    bounds: tuple[int, int, int, int]
    image_size: int

    def crop(self, frame):
        left, top, right, bottom = self.bounds
        return np.ascontiguousarray(frame[top:bottom, left:right])

    def restore_boxes(self, boxes):
        """Return full processing-frame boxes without changing detector results."""
        restored = np.asarray(boxes, dtype=float).copy()
        restored[:, [0, 2]] += self.bounds[0]
        restored[:, [1, 3]] += self.bounds[1]
        return restored


def make_simulation_plan(frame_size, points, fps, *, target_fps, padding,
                         maximum_image_size, fallback_step):
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("SIMULATION_TARGET_FPS must be positive and finite.")
    if not isinstance(padding, int) or padding < 0:
        raise ValueError("SIMULATION_ROAD_PADDING must be a nonnegative integer.")
    # Missing FPS metadata must not invent timestamps; retain the configured stride.
    step = max(1, round(fps / target_fps)) if math.isfinite(fps) and fps > 0 else fallback_step
    if math.isfinite(fps) and fps > 0:
        # Keep nominal samples comfortably inside the speed tracker's gap limit.
        step = min(step, max(1, math.floor(fps * MAX_TRACK_GAP_SECONDS * 0.8)))
    width, height = frame_size
    corners = np.asarray(points, dtype=float)
    left, top = np.maximum(0, np.floor(corners.min(axis=0) - padding)).astype(int)
    right, bottom = np.minimum([width, height], np.ceil(corners.max(axis=0) + padding + 1)).astype(int)
    image_size = min(maximum_image_size, math.ceil(max(right-left, bottom-top) / 32) * 32)
    return SimulationPlan(step, (int(left), int(top), int(right), int(bottom)), image_size)
