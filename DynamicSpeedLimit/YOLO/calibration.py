"""Interactive road-plane calibration, with profiles tied to a specific video."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import SIGN_WIDTH, VIDEO_H, VIDEO_W, WINDOW_NAME
from ui import resize_to_height


@dataclass
class RoadCalibration:
    points: list
    road_width_m: float
    road_length_m: float

    def __post_init__(self):
        points = np.asarray(self.points, dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("Select four finite road corner coordinates.")
        if not cv2.isContourConvex(points) or cv2.contourArea(points) < 100:
            raise ValueError("Road corners must form a convex, nondegenerate quadrilateral.")
        self.road_width_m = float(self.road_width_m)
        self.road_length_m = float(self.road_length_m)
        if not (np.isfinite([self.road_width_m, self.road_length_m]).all()
                and self.road_width_m > 0 and self.road_length_m > 0):
            raise ValueError("Measured road width and length must be positive and finite.")
        self.points = points.tolist()

    def perspective_matrix(self):
        destination = np.float32([
            [0, 0], [self.road_width_m, 0],
            [self.road_width_m, self.road_length_m], [0, self.road_length_m],
        ])
        matrix = cv2.getPerspectiveTransform(np.float32(self.points), destination)
        if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
            raise ValueError("Could not compute a valid road-plane transform.")
        return matrix


def video_identity(video_path):
    path = Path(video_path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def prompt_distance(label):
    while True:
        raw = input(f"Measured {label} in meters (Q to quit): ").strip()
        if raw.lower() == "q":
            raise KeyboardInterrupt
        try:
            value = float(raw)
            if np.isfinite(value) and value > 0:
                return value
        except ValueError:
            pass
        print("Enter a positive measured distance, e.g. 12.5.")


def window_is_open():
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


CORNER_NAMES = ("Far-left", "Far-right", "Near-right", "Near-left")
# OpenCV BGR colors: red, green, blue, yellow.
CORNER_COLORS = ((40, 40, 255), (50, 220, 50), (255, 140, 40), (0, 230, 255))


def default_corners(width=VIDEO_W, height=VIDEO_H):
    return [[width * .35, height * .28], [width * .65, height * .28],
            [width * .85, height * .9], [width * .15, height * .9]]


class CornerEditor:
    """Four fixed-identity handles; clicks away from a handle never add points."""
    def __init__(self, points, width=VIDEO_W, height=VIDEO_H):
        self.points = [list(map(float, point)) for point in points]
        self.width = width
        self.height = height
        self.initial = [point[:] for point in self.points]
        self.active = None
        self._before_drag = None
        self._undo = []

    def _move(self, x, y):
        self.points[self.active] = [float(np.clip(x, 0, self.width - 1)),
                                    float(np.clip(y, 0, self.height - 1))]

    def on_mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            distances = [np.hypot(x - px, y - py) for px, py in self.points]
            index = int(np.argmin(distances))
            if distances[index] <= 18:
                self.active = index
                self._before_drag = [point[:] for point in self.points]
        elif event == cv2.EVENT_MOUSEMOVE and self.active is not None:
            self._move(x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.active is not None:
            self._move(x, y)
            if self.points != self._before_drag:
                self._undo.append(self._before_drag)
            self.active = None

    def reset(self):
        self._undo.append([point[:] for point in self.points])
        self.points = [point[:] for point in self.initial]
        self.active = None

    def undo(self):
        if self._undo:
            self.points = self._undo.pop()
        self.active = None


def draw_road_corners(frame, points):
    cv2.polylines(frame, [np.int32(points)], True, (220, 220, 220), 2)
    for index, (point, color) in enumerate(zip(points, CORNER_COLORS)):
        pixel = tuple(map(int, point))
        cv2.circle(frame, pixel, 13, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(frame, pixel, 10, color, -1, cv2.LINE_AA)
        cv2.putText(frame, str(index + 1), (pixel[0] - 6, pixel[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 2, cv2.LINE_AA)


def calibration_panel(width_m, length_m, message="", height=VIDEO_H):
    panel = np.full((height, SIGN_WIDTH, 3), 25, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, "ROAD CALIBRATION", (22, 42), font, .8, (255, 255, 255), 2)
    for index, (name, color) in enumerate(zip(CORNER_NAMES, CORNER_COLORS)):
        y = 95 + index * 43
        cv2.circle(panel, (32, y - 6), 9, color, -1)
        cv2.putText(panel, f"{index + 1}. {name}", (53, y), font, .7, color, 2)
    lines = [f"Width: {width_m:g} m (1-2 / 4-3)",
             f"Length: {length_m:g} m (1-4 / 2-3)",
             "Drag each colored corner",
             "ENTER: confirm rectangle",
             "R: reset   Backspace: undo",
             "Q / Escape: quit"]
    for index, line in enumerate(lines):
        cv2.putText(panel, line, (22, 295 + index * 30), font, .55, (230, 230, 230), 1, cv2.LINE_AA)
    if message:
        cv2.putText(panel, message, (14, height - 22), font, .48, (80, 80, 255), 1)
    return panel


def run_calibration(first_frame):
    height, width = first_frame.shape[:2]
    display_frame = resize_to_height(first_frame, VIDEO_H)
    display_height, display_width = display_frame.shape[:2]
    print("\nUse a stationary camera and a flat road rectangle with MEASURED dimensions.")
    print("Width: across the road. Length: along the road. Do not guess dimensions.")
    width_m = prompt_distance("road width")
    length_m = prompt_distance("road length")
    points = default_corners(display_width, display_height)
    print("Drag: 1 RED far-left, 2 GREEN far-right, 3 BLUE near-right, 4 YELLOW near-left.")
    print("ENTER accepts; R resets; Backspace undoes a drag; Q / Escape quits.")

    editor = CornerEditor(points, display_width, display_height)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, editor.on_mouse)
    message = ""
    try:
        while True:
            frame = display_frame.copy()
            draw_road_corners(frame, editor.points)
            cv2.imshow(WINDOW_NAME, np.hstack((frame, calibration_panel(
                width_m, length_m, message, height=display_height))))
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27) or not window_is_open():
                return None
            if key == ord("r"):
                editor.reset()
                message = ""
            elif key in (8, 127):
                editor.undo()
                message = ""
            elif key in (10, 13) and editor.active is None:
                try:
                    source_points = [[x * width / display_width, y * height / display_height]
                                     for x, y in editor.points]
                    calibration = RoadCalibration(source_points, width_m, length_m)
                    calibration.perspective_matrix()
                except ValueError:
                    message = "Invalid shape: adjust corner order."
                    continue
                return calibration
    finally:
        if window_is_open():
            cv2.setMouseCallback(WINDOW_NAME, lambda *args: None)
