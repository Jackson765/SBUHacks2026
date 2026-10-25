"""Editing controls for --calibrate playback of a prerecorded video file."""
import cv2
import numpy as np

from calibration import CornerEditor, RoadCalibration, draw_road_corners
from config import DISPLAY_HEIGHT, SIGN_WIDTH, WINDOW_NAME
from ratio_predict import validate_ratio


class VideoCalibrationControls:
    SLIDERS = ("Scale %", "Width %", "Length %", "Ratio %")

    def __init__(self, calibration, ratio, frame_size):
        self.frame_size = tuple(frame_size)
        width, height = self.frame_size
        self.display_size = (max(1, round(width * DISPLAY_HEIGHT / height)), DISPLAY_HEIGHT)
        self.initial = RoadCalibration(calibration.points, calibration.road_width_m,
                                       calibration.road_length_m)
        self.calibration = self.initial
        self.initial_ratio = self.ratio = validate_ratio(ratio)
        points = [[x * self.display_size[0] / width, y * DISPLAY_HEIGHT / height]
                  for x, y in calibration.points]
        self.editor = CornerEditor(points, *self.display_size)
        self.percent = {name: 100 for name in self.SLIDERS}
        self._last_input = self._signature()
        self.attached = False
        self.valid = True
        self.needs_redraw = True
        self.message = "Edits restart the 15-minute collection."

    def _signature(self):
        return (tuple(tuple(point) for point in self.editor.points), tuple(self.percent.values()))

    def set_percent(self, name, value):
        self.percent[name] = max(1, min(400, int(value)))

    def attach(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        for name in self.SLIDERS:
            cv2.createTrackbar(name, WINDOW_NAME, 100, 400,
                               lambda value, name=name: self.set_percent(name, value))
            cv2.setTrackbarMin(name, WINDOW_NAME, 1)
        self.attached = True

    def on_mouse(self, event, x, y, flags, userdata):
        # The side panel, footer, and sliders are outside the editable video.
        if self.editor.active is None and not (0 <= x < self.display_size[0]
                                                and 0 <= y < self.display_size[1]):
            return
        self.editor.on_mouse(event, x, y, flags, userdata)

    def poll(self):
        """Commit only valid changes; return True when the active setup changes."""
        signature = self._signature()
        self.needs_redraw = signature != self._last_input
        if not self.needs_redraw:
            return False
        self._last_input = signature
        width, height = self.frame_size
        points = [[min(width - 1., x * width / self.display_size[0]),
                   min(height - 1., y * height / self.display_size[1])]
                  for x, y in self.editor.points]
        scale = self.percent["Scale %"] / 100
        try:
            candidate = RoadCalibration(
                points, self.initial.road_width_m * scale * self.percent["Width %"] / 100,
                self.initial.road_length_m * scale * self.percent["Length %"] / 100)
            candidate.perspective_matrix()
            ratio = validate_ratio(self.initial_ratio * self.percent["Ratio %"] / 100)
        except ValueError:
            self.valid = False
            self.message = "Invalid shape: last valid calibration still applies."
            return False
        self.valid = True
        changed = (not np.allclose(candidate.points, self.calibration.points, rtol=0, atol=1e-5)
                   or candidate.road_width_m != self.calibration.road_width_m
                   or candidate.road_length_m != self.calibration.road_length_m
                   or ratio != self.ratio)
        self.calibration, self.ratio = candidate, ratio
        self.message = "Edits restart the 15-minute collection."
        return changed

    def reset(self):
        self.editor.reset()
        for name in self.SLIDERS:
            self.percent[name] = 100
            if self.attached:
                cv2.setTrackbarPos(name, WINDOW_NAME, 100)

    def draw_points(self, display_frame):
        draw_road_corners(display_frame, self.editor.points)

    def footer(self, average_speed, paused):
        width = self.display_size[0] + SIGN_WIDTH
        panel = np.full((100, width, 3), 25, np.uint8)
        scaled = "--" if average_speed is None else f"{average_speed * self.ratio:.2f}"
        lines = [
            f"Width {self.calibration.road_width_m:.4g} m | Length {self.calibration.road_length_m:.4g} m"
            f" | Ratio {self.ratio:.4g} | Scaled avg {scaled}",
            f"{'PAUSED' if paused else 'PLAYING'} | Drag corners | SPACE pause/resume | R reset | U undo drag",
            "Sliders: % of starting values | S save settings | Q/Esc quit",
            self.message,
        ]
        for index, text in enumerate(lines):
            cv2.putText(panel, text, (10, 20 + index * 23), cv2.FONT_HERSHEY_SIMPLEX,
                        .46, (210, 210, 210) if self.valid else (80, 160, 255), 1, cv2.LINE_AA)
        return panel
