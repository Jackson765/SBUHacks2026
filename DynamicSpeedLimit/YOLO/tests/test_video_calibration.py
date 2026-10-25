"""Standalone geometry controls and retained speed-history reprojection."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from calibration import RoadCalibration
from video_calibration import VideoCalibrationControls
from tracker import SpeedTracker


POINTS = [[50, 50], [450, 50], [450, 450], [50, 450]]


def moving_track():
    tracker = SpeedTracker(POINTS, 40, 80)
    for time in np.arange(7) / 10:
        x = 100 + 20 * time
        tracker.update_vehicle_speed(1, [x - 5, 100, x + 5, 200], float(time))
    return tracker


class VideoCalibrationTests(unittest.TestCase):
    def controls(self):
        return VideoCalibrationControls(RoadCalibration(POINTS, 40, 80), 2, (500, 500))

    def test_scale_recomputes_paused_speed_without_adding_samples(self):
        tracker, controls = moving_track(), self.controls()
        before = tracker.current_speed(1)
        count = len(tracker.track_history[1])
        controls.set_percent("Scale %", 200)
        self.assertTrue(controls.poll())
        calibration = controls.calibration
        tracker.recalibrate(calibration.points, calibration.road_width_m, calibration.road_length_m)
        self.assertAlmostEqual(tracker.current_speed(1), before * 2, delta=.001)
        self.assertEqual(len(tracker.track_history[1]), count)
        self.assertEqual(len(tracker.pixel_history[1]), count)
        self.assertIsNone(tracker.current_speed(1, 2))  # Stale positions are not revived.

    def test_moving_road_corners_changes_projection_of_retained_tracks(self):
        tracker = moving_track()
        before = tracker.current_speed(1)
        tracker.recalibrate([[50, 50], [250, 50], [250, 450], [50, 450]], 40, 80)
        self.assertAlmostEqual(tracker.current_speed(1), before * 2, delta=.001)
        tracker.recalibrate([[300, 50], [450, 50], [450, 450], [300, 450]], 40, 80)
        self.assertNotIn(1, tracker.track_history)
        self.assertNotIn(1, tracker.pixel_history)
        self.assertNotIn(1, tracker.last_seen)

    def test_width_and_length_scale_their_own_axes(self):
        controls = self.controls()
        controls.set_percent("Width %", 150)
        controls.set_percent("Length %", 75)
        self.assertTrue(controls.poll())
        self.assertEqual(controls.calibration.road_width_m, 60)
        self.assertEqual(controls.calibration.road_length_m, 60)
        self.assertEqual(controls.ratio, 2)

    def test_ratio_does_not_change_physical_speed_calibration(self):
        controls = self.controls()
        controls.set_percent("Ratio %", 250)
        self.assertTrue(controls.poll())
        self.assertEqual(controls.ratio, 5)
        self.assertEqual(controls.calibration.road_width_m, 40)
        self.assertEqual(controls.calibration.road_length_m, 80)

    def test_drag_uses_display_coordinates_for_portrait_video_and_ignores_panel(self):
        controls = VideoCalibrationControls(RoadCalibration(
            [[50, 100], [490, 100], [490, 850], [50, 850]], 4, 50), 1, (540, 960))
        self.assertEqual(controls.display_size, (304, 540))
        controls.on_mouse(cv2.EVENT_LBUTTONDOWN, 500, 200, 0, None)
        self.assertIsNone(controls.editor.active)
        start = controls.editor.points[0]
        controls.on_mouse(cv2.EVENT_LBUTTONDOWN, *start, 0, None)
        controls.on_mouse(cv2.EVENT_MOUSEMOVE, 40, 70, cv2.EVENT_FLAG_LBUTTON, None)
        controls.on_mouse(cv2.EVENT_LBUTTONUP, 40, 70, 0, None)
        self.assertTrue(controls.poll())
        np.testing.assert_allclose(controls.calibration.points[0], [40*540/304, 70*960/540], atol=1e-4)
        controls.editor.undo()
        self.assertTrue(controls.poll())
        np.testing.assert_allclose(controls.calibration.points[0], [50, 100], atol=1e-4)

    def test_invalid_drag_keeps_valid_transform_and_reset_restores_all_settings(self):
        controls = self.controls()
        original = controls.calibration.perspective_matrix()
        controls.editor.points[0] = controls.editor.points[2][:]
        self.assertFalse(controls.poll())
        self.assertFalse(controls.valid)
        np.testing.assert_array_equal(controls.calibration.perspective_matrix(), original)
        controls.reset()
        controls.set_percent("Width %", 200)
        self.assertTrue(controls.poll())
        self.assertTrue(controls.valid)
        controls.reset()
        self.assertTrue(controls.poll())
        self.assertEqual(controls.calibration.road_width_m, 40)
        self.assertEqual(controls.ratio, 2)
        self.assertFalse(controls.poll())

    def test_raw_history_is_pruned_and_forgotten_with_metric_history(self):
        tracker = moving_track()
        for time in np.arange(7, 21) / 10:
            x = 100 + 20 * time
            tracker.update_vehicle_speed(1, [x-5, 100, x+5, 200], float(time))
        self.assertEqual(len(tracker.pixel_history[1]), len(tracker.track_history[1]))
        self.assertGreaterEqual(tracker.pixel_history[1][0][0], 1)
        tracker.remove_inactive_tracks(10)
        self.assertFalse(tracker.pixel_history)



if __name__ == "__main__":
    unittest.main()
