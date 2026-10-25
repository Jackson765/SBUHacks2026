"""Unique in-area totals and actual-mph speeding thresholds."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "YOLO"))

from backend.counters import AreaVehicleCounter
from calibration import RoadCalibration


def detection(track_id, speed, x=10, y=10):
    return ([x - 2, y - 4, x + 2, y], track_id, speed)


class AreaVehicleCounterTests(unittest.TestCase):
    def setUp(self):
        self.calibration = RoadCalibration([[0, 0], [20, 0], [20, 20], [0, 20]], 10, 20)
        self.counter = AreaVehicleCounter(self.calibration, 30)

    def test_threshold_is_inclusive_and_uses_raw_mph(self):
        self.counter.update([detection(1, 39.999), detection(2, 40), detection(3, 45)])
        result = self.counter.snapshot()
        self.assertEqual(result["detected_vehicle_count"], 3)
        self.assertEqual(result["speeding_vehicle_count"], 2)
        self.assertAlmostEqual(result["current_average_mph"], (39.999 + 40 + 45) / 3)
        other_limit = AreaVehicleCounter(self.calibration, 45)
        other_limit.update([detection(1, 54.999), detection(2, 55)])
        self.assertEqual(other_limit.snapshot()["speeding_vehicle_count"], 1)

    def test_outside_fast_vehicle_is_excluded_from_every_metric(self):
        inside = detection(1, 20)
        outside = detection(2, 100, x=30)
        # Most of this box overlaps the road, but its road contact is outside.
        overlapping = ([8, 5, 12, 24], 3, 90)
        self.assertEqual(self.counter.update([inside, outside, overlapping]), [inside])
        self.assertEqual(self.counter.snapshot(), dict(detected_vehicle_count=1,
                         speeding_vehicle_count=0, visible_vehicles=1, current_average_mph=20))

    def test_warmup_counts_detection_and_later_speed_only_once(self):
        self.counter.update([detection(1, None)])
        self.assertEqual(self.counter.snapshot()["detected_vehicle_count"], 1)
        self.assertEqual(self.counter.snapshot()["speeding_vehicle_count"], 0)
        self.assertIsNone(self.counter.snapshot()["current_average_mph"])
        for speed in (41, 45, 30, 60):
            self.counter.update([detection(1, speed)])
        self.assertEqual(self.counter.snapshot()["detected_vehicle_count"], 1)
        self.assertEqual(self.counter.snapshot()["speeding_vehicle_count"], 1)

    def test_duplicate_current_ids_do_not_duplicate_totals_or_average_weight(self):
        self.counter.update([detection(1, 50), detection(1, 50), detection(1, None), detection(2, 10)])
        self.assertEqual(self.counter.snapshot(), dict(detected_vehicle_count=2,
                         speeding_vehicle_count=1, visible_vehicles=2, current_average_mph=30))

    def test_cached_frames_retain_the_snapshot(self):
        self.counter.update([detection(1, 45)])
        before = self.counter.snapshot()
        self.assertIsNone(self.counter.update(None))
        self.assertEqual(self.counter.snapshot(), before)

    def test_empty_frame_and_reentry_preserve_run_totals(self):
        self.counter.update([detection(1, 45)])
        self.assertEqual(self.counter.update([detection(1, 100, x=30)]), [])
        outside = self.counter.snapshot()
        self.assertEqual(outside["visible_vehicles"], 0)
        self.assertIsNone(outside["current_average_mph"])
        self.counter.update([])
        self.counter.update([detection(1, None)])
        # Repeated updates across time-window boundaries use these same totals.
        self.counter.update([detection(1, 45), detection(2, 41)])
        self.assertEqual(self.counter.snapshot()["detected_vehicle_count"], 2)
        self.assertEqual(self.counter.snapshot()["speeding_vehicle_count"], 2)

    def test_invalid_speeds_are_not_speeders_or_average_samples_but_zero_is_valid(self):
        self.counter.update([detection(1, float("nan")), detection(2, float("inf")),
                             detection(3, -1), detection(4, None), detection(5, 0)])
        self.assertEqual(self.counter.snapshot(), dict(detected_vehicle_count=5,
                         speeding_vehicle_count=0, visible_vehicles=5, current_average_mph=0))

    def test_polygon_boundary_counts_inside(self):
        self.counter.update([detection(1, 40, x=0, y=10), detection(2, 40, x=20, y=20)])
        self.assertEqual(self.counter.snapshot()["detected_vehicle_count"], 2)
        self.assertEqual(self.counter.snapshot()["speeding_vehicle_count"], 2)

    def test_nonfinite_box_does_not_enter_any_count(self):
        self.counter.update([([float("nan"), 0, 10, 10], 1, 90)])
        self.assertEqual(self.counter.snapshot()["detected_vehicle_count"], 0)

    def test_new_job_instance_starts_at_zero(self):
        self.counter.update([detection(1, 60)])
        other = AreaVehicleCounter(self.calibration, 30)
        self.assertEqual(other.snapshot(), dict(detected_vehicle_count=0,
                         speeding_vehicle_count=0, visible_vehicles=0, current_average_mph=None))


if __name__ == "__main__":
    unittest.main()
