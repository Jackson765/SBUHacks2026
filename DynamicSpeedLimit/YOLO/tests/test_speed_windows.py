"""Verify aggregation independently of camera calibration and detector runtime."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speed_windows import FiveMinuteAverages
from main import forecast_collected_averages, report_windows, VideoClock
from ratio_predict import predict_scaled_averages


class SpeedWindowTests(unittest.TestCase):
    def test_equal_car_weight_even_when_one_car_has_more_detections(self):
        collector = FiveMinuteAverages()
        for t in range(10):
            collector.add_measurements(t, [(1, 10), (3, None)])
        collector.add_measurements(20, [(2, 30)])
        windows = collector.advance(300)
        self.assertEqual(windows[0].average_speed, 20)
        self.assertEqual(windows[0].vehicle_count, 2)

    def test_three_bins_boundary_assignment_and_cutoff(self):
        collector = FiveMinuteAverages()
        collector.add_measurements(0, [(1, 10)])
        first = collector.add_measurements(300, [(1, 20)])
        second = collector.add_measurements(600, [(1, 30)])
        third = collector.add_measurements(900, [(1, 999)])
        self.assertEqual([first[0].average_speed, second[0].average_speed, third[0].average_speed], [10, 20, 30])
        self.assertTrue(collector.complete)
        self.assertEqual(collector.advance(1200), [])
        self.assertEqual(collector.averages, [10, 20, 30])

    def test_empty_bins_are_missing_and_stopped_cars_are_zero(self):
        collector = FiveMinuteAverages()
        collector.add_measurements(0, [(1, 0), (2, None)])
        collector.advance(900)
        self.assertEqual(collector.averages, [0, None, None])
        with patch("main.predict_scaled_averages") as predictor, contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(forecast_collected_averages(collector.averages, 2))
        predictor.assert_not_called()

    def test_short_video_does_not_finalize_partial_bin(self):
        collector = FiveMinuteAverages()
        collector.add_measurements(0, [(1, 20)])
        collector.advance(899.99)
        self.assertFalse(collector.complete)
        self.assertEqual(len(collector.windows), 2)

    def test_irregular_timing_and_missed_frames(self):
        collector = FiveMinuteAverages()
        collector.add_measurements(1, [(1, 10)])
        collector.add_measurements(280, [(2, 30)])
        collector.add_measurements(305, [(3, 40)])
        collector.add_measurements(680, [(4, 0)])
        collector.advance(900)
        self.assertEqual(collector.averages, [20, 40, 0])

    def test_invalid_times_and_speeds(self):
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                FiveMinuteAverages().advance(value)
            with self.assertRaises(ValueError):
                FiveMinuteAverages().add_measurements(0, [(1, value)])
        collector = FiveMinuteAverages()
        collector.advance(10)
        with self.assertRaises(ValueError):
            collector.advance(9)

    def test_prediction_gets_scaled_bins_and_unscales_once(self):
        model = Mock(return_value={"predicted_speed": 55})
        with patch("main.predict_scaled_averages",
                   side_effect=lambda speeds, ratio: predict_scaled_averages(speeds, ratio, predictor=model)):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                result = forecast_collected_averages([25, 28, 30], 2)
        np.testing.assert_array_equal(model.call_args.args[0], [50, 56, 60])
        self.assertEqual(result["predicted_speed"], 27.5)
        self.assertIn("Three scaled averages", output.getvalue())
        self.assertIn("27.500 mph", output.getvalue())

    def test_exact_fifteen_minute_cfr_eof_includes_last_frame_duration(self):
        clock = VideoClock(1, source="video")
        for frame in range(900):
            clock.timestamp(frame, frame * 1000)
        self.assertEqual(clock.previous, 899)
        self.assertEqual(clock.end_timestamp(), 900)

    def test_final_ten_to_fifteen_minute_average_is_reported_when_eof_closes_it(self):
        collector = FiveMinuteAverages()
        clock = VideoClock(1, source="video")
        for frame in range(900):
            timestamp = clock.timestamp(frame, frame * 1000)
            collector.add_measurements(timestamp, [(1, 10 * (1 + frame // 300))])
        self.assertEqual(len(collector.windows), 2)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            report_windows(collector.advance(clock.end_timestamp()), 2)
        self.assertIn('10-15 minutes: average 30.000 mph; scaled 60.000', output.getvalue())
        self.assertEqual(collector.averages, [10, 20, 30])
        self.assertTrue(collector.complete)

    def test_final_average_is_visible_immediately_in_buffered_terminal_output(self):
        collector = FiveMinuteAverages()
        collector.add_measurements(600, [(1, 30)])
        completed = collector.advance(900)
        raw_output = io.BytesIO()
        with io.TextIOWrapper(raw_output, encoding='utf-8') as buffered_output:
            with contextlib.redirect_stdout(buffered_output):
                report_windows(completed, 2)
            # Check the underlying bytes before closing/flushing the wrapper.
            self.assertIn('10-15 minutes: average 30.000 mph; scaled 60.000',
                          raw_output.getvalue().decode('utf-8'))

    def test_variable_frame_rate_does_not_invent_eof_duration(self):
        clock = VideoClock(30, source="video")
        for frame, milliseconds in enumerate([0, 40, 100, 899900]):
            clock.timestamp(frame, milliseconds)
        self.assertEqual(clock.end_timestamp(), 899.9)


if __name__ == "__main__":
    unittest.main()
