"""Prerecorded decoder preroll preserves source pixels and presentation times."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from video_input import VideoClock
from speed_windows import FiveMinuteAverages


class VideoInputTests(unittest.TestCase):
    def test_exact_fifteen_minutes_with_rounded_fps_closes_all_windows(self):
        clock = VideoClock(29.999542600343624)
        collector = FiveMinuteAverages()
        for index in range(27000):
            clock.timestamp(index, index * 1000 / 30)
            if index in (0,9000,18000):
                collector.add_measurements(index/30, [(1,20+5*(index//9000))])
        self.assertTrue(clock.constant_frame_timing)
        self.assertAlmostEqual(clock.end_timestamp(), 900)
        collector.advance(clock.end_timestamp())
        self.assertTrue(collector.complete)
        self.assertEqual(collector.averages,[20,25,30])

    def test_decoder_tick_rounding_keeps_valid_last_frame_duration(self):
        clock = VideoClock(30)
        for index in range(90):
            clock.timestamp(index, round(index * 1000 / 30, 2))
        self.assertTrue(clock.constant_frame_timing)
        self.assertAlmostEqual(clock.end_timestamp(), 3, places=4)

    def test_quantized_standard_cadences_do_not_lose_the_fifteenth_minute(self):
        for fps in (24,25,30,60):
            with self.subTest(fps=fps):
                clock=VideoClock(fps)
                for index in range(fps*900):
                    clock.timestamp(index,round(index*1000/fps))
                self.assertTrue(clock.constant_frame_timing)
                collector=FiveMinuteAverages()
                collector.advance(clock.end_timestamp())
                self.assertTrue(collector.complete)

    def test_genuinely_variable_or_unknown_timing_does_not_invent_eof_duration(self):
        for fps in (30, 0):
            clock = VideoClock(fps)
            for index, timestamp in enumerate([0, 33.333, 100, 133.333]):
                clock.timestamp(index, timestamp)
            self.assertFalse(clock.constant_frame_timing)
            self.assertAlmostEqual(clock.end_timestamp(), .133333)

    def test_cfr_frame_indices_can_be_sparse_without_changing_duration(self):
        clock = VideoClock(30)
        for index in (0, 5, 10, 89):
            clock.timestamp(index, index * 1000 / 30)
        self.assertAlmostEqual(clock.end_timestamp(), 3)

    def capture(self, timestamps):
        frames = [np.full((3, 4, 3), index + 1, np.uint8) for index in range(len(timestamps))]
        state = {'index': -1}
        capture = Mock()

        def read():
            state['index'] += 1
            if state['index'] == len(frames):
                return False, None
            return True, frames[state['index']]

        def get(prop):
            self.assertEqual(prop, cv2.CAP_PROP_POS_MSEC)
            return timestamps[state['index']]

        capture.read.side_effect = read
        capture.get.side_effect = get
        return capture, frames

    def test_playback_skips_negative_prefix_without_rewriting_frame_or_timestamp(self):
        capture, frames = self.capture([-35.7667, -2.4333333, 30.9001234, 64.2334567])
        with contextlib.redirect_stdout(io.StringIO()):
            frame, timestamp = main.read_initial_frame(capture, skip_preroll=True)
        self.assertIs(frame, frames[2])
        self.assertEqual(timestamp, 30.9001234)
        self.assertEqual(capture.read.call_count, 3)
        self.assertEqual(capture.get.call_count, 3)
        capture.grab.assert_not_called()

    def test_default_skips_negative_prefix_and_starts_clock_at_actual_pts(self):
        capture, frames = self.capture([-35.7667, 30.9001234])
        with contextlib.redirect_stdout(io.StringIO()):
            frame, timestamp = main.read_initial_frame(capture)
        self.assertIs(frame, frames[1])
        self.assertEqual(timestamp, 30.9001234)
        self.assertEqual(capture.read.call_count, 2)
        clock = main.VideoClock(30)
        self.assertEqual(clock.timestamp(0, timestamp), .0309001234)
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            clock.timestamp(1, -1)

    def test_zero_is_a_valid_first_playback_timestamp(self):
        capture, frames = self.capture([-20, 0, 33.3333])
        with contextlib.redirect_stdout(io.StringIO()):
            frame, timestamp = main.read_initial_frame(capture, skip_preroll=True)
        self.assertIs(frame, frames[1])
        self.assertEqual(timestamp, 0)
        self.assertEqual(capture.read.call_count, 2)

    def test_nonfinite_timestamp_is_left_for_clock_validation(self):
        for value in (float('nan'), float('-inf'), float('inf')):
            with self.subTest(value=value):
                capture, frames = self.capture([value, 0])
                frame, timestamp = main.read_initial_frame(capture, skip_preroll=True)
                self.assertIs(frame, frames[0])
                capture.read.assert_called_once_with()
                with self.assertRaisesRegex(ValueError, 'timestamps'):
                    main.VideoClock(30).timestamp(0, timestamp)

    def test_empty_video_and_negative_only_preroll_fail_without_inventing_frame(self):
        for times in ([], [-35.7667, -2.4333333]):
            with self.subTest(times=times):
                capture, _ = self.capture(times)
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'frame'):
                    main.read_initial_frame(capture, skip_preroll=True)
                self.assertEqual(capture.read.call_count, len(times) + 1)


if __name__ == '__main__':
    unittest.main()
