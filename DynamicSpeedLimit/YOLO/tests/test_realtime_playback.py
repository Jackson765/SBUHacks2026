"""Deterministic pacing tests, including slow inference and cancellation."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realtime_playback import RealtimePlaybackClock


class RealtimePlaybackTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.waits = []
        self.playback = RealtimePlaybackClock(clock=lambda: self.now, sleeper=self.sleep)

    def sleep(self, delay):
        self.waits.append(delay)
        self.now += delay

    def test_source_intervals_take_at_least_their_wall_time(self):
        for elapsed in [0, .025, .2, .7, 1]:
            self.assertTrue(self.playback.wait(elapsed))
            self.assertAlmostEqual(self.now, 100 + elapsed)
        self.assertAlmostEqual(sum(self.waits), 1)

    def test_thirty_fps_releases_thirty_intervals_in_one_wall_second(self):
        for frame_index in range(31):
            elapsed = frame_index / 30
            self.assertTrue(self.playback.wait(elapsed))
            self.assertAlmostEqual(self.now - 100, elapsed)
        self.assertAlmostEqual(sum(self.waits), 1)

    def test_thirty_fps_cannot_speed_up_after_a_one_second_detector_stall(self):
        self.playback.wait(0)
        self.now += 1
        self.playback.wait(1 / 30)
        released_after_stall = self.now
        for frame_index in range(2, 32):
            self.playback.wait(frame_index / 30)
        self.assertAlmostEqual(self.now - released_after_stall, 1)

    def test_processing_time_is_included_in_each_interval(self):
        self.playback.wait(0)
        self.now += .2
        self.playback.wait(.5)
        self.assertAlmostEqual(self.now, 100.5)
        self.assertAlmostEqual(sum(self.waits), .3)

    def test_late_inference_does_not_cause_catch_up_bursts(self):
        self.playback.wait(0)
        self.now += 3
        self.playback.wait(1)
        self.assertEqual(self.waits, [])
        self.playback.wait(2)
        self.assertAlmostEqual(self.now, 104)
        self.assertAlmostEqual(sum(self.waits), 1)

    def test_wait_can_be_cancelled_during_a_large_timestamp_gap(self):
        stopped = False

        def cancel_after_sleep(delay):
            nonlocal stopped
            self.sleep(delay)
            stopped = True

        playback = RealtimePlaybackClock(clock=lambda: self.now, sleeper=cancel_after_sleep)
        self.assertFalse(playback.wait(300, should_stop=lambda: stopped))
        self.assertLessEqual(sum(self.waits), .05)

    def test_invalid_source_time_is_rejected(self):
        self.playback.wait(.1)
        for elapsed in [-1, .09, float('nan'), float('inf')]:
            with self.subTest(elapsed=elapsed), self.assertRaises(ValueError):
                self.playback.wait(elapsed)
