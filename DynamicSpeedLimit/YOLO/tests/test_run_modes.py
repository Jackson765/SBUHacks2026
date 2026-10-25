"""Visual playback, pause behavior, and explicit calibration mode."""
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from ui import create_dynamic_speed_sign
from settings_fixture import store_with_defaults
from calibration import RoadCalibration
from video_settings import VideoSettingsStore
from speed_windows import FiveMinuteAverages
from video_calibration import VideoCalibrationControls
from video_input import VideoClock
from realtime_playback import RealtimePlaybackClock


class RunModeTests(unittest.TestCase):
    def test_quick_entry_point_disables_pacing_without_mutating_args(self):
        for full_analysis in (None, False, True):
            with self.subTest(full_analysis=full_analysis):
                args = SimpleNamespace(visual=False, calibrate=False, real_time=True)
                if full_analysis is not None:
                    args.full_analysis = full_analysis
                original = vars(args).copy()
                with patch('main.process_video', return_value='quick result') as process:
                    result = main.process_video_quick(None, None, 0, None, None, None, None, args)
                self.assertEqual(result, 'quick result')
                actual = process.call_args.args[7]
                self.assertFalse(actual.real_time)
                self.assertEqual(actual.full_analysis, bool(full_analysis))
                self.assertEqual(vars(args), original)

    def test_realtime_entry_point_sets_full_frame_pacing_without_mutating_args(self):
        args = SimpleNamespace(visual=False, calibrate=False, full_analysis=False, real_time=False)
        with patch('main.process_video', return_value='result') as process:
            result = main.process_video_realtime(None, None, 0, None, None, None, None, args)
        self.assertEqual(result, 'result')
        actual = process.call_args.args[7]
        self.assertTrue(actual.real_time)
        self.assertTrue(actual.full_analysis)
        self.assertFalse(args.real_time)
        self.assertFalse(args.full_analysis)

    def test_cli_real_time_selects_dedicated_entry_point(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            video = Path(temp) / 'clip.mp4'
            video.touch()
            capture = Mock()
            capture.isOpened.return_value = True
            capture.read.return_value = True, np.zeros((500, 500, 3), np.uint8)
            capture.get.side_effect = lambda prop: 10 if prop == cv2.CAP_PROP_FPS else 0
            settings = Mock()
            settings.source, settings.ratio, settings.base_speed_limit = 'saved settings', 1, 30
            stack.enter_context(patch('main.cv2.VideoCapture', return_value=capture))
            stack.enter_context(patch('main.VideoSettingsStore', return_value=Mock(load=Mock(return_value=settings))))
            real_time = stack.enter_context(patch('main.process_video_realtime', return_value='paced result'))
            quick = stack.enter_context(patch('main.process_video'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            result = main.main(['--real-time', '--video', str(video)])
            self.assertEqual(result, 'paced result')
            real_time.assert_called_once()
            self.assertTrue(real_time.call_args.args[7].real_time)
            quick.assert_not_called()
            capture.release.assert_called_once()

    def test_realtime_windows_wait_for_five_ten_and_fifteen_wall_minutes(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        settings = SimpleNamespace(calibration=RoadCalibration(
            [[20, 20], [140, 20], [140, 100], [20, 100]], 12, 20), ratio=1, base_speed_limit=30)
        args = SimpleNamespace(visual=False, calibrate=False, full_analysis=False, weather=False)
        capture = Mock()
        capture.read.return_value = True, frame
        capture.get.side_effect = [300000, 600000, 900000]
        detector = Mock()
        detector.track.return_value = [SimpleNamespace(boxes=None)]
        now = [0.0]

        def sleep(delay):
            now[0] += delay

        pacing = RealtimePlaybackClock(clock=lambda: now[0], sleeper=sleep)
        observations = []

        def observe(elapsed, collector, frame, vehicles):
            observations.append((elapsed, now[0], len(collector.windows)))

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch('main.RealtimePlaybackClock', return_value=pacing))
            stack.enter_context(patch('main.FRAME_SKIP', 1))
            stack.enter_context(patch('main.YOLO_MODEL_PATH', Path(__file__)))
            stack.enter_context(patch('main.make_simulation_plan', side_effect=AssertionError('Fast plan used')))
            stack.enter_context(patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=Mock(return_value=detector))}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            main.process_video_realtime(capture, frame, 0, VideoClock(0), None, settings, None, args,
                                        observer=observe)
        self.assertEqual([item[0] for item in observations], [0, 300, 600, 900])
        self.assertEqual([item[2] for item in observations], [0, 1, 2, 3])
        for elapsed, wall, _ in observations:
            self.assertAlmostEqual(elapsed, wall)
        self.assertGreaterEqual(now[0], 900)

    def run_thirty_fps_timeline(self, real_time):
        """Run the real loop over three seconds of 30-FPS decoded timestamps."""
        frame = np.zeros((120, 160, 3), np.uint8)
        settings = SimpleNamespace(calibration=RoadCalibration(
            [[20, 20], [140, 20], [140, 100], [20, 100]], 12, 20), ratio=1, base_speed_limit=30)
        args = SimpleNamespace(visual=False, calibrate=False, full_analysis=False, weather=False)
        state = {'index': 0, 'wall': 0.0}
        waits, observations = [], []

        def advance():
            state['index'] += 1
            return state['index'] < 90

        capture = Mock()
        capture.read.side_effect = lambda: (advance(), frame)
        capture.grab.side_effect = advance
        capture.get.side_effect = lambda prop: state['index'] * 1000 / 30
        detector = Mock()
        detector.track.return_value = [SimpleNamespace(boxes=None)]

        def sleep(delay):
            waits.append(delay)
            state['wall'] += delay

        def observe(elapsed, collector, preview_frame, vehicles):
            observations.append((elapsed, state['wall'], preview_frame is not None))

        with contextlib.ExitStack() as stack:
            if real_time:
                pacing = RealtimePlaybackClock(clock=lambda: state['wall'], sleeper=sleep)
                stack.enter_context(patch('main.RealtimePlaybackClock', return_value=pacing))
                stack.enter_context(patch('main.make_simulation_plan', side_effect=AssertionError('Fast plan used')))
            else:
                stack.enter_context(patch('main.RealtimePlaybackClock', side_effect=AssertionError('Quick mode paced')))
            stack.enter_context(patch('main.FRAME_SKIP', 2))
            stack.enter_context(patch('main.YOLO_MODEL_PATH', Path(__file__)))
            stack.enter_context(patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=Mock(return_value=detector))}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            processor = main.process_video_realtime if real_time else main.process_video_quick
            processor(capture, frame, 0, VideoClock(30, source='video'), None, settings, None, args,
                      observer=observe)
        return state, waits, observations, detector, capture

    def test_regular_thirty_fps_timeline_publishes_every_frame_at_source_speed(self):
        state, waits, observations, detector, capture = self.run_thirty_fps_timeline(True)
        published = [item for item in observations if item[2]]
        self.assertEqual(len(published), 90)
        for index, (elapsed, wall, _) in enumerate(published):
            self.assertAlmostEqual(elapsed, index / 30)
            self.assertGreaterEqual(wall + 1e-9, elapsed)
        self.assertAlmostEqual(observations[-1][0], 3)
        self.assertAlmostEqual(state['wall'], 3)
        self.assertAlmostEqual(sum(waits), 3)
        self.assertEqual(detector.track.call_count, 45)
        capture.grab.assert_not_called()

    def test_quick_thirty_fps_timeline_can_finish_without_wall_waits(self):
        state, waits, observations, detector, capture = self.run_thirty_fps_timeline(False)
        self.assertAlmostEqual(observations[-1][0], 3)
        self.assertEqual(state['wall'], 0)
        self.assertEqual(waits, [])
        self.assertLess(len([item for item in observations if item[2]]), 90)
        self.assertGreater(detector.track.call_count, 0)
        self.assertGreater(capture.grab.call_count, 0)

    def test_paused_calibration_reprojects_speed_and_saves_without_more_inference(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            video = Path(temp) / 'clip.mp4'
            video.touch()
            points = [[50, 50], [450, 50], [450, 450], [50, 450]]
            calibration = RoadCalibration(points, 40, 80)
            controls = VideoCalibrationControls(calibration, 2, (500, 500))
            frame = np.zeros((500, 500, 3), np.uint8)
            state = {'index': -1, 'key': -1}
            capture = Mock()
            capture.isOpened.return_value = True

            def read():
                state['index'] += 1
                return True, frame

            capture.read.side_effect = read
            capture.get.side_effect = lambda prop: 10 if prop == cv2.CAP_PROP_FPS else state['index'] * 100

            def track(*args, **kwargs):
                # 400 image pixels cover 80m; 50px/s therefore means 10m/s.
                y = 100 + 50 * state['index'] / 10
                result = Mock()
                result.plot.return_value = frame.copy()
                result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[145, y-10, 155, y]])
                result.boxes.id.int.return_value.cpu.return_value.tolist.return_value = [1]
                return [result]

            detector = Mock()
            detector.track.side_effect = track

            def wait_key(*args):
                state['key'] += 1
                if state['key'] < 10:
                    return -1
                self.assertEqual((capture.read.call_count, detector.track.call_count), (11, 11))
                if state['key'] == 10:
                    return ord(' ')
                if state['key'] == 11:
                    controls.set_percent('Scale %', 200)
                    controls.set_percent('Ratio %', 150)
                    return ord('s')
                return ord('q')

            stack.enter_context(patch.object(sys, 'argv', ['main.py', '--calibrate', '--video', str(video)]))
            stack.enter_context(patch('main.CALIBRATION_PATH', Path(temp) / 'calibration.json'))
            stack.enter_context(patch('main.prompt_ratio', return_value=2))
            stack.enter_context(patch('main.prompt_base_speed_limit', return_value=45))
            stack.enter_context(patch('main.run_calibration', return_value=calibration))
            stack.enter_context(patch('main.VideoCalibrationControls', return_value=controls))
            stack.enter_context(patch.object(controls, 'attach'))
            stack.enter_context(patch('main.FRAME_SKIP', 1))
            stack.enter_context(patch('main.YOLO_MODEL_PATH', Path(__file__)))
            stack.enter_context(patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=Mock(return_value=detector))}))
            stack.enter_context(patch('main.cv2.VideoCapture', return_value=capture))
            stack.enter_context(patch('main.cv2.imshow'))
            stack.enter_context(patch('main.playback_key', side_effect=wait_key))
            stack.enter_context(patch('main.cv2.destroyAllWindows'))
            stack.enter_context(patch('main.window_is_open', return_value=True))
            save = stack.enter_context(patch.object(VideoSettingsStore, 'save', autospec=True, side_effect=VideoSettingsStore.save))
            panel = stack.enter_context(patch('ui.create_dynamic_speed_sign', wraps=create_dynamic_speed_sign))
            forecast = stack.enter_context(patch('main.forecast_collected_averages'))
            advisory = stack.enter_context(patch('main.display_weather_advisory'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

            main.main()

            self.assertEqual((capture.read.call_count, detector.track.call_count), (11, 11))
            self.assertAlmostEqual(panel.call_args_list[-2].args[0], 10 * 2.2369362921, delta=.001)
            self.assertAlmostEqual(panel.call_args.args[0], 20 * 2.2369362921, delta=.001)
            self.assertTrue(all(call.args[1] == 45 for call in panel.call_args_list))
            self.assertEqual(save.call_count, 2)
            self.assertFalse((Path(temp) / 'calibration.json').exists())
            settings = VideoSettingsStore(Path(temp) / 'calibrationtemp.json').load(video, (500, 500))
            saved = settings.calibration
            self.assertEqual((settings.ratio, settings.base_speed_limit), (3, 45))
            self.assertEqual((saved.road_width_m, saved.road_length_m), (80, 160))
            self.assertEqual(saved.points, calibration.points)
            forecast.assert_not_called()
            advisory.assert_not_called()
            capture.release.assert_called_once()

    def test_visual_opens_file_picker_and_locks_preset_controls(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            video = Path(temp) / 'clip.mp4'
            video.touch()
            entry = dict(ratio=2, base_speed_limit=45, road_width_m=40, road_length_m=80,
                         frame_size=[500, 500], points=[[50, 50], [450, 50], [450, 450], [50, 450]])
            original = copy.deepcopy(entry)
            frame = np.zeros((500, 500, 3), np.uint8)
            capture = Mock()
            capture.isOpened.return_value = True
            state = {'index': -1}

            def read():
                state['index'] += 1
                return True, frame

            capture.read.side_effect = read
            capture.get.side_effect = lambda prop: 10 if prop == cv2.CAP_PROP_FPS else state['index'] * 100
            result = Mock()
            result.boxes = None
            result.plot.return_value = frame.copy()
            detector = Mock()
            detector.track.return_value = [result]
            collectors = []

            def collector_factory():
                collectors.append(FiveMinuteAverages())
                return collectors[-1]

            stack.enter_context(patch.object(sys, 'argv', ['main.py', '--visual']))
            stack.enter_context(patch('main.VIDEO_PATH', video))
            stack.enter_context(patch('main.VideoSettingsStore', return_value=store_with_defaults({video.name: entry})))
            stack.enter_context(patch('main.YOLO_MODEL_PATH', Path(__file__)))
            stack.enter_context(patch('main.FRAME_SKIP', 1))
            stack.enter_context(patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=Mock(return_value=detector))}))
            stack.enter_context(patch('main.cv2.VideoCapture', return_value=capture))
            stack.enter_context(patch('main.FiveMinuteAverages', side_effect=collector_factory))
            choose_video = stack.enter_context(patch('main.choose_video', return_value=video))
            for target in ('main.prompt_ratio', 'main.prompt_base_speed_limit', 'main.run_calibration',
                           'main.VideoCalibrationControls', 'main.cv2.createTrackbar', 'main.cv2.setMouseCallback'):
                stack.enter_context(patch(target, side_effect=AssertionError(f'Locked playback called {target}')))
            save = stack.enter_context(patch('video_settings.VideoSettingsStore.save'))
            stack.enter_context(patch('main.cv2.imshow'))
            # Pause, try the former editing keys, resume, then quit on frame 2.
            stack.enter_context(patch('main.playback_key', side_effect=[32, ord('r'), ord('u'), ord('s'), 32, -1, ord('q')]))
            cleanup = stack.enter_context(patch('main.cv2.destroyAllWindows'))
            stack.enter_context(patch('main.window_is_open', return_value=True))
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            main.main()
            choose_video.assert_called_once_with()
            self.assertEqual(capture.read.call_count, 3)
            self.assertEqual(detector.track.call_count, 3)
            self.assertEqual(len(collectors), 1)
            self.assertAlmostEqual(collectors[0]._last_time, .2)
            self.assertEqual(entry, original)
            self.assertIn('Calibration is locked; use --calibrate to edit', output.getvalue())
            save.assert_not_called()
            capture.release.assert_called_once()
            cleanup.assert_called_once()

if __name__ == '__main__':
    unittest.main()
