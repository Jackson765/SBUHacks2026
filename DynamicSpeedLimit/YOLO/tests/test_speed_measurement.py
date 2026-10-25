"""Independent trajectories exercise units, perspective, timing, and track lifecycle."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration import CornerEditor, RoadCalibration, default_corners, prompt_distance, run_calibration
from video_settings import VideoSettings, VideoSettingsStore
from config import MAX_INACTIVE_SECONDS, VIDEO_H, VIDEO_W
from main import VideoClock, resize_for_processing
from settings_fixture import store_with_defaults
import main as application
from tracker import SpeedTracker
from ui import create_dynamic_speed_sign, draw_speed_label, resize_to_height

MPS_TO_MPH = 2.2369362921
# A trapezoid in the camera image represents a measured 20 x 60 meter rectangle.
ROAD_POINTS = [[240, 30], [500, 30], [820, 520], [30, 520]]
WORLD_POINTS = [[0, 0], [20, 0], [20, 60], [0, 60]]
WORLD_TO_IMAGE = cv2.getPerspectiveTransform(np.float32(WORLD_POINTS), np.float32(ROAD_POINTS))


def bbox_at(x_m, y_m):
    point = cv2.perspectiveTransform(np.float32([[[x_m, y_m]]]), WORLD_TO_IMAGE)[0, 0]
    x, y = map(float, point)
    return (x - 5, y - 10, x + 5, y)


class SpeedTests(unittest.TestCase):
    def setUp(self):
        self.tracker = SpeedTracker(ROAD_POINTS, 20, 60)

    def test_known_speed_across_perspective_and_frame_skips(self):
        # 10 m/s = 22.3694 mph, at different depths and detection rates.
        for start in (3, 40):
            for skip in (1, 2, 5):
                tracker = SpeedTracker(ROAD_POINTS, 20, 60)
                for frame in range(0, 51, skip):
                    t = frame / 50
                    speed = tracker.update_vehicle_speed(7, bbox_at(10, start + 10 * t), t)
                self.assertAlmostEqual(speed, 10 * MPS_TO_MPH, delta=0.001)

    def test_sub_mph_movement_is_preserved(self):
        for frame in range(51):
            t = frame / 50
            speed = self.tracker.update_vehicle_speed(1, bbox_at(10, 10 + 0.1 * t), t)
        self.assertAlmostEqual(speed, 0.1 * MPS_TO_MPH, delta=0.001)

    def test_both_axes_and_reverse_motion(self):
        for frame in range(51):
            t = frame / 50
            speed = self.tracker.update_vehicle_speed(1, bbox_at(10 + 3*t, 30 - 4*t), t)
        self.assertAlmostEqual(speed, 5 * MPS_TO_MPH, delta=0.001)

    def test_variable_intervals_use_elapsed_video_time(self):
        for t in [0, .02, .07, .14, .31, .52, .63, .91, 1.0]:
            speed = self.tracker.update_vehicle_speed(1, bbox_at(10, 10 + 6 * t), t)
        self.assertAlmostEqual(speed, 6 * MPS_TO_MPH, delta=0.001)

    def test_new_track_requires_time_and_samples(self):
        for t in (0, .1, .2, .3, .4):
            self.assertIsNone(self.tracker.update_vehicle_speed(1, bbox_at(10, 10 + t), t))
        self.assertIsNotNone(self.tracker.update_vehicle_speed(1, bbox_at(10, 10.5), .5))

    def test_moving_vehicle_reaches_zero_after_stopping(self):
        for frame in range(101):
            t = frame / 50
            speed = self.tracker.update_vehicle_speed(1, bbox_at(10, 10 + 10 * min(t, 1)), t)
            if frame == 50:
                self.assertGreater(speed, 20)
        self.assertEqual(speed, 0.0)

    def test_stationary_track(self):
        for t in np.linspace(0, 1, 51):
            speed = self.tracker.update_vehicle_speed(1, bbox_at(10, 10), t)
        self.assertEqual(speed, 0.0)

    def test_zero_mean_position_jitter_is_reduced(self):
        rng = np.random.default_rng(12)
        for t in np.linspace(0, 1, 51):
            noise_x, noise_y = rng.normal(0, .03, 2)
            speed = self.tracker.update_vehicle_speed(1, bbox_at(10 + noise_x, 10 + 10*t + noise_y), t)
        self.assertAlmostEqual(speed, 10 * MPS_TO_MPH, delta=.15)

    def test_impossible_jump_is_ignored_and_track_recovers(self):
        for t in np.linspace(0, .6, 31):
            self.tracker.update_vehicle_speed(1, bbox_at(10, 10 + t), t)
        self.assertIsNone(self.tracker.update_vehicle_speed(1, bbox_at(10, 50), .62))
        self.assertAlmostEqual(self.tracker.last_seen[1], .6)
        speed = self.tracker.update_vehicle_speed(1, bbox_at(10, 10.64), .64)
        self.assertAlmostEqual(speed, MPS_TO_MPH, delta=.001)

    def test_exit_gap_and_timestamp_reset_require_new_history(self):
        for t in np.linspace(0, 1, 51):
            self.tracker.update_vehicle_speed(1, bbox_at(10, 10 + t), t)
        self.assertIsNone(self.tracker.update_vehicle_speed(1, bbox_at(25, 11), 1.02))
        self.assertNotIn(1, self.tracker.track_history)
        self.assertIsNone(self.tracker.update_vehicle_speed(1, bbox_at(10, 11), 1.04))
        self.assertIsNone(self.tracker.update_vehicle_speed(1, bbox_at(10, 11), 2.0))
        self.assertEqual(len(self.tracker.track_history[1]), 1)
        self.assertIsNone(self.tracker.update_vehicle_speed(1, bbox_at(10, 11), 0.0))
        self.assertEqual(len(self.tracker.track_history[1]), 1)

    def test_inactive_tracks_are_removed(self):
        self.tracker.update_vehicle_speed(1, bbox_at(10, 10), 0)
        self.tracker.remove_inactive_tracks(MAX_INACTIVE_SECONDS + .01)
        self.assertFalse(self.tracker.track_history)
        self.assertFalse(self.tracker.last_seen)

    def test_invalid_measurements(self):
        for timestamp in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.tracker.update_vehicle_speed(1, bbox_at(10, 10), timestamp)
        self.assertIsNone(self.tracker.update_vehicle_speed(1, [float('nan'), 0, 100, 100], 0))


class CalibrationTests(unittest.TestCase):
    def test_corners_map_to_measured_meters(self):
        calibration = RoadCalibration(ROAD_POINTS, 20, 60)
        transformed = cv2.perspectiveTransform(np.float32([ROAD_POINTS]), calibration.perspective_matrix())
        np.testing.assert_allclose(transformed[0], WORLD_POINTS, atol=1e-5)

    def test_invalid_geometry_and_dimensions(self):
        for points in ([[0, 0]] * 4, [[0, 0], [100, 100], [100, 0], [0, 100]],
                       [[0, 0], [100, 0], [float('nan'), 100], [0, 100]], [[0, 0]]):
            with self.assertRaises(ValueError):
                RoadCalibration(points, 20, 60)
        for width, length in [(0, 1), (1, -1), (float('nan'), 1), (1, float('inf'))]:
            with self.assertRaises(ValueError):
                RoadCalibration(ROAD_POINTS, width, length)

    def test_profile_round_trip_and_mismatch_protection(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            video = Path(temp) / 'test.mp4'
            video.write_bytes(b'video fixture')
            profile = Path(temp) / 'calibration.json'
            store = VideoSettingsStore(profile)
            calibration = RoadCalibration(ROAD_POINTS, 20, 60)
            store.save(video, (960, 540), VideoSettings(calibration, 1, 45))
            loaded = VideoSettingsStore(profile).load(video, (960, 540)).calibration
            self.assertEqual(loaded.points, calibration.points)
            self.assertEqual(loaded.road_length_m, 60)
            other = Path(temp) / 'other.mp4'
            other.write_bytes(b'video fixture')
            with self.assertRaisesRegex(ValueError, '--calibrate'):
                store.load(other, (960, 540))
            with self.assertRaisesRegex(ValueError, 'processing size'):
                store.load(video, (1, 1))
            video.write_bytes(b'changed video fixture')
            with self.assertRaisesRegex(ValueError, '--calibrate'):
                store.load(video, (960, 540))
            profile.write_text('{bad JSON')
            with self.assertRaisesRegex(ValueError, 'Invalid video settings'):
                VideoSettingsStore(profile)

    def test_profile_round_trip_at_portrait_processing_resolution(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            video = Path(temp) / 'portrait.mp4'
            video.write_bytes(b'video fixture')
            store = VideoSettingsStore(Path(temp) / 'calibration.json')
            calibration = RoadCalibration(default_corners(540, 960), 20, 60)
            store.save(video, (540, 960), VideoSettings(calibration, 1, 45))
            self.assertEqual(store.load(video, (540, 960)).calibration.points, calibration.points)
            with self.assertRaisesRegex(ValueError, 'processing size'):
                store.load(video, (960, 540))

    def test_distance_prompt_rejects_invalid_values_and_can_cancel(self):
        with patch('builtins.input', side_effect=['nan', '-5', 'abc', '12.5']):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(prompt_distance('width'), 12.5)
        with patch('builtins.input', return_value='q'):
            with self.assertRaises(KeyboardInterrupt):
                prompt_distance('width')

    def test_interactive_drag_and_accept(self):
        callbacks = []
        positions = default_corners()
        actions = list(zip(positions, ROAD_POINTS)) + ['accept']

        def wait_key(delay):
            action = actions.pop(0)
            if action != 'accept':
                start, end = action
                callbacks[0](cv2.EVENT_LBUTTONDOWN, *start, None, None)
                callbacks[0](cv2.EVENT_MOUSEMOVE, *end, cv2.EVENT_FLAG_LBUTTON, None)
                callbacks[0](cv2.EVENT_LBUTTONUP, *end, None, None)
                return -1
            return 13

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch('calibration.prompt_distance', side_effect=[20, 60]))
            stack.enter_context(patch('calibration.cv2.namedWindow'))
            stack.enter_context(patch('calibration.cv2.setMouseCallback', side_effect=lambda *args: callbacks.append(args[1])))
            stack.enter_context(patch('calibration.cv2.imshow'))
            stack.enter_context(patch('calibration.cv2.waitKey', side_effect=wait_key))
            stack.enter_context(patch('calibration.window_is_open', return_value=True))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            result = run_calibration(np.zeros((VIDEO_H, VIDEO_W, 3), np.uint8))
        self.assertEqual(result.points, ROAD_POINTS)
        self.assertEqual(result.road_width_m, 20)

    def test_drag_preserves_corner_identity_clamps_and_undoes(self):
        editor = CornerEditor(ROAD_POINTS)
        editor.on_mouse(cv2.EVENT_LBUTTONDOWN, 900, 100, 0, None)
        editor.on_mouse(cv2.EVENT_MOUSEMOVE, 100, 100, 0, None)
        self.assertEqual(editor.points, ROAD_POINTS)  # Clicking away from handles adds nothing.
        editor.on_mouse(cv2.EVENT_LBUTTONDOWN, *ROAD_POINTS[0], 0, None)
        editor.on_mouse(cv2.EVENT_MOUSEMOVE, -20, VIDEO_H + 20, cv2.EVENT_FLAG_LBUTTON, None)
        editor.on_mouse(cv2.EVENT_LBUTTONUP, -20, VIDEO_H + 20, 0, None)
        self.assertEqual(editor.points[0], [0, VIDEO_H - 1])
        self.assertEqual(editor.points[1:], ROAD_POINTS[1:])
        editor.undo()
        self.assertEqual(editor.points, ROAD_POINTS)
        editor.on_mouse(cv2.EVENT_LBUTTONDOWN, *ROAD_POINTS[1], 0, None)
        editor.on_mouse(cv2.EVENT_LBUTTONUP, 600, 100, 0, None)
        editor.reset()
        self.assertEqual(editor.points, ROAD_POINTS)

    def test_portrait_corner_editor_clamps_to_portrait_frame(self):
        editor = CornerEditor(default_corners(540, 960), 540, 960)
        editor.on_mouse(cv2.EVENT_LBUTTONDOWN, *editor.points[0], 0, None)
        editor.on_mouse(cv2.EVENT_MOUSEMOVE, 600, 1000, cv2.EVENT_FLAG_LBUTTON, None)
        editor.on_mouse(cv2.EVENT_LBUTTONUP, 600, 1000, 0, None)
        self.assertEqual(editor.points[0], [539.0, 959.0])

    def test_portrait_calibration_displays_fixed_height_and_saves_source_coordinates(self):
        callbacks = []
        display_size = (304, 540)
        target_source_points = [[50, 100], [490, 100], [490, 850], [50, 850]]
        target_display_points = [
            (round(x * display_size[0] / 540), round(y * display_size[1] / 960))
            for x, y in target_source_points
        ]
        actions = list(zip(default_corners(*display_size), target_display_points)) + ['accept']
        shown_shapes = []

        def wait_key(delay):
            action = actions.pop(0)
            if action != 'accept':
                start, end = action
                callbacks[0](cv2.EVENT_LBUTTONDOWN, *start, None, None)
                callbacks[0](cv2.EVENT_MOUSEMOVE, *end, cv2.EVENT_FLAG_LBUTTON, None)
                callbacks[0](cv2.EVENT_LBUTTONUP, *end, None, None)
                return -1
            return 13

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch('calibration.prompt_distance', side_effect=[20, 60]))
            stack.enter_context(patch('calibration.cv2.namedWindow'))
            stack.enter_context(patch('calibration.cv2.setMouseCallback',
                                      side_effect=lambda *args: callbacks.append(args[1])))
            stack.enter_context(patch('calibration.cv2.imshow',
                                      side_effect=lambda _, image: shown_shapes.append(image.shape[:2])))
            stack.enter_context(patch('calibration.cv2.waitKey', side_effect=wait_key))
            stack.enter_context(patch('calibration.window_is_open', return_value=True))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            result = run_calibration(np.zeros((960, 540, 3), np.uint8))

        np.testing.assert_allclose(result.points, target_source_points, atol=2)
        self.assertEqual(shown_shapes[0], (540, 304 + 400))


class TimingAndDisplayTests(unittest.TestCase):
    def test_processing_resize_preserves_portrait_and_landscape_aspect_ratios(self):
        portrait = resize_for_processing(np.zeros((1920, 1080, 3), dtype=np.uint8))
        landscape = resize_for_processing(np.zeros((1080, 1920, 3), dtype=np.uint8))
        self.assertEqual(portrait.shape[:2], (960, 540))
        self.assertEqual(landscape.shape[:2], (540, 960))

    def test_display_resize_fixes_height_and_adapts_width(self):
        portrait = resize_to_height(np.zeros((960, 540, 3), dtype=np.uint8), 540)
        landscape = resize_to_height(np.zeros((540, 960, 3), dtype=np.uint8), 540)
        self.assertEqual(portrait.shape[:2], (540, 304))
        self.assertEqual(landscape.shape[:2], (540, 960))

    def test_presentation_timestamps_take_priority_over_fps(self):
        clock = VideoClock(float('nan'), source='video')
        for frame, milliseconds in enumerate([0, 20, 70, 95, 150]):
            self.assertEqual(clock.timestamp(frame, milliseconds), milliseconds / 1000)
        with self.assertRaises(ValueError):
            clock.timestamp(5, 150)

    def test_explicit_fps_mode_and_invalid_timing(self):
        clock = VideoClock(50, source='fps')
        self.assertEqual(clock.timestamp(0, float('nan')), 0)
        self.assertEqual(clock.timestamp(2, float('nan')), .04)
        for fps in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                VideoClock(fps, source='fps')
        with self.assertRaises(ValueError):
            VideoClock(30, source='unknown')
        for timestamp in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                VideoClock(30, source='video').timestamp(0, timestamp)

    def test_display_handles_unknown_zero_slow_and_positive_speeds(self):
        for speed in (None, 0.0, .2, 35.0):
            canvas = create_dynamic_speed_sign(speed)
            self.assertEqual(canvas.shape, (540, 400, 3))
            draw_speed_label(np.zeros((540, 960, 3), np.uint8), (100, 100, 200, 200), speed, False)


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        base_limit_prompt = patch('main.prompt_base_speed_limit', return_value=30)
        base_limit_prompt.start()
        self.addCleanup(base_limit_prompt.stop)

    def run_fifteen_minute_collection(self, empty_second_window=False, visual=False):
        frame = np.zeros((VIDEO_H, VIDEO_W, 3), np.uint8)
        # Boundary frames at 05:00 and 15:00 are skipped by detection. The clock
        # must still close the windows; the timestamp origin is deliberately nonzero.
        times = [10, 110, 290, 310, 315, 610, 690, 910, 1010]
        state = {'index': -1}
        capture = Mock()
        capture.isOpened.return_value = True

        def read():
            state['index'] += 1
            return state['index'] < len(times), frame

        capture.read.side_effect = read
        capture.get.side_effect = lambda prop: 50 if prop == cv2.CAP_PROP_FPS else times[state['index']] * 1000
        result = Mock()
        result.plot.return_value = frame.copy()
        result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([bbox_at(10, 10), bbox_at(12, 10)])
        result.boxes.id.int.return_value.cpu.return_value.tolist.return_value = [1, 2]
        detector = Mock()
        detector.track.return_value = [result]
        tracker = Mock()
        tracker.point_in_road.return_value = True

        def speed(track_id, bbox, timestamp):
            window = int((timestamp - times[0]) / 300)
            if window == 1 and empty_second_window:
                return None
            return 10 + 10 * window + (20 if track_id == 2 else 0)

        tracker.update_vehicle_speed.side_effect = speed
        source_model = Mock(return_value={'predicted_speed': 55})
        with contextlib.ExitStack() as stack:
            arguments = ['main.py', '--video', __file__, '--ratio', '2']
            if visual:
                arguments.append('--visual')
            else:
                arguments.append('--full-analysis')
            stack.enter_context(patch.object(sys, 'argv', arguments))
            stack.enter_context(patch('main.VideoSettingsStore', return_value=store_with_defaults({Path(__file__).name: {
                'ratio': 1, 'base_speed_limit': 30, 'road_width_m': 20, 'road_length_m': 60,
                'frame_size': [VIDEO_W, VIDEO_H], 'points': ROAD_POINTS,
            }})))
            stack.enter_context(patch('main.YOLO_MODEL_PATH', Path(__file__)))
            stack.enter_context(patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=Mock(return_value=detector))}))
            stack.enter_context(patch('main.cv2.VideoCapture', return_value=capture))
            stack.enter_context(patch('main.run_calibration', return_value=RoadCalibration(ROAD_POINTS, 20, 60)))
            stack.enter_context(patch('main.SpeedTracker', return_value=tracker))
            stack.enter_context(patch('main.FRAME_SKIP', 2))
            stack.enter_context(patch('main.window_is_open', return_value=True))
            wait_key = stack.enter_context(patch('main.playback_key', return_value=-1))
            show = stack.enter_context(patch('main.cv2.imshow'))
            cleanup = stack.enter_context(patch('main.cv2.destroyAllWindows'))
            stack.enter_context(patch('ratio_predict._model_b_predictor', return_value=source_model))
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            forecast = application.main()
        self.assertEqual(capture.read.call_count, 8)  # Never reads the post-15-minute frame.
        self.assertEqual(detector.track.call_count, 4)
        self.assertEqual(detector.track.call_args.kwargs['conf'], .25)
        self.assertEqual(detector.track.call_args.kwargs['imgsz'], 960)
        if not visual:
            result.plot.assert_not_called()
            wait_key.assert_not_called()
            show.assert_not_called()
        capture.release.assert_called_once()
        if visual:
            cleanup.assert_called_once()
        else:
            cleanup.assert_not_called()
        return forecast, source_model, output.getvalue()

    def test_fifteen_minute_flow_predicts_once_and_stops_on_skipped_frame(self):
        result, predictor, output = self.run_fifteen_minute_collection()
        predictor.assert_called_once()
        np.testing.assert_array_equal(predictor.call_args.args[0], [40, 60, 80])
        self.assertEqual(result['predicted_speed'], 27.5)
        self.assertIn('27.500 mph', output)
        self.assertIn('Collecting 10-15 minutes', output)
        self.assertIn('10-15 minutes: average 40.000 mph; scaled 80.000', output)
        self.assertLess(output.index('10-15 minutes: average'), output.index('Three scaled averages sent to Model B'))

    def test_missing_five_minute_window_does_not_call_prediction_model(self):
        result, predictor, output = self.run_fifteen_minute_collection(empty_second_window=True)
        self.assertIsNone(result)
        predictor.assert_not_called()
        self.assertIn('Prediction unavailable', output)

    def test_visual_uses_the_same_three_scaled_inputs_and_unscaled_forecast(self):
        result, predictor, output = self.run_fifteen_minute_collection(visual=True)
        predictor.assert_called_once()
        np.testing.assert_array_equal(predictor.call_args.args[0], [40, 60, 80])
        self.assertEqual(result['predicted_speed'], 27.5)
        self.assertIn('27.500 mph', output)

    def test_main_loop_uses_video_time_and_skips_frames(self):
        frame = np.zeros((VIDEO_H, VIDEO_W, 3), np.uint8)
        capture = Mock()
        capture.isOpened.return_value = True
        state = {'index': -1}

        def read():
            state['index'] += 1
            return state['index'] <= 50, frame

        capture.read.side_effect = read
        capture.get.side_effect = lambda prop: 50 if prop == cv2.CAP_PROP_FPS else state['index'] * 20
        detector = Mock()

        def track(*args, **kwargs):
            result = Mock()
            result.plot.return_value = frame.copy()
            t = state['index'] / 50
            # Second vehicle is outside the calibrated region and must be ignored.
            result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([
                bbox_at(10, 10 + 10*t), bbox_at(25, 10 + 10*t),
            ])
            result.boxes.id.int.return_value.cpu.return_value.tolist.return_value = [1, 2]
            return [result]

        detector.track.side_effect = track
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', ['main.py', '--visual', '--video', __file__, '--ratio', '2']))
            stack.enter_context(patch('main.VideoSettingsStore', return_value=store_with_defaults({Path(__file__).name: {
                'ratio': 1, 'base_speed_limit': 30, 'road_width_m': 20, 'road_length_m': 60,
                'frame_size': [VIDEO_W, VIDEO_H], 'points': ROAD_POINTS,
            }})))
            stack.enter_context(patch('main.VIDEO_PATH', Path(__file__)))
            stack.enter_context(patch('main.YOLO_MODEL_PATH', Path(__file__)))
            stack.enter_context(patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=Mock(return_value=detector))}))
            stack.enter_context(patch('main.cv2.VideoCapture', return_value=capture))
            stack.enter_context(patch('main.run_calibration', return_value=RoadCalibration(ROAD_POINTS, 20, 60)))
            stack.enter_context(patch('main.FRAME_SKIP', 2))
            stack.enter_context(patch('main.window_is_open', return_value=True))
            stack.enter_context(patch('main.playback_key', return_value=-1))
            stack.enter_context(patch('main.cv2.imshow'))
            cleanup = stack.enter_context(patch('main.cv2.destroyAllWindows'))
            panel = stack.enter_context(patch('ui.create_dynamic_speed_sign', wraps=create_dynamic_speed_sign))
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            application.main()
        self.assertEqual(detector.track.call_count, 26)
        self.assertIsNone(panel.call_args_list[0].args[0])
        self.assertAlmostEqual(panel.call_args.args[0], 10 * MPS_TO_MPH, delta=.001)
        self.assertEqual(panel.call_args.args[1], 30)  # Outside vehicle never enters the count.
        capture.release.assert_called_once()
        cleanup.assert_called_once()
        self.assertIn('0/3 complete five-minute averages', output.getvalue())
        self.assertIn('no forecast was generated', output.getvalue())

    def test_cancelled_calibration_releases_video_without_loading_detector(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: 50 if prop == cv2.CAP_PROP_FPS else 0
        capture.read.return_value = (True, np.zeros((VIDEO_H, VIDEO_W, 3), np.uint8))
        yolo = Mock()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', ['main.py', '--calibrate', '--default-video', '--ratio', '2']))
            stack.enter_context(patch('main.VIDEO_PATH', Path(__file__)))
            stack.enter_context(patch('main.YOLO_MODEL_PATH', Path(__file__)))
            stack.enter_context(patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=yolo)}))
            stack.enter_context(patch('main.cv2.VideoCapture', return_value=capture))
            stack.enter_context(patch('main.run_calibration', return_value=None))
            cleanup = stack.enter_context(patch('main.cv2.destroyAllWindows'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            application.main()
        yolo.assert_not_called()
        capture.release.assert_called_once()
        cleanup.assert_called_once()


if __name__ == '__main__':
    unittest.main()
