"""Per-video persistence, validation, and command-mode behavior."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from calibration import RoadCalibration, video_identity
from video_settings import VideoSettings, VideoSettingsStore, prompt_base_speed_limit
from config import DEFAULT_SPEED_LIMIT
from settings_fixture import store_with_defaults
from ui import create_dynamic_speed_sign

ENTRY = {
    'ratio': 1.0, 'base_speed_limit': 45, 'road_width_m': 20, 'road_length_m': 60,
    'frame_size': [960, 540],
    'points': [[240, 30], [500, 30], [820, 520], [30, 520]],
}


class VideoPresetTests(unittest.TestCase):
    def test_legacy_profile_migrates_geometry_and_inputs_without_losing_them(self):
        for inputs in (None, dict(ratio=2.5, base_speed_limit=35)):
            with self.subTest(inputs=inputs), tempfile.TemporaryDirectory() as temp:
                video = Path(temp) / 'traffic4.mp4'
                video.touch()
                path = Path(temp) / 'calibration.json'
                legacy = dict(version=1, video=video_identity(video),
                              **{k: ENTRY[k] for k in ('frame_size', 'points', 'road_width_m', 'road_length_m')})
                if inputs is not None:
                    legacy['run_settings'] = inputs
                path.write_text(json.dumps(legacy))
                settings = VideoSettingsStore(path).load(video, (960, 540))
                self.assertEqual(settings.calibration.points, ENTRY['points'])
                self.assertEqual((settings.ratio, settings.base_speed_limit),
                                 (2.5, 35) if inputs else (1, 45))
                self.assertEqual(json.loads(path.read_text()), legacy)  # Read-only until the next save.

    def test_defaults_preserve_points_and_dimensions_without_mutating_data(self):
        store = store_with_defaults({'clip.mp4': ENTRY})
        settings = store.load('clip.mp4', (960, 540))
        self.assertEqual((settings.ratio, settings.base_speed_limit), (1, 45))
        self.assertEqual(settings.calibration.points, ENTRY['points'])
        self.assertEqual((settings.calibration.road_width_m, settings.calibration.road_length_m), (20, 60))
        settings.calibration.points[0][0] = 1
        self.assertEqual(store.defaults['clip.mp4'], ENTRY)

    def test_explicit_ratio_overrides_default_without_persisting_it(self):
        store = store_with_defaults({'clip.mp4': ENTRY})
        self.assertEqual(store.load('clip.mp4', (960, 540), 2.5).ratio, 2.5)
        self.assertEqual(store.load('clip.mp4', (960, 540)).ratio, 1)

    def test_profiles_retain_multiple_videos_and_override_defaults_in_all_modes(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            path = Path(temp) / 'calibration.json'
            video = Path(temp) / 'clip.mp4'
            other = Path(temp) / 'other.mp4'
            video.touch()
            other.touch()
            store = VideoSettingsStore(path)
            store.defaults = {'clip.mp4': copy.deepcopy(ENTRY)}
            changed = VideoSettings(RoadCalibration(ENTRY['points'], 25, 70), 2, 35)
            store.save(video, (960, 540), changed)
            store.save(other, (960, 540), VideoSettings(RoadCalibration(ENTRY['points'], 10, 30), 4, 15))
            reloaded = VideoSettingsStore(path)
            saved = reloaded.load(video, (960, 540))
            self.assertEqual((saved.ratio, saved.base_speed_limit), (2, 35))
            self.assertEqual((saved.calibration.road_width_m, saved.calibration.road_length_m), (25, 70))
            self.assertEqual(reloaded.load(other, (960, 540)).ratio, 4)
            self.assertEqual(reloaded.load(video, (960, 540), 3).ratio, 3)
            self.assertEqual(VideoSettingsStore(path).load(video, (960, 540)).ratio, 2)
            self.assertEqual(store.defaults['clip.mp4'], ENTRY)

    def test_saved_profile_cannot_apply_to_a_different_file_with_same_name(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            first = Path(temp) / 'clip.mp4'
            directory = Path(temp) / 'other'
            directory.mkdir()
            second = directory / 'clip.mp4'
            first.touch()
            second.touch()
            store = VideoSettingsStore(Path(temp) / 'settings.json')
            store.defaults['clip.mp4'] = copy.deepcopy(ENTRY)
            store.save(first, (960, 540), VideoSettings(RoadCalibration(ENTRY['points'], 10, 30), 5, 30))
            self.assertEqual(store.load(first, (960, 540)).ratio, 5)
            self.assertEqual(store.load(second, (960, 540)).ratio, 1)

    def test_missing_settings_explain_how_to_calibrate(self):
        with self.assertRaisesRegex(ValueError, '--calibrate'):
            store_with_defaults({}).load('unknown.mp4', (960, 540))

    def test_portrait_and_landscape_coordinates_cannot_be_mixed(self):
        with self.assertRaisesRegex(ValueError, 'does not match processing size'):
            store_with_defaults({'clip.mp4': ENTRY}).load('clip.mp4', (540, 960))

    def test_invalid_values_do_not_start_with_guessed_settings(self):
        for change in ({'ratio': 0}, {'road_width_m': -1}, {'road_width_m': None},
                       {'points': [[0, 0]] * 4}, {'points': [[-10, 30], [500, 30], [820, 520], [30, 520]]}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'Invalid settings'):
                store_with_defaults({'clip.mp4': dict(ENTRY, **change)}).load('clip.mp4', (960, 540))

    def test_settings_files_are_read_once_then_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'settings.json'
            path.write_text(json.dumps(dict(version=2, profiles={})))
            original_read = Path.read_text
            with patch.object(Path, 'read_text', autospec=True, side_effect=original_read) as read:
                store = VideoSettingsStore(path)
                store.defaults = {'clip.mp4': ENTRY}
                store.load('clip.mp4', (960, 540))
                store.load('clip.mp4', (960, 540), 2)
                self.assertEqual(len([call for call in read.call_args_list if call.args[0] == path]), 1)
                self.assertEqual(read.call_count, 2)  # Defaults and profiles each read once.

    def run_preset_mode(self, extra_args=(), base_limit=45, vehicle_count=0, expected_limit=None, saved_profile=False):
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            video = Path(temp) / 'clip.mp4'
            video.touch()
            frame = np.zeros((540, 960, 3), np.uint8)
            capture = Mock()
            capture.isOpened.return_value = True
            capture.read.side_effect = [(True, frame), (False, None)]
            capture.grab.return_value = False
            capture.get.side_effect = lambda prop: 30 if prop == main.cv2.CAP_PROP_FPS else 0
            result = Mock()
            result.plot.return_value = frame.copy()
            if vehicle_count:
                result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([
                    [330 + i * 10, 160, 340 + i * 10, 200] for i in range(vehicle_count)
                ])
                result.boxes.id.int.return_value.cpu.return_value.tolist.return_value = list(range(vehicle_count))
            else:
                result.boxes = None
            detector = Mock()
            detector.track.return_value = [result]
            stack.enter_context(patch.object(sys, 'argv', ['main.py', *extra_args]))
            stack.enter_context(patch('main.VIDEO_PATH', video))
            store = store_with_defaults({'clip.mp4': dict(ENTRY, base_speed_limit=base_limit)})
            ratio = 1
            if saved_profile:
                store.path = Path(temp) / 'calibration.json'
                ratio, base_limit = 2, 35
                with contextlib.redirect_stdout(io.StringIO()):
                    store.save(video, (960, 540), VideoSettings(RoadCalibration(ENTRY['points'], 25, 70), ratio, base_limit))
                stack.enter_context(patch('main.CALIBRATION_PATH', store.path))
            else:
                stack.enter_context(patch('main.VideoSettingsStore', return_value=store))
            stack.enter_context(patch.dict(sys.modules, {'api_code': None, 'decision_engine': None}))
            tracker = stack.enter_context(patch('main.SpeedTracker', wraps=main.SpeedTracker))
            choose_video = stack.enter_context(patch('main.choose_video', return_value=video))
            stack.enter_context(patch('main.YOLO_MODEL_PATH', Path(__file__)))
            stack.enter_context(patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=Mock(return_value=detector))}))
            stack.enter_context(patch('main.cv2.VideoCapture', return_value=capture))
            stack.enter_context(patch('main.cv2.imshow'))
            stack.enter_context(patch('main.playback_key', return_value=-1))
            stack.enter_context(patch('main.cv2.destroyAllWindows'))
            stack.enter_context(patch('main.window_is_open', return_value=True))
            panel = stack.enter_context(patch('ui.create_dynamic_speed_sign', wraps=create_dynamic_speed_sign))
            prompts = [stack.enter_context(patch(target, side_effect=AssertionError('Unexpected setup prompt')))
                       for target in ['main.prompt_ratio', 'main.prompt_base_speed_limit',
                                      'main.run_calibration', 'builtins.input']]
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            main.main()
            if '--visual' in extra_args and '--default-video' not in extra_args:
                choose_video.assert_called_once_with()
            else:
                choose_video.assert_not_called()
            for prompt in prompts:
                prompt.assert_not_called()
            self.assertIn(f'speed multiplier: {ratio}', output.getvalue())
            self.assertIn('Using default calibration' if saved_profile else 'Using migrated defaults', output.getvalue())
            if saved_profile:
                self.assertEqual(tracker.call_args.args[1:], (25, 70))
            self.assertNotIn('UNVALIDATED DEMO', output.getvalue())
            self.assertNotIn('Paste this entry into VIDEO_PRESETS', output.getvalue())
            if '--visual' in extra_args:
                self.assertEqual(panel.call_args.args[1], base_limit if expected_limit is None else expected_limit)
            else:
                panel.assert_not_called()
                result.plot.assert_not_called()
            capture.release.assert_called_once()

    def test_preset_mode_starts_without_any_setup_prompts(self):
        self.run_preset_mode()

    def test_headless_and_visual_use_the_same_saved_settings(self):
        for args in ([], ['--visual']):
            with self.subTest(args=args):
                self.run_preset_mode(args, saved_profile=True)

    def test_visual_default_video_bypasses_picker(self):
        self.run_preset_mode(['--visual', '--default-video'])

    def test_dense_traffic_never_increases_a_low_base_limit(self):
        self.run_preset_mode(['--visual'], base_limit=15, vehicle_count=3, expected_limit=15)

    def test_dense_traffic_keeps_the_input_limit(self):
        self.run_preset_mode(['--visual'], base_limit=45, vehicle_count=3, expected_limit=45)

    def test_base_limit_prompt_validates_retries_defaults_and_can_cancel(self):
        with patch('builtins.input', side_effect=['0', '-5', 'nan', '35.5', '40']):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(prompt_base_speed_limit(), 40)
        with patch('builtins.input', return_value=''):
            self.assertEqual(prompt_base_speed_limit(), DEFAULT_SPEED_LIMIT)
        with patch('builtins.input', return_value='q'):
            with self.assertRaises(KeyboardInterrupt):
                prompt_base_speed_limit()

    def test_invalid_base_limits_are_rejected(self):
        for value in (0, -5, float('nan'), float('inf'), 35.5, True, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Base speed limit'):
                store_with_defaults({'clip.mp4': dict(ENTRY, base_speed_limit=value)}).load('clip.mp4', (960, 540))

    def test_recalibrate_explicitly_opens_setup(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            video = Path(temp) / 'clip.mp4'
            video.touch()
            capture = Mock()
            capture.isOpened.return_value = True
            capture.get.side_effect = lambda prop: 30 if prop == main.cv2.CAP_PROP_FPS else 0
            capture.read.return_value = (True, np.zeros((540, 960, 3), np.uint8))
            stack.enter_context(patch.object(sys, 'argv', ['main.py', '--recalibrate']))
            stack.enter_context(patch('main.YOLO_MODEL_PATH', Path(__file__)))
            picker = stack.enter_context(patch('main.choose_video', return_value=video))
            ratio_prompt = stack.enter_context(patch('main.prompt_ratio', return_value=1))
            base_limit_prompt = stack.enter_context(patch('main.prompt_base_speed_limit', return_value=45))
            calibration_prompt = stack.enter_context(patch('main.run_calibration', return_value=None))
            stack.enter_context(patch('main.cv2.VideoCapture', return_value=capture))
            stack.enter_context(patch('main.cv2.destroyAllWindows'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            main.main()
            picker.assert_called_once()
            ratio_prompt.assert_called_once()
            base_limit_prompt.assert_called_once()
            capture.release.assert_called_once()

    def test_calibrate_saves_setup_then_plays_short_video_with_calibration_controls(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            video = Path(temp) / 'unconfigured-short-clip.mp4'
            video.touch()
            frames = [np.full((540, 960, 3), index, np.uint8) for index in range(4)]
            state = {'index': -1}
            capture = Mock()
            capture.isOpened.return_value = True

            def read():
                state['index'] += 1
                if state['index'] == len(frames):
                    return False, None
                return True, frames[state['index']]

            capture.read.side_effect = read
            capture.get.side_effect = lambda prop: (
                1 if prop == main.cv2.CAP_PROP_FPS else
                len(frames) if prop == main.cv2.CAP_PROP_FRAME_COUNT else state['index'] * 1000
            )
            result = Mock()
            result.plot.side_effect = lambda **kwargs: frames[state['index']].copy()
            result.boxes = None
            detector = Mock()
            detector.track.return_value = [result]
            calibration = RoadCalibration(ENTRY['points'], ENTRY['road_width_m'], ENTRY['road_length_m'])
            stack.enter_context(patch.object(
                sys, 'argv', ['main.py', '--calibrate', '--video', str(video)]
            ))
            stack.enter_context(patch('main.prompt_ratio', return_value=2))
            stack.enter_context(patch('main.prompt_base_speed_limit', return_value=45))
            calibration_prompt = stack.enter_context(patch('main.run_calibration', return_value=calibration))
            save = stack.enter_context(patch('video_settings.VideoSettingsStore.save'))
            stack.enter_context(patch('main.FRAME_SKIP', 1))
            stack.enter_context(patch('main.YOLO_MODEL_PATH', Path(__file__)))
            stack.enter_context(patch.dict(
                sys.modules, {'ultralytics': SimpleNamespace(YOLO=Mock(return_value=detector))}
            ))
            stack.enter_context(patch('main.cv2.VideoCapture', return_value=capture))
            display = stack.enter_context(patch('main.cv2.imshow'))
            panel = stack.enter_context(patch('ui.create_dynamic_speed_sign', wraps=create_dynamic_speed_sign))
            controls = stack.enter_context(patch('main.VideoCalibrationControls.attach'))
            stack.enter_context(patch('main.playback_key', return_value=-1))
            stack.enter_context(patch('main.cv2.destroyAllWindows'))
            stack.enter_context(patch('main.window_is_open', return_value=True))
            forecast = stack.enter_context(patch('main.forecast_collected_averages'))
            advisory = stack.enter_context(patch('main.display_weather_advisory'))
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

            self.assertIsNone(main.main())

            calibration_prompt.assert_called_once()
            save.assert_called_once()
            self.assertEqual(save.call_args.args[:2], (video, (960, 540)))
            saved = save.call_args.args[2]
            self.assertEqual((saved.calibration, saved.ratio, saved.base_speed_limit), (calibration, 2, 45))
            controls.assert_called_once()
            self.assertEqual(detector.track.call_count, len(frames))
            self.assertEqual(display.call_count, len(frames))
            self.assertTrue(all(call.args[1] == 45 for call in panel.call_args_list))
            forecast.assert_not_called()
            advisory.assert_not_called()
            self.assertIn('0/3 complete five-minute averages', output.getvalue())
            self.assertIn('no forecast was generated.', output.getvalue())
            capture.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
