"""Avoid per-frame GUI sleeps and repaint paused video only when needed."""
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from ui import create_dynamic_speed_sign
from settings_fixture import store_with_defaults
from calibration import RoadCalibration
from video_calibration import VideoCalibrationControls


class PlaybackEventTests(unittest.TestCase):
    def test_playing_pumps_events_without_waiting(self):
        with patch('main.cv2.pollKey', return_value=ord('q')) as poll, \
                patch('main.cv2.waitKey', side_effect=AssertionError('Unexpected playback sleep')) as wait:
            self.assertEqual(main.playback_key(), ord('q'))
        poll.assert_called_once_with()
        wait.assert_not_called()

    def test_paused_wait_is_bounded_and_keeps_keyboard_events(self):
        with patch('main.cv2.pollKey') as poll, patch('main.cv2.waitKey', return_value=32) as wait:
            self.assertEqual(main.playback_key(paused=True), 32)
        wait.assert_called_once_with(20)
        poll.assert_not_called()

    def test_older_opencv_without_pollkey_keeps_compatible_fallback(self):
        with patch('main.cv2.pollKey', None), patch('main.cv2.waitKey', return_value=-1) as wait:
            self.assertEqual(main.playback_key(), 255)
        wait.assert_called_once_with(1)

    def test_invalid_drag_requests_one_repaint_without_committing_bad_geometry(self):
        controls = VideoCalibrationControls(RoadCalibration(
            [[50, 50], [450, 50], [450, 450], [50, 450]], 40, 80), 1, (500, 500))
        controls.editor.points[0] = controls.editor.points[2][:]
        self.assertFalse(controls.poll())
        self.assertFalse(controls.valid)
        self.assertTrue(controls.needs_redraw)
        self.assertFalse(controls.poll())
        self.assertFalse(controls.needs_redraw)

    def test_idle_pause_does_not_redraw_decode_or_infer_on_each_gui_poll(self):
        frame = np.zeros((540, 960, 3), np.uint8)
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, frame)
        capture.get.side_effect = lambda prop: 30 if prop == main.cv2.CAP_PROP_FPS else 0
        result = Mock()
        result.boxes = None
        result.plot.return_value = frame.copy()
        model = Mock()
        model.track.return_value = [result]
        entry = dict(ratio=1, base_speed_limit=45, road_width_m=10, road_length_m=25,
                     frame_size=[960, 540], points=[[50, 50], [900, 50], [900, 500], [50, 500]])
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', ['main.py', '--visual', '--video', __file__]))
            stack.enter_context(patch('main.YOLO_MODEL_PATH', Path(__file__)))
            stack.enter_context(patch('main.VideoSettingsStore', return_value=store_with_defaults({Path(__file__).name: entry})))
            stack.enter_context(patch('main.cv2.VideoCapture', return_value=capture))
            stack.enter_context(patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=Mock(return_value=model))}))
            stack.enter_context(patch('main.playback_key', side_effect=[32]+[-1]*20+[ord('q')]))
            display = stack.enter_context(patch('main.cv2.imshow'))
            panel = stack.enter_context(patch('ui.create_dynamic_speed_sign', wraps=create_dynamic_speed_sign))
            stack.enter_context(patch('main.cv2.destroyAllWindows'))
            stack.enter_context(patch('main.window_is_open', return_value=True))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            main.main()
        capture.read.assert_called_once()
        model.track.assert_called_once()
        self.assertEqual(display.call_count, 2)  # Initial frame, then the paused state.
        self.assertEqual(panel.call_count, 2)
        capture.release.assert_called_once()

    def test_preview_shows_fresh_skipped_frames_without_extra_inference(self):
        frames = [np.full((540, 960, 3), value, np.uint8) for value in range(4)]
        state = {'index': -1}
        capture = Mock()
        capture.isOpened.return_value = True

        def read():
            state['index'] += 1
            return (True, frames[state['index']]) if state['index'] < 4 else (False, None)

        capture.read.side_effect = read
        capture.get.side_effect = lambda prop: 30 if prop == main.cv2.CAP_PROP_FPS else state['index']*1000/30
        result = Mock()
        result.boxes = None
        observed = []

        def plot(*, img, labels):
            observed.append(int(img[0, 0, 0]))
            return img.copy()

        result.plot.side_effect = plot
        model = Mock()
        model.track.return_value = [result]
        entry = dict(ratio=1, base_speed_limit=45, road_width_m=10, road_length_m=25,
                     frame_size=[960, 540], points=[[50, 50], [900, 50], [900, 500], [50, 500]])
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', ['main.py', '--visual', '--video', __file__]))
            stack.enter_context(patch('main.FRAME_SKIP', 2))
            stack.enter_context(patch('main.YOLO_MODEL_PATH', Path(__file__)))
            stack.enter_context(patch('main.VideoSettingsStore', return_value=store_with_defaults({Path(__file__).name: entry})))
            stack.enter_context(patch('main.cv2.VideoCapture', return_value=capture))
            stack.enter_context(patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=Mock(return_value=model))}))
            stack.enter_context(patch('main.playback_key', return_value=255))
            display = stack.enter_context(patch('main.cv2.imshow'))
            stack.enter_context(patch('main.cv2.destroyAllWindows'))
            stack.enter_context(patch('main.window_is_open', return_value=True))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            main.main()
        self.assertEqual(observed, [0, 1, 2, 3])
        self.assertEqual(display.call_count, 4)
        self.assertEqual(model.track.call_count, 2)
        self.assertEqual(capture.read.call_count, 5)


if __name__ == '__main__':
    unittest.main()
