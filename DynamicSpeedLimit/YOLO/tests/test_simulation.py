"""Fast frame sampling must preserve source time, calibration, and all three bins."""
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from settings_fixture import store_with_defaults
from simulation import make_simulation_plan


POINTS = [[200, 300], [400, 300], [400, 500], [200, 500]]


class SimulationTests(unittest.TestCase):
    def plan(self, fps=30, **kwargs):
        settings = dict(target_fps=5, padding=128, maximum_image_size=960, fallback_step=2)
        settings.update(kwargs)
        return make_simulation_plan((640, 640), POINTS, fps, **settings)

    def test_crop_preserves_pixels_and_restores_global_box_coordinates(self):
        plan = self.plan()
        self.assertEqual(plan.frame_step, 6)
        frame = np.arange(640*640*3, dtype=np.int32).reshape(640, 640, 3)
        crop = plan.crop(frame)
        np.testing.assert_array_equal(crop[0, 0], frame[172, 72])
        self.assertTrue(crop.flags.c_contiguous)
        self.assertEqual(plan.image_size, 480)
        boxes = np.array([[128, 228, 148, 248]], dtype=np.float32)
        restored = plan.restore_boxes(boxes)
        np.testing.assert_array_equal(restored, [[200, 400, 220, 420]])
        np.testing.assert_array_equal(boxes, [[128, 228, 148, 248]])
        self.assertEqual(plan.restore_boxes(np.empty((0, 4))).shape, (0, 4))

    def test_roi_is_clipped_to_frame_and_unknown_fps_does_not_invent_a_rate(self):
        plan = self.plan(padding=1000)
        self.assertEqual(plan.bounds, (0, 0, 640, 640))
        self.assertEqual(plan.image_size, 640)
        for fps in (0, float('nan')):
            self.assertEqual(self.plan(fps).frame_step, 2)

    def test_sparse_setting_stays_inside_tracker_gap_limit(self):
        self.assertLessEqual(self.plan(target_fps=.1).frame_step / 30, .4)
        self.assertEqual(self.plan(fps=5).frame_step, 1)
        for target in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                self.plan(target_fps=target)

    def test_fast_default_grabs_skipped_frames_and_predicts_all_three_scaled_bins(self):
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        # Deliberately sparse timestamps cover all three windows. The 15:00
        # boundary is a grabbed frame: it must finalize bin 3 without inference.
        times = [i/30 for i in range(7)] + [300+i/30 for i in range(6)] + [600+i/30 for i in range(6)] + [900]
        state = {'index': -1}
        capture = Mock()
        capture.isOpened.return_value = True

        def grab():
            state['index'] += 1
            return state['index'] < len(times)

        capture.grab.side_effect = grab
        capture.read.side_effect = lambda: (grab(), frame)
        capture.get.side_effect = lambda prop: 30 if prop == cv2.CAP_PROP_FPS else times[state['index']] * 1000
        result = Mock()
        raw_boxes = np.array([[128, 228, 148, 248]], dtype=float)
        result.boxes.xyxy.cpu.return_value.numpy.return_value = raw_boxes
        result.boxes.id.int.return_value.cpu.return_value.tolist.return_value = [1]
        model = Mock()
        model.track.return_value = [result]
        tracker = Mock()
        tracker.point_in_road.return_value = True
        tracker.update_vehicle_speed.side_effect = lambda track_id, box, timestamp: 10*(1+int(timestamp//300))
        predictor = Mock(return_value={'predicted_speed': 50})
        entry = dict(ratio=2, base_speed_limit=45, road_width_m=10, road_length_m=25,
                     frame_size=[640, 640], points=POINTS)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', ['main.py', '--video', __file__]))
            stack.enter_context(patch('main.VideoSettingsStore', return_value=store_with_defaults({Path(__file__).name: entry})))
            stack.enter_context(patch('main.YOLO_MODEL_PATH', Path(__file__)))
            stack.enter_context(patch('main.cv2.VideoCapture', return_value=capture))
            stack.enter_context(patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=Mock(return_value=model))}))
            stack.enter_context(patch('main.SpeedTracker', return_value=tracker))
            stack.enter_context(patch('ratio_predict._model_b_predictor', return_value=predictor))
            for target in ('main.VideoCalibrationControls', 'main.playback_key', 'main.cv2.imshow',
                           'main.cv2.createTrackbar', 'main.cv2.setMouseCallback'):
                stack.enter_context(patch(target, side_effect=AssertionError(f'Fast mode called {target}')))
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            forecast = main.main()
        self.assertEqual(capture.read.call_count, 4)
        self.assertEqual(capture.grab.call_count, 16)
        self.assertEqual(model.track.call_count, 4)
        self.assertEqual(model.track.call_args.args[0].shape, (457, 457, 3))
        self.assertEqual(model.track.call_args.kwargs['imgsz'], 480)
        self.assertEqual([call.args[2] for call in tracker.update_vehicle_speed.call_args_list], [0, .2, 300+5/30, 600+5/30])
        for call in tracker.update_vehicle_speed.call_args_list:
            np.testing.assert_array_equal(call.args[1], [200, 400, 220, 420])
        np.testing.assert_array_equal(raw_boxes, [[128, 228, 148, 248]])
        predictor.assert_called_once()
        np.testing.assert_array_equal(predictor.call_args.args[0], [20, 40, 60])
        self.assertEqual(forecast['predicted_speed'], 25)
        self.assertIn('10-15 minutes: average 30.000 mph; scaled 60.000', output.getvalue())
        result.plot.assert_not_called()
        capture.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
