"""Existing runner -> one Model B prediction -> weather -> advisory console display."""
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import api_code
import main
from settings_fixture import store_with_defaults
from decision_engine import evaluate_advisory
from ui import create_dynamic_speed_sign, format_advisory
from test_decision_engine import NOW
from test_decision_engine import observation
from test_weather_api import api_payload


class AdvisoryIntegrationTests(unittest.TestCase):
    def run_app(self, *, enabled=True, weather_error=None, missing_window=False, short=False):
        times = [0, 100, 300, 400, 600, 700, 900]
        if short:
            times = times[:2]
        frame = np.zeros((540, 960, 3), np.uint8)
        state = {'index': -1}
        capture = Mock()
        capture.isOpened.return_value = True

        def read():
            state['index'] += 1
            return state['index'] < len(times), frame

        capture.read.side_effect = read
        capture.get.side_effect = lambda prop: 30 if prop == cv2.CAP_PROP_FPS else times[state['index']]*1000
        result = Mock()
        result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[250, 300, 300, 400]])
        result.boxes.id.int.return_value.cpu.return_value.tolist.return_value = [1]
        vehicle_model = Mock()
        vehicle_model.track.return_value = [result]
        tracker = Mock()
        tracker.point_in_road.return_value = True

        def speed(track_id, box, timestamp):
            window = int(timestamp // 300)
            return None if missing_window and window == 1 else [25, 28, 30][window]

        tracker.update_vehicle_speed.side_effect = speed
        predictor = Mock(return_value={'predicted_speed': 70., 'congestion_probability': .999})
        response = Mock(json=Mock(return_value=api_payload(weather_code=61, precipitation=.2, rain=.2)))

        def http_get(*args, **kwargs):
            # Assert it happens after the complete collection and the existing inference.
            self.assertEqual(state['index'], 6)
            self.assertEqual(predictor.call_count, 1)
            if weather_error:
                raise weather_error
            return response

        preset = dict(ratio=2, base_speed_limit=65, road_width_m=10, road_length_m=25,
                      frame_size=[960, 540], points=[[0, 0], [959, 0], [959, 539], [0, 539]])
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', ['main.py', '--full-analysis', '--video', __file__] + (['--weather'] if enabled else [])))
            stack.enter_context(patch('main.VideoSettingsStore', return_value=store_with_defaults({Path(__file__).name: preset})))
            stack.enter_context(patch('main.YOLO_MODEL_PATH', Path(__file__)))
            stack.enter_context(patch('main.FRAME_SKIP', 1))
            stack.enter_context(patch('main.cv2.VideoCapture', return_value=capture))
            stack.enter_context(patch('main.SpeedTracker', return_value=tracker))
            stack.enter_context(patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=Mock(return_value=vehicle_model))}))
            stack.enter_context(patch('ratio_predict._model_b_predictor', return_value=predictor))
            stack.enter_context(patch('api_code.utc_now', return_value=NOW))
            stack.enter_context(patch('api_code._client', api_code.OpenMeteoClient()))
            network = stack.enter_context(patch('api_code.requests.get', side_effect=http_get))
            engine = stack.enter_context(patch('decision_engine.evaluate_advisory', wraps=evaluate_advisory))
            display = stack.enter_context(patch('ui.format_advisory', wraps=format_advisory))
            stack.enter_context(patch('config.ADVISORY_FREE_FLOW_SPEED_MPH', 65))
            stack.enter_context(patch('config.ADVISORY_ROAD_TYPE', 'freeway'))
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            forecast = main.main()
        capture.release.assert_called_once()
        return forecast, output.getvalue(), predictor, network, engine, display

    def test_existing_prediction_is_unscaled_once_and_consumed_once_after_frame_loop(self):
        forecast, output, predictor, network, engine, display = self.run_app()
        predictor.assert_called_once()
        np.testing.assert_array_equal(predictor.call_args.args[0], [50, 56, 60])
        network.assert_called_once()
        engine.assert_called_once()
        self.assertEqual(engine.call_args.args[0], 35)
        self.assertEqual(engine.call_args.args[1:3], (65, 65))
        self.assertEqual(forecast['predicted_speed'], 35)
        display.assert_called_once()
        self.assertEqual(display.call_args.args[0]['advisory_speed_mph'], 45)
        self.assertIn('Weather: adverse', output)
        self.assertIn('Advisory: 45 mph', output)
        self.assertIn('current weather applied to recorded video', output)
        self.assertIn('not historically matched', output)

    def test_disabled_and_weather_failure_preserve_exact_forecast(self):
        disabled, output, predictor, network, engine, display = self.run_app(enabled=False)
        predictor.assert_called_once()
        network.assert_not_called()
        engine.assert_not_called()
        display.assert_not_called()
        self.assertNotIn('Advisory:', output)
        partial, output, predictor, network, engine, display = self.run_app(weather_error=requests.Timeout('timeout'))
        self.assertEqual(disabled, partial)
        predictor.assert_called_once()
        self.assertEqual(display.call_args.args[0]['status'], 'partial_data')
        self.assertIn('Traffic-only candidate: 45 mph', output)
        self.assertIsNone(display.call_args.args[0]['advisory_speed_mph'])

    def test_missing_history_never_fetches_weather_or_calls_engine_with_fake_prediction(self):
        for kwargs in [dict(missing_window=True), dict(short=True)]:
            forecast, output, predictor, network, engine, display = self.run_app(**kwargs)
            self.assertIsNone(forecast)
            predictor.assert_not_called()
            network.assert_not_called()
            engine.assert_not_called()
            display.assert_not_called()
            self.assertNotIn('Advisory:', output)

    def test_optional_display_layer_failure_does_not_erase_or_repeat_prediction(self):
        forecast, output, predictor, network, engine, display = self.run_app(weather_error=RuntimeError('optional client failure'))
        self.assertEqual(forecast['predicted_speed'], 35)
        predictor.assert_called_once()
        self.assertIn('Existing Model B result is retained', output)

    def test_missing_free_flow_configuration_displays_insufficient_data(self):
        weather = observation()
        decision = evaluate_advisory(35, None, 45, weather, 'freeway')
        output = format_advisory(decision, weather)
        self.assertIn('insufficient_data', output)
        self.assertIn('Advisory: unavailable', output)


if __name__ == '__main__':
    unittest.main()
