"""Manual ratio: preserve input units and scale/unscale exactly once."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ratio_predict import (prompt_ratio, predict_with_ratio, predict_scaled_averages,
                           scale_average_speeds, validate_ratio)


class RatioPredictionTests(unittest.TestCase):
    def test_manual_ratio_scales_inputs_and_unscales_forecast(self):
        model = Mock(return_value={"predicted_speed": 55, "congestion_probability": .7})
        history = np.array([25., 28., 30.])
        result = predict_with_ratio(history, 2, predictor=model)
        np.testing.assert_array_equal(model.call_args.args[0], [50, 56, 60])
        np.testing.assert_array_equal(history, [25, 28, 30])
        self.assertEqual(result["ratio"], 2)
        self.assertEqual(result["predicted_speed"], 27.5)
        self.assertNotIn("congestion_probability", result)

    def test_already_scaled_values_are_not_scaled_twice(self):
        model = Mock(return_value={"predicted_speed": 55})
        result = predict_scaled_averages([50, 56, 60], 2, predictor=model)
        np.testing.assert_array_equal(model.call_args.args[0], [50, 56, 60])
        self.assertEqual(result["predicted_speed"], 27.5)

    def test_small_ratio_identity_and_zero_speeds(self):
        for ratio, history, prediction, expected in [
            (.5, [30, 40, 50], 40, 80), (1, [30, 40, 50], 40, 40),
            (2, [0, 0, 0], 0, 0),
        ]:
            model = Mock(return_value={"predicted_speed": prediction})
            result = predict_with_ratio(history, ratio, predictor=model)
            np.testing.assert_array_equal(model.call_args.args[0], np.array(history) * ratio)
            self.assertEqual(result["predicted_speed"], expected)

    def test_invalid_ratios(self):
        model = Mock()
        for ratio in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                predict_with_ratio([25, 28, 30], ratio, predictor=model)
        model.assert_not_called()

    def test_invalid_inputs_and_overflow(self):
        for history in ([], [1, 2], [1, 2, 3, 4], [1, -2, 3], [1, float("nan"), 3]):
            with self.assertRaises(ValueError):
                scale_average_speeds(history, 2)
        with self.assertRaises(ValueError):
            scale_average_speeds([25, 28, 30], 1e308)

    def test_invalid_model_output(self):
        for value in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                predict_scaled_averages([50, 56, 60], 2,
                                        predictor=lambda history: {"predicted_speed": value})

    def test_prompt_retries_and_cancels(self):
        with patch("builtins.input", side_effect=["0", "nan", "abc", "2.5"]):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(prompt_ratio(), 2.5)
        with patch("builtins.input", return_value="q"):
            with self.assertRaises(KeyboardInterrupt):
                prompt_ratio()
        self.assertEqual(validate_ratio("2"), 2)


if __name__ == "__main__":
    unittest.main()
