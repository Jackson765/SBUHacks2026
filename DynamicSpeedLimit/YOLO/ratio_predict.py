"""Scale three five-minute average speeds using a user-entered ratio."""
import argparse
from functools import lru_cache
import importlib.util
import json
from pathlib import Path

import numpy as np


def validate_ratio(ratio):
    ratio = float(ratio)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("Ratio must be positive and finite (for example, 2).")
    return ratio


def prompt_ratio():
    while True:
        raw = input("Speed scaling ratio (e.g. 2; Q to quit): ").strip()
        if raw.lower() == "q":
            raise KeyboardInterrupt
        try:
            return validate_ratio(raw)
        except ValueError as error:
            print(error)


def _three_speeds(values):
    speeds = np.asarray(values, dtype=float).reshape(-1)
    if speeds.size != 3 or not np.isfinite(speeds).all() or (speeds < 0).any():
        raise ValueError("Exactly three finite, nonnegative average speeds are required.")
    return speeds


def scale_average_speeds(average_speeds, ratio):
    ratio = validate_ratio(ratio)
    with np.errstate(over="ignore"):
        scaled = _three_speeds(average_speeds) * ratio
    if not np.isfinite(scaled).all():
        raise ValueError("Scaled average speeds overflow; use a smaller ratio.")
    return scaled


@lru_cache(maxsize=1)
def _model_b_predictor():
    # Reuse the existing predictor; no dataset statistics or training changes.
    path = Path(__file__).resolve().parent.parent / "model_b" / "predict.py"
    spec = importlib.util.spec_from_file_location("ratio_adapter_model_b", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.predict_next_step


def predict_scaled_averages(scaled_average_speeds, ratio, *, predictor=None):
    """Call Model B with already-scaled values, then return its forecast / ratio.

    Inputs are oldest to newest, representing three completed five-minute bins.
    The model forecasts ten minutes after the final bin's endpoint.
    """
    ratio = validate_ratio(ratio)
    scaled = _three_speeds(scaled_average_speeds)
    model_predict = _model_b_predictor() if predictor is None else predictor
    model_speed = float(model_predict(scaled.copy())["predicted_speed"])
    if not np.isfinite(model_speed) or model_speed < 0:
        raise ValueError("Model B returned an invalid predicted speed.")
    local_speed = model_speed / ratio
    if not np.isfinite(local_speed):
        raise ValueError("Unscaled prediction overflowed; check the ratio.")
    return {
        "ratio": ratio,
        "scaled_average_speeds": scaled.tolist(),
        "scaled_predicted_speed": model_speed,
        "predicted_speed": local_speed,
    }


def predict_with_ratio(average_speeds, ratio, *, predictor=None):
    """Scale raw five-minute averages exactly once and return a local-mph forecast."""
    scaled = scale_average_speeds(average_speeds, ratio)
    return predict_scaled_averages(scaled, ratio, predictor=predictor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratio", type=float, help="Positive speed multiplier; prompted if omitted")
    parser.add_argument("--history", type=float, nargs=3, metavar=("FIRST", "SECOND", "THIRD"),
                        help="Three five-minute average speeds in local mph")
    args = parser.parse_args()
    ratio = prompt_ratio() if args.ratio is None else validate_ratio(args.ratio)
    history = args.history
    if history is None:
        history = [float(value) for value in input("Three average speeds, oldest first (e.g. 25 28 30): ").split()]
    print(json.dumps(predict_with_ratio(history, ratio), indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
