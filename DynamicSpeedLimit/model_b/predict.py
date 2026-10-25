from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "models"


def load_models():
    speed_model = joblib.load(MODEL_DIR / "speed_model.joblib")
    congestion_model = joblib.load(MODEL_DIR / "congestion_model.joblib")
    return speed_model, congestion_model


def predict_next_step(history_window):
    history_array = np.asarray(history_window, dtype=float).reshape(-1)

    if history_array.size == 0:
        raise ValueError("history_window must contain at least one value.")

    if history_array.size != 3:
        raise ValueError(f"YOLO-style input must contain exactly 3 values, received {history_array.size}.")

    speed_model, congestion_model = load_models()
    sample = history_array.reshape(1, -1)

    speed_pred = float(speed_model.predict(sample)[0])
    congestion_probability = float(congestion_model.predict_proba(sample)[0, 1])

    return {
        "predicted_speed": speed_pred,
        "congestion_probability": congestion_probability,
    }


if __name__ == "__main__":
    example_history = np.array([62.0, 57.0, 50.0], dtype=float)
    result = predict_next_step(example_history)
    print("Example YOLO-style input:", example_history.tolist())
    print("Prediction:", result)
