
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

DATA_PATH = Path(__file__).resolve().parent / "data" / "METR-LA.csv"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_STEPS = 3
FORECAST_STEPS = 2
TIME_STEP_MINUTES = 5
CONGESTION_THRESHOLD_MPH = 40.0


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError(f"Dataset is empty: {DATA_PATH}")

    if "Unnamed: 0" in df.columns:
        df["Unnamed: 0"] = pd.to_datetime(df["Unnamed: 0"], errors="coerce")

    df = df.sort_values("Unnamed: 0").reset_index(drop=True)
    return df


def make_training_examples(
    df: pd.DataFrame,
    history_steps: int = HISTORY_STEPS,
    forecast_steps: int = FORECAST_STEPS,
):
    sensor_cols = [col for col in df.columns if col != "Unnamed: 0"]
    values = df[sensor_cols].to_numpy(dtype=float)

    X, y_speed = [], []

    for sensor_idx in range(values.shape[1]):
        for start_idx in range(0, len(values) - history_steps - forecast_steps + 1):
            history = values[start_idx : start_idx + history_steps, sensor_idx]
            target_idx = start_idx + history_steps + forecast_steps - 1
            target = values[target_idx, sensor_idx]

            if np.isnan(history).any() or np.isnan(target):
                continue

            X.append(history)
            y_speed.append(float(target))

    return np.asarray(X, dtype=float), np.asarray(y_speed, dtype=float)


def train_models() -> None:
    df = load_dataset()

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train, y_speed_train = make_training_examples(train_df)
    X_test, y_speed_test = make_training_examples(test_df)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Not enough valid training samples could be created from the chronological split.")

    y_congestion_train = (y_speed_train < CONGESTION_THRESHOLD_MPH).astype(int)
    y_congestion_test = (y_speed_test < CONGESTION_THRESHOLD_MPH).astype(int)

    speed_model = lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.05,
        n_estimators=300,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )
    speed_model.fit(X_train, y_speed_train)

    congestion_model = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        n_estimators=300,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    congestion_model.fit(X_train, y_congestion_train)

    speed_preds = speed_model.predict(X_test)
    congestion_preds = congestion_model.predict(X_test)

    mae = mean_absolute_error(y_speed_test, speed_preds)
    r2 = r2_score(y_speed_test, speed_preds)
    clf_acc = accuracy_score(y_congestion_test, congestion_preds)

    joblib.dump(speed_model, MODEL_DIR / "speed_model.joblib")
    joblib.dump(congestion_model, MODEL_DIR / "congestion_model.joblib")

    print(f"Dataset rows: {len(df)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Input window: {HISTORY_STEPS} time steps")
    print(f"Forecast horizon: {FORECAST_STEPS * TIME_STEP_MINUTES} minutes")
    print(f"Congestion threshold: {CONGESTION_THRESHOLD_MPH:.1f} mph")
    print(f"Speed model MAE: {mae:.3f}")
    print(f"Speed model R^2: {r2:.3f}")
    print(f"Congestion model accuracy: {clf_acc:.3f}")
    print(f"Saved speed model to: {MODEL_DIR / 'speed_model.joblib'}")
    print(f"Saved congestion model to: {MODEL_DIR / 'congestion_model.joblib'}")


if __name__ == "__main__":
    train_models()
