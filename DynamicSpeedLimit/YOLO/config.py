"""Video, calibration, and speed-estimation settings."""
from pathlib import Path

YOLO_DIR = Path(__file__).resolve().parent
VIDEO_PATH = YOLO_DIR.parent / "traffic" / "traffic4.mp4"
YOLO_MODEL_PATH = YOLO_DIR / "yolov8n.pt"
CALIBRATION_PATH = YOLO_DIR / "calibration.json"
VIDEO_DEFAULTS_PATH = YOLO_DIR / "video_defaults.json"
WINDOW_NAME = "Stabilized Dynamic Speed Sign System"
VIDEO_W = 960
VIDEO_H = 540
DISPLAY_HEIGHT = 540
SIGN_WIDTH = 400

# Default prompt value; each video's input/preset limit stays fixed during a run.
DEFAULT_SPEED_LIMIT = 30

# Advisory output is separate from the fixed input sign. Explicit demo assumptions:
# neither free-flow speed nor the legal posted limit has been independently verified.
ADVISORY_FREE_FLOW_SPEED_MPH = 55.0
ADVISORY_ROAD_TYPE = "unknown"
ADVISORY_POLICY = {
    "normal_flow_ratio": 0.80, "moderate_flow_ratio": 0.60,
    "normal_reduction_mph": 0.0, "moderate_reduction_mph": 10.0, "severe_reduction_mph": 20.0,
    "rain_reduction_mph": 10.0, "severe_weather_reduction_mph": 20.0,
    "increment_mph": 5.0, "minimum_supported_limit_mph": 45.0,
    "allow_unsupported_road_demo": False,
    "freezing_temperature_f": 32.0, "low_visibility_m": 500.0,
    "weather_max_age_seconds": 1800.0, "prediction_max_age_seconds": 300.0,
    "future_tolerance_seconds": 60.0,
}
WEATHER_LOCATION = {"name": "Stony Brook, New York", "latitude": 40.91, "longitude": -73.14}
WEATHER_TIMEOUT = (2.0, 3.0)  # Connect/read seconds; only requested after prediction.
WEATHER_CACHE_SECONDS = 600.0
WEATHER_LOCATION_TOLERANCE_DEGREES = 0.25
WEATHER_EXPECTED_INTERVAL_SECONDS = 900

FRAME_SKIP = 2
# Fast simulation samples uniformly across the entire source, without playback waits.
SIMULATION_TARGET_FPS = 5.0
SIMULATION_ROAD_PADDING = 128  # Processing pixels; keep vehicle bodies/context above road contacts.
DETECTION_CONFIDENCE = 0.25
DETECTION_IMAGE_SIZE = 960
MAX_PROCESSING_DIMENSION = 960
VEHICLE_CLASSES = [2, 3, 5, 7]

# Prefer presentation timestamps, including for variable-frame-rate videos.
# Set to "fps" ONLY for footage known to have a constant frame rate.
TIMESTAMP_SOURCE = "video"
# Optional verified constant FPS; used only with TIMESTAMP_SOURCE = "fps".
FPS_OVERRIDE = None

# Distances below are meters; timing is VIDEO seconds, independent of YOLO runtime.
TRACK_WINDOW_SECONDS = 1.0
MIN_TRACK_SECONDS = 0.5
MIN_TRACK_SAMPLES = 3
MAX_TRACK_GAP_SECONDS = 0.5
MAX_INACTIVE_SECONDS = 3.0
MAX_REASONABLE_SPEED_MPH = 120.0
# Allow some bounding-box noise when checking for impossible position jumps.
POSITION_JUMP_TOLERANCE_METERS = 0.5
