# config.py

# Video & Window Settings
VIDEO_PATH = "traffic/traffic3.mp4"
WINDOW_NAME = "Stabilized Dynamic Speed Sign System"
VIDEO_W = 960
VIDEO_H = 540
SIGN_WIDTH = 400

# Speed Sign Thresholds
DEFAULT_SPEED_LIMIT = 30  # Baseline limit (MPH)
DENSE_TRAFFIC_LIMIT = 20  # Reduced limit when crowded (MPH)
DENSE_TRAFFIC_THRESHOLD = 3  # Car count trigger for dense traffic

# Perspective & Velocity Physics
PIXELS_PER_METER = 6.5  # Scale factor on bird's-eye transformed plane
DST_POINTS = [[0, 0], [200, 0], [200, 600], [0, 600]]  # Virtual grid plane

# Stability & Performance Controls
ALPHA = 0.3  # Position EMA smoothing factor (0.1 = heavy, 1.0 = raw)
MIN_PIXEL_SHIFT = 2.0  # Ignore movement below this pixel distance
FRAME_SKIP = 2  # Run YOLO detection every N frames
SPEED_QUEUE_DEPTH = 5  # Moving average window length for velocity