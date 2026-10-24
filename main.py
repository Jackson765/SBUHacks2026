from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
SPEED_LIMIT = 30  # Default baseline speed limit in MPH
PIXELS_PER_METER = 6.5  # Transformed plane scale factor

# Noise & Smoothing Controls
ALPHA = 0.3  # EMA position smoothing factor (0.1 = heavy smoothing, 1.0 = raw input)
MIN_PIXEL_SHIFT = 2.0  # Ignore movement jitter under 2 transformed pixels
FRAME_SKIP = 2  # Process model every N frames

src_points = []


def select_points(event, x, y, flags, param):
    """Mouse callback function to record 4 calibration clicks on video side."""
    if event == cv2.EVENT_LBUTTONDOWN and len(src_points) < 4:
        if x < 960:
            src_points.append([x, y])
            print(f"Point {len(src_points)} selected at: ({x}, {y})")


def create_dynamic_speed_sign(
    current_max_speed, speed_limit=SPEED_LIMIT, width=400, height=540
):
    """Generates the dynamic digital speed limit dashboard panel."""
    canvas = np.full((height, width, 3), (25, 25, 25), dtype=np.uint8)

    sign_margin = 30
    cv2.rectangle(
        canvas,
        (sign_margin, 20),
        (width - sign_margin, height - 20),
        (240, 240, 240),
        -1,
    )
    cv2.rectangle(
        canvas,
        (sign_margin + 8, 28),
        (width - sign_margin - 8, height - 28),
        (10, 10, 10),
        4,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas, "SPEED", (width // 2 - 75, 85), font, 1.3, (10, 10, 10), 4
    )
    cv2.putText(
        canvas, "LIMIT", (width // 2 - 70, 140), font, 1.3, (10, 10, 10), 4
    )
    cv2.putText(
        canvas, f"{speed_limit}", (width // 2 - 80, 300), font, 3.8, (10, 10, 10), 10
    )

    panel_y1, panel_y2 = 350, height - 45
    cv2.rectangle(
        canvas,
        (sign_margin + 20, panel_y1),
        (width - sign_margin - 20, panel_y2),
        (0, 0, 0),
        -1,
    )

    is_speeding = current_max_speed > speed_limit
    led_color = (0, 0, 255) if is_speeding else (0, 255, 0)

    speed_text = f"{int(current_max_speed)}"
    text_size, _ = cv2.getTextSize(speed_text, font, 2.5, 6)
    text_x = (width - text_size[0]) // 2
    cv2.putText(
        canvas, speed_text, (text_x, panel_y1 + 80), font, 2.5, led_color, 6
    )

    unit_text = "YOUR SPEED"
    unit_size, _ = cv2.getTextSize(unit_text, font, 0.6, 2)
    unit_x = (width - unit_size[0]) // 2
    cv2.putText(
        canvas,
        unit_text,
        (unit_x, panel_y1 + 115),
        font,
        0.6,
        (200, 200, 200),
        2,
    )

    return canvas


# 1. Initialize Detection Model & Video
model = YOLO("yolov8n.pt")

WINDOW_NAME = "Stabilized Dynamic Speed Sign System"
cap = cv2.VideoCapture("traffic/traffic3.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or np.isnan(fps):
    fps = 30.0
FRAME_TIME = 1.0 / fps

ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read video file.")
    cap.release()
    exit()

VIDEO_W, VIDEO_H = 960, 540
first_frame = cv2.resize(first_frame, (VIDEO_W, VIDEO_H))

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1360, 540)
cv2.setMouseCallback(WINDOW_NAME, select_points)

print("\n=== CALIBRATION REQUIRED ===")
print(
    "Click 4 points on the road in order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left"
)

# 2. Interactive Calibration Loop
while len(src_points) < 4:
    display_video = first_frame.copy()

    for i, pt in enumerate(src_points):
        cv2.circle(display_video, tuple(pt), 7, (0, 0, 255), -1)
        if i > 0:
            cv2.line(display_video, tuple(src_points[i - 1]), tuple(pt), (0, 0, 255), 2)

    if len(src_points) == 4:
        cv2.line(
            display_video, tuple(src_points[3]), tuple(src_points[0]), (0, 0, 255), 2
        )

    cv2.putText(
        display_video,
        f"Click point {len(src_points) + 1}/4 on road corners",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    sign_canvas = create_dynamic_speed_sign(0, SPEED_LIMIT, width=400, height=VIDEO_H)
    cv2.imshow(
        WINDOW_NAME, np.hstack((display_video, sign_canvas))
    )

    if cv2.waitKey(10) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.setMouseCallback(WINDOW_NAME, lambda *args: None)

# 3. Perspective Warp Setup
SRC_POINTS = np.float32(src_points)
DST_POINTS = np.float32([[0, 0], [200, 0], [200, 600], [0, 600]])
M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)


def transform_point(x, y, matrix):
    """Transforms 2D screen coordinates to bird's-eye view plane."""
    pt = np.array([[[x, y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, matrix)
    return transformed[0][0][0], transformed[0][0][1]


frame_count = 0
track_history = {}
speed_queues = {}

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# 4. Main Processing Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
    frame_count += 1

    if frame_count % FRAME_SKIP == 0:
        results = model.track(
            frame,
            classes=[2, 3, 5, 7],
            conf=0.4,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        annotated_frame = results[0].plot(labels=False)

        # Draw calibration box (green)
        cv2.polylines(
            annotated_frame,
            [np.int32(src_points)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

        current_frame_max_speed = 0.0

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Dynamic speed limit adjustment based on traffic density
            current_speed_limit = 20 if len(track_ids) >= 3 else 30

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box

                # Bottom-center point (where tires hit road)
                cx = int((x1 + x2) / 2)
                cy = int(y2)

                raw_flat_cx, raw_flat_cy = transform_point(cx, cy, M)

                raw_speed_mph = 0.0

                if track_id in track_history:
                    prev_flat_cx, prev_flat_cy, prev_frame_num = track_history[track_id]

                    # Exponential Moving Average (EMA) position smoothing
                    flat_cx = ALPHA * raw_flat_cx + (1 - ALPHA) * prev_flat_cx
                    flat_cy = ALPHA * raw_flat_cy + (1 - ALPHA) * prev_flat_cy

                    effective_dt = (frame_count - prev_frame_num) * FRAME_TIME
                    pixel_distance = (
                        (flat_cx - prev_flat_cx) ** 2
                        + (flat_cy - prev_flat_cy) ** 2
                    ) ** 0.5

                    # Apply minimum movement threshold to filter jitter
                    if pixel_distance >= MIN_PIXEL_SHIFT and effective_dt > 0:
                        meters = pixel_distance / PIXELS_PER_METER
                        meters_per_sec = meters / effective_dt
                        raw_speed_mph = meters_per_sec * 2.23694
                else:
                    flat_cx, flat_cy = raw_flat_cx, raw_flat_cy

                track_history[track_id] = (flat_cx, flat_cy, frame_count)

                if track_id not in speed_queues:
                    speed_queues[track_id] = deque(maxlen=5)

                if raw_speed_mph > 0:
                    speed_queues[track_id].append(raw_speed_mph)

                display_speed = (
                    sum(speed_queues[track_id]) / len(speed_queues[track_id])
                    if len(speed_queues[track_id]) > 0
                    else 0.0
                )

                if display_speed > current_frame_max_speed:
                    current_frame_max_speed = display_speed

                label_text = (
                    f"{int(display_speed)} MPH" if display_speed > 1.0 else "0 MPH"
                )

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 3

                text_size, baseline = cv2.getTextSize(
                    label_text, font, font_scale, thickness
                )
                text_w, text_h = text_size
                tx, ty = int(x1), int(y1) - 15

                cv2.rectangle(
                    annotated_frame,
                    (tx, ty - text_h - 10),
                    (tx + text_w + 12, ty + baseline + 5),
                    (0, 0, 0),
                    -1,
                )

                text_color = (
                    (0, 0, 255)
                    if display_speed > current_speed_limit
                    else (0, 255, 255)
                )

                cv2.putText(
                    annotated_frame,
                    label_text,
                    (tx + 6, ty - 2),
                    font,
                    font_scale,
                    text_color,
                    thickness,
                )

        speed_sign_canvas = create_dynamic_speed_sign(
            current_frame_max_speed,
            current_speed_limit,
            width=400,
            height=VIDEO_H,
        )

        cv2.imshow(
            WINDOW_NAME, np.hstack((annotated_frame, speed_sign_canvas))
        )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()