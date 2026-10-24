from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO

# Global list to hold calibration points
src_points = []


def select_points(event, x, y, flags, param):
    """Mouse callback function to record 4 calibration clicks."""
    if event == cv2.EVENT_LBUTTONDOWN and len(src_points) < 4:
        src_points.append([x, y])
        print(f"Point {len(src_points)} selected at: ({x}, {y})")


# 1. Load YOLO Model
model = YOLO("yolov8n.pt")

WINDOW_NAME = "Dynamic Speed Sign - Performance & Speed Calibrated"
cap = cv2.VideoCapture("traffic/traffic3.mp4")

# Automatically detect video Framerate (FPS)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or np.isnan(fps):
    fps = 30.0  # Fallback standard FPS

FRAME_TIME = 1.0 / fps

ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read video file.")
    cap.release()
    exit()

# 2. Interactive Calibration Phase
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 960, 540)
cv2.setMouseCallback(WINDOW_NAME, select_points)

print("\n=== CALIBRATION REQUIRED ===")
print(
    "Click 4 points on the road in order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left"
)

while len(src_points) < 4:
    display_frame = first_frame.copy()

    for i, pt in enumerate(src_points):
        cv2.circle(display_frame, tuple(pt), 7, (0, 0, 255), -1)
        if i > 0:
            cv2.line(display_frame, tuple(src_points[i - 1]), tuple(pt), (0, 0, 255), 2)

    if len(src_points) == 4:
        cv2.line(
            display_frame, tuple(src_points[3]), tuple(src_points[0]), (0, 0, 255), 2
        )

    cv2.putText(
        display_frame,
        f"Click point {len(src_points) + 1}/4 on road corners",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )
    cv2.imshow(WINDOW_NAME, display_frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Remove mouse listener
cv2.setMouseCallback(WINDOW_NAME, lambda *args: None)

# 3. Compute Perspective Transform Matrix
SRC_POINTS = np.float32(src_points)
DST_POINTS = np.float32([[0, 0], [200, 0], [200, 600], [0, 600]])
M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)


def transform_point(x, y, matrix):
    """Transforms 2D screen coordinates to bird's-eye view plane."""
    pt = np.array([[[x, y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, matrix)
    return transformed[0][0][0], transformed[0][0][1]


# 4. Calibration & Tracking Setup
# ADJUSTED: Reduced from 20.0 to 6.5 so a ~10 MPH output scales up 3x to ~30 MPH
PIXELS_PER_METER = 6.5

# Performance knob: Process model inference every N frames (1 = every frame, 2 = every 2nd frame)
FRAME_SKIP = 2
frame_count = 0

track_history = {}
speed_queues = {}  # Smooths speed jumps using rolling average

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run inference only on target skipped frames to boost processing FPS
    if frame_count % FRAME_SKIP == 0:
        results = model.track(
            frame,
            classes=[2, 3, 5, 7],
            conf=0.4,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,  # Suppress console log output to save CPU cycles
        )

        annotated_frame = results[0].plot(labels=False)

        # Draw calibration boundary
        cv2.polylines(
            annotated_frame,
            [np.int32(src_points)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box

                cx = int((x1 + x2) / 2)
                cy = int(y2)

                flat_cx, flat_cy = transform_point(cx, cy, M)

                raw_speed_mph = 0.0

                if track_id in track_history:
                    prev_flat_cx, prev_flat_cy, prev_frame_num = track_history[track_id]

                    # Account for skipped frames in time delta
                    effective_dt = (frame_count - prev_frame_num) * FRAME_TIME

                    pixel_distance = (
                        (flat_cx - prev_flat_cx) ** 2 + (flat_cy - prev_flat_cy) ** 2
                    ) ** 0.5

                    meters = pixel_distance / PIXELS_PER_METER
                    meters_per_sec = meters / effective_dt
                    raw_speed_mph = meters_per_sec * 2.23694

                track_history[track_id] = (flat_cx, flat_cy, frame_count)

                # Smooth speed using rolling queue across frames
                if track_id not in speed_queues:
                    speed_queues[track_id] = deque(maxlen=5)

                if raw_speed_mph > 0:
                    speed_queues[track_id].append(raw_speed_mph)

                display_speed = (
                    sum(speed_queues[track_id]) / len(speed_queues[track_id])
                    if len(speed_queues[track_id]) > 0
                    else 0.0
                )

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

                # Dark background pill
                cv2.rectangle(
                    annotated_frame,
                    (tx, ty - text_h - 10),
                    (tx + text_w + 12, ty + baseline + 5),
                    (0, 0, 0),
                    -1,
                )

                # Prominent yellow text
                cv2.putText(
                    annotated_frame,
                    label_text,
                    (tx + 6, ty - 2),
                    font,
                    font_scale,
                    (0, 255, 255),
                    thickness,
                )

        cv2.imshow(WINDOW_NAME, annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()