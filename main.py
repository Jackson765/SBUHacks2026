# main.py
import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    DEFAULT_SPEED_LIMIT,
    DENSE_TRAFFIC_LIMIT,
    DENSE_TRAFFIC_THRESHOLD,
    FRAME_SKIP,
    SIGN_WIDTH,
    VIDEO_H,
    VIDEO_PATH,
    VIDEO_W,
    WINDOW_NAME,
)
from tracker import SpeedTracker
from ui import create_dynamic_speed_sign, draw_speed_label

# Interactive calibration container
src_points = []


def select_points(event, x, y, flags, param):
    """Mouse callback to capture 4 perspective corner points."""
    if event == cv2.EVENT_LBUTTONDOWN and len(src_points) < 4:
        if x < VIDEO_W:
            src_points.append([x, y])
            print(f"Point {len(src_points)} selected at: ({x}, {y})")


def run_calibration(cap: cv2.VideoCapture, first_frame: np.ndarray):
    """Loops until 4 road points are selected by the user."""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, VIDEO_W + SIGN_WIDTH, VIDEO_H)
    cv2.setMouseCallback(WINDOW_NAME, select_points)

    print("\n=== CALIBRATION REQUIRED ===")
    print(
        "Click 4 points on the road in order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left"
    )

    while len(src_points) < 4:
        display_video = first_frame.copy()

        for i, pt in enumerate(src_points):
            cv2.circle(display_video, tuple(pt), 7, (0, 0, 255), -1)
            if i > 0:
                cv2.line(
                    display_video,
                    tuple(src_points[i - 1]),
                    tuple(pt),
                    (0, 0, 255),
                    2,
                )

        if len(src_points) == 4:
            cv2.line(
                display_video,
                tuple(src_points[3]),
                tuple(src_points[0]),
                (0, 0, 255),
                2,
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

        sign_canvas = create_dynamic_speed_sign(
            0, DEFAULT_SPEED_LIMIT, width=SIGN_WIDTH, height=VIDEO_H
        )
        cv2.imshow(WINDOW_NAME, np.hstack((display_video, sign_canvas)))

        if cv2.waitKey(10) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cv2.setMouseCallback(WINDOW_NAME, lambda *args: None)


def main():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    frame_time = 1.0 / fps

    ret, first_frame = cap.read()
    if not ret:
        print(f"Error: Could not read video file at '{VIDEO_PATH}'.")
        cap.release()
        return

    first_frame = cv2.resize(first_frame, (VIDEO_W, VIDEO_H))

    # Run calibration
    run_calibration(cap, first_frame)

    # Initialize Tracker with calibrated points
    tracker = SpeedTracker(src_points)

    frame_count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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

            # Draw calibration polygon
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

                # Dynamic speed limit threshold
                current_speed_limit = (
                    DENSE_TRAFFIC_LIMIT
                    if len(track_ids) >= DENSE_TRAFFIC_THRESHOLD
                    else DEFAULT_SPEED_LIMIT
                )

                for box, track_id in zip(boxes, track_ids):
                    display_speed = tracker.update_vehicle_speed(
                        track_id, box, frame_count, frame_time
                    )

                    if display_speed > current_frame_max_speed:
                        current_frame_max_speed = display_speed

                    is_speeding = display_speed > current_speed_limit
                    draw_speed_label(
                        annotated_frame, box, display_speed, is_speeding
                    )
            else:
                current_speed_limit = DEFAULT_SPEED_LIMIT

            speed_sign_canvas = create_dynamic_speed_sign(
                current_frame_max_speed,
                current_speed_limit,
                width=SIGN_WIDTH,
                height=VIDEO_H,
            )

            cv2.imshow(
                WINDOW_NAME, np.hstack((annotated_frame, speed_sign_canvas))
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()