# ui.py
import cv2
import numpy as np
from config import DEFAULT_SPEED_LIMIT, DISPLAY_HEIGHT, SIGN_WIDTH


def resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    """Resize a frame to a fixed display height without changing its aspect ratio."""
    source_height, source_width = frame.shape[:2]
    width = max(1, round(source_width * height / source_height))
    interpolation = cv2.INTER_AREA if height < source_height else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def create_dynamic_speed_sign(
    average_speed,
    speed_limit: int = DEFAULT_SPEED_LIMIT,
    width: int = 400,
    height: int = 540,
) -> np.ndarray:
    """Renders the digital speed limit dashboard side panel."""
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
    first_label, second_label = "SPEED", "LIMIT"
    first_size, _ = cv2.getTextSize(first_label, font, 1.3, 4)
    second_size, _ = cv2.getTextSize(second_label, font, 1.3, 4)
    cv2.putText(
        canvas, first_label, ((width - first_size[0]) // 2, 85), font, 1.3, (10, 10, 10), 4
    )
    cv2.putText(
        canvas, second_label, ((width - second_size[0]) // 2, 140), font, 1.3, (10, 10, 10), 4
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

    is_speeding = average_speed is not None and average_speed > speed_limit
    led_color = ((180, 180, 180) if average_speed is None
                 else (0, 0, 255) if is_speeding else (0, 255, 0))

    speed_text = "--" if average_speed is None else f"{average_speed:.0f}"
    text_size, _ = cv2.getTextSize(speed_text, font, 2.5, 6)
    text_x = (width - text_size[0]) // 2
    cv2.putText(
        canvas, speed_text, (text_x, panel_y1 + 80), font, 2.5, led_color, 6
    )

    unit_text = "CURRENT AVG (MPH)"
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


def format_advisory(decision, weather):
    """Brief optional weather output; retain availability and hazard reasons."""
    location = (weather.get("location") or {}).get("name", "unknown area")
    scenario = "simulated" if weather.get("scenario") == "simulated" else "current weather applied to recorded video"
    lines = [f"Weather: {decision['weather_category']} | {location} | {scenario}."]
    advisory = decision["advisory_speed_mph"]
    value = f"{advisory:g} mph" if advisory is not None else "unavailable"
    lines.append(f"Advisory: {value} | {decision['status']}.")
    if advisory is None and decision["candidate_speed_mph"] is not None:
        lines.append(f"Traffic-only candidate: {decision['candidate_speed_mph']:g} mph; weather unavailable.")
    details = decision["reasons"] + decision["warnings"]
    if details:
        lines.append("; ".join(dict.fromkeys(details)))
    return "\n".join(lines)


def compose_playback(annotated, vehicles, base_limit, elapsed, completed_windows, ratio, *, controls=None, paused=False):
    """Compose a preview only after a frame, calibration, or pause-state change."""
    canvas = annotated.copy()
    speeds = [speed for _, _, speed in vehicles if speed is not None]
    average = float(np.mean(speeds)) if speeds else None
    for box, _, speed in vehicles:
        draw_speed_label(canvas, box, speed, speed is not None and speed > base_limit)
    panel = create_dynamic_speed_sign(average, base_limit, width=SIGN_WIDTH, height=DISPLAY_HEIGHT)
    display_frame = resize_to_height(canvas, DISPLAY_HEIGHT)
    seconds = int(elapsed)
    status = f"{seconds//60:02d}:{seconds%60:02d}/15:00 | {completed_windows}/3 | x{ratio:g}"
    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(display_frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                .45, (255, 255, 255), 1, cv2.LINE_AA)
    if controls is not None:
        controls.draw_points(display_frame)
        return np.vstack((np.hstack((display_frame, panel)), controls.footer(average, paused)))
    return np.hstack((display_frame, panel))


def draw_speed_label(
    frame: np.ndarray, bbox: tuple, speed_mph, is_speeding: bool
):
    """Draws a high-visibility speed overlay box above a vehicle bounding box."""
    x1, y1, _, _ = bbox
    label_text = "-- MPH" if speed_mph is None else f"{speed_mph:.1f} MPH"

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
        frame,
        (tx, ty - text_h - 10),
        (tx + text_w + 12, ty + baseline + 5),
        (0, 0, 0),
        -1,
    )

    # Text color (Red if speeding, Yellow if compliant)
    text_color = ((180, 180, 180) if speed_mph is None
                  else (0, 0, 255) if is_speeding else (0, 255, 255))

    cv2.putText(
        frame,
        label_text,
        (tx + 6, ty - 2),
        font,
        font_scale,
        text_color,
        thickness,
    )
