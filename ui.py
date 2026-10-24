# ui.py
import cv2
import numpy as np
from config import DEFAULT_SPEED_LIMIT


def create_dynamic_speed_sign(
    current_max_speed: float,
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


def draw_speed_label(
    frame: np.ndarray, bbox: tuple, speed_mph: float, is_speeding: bool
):
    """Draws a high-visibility speed overlay box above a vehicle bounding box."""
    x1, y1, _, _ = bbox
    label_text = f"{int(speed_mph)} MPH" if speed_mph > 1.0 else "0 MPH"

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
    text_color = (0, 0, 255) if is_speeding else (0, 255, 255)

    cv2.putText(
        frame,
        label_text,
        (tx + 6, ty - 2),
        font,
        font_scale,
        text_color,
        thickness,
    )