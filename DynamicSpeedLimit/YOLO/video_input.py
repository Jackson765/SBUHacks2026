"""Prerecorded video selection, resizing, timing, and playback events."""
from pathlib import Path
import cv2
import numpy as np
from config import MAX_PROCESSING_DIMENSION, TIMESTAMP_SOURCE, VIDEO_PATH


class VideoClock:
    """Require monotonic presentation times or an explicit constant-FPS mode."""
    def __init__(self, fps, source=TIMESTAMP_SOURCE):
        if source not in ("video", "fps"):
            raise ValueError("TIMESTAMP_SOURCE must be 'video' or 'fps'.")
        if source == "fps" and (not np.isfinite(fps) or fps <= 0):
            raise ValueError("Constant-FPS timing requires a valid verified FPS.")
        self.source = source
        self.fps = fps
        self.previous = None
        self.first = None
        self.first_index = None
        self.constant_frame_timing = bool(np.isfinite(fps) and fps > 0)

    def timestamp(self, frame_index, position_msec):
        value = frame_index / self.fps if self.source == "fps" else position_msec / 1000.0
        if (not np.isfinite(value) or value < 0
                or (self.previous is not None and value <= self.previous)):
            raise ValueError(
                "Video timestamps are missing or not increasing. For VERIFIED constant-frame-rate "
                "footage, set TIMESTAMP_SOURCE = 'fps' and, if needed, FPS_OVERRIDE in config.py."
            )
        if self.first is None:
            self.first, self.first_index = value, frame_index
        if self.constant_frame_timing:
            expected = self.first + (frame_index - self.first_index) / self.fps
            self.constant_frame_timing = abs(value - expected) < 1e-5
        self.previous = value
        return value

    def end_timestamp(self):
        """Account for the final frame's duration only with confirmed constant timing.

        A 15-minute CFR file's last frame typically STARTS just before 15:00.
        With variable/unknown frame durations, do not invent extra coverage.
        """
        if self.previous is None:
            return 0.0
        return self.previous + 1.0 / self.fps if self.constant_frame_timing else self.previous


def prompt_video_path():
    raw = input("Video file path (leave blank to cancel): ").strip().strip('"')
    return Path(raw).expanduser() if raw else None


def resize_for_processing(frame):
    """Downscale without distorting the source aspect ratio."""
    height, width = frame.shape[:2]
    scale = min(1.0, MAX_PROCESSING_DIMENSION / max(width, height))
    target_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if target_size == (width, height):
        return frame
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


def choose_video():
    """Open a native file picker; offer terminal input if Tk is unavailable."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("File picker is unavailable; enter the video path in the terminal.")
        return prompt_video_path()

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            parent=root,
            title="Choose a traffic video",
            initialdir=str(Path(VIDEO_PATH).parent),
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.webm *.mpeg *.mpg"),
                ("All files", "*.*"),
            ],
        )
        return Path(selected) if selected else None
    except tk.TclError as error:
        print(f"File picker is unavailable ({error}); enter the path in the terminal.")
        return prompt_video_path()
    finally:
        if root is not None:
            root.destroy()


def read_initial_frame(cap, *, skip_preroll=True):
    """Skip leading finite negative decoder timestamps; keep the retained PTS."""
    skipped = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            raise ValueError("Could not read an initial video frame with usable timing.")
        position_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if not skip_preroll or not np.isfinite(position_msec) or position_msec >= 0:
            if skipped:
                print(f"Skipped {skipped} leading frame(s) with negative decoder timestamps.")
            return frame, position_msec
        skipped += 1


def playback_key(paused=False):
    """Pump GUI events without Windows' extra wait on each playing frame."""
    if paused:
        return cv2.waitKey(20) & 0xFF
    poll = getattr(cv2, "pollKey", None)
    return (poll() if callable(poll) else cv2.waitKey(1)) & 0xFF


