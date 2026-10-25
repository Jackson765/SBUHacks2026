"""One per-video store for road geometry, scaling, and the fixed input limit."""
from dataclasses import dataclass
import json
from pathlib import Path
from shutil import copy2
from uuid import uuid4

import numpy as np

from calibration import RoadCalibration, video_identity
from config import CALIBRATION_PATH, DEFAULT_SPEED_LIMIT, VIDEO_DEFAULTS_PATH
from ratio_predict import validate_ratio


def validate_base_speed_limit(value):
    try:
        limit = float(value)
    except (TypeError, ValueError):
        raise ValueError("Base speed limit must be a positive whole number in mph.") from None
    if isinstance(value, bool) or not np.isfinite(limit) or limit <= 0 or not limit.is_integer():
        raise ValueError("Base speed limit must be a positive whole number in mph.")
    return int(limit)


def prompt_base_speed_limit():
    while True:
        raw = input(f"Base speed limit in mph [{DEFAULT_SPEED_LIMIT}] (Q to quit): ").strip()
        if raw.lower() == "q":
            raise KeyboardInterrupt
        try:
            return validate_base_speed_limit(raw or DEFAULT_SPEED_LIMIT)
        except ValueError as error:
            print(error)


def scale_for_speed_limit(value):
    """Browser forecast scale: entered limit in mph divided by the 45 mph reference."""
    return validate_base_speed_limit(value) / 45.0


@dataclass
class VideoSettings:
    calibration: RoadCalibration
    ratio: float
    base_speed_limit: int
    source: str = "saved settings"


def validate_entry(entry, frame_size, ratio_override=None):
    if list(entry["frame_size"]) != list(frame_size):
        raise ValueError(f"Settings frame_size {entry['frame_size']} does not match processing size {list(frame_size)}.")
    calibration = RoadCalibration(entry["points"], entry["road_width_m"], entry["road_length_m"])
    points = np.asarray(calibration.points)
    if ((points < 0).any() or (points[:, 0] >= frame_size[0]).any()
            or (points[:, 1] >= frame_size[1]).any()):
        raise ValueError("Road corners must lie inside the processing frame.")
    calibration.perspective_matrix()
    return VideoSettings(calibration,
                         validate_ratio(entry["ratio"] if ratio_override is None else ratio_override),
                         validate_base_speed_limit(entry["base_speed_limit"]))


class VideoSettingsStore:
    """Read once per run; saved profiles override migrated filename defaults.

    Profiles include file identity so replacing a video cannot reuse old geometry.
    Atomic writes retain profiles for other videos and the original defaults.
    """
    def __init__(self, path=CALIBRATION_PATH, *, recover_invalid=False):
        self.path = Path(path)
        self.load_error = None
        self.backup_path = None
        self.defaults = json.loads(Path(VIDEO_DEFAULTS_PATH).read_text(encoding="utf-8"))
        if not isinstance(self.defaults, dict):
            raise ValueError(f"Video defaults in {VIDEO_DEFAULTS_PATH} must be an object keyed by filename.")
        self.data = {"version": 2, "profiles": {}}
        if self.path.is_file():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
                if self.data["version"] == 1:
                    legacy = self.data
                    filename = Path(legacy["video"]["path"]).name
                    fallback = self.defaults.get(filename, {})
                    inputs = legacy.get("run_settings", {})
                    if not isinstance(inputs, dict):
                        raise ValueError("Saved run_settings must be an object.")
                    entry = {key: legacy[key] for key in
                             ("video", "frame_size", "points", "road_width_m", "road_length_m")}
                    entry.update(ratio=inputs.get("ratio", fallback.get("ratio", 1.0)),
                                 base_speed_limit=inputs.get("base_speed_limit", fallback.get("base_speed_limit", DEFAULT_SPEED_LIMIT)))
                    self.data = {"version": 2, "profiles": {self.key(legacy["video"]["path"]): entry}}
                if self.data["version"] != 2 or not isinstance(self.data["profiles"], dict):
                    raise ValueError("Expected version 2 with a profiles object.")
            except (KeyError, TypeError, ValueError) as error:
                message = f"Invalid video settings in {self.path}: {error}"
                if not recover_invalid:
                    raise ValueError(message) from error
                self.load_error = message
                self.data = {"version": 2, "profiles": {}}

    @staticmethod
    def key(video_path):
        # Profiles in this Windows project use case-insensitive absolute paths.
        return str(Path(video_path).resolve()).casefold()

    def entry(self, video_path):
        saved = self.data["profiles"].get(self.key(video_path))
        if saved is not None and not isinstance(saved, dict):
            raise ValueError(f"Invalid saved settings for {Path(video_path).name!r}; run --calibrate to replace them.")
        if saved is not None and saved.get("video") == video_identity(video_path):
            return saved, "saved settings"
        if saved is not None:
            print("Video file changed; using its original defaults. Use --calibrate to save new settings.")
        return self.defaults.get(Path(video_path).name), "migrated defaults"

    def load(self, video_path, frame_size, ratio_override=None, *, automatic_ratio=False):
        entry, source = self.entry(video_path)
        if entry is None:
            raise ValueError(f"No settings for {Path(video_path).name!r}; run --calibrate --video \"{video_path}\".")
        try:
            if automatic_ratio:
                ratio_override = scale_for_speed_limit(entry["base_speed_limit"])
            settings = validate_entry(entry, frame_size, ratio_override)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid settings for {Path(video_path).name!r}: {error} Run --calibrate to replace them.") from error
        settings.source = source
        return settings

    def save(self, video_path, frame_size, settings):
        calibration = settings.calibration
        entry = dict(video=video_identity(video_path), frame_size=list(frame_size),
                     points=calibration.points, road_width_m=calibration.road_width_m,
                     road_length_m=calibration.road_length_m, ratio=settings.ratio,
                     base_speed_limit=settings.base_speed_limit)
        validate_entry(entry, frame_size)
        # Keep the unreadable original before any explicit save/revert replaces it.
        if getattr(self, "load_error", None) and self.path.is_file():
            backup = self.path.with_name(f"{self.path.stem}.invalid-{uuid4().hex}.json")
            copy2(self.path, backup)
            self.backup_path = backup
            self.load_error = None
        self.data["profiles"][self.key(video_path)] = entry
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(self.path)
        print(f"Saved settings for {Path(video_path).name} to {self.path}.", flush=True)


class WorkingVideoSettingsStore(VideoSettingsStore):
    """Read defaults from calibration.json; save edits only to calibrationtemp.json."""

    def __init__(self, baseline_path=CALIBRATION_PATH, temp_path=None):
        baseline_path = Path(baseline_path)
        temp_path = Path(temp_path) if temp_path is not None else baseline_path.with_name("calibrationtemp.json")
        if baseline_path.resolve() == temp_path.resolve():
            raise ValueError("Default and working calibration files must have different paths.")
        self.baseline = VideoSettingsStore(baseline_path)
        super().__init__(temp_path, recover_invalid=True)

    def entry(self, video_path):
        saved = self.data["profiles"].get(self.key(video_path))
        if saved is not None and not isinstance(saved, dict):
            raise ValueError(f"Invalid working settings for {Path(video_path).name!r}; recalibrate or revert to default.")
        if saved is not None and saved.get("video") == video_identity(video_path):
            return saved, "saved settings"
        entry, source = self.baseline.entry(video_path)
        return entry, "default calibration" if source == "saved settings" else source

    def load_default(self, video_path, frame_size, *, automatic_ratio=False):
        """Use the baseline profile, then the original filename preset."""
        try:
            settings = self.baseline.load(video_path, frame_size, automatic_ratio=automatic_ratio)
        except ValueError as error:
            raise ValueError(f"No valid default calibration for {Path(video_path).name!r}. {error}") from error
        settings.source = "default calibration"
        return settings

    def revert_to_default(self, video_path, frame_size):
        """Replace only this video's working profile; preserve defaults and other videos."""
        settings = self.load_default(video_path, frame_size)
        self.save(video_path, frame_size, settings)
        return settings
