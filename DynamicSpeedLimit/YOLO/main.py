"""Analyze prerecorded traffic videos or calibrate them during playback."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

# Avoid the Conda MKL/OpenMP conflict with PyTorch DLLs on Windows.
# This must precede NumPy and calibration imports; preserve explicit user settings.
if os.name == "nt":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import cv2
import numpy as np
from calibration import run_calibration, window_is_open
from video_calibration import VideoCalibrationControls
from config import (CALIBRATION_PATH, SIMULATION_ROAD_PADDING, SIMULATION_TARGET_FPS,
                    DETECTION_CONFIDENCE, DETECTION_IMAGE_SIZE, FPS_OVERRIDE, FRAME_SKIP,
                    VEHICLE_CLASSES, VIDEO_PATH, WINDOW_NAME, YOLO_MODEL_PATH)
from tracker import SpeedTracker
from ratio_predict import prompt_ratio, validate_ratio, scale_average_speeds, predict_scaled_averages
from speed_windows import WINDOW_SECONDS, FiveMinuteAverages
from simulation import make_simulation_plan
from video_input import VideoClock, choose_video, playback_key, read_initial_frame, resize_for_processing
from video_settings import VideoSettings, VideoSettingsStore, prompt_base_speed_limit
from ui import compose_playback


def report_windows(windows, ratio):
    for window in windows:
        label = f"{window.start_seconds / 60:g}-{window.end_seconds / 60:g} minutes"
        if window.average_speed is None:
            print(f"{label}: no valid car-speed measurements.", flush=True)
        else:
            scaled = window.average_speed * ratio
            if not np.isfinite(scaled):
                raise ValueError("Scaled average overflowed; use a smaller ratio.")
            print(f"{label}: average {window.average_speed:.3f} mph; "
                  f"scaled {scaled:.3f}; {window.vehicle_count} tracked cars.", flush=True)


def forecast_collected_averages(averages, ratio):
    """Output the three scaled bins, invoke Model B, then unscale its forecast."""
    if len(averages) != 3 or any(value is None for value in averages):
        print("Prediction unavailable: all three five-minute windows need valid car speeds.")
        return None
    scaled = scale_average_speeds(averages, ratio)
    print("Three five-minute averages (local mph):", json.dumps(averages))
    print("Three scaled averages sent to Model B:", json.dumps(scaled.tolist()))
    result = predict_scaled_averages(scaled, ratio)
    print(f"Model output (scaled): {result['scaled_predicted_speed']:.3f}")
    print(f"Predicted average speed 10 minutes after collection: {result['predicted_speed']:.3f} mph")
    return result


def display_weather_advisory(forecast, base_speed_limit):
    """Optional dependencies and network access are used only after a forecast."""
    if forecast is None:
        return None
    started = perf_counter()
    try:
        import api_code
        from decision_engine import evaluate_advisory
        from config import ADVISORY_FREE_FLOW_SPEED_MPH, ADVISORY_ROAD_TYPE
        from ui import format_advisory
        weather = api_code.get_current_weather()
        decision = evaluate_advisory(forecast["predicted_speed"], ADVISORY_FREE_FLOW_SPEED_MPH,
                                     base_speed_limit, weather, ADVISORY_ROAD_TYPE,
                                     prediction_age_seconds=perf_counter()-started)
        print(format_advisory(decision, weather), flush=True)
        return decision
    except Exception as error:
        print(f"Advisory unavailable: {error}. Existing Model B result is retained.", flush=True)
        return None


def process_video(cap, first_frame, first_timestamp_msec, clock, video_path, settings, store, args):
    """One inference/measurement loop shared by analysis and both preview modes."""
    visual = args.visual or args.calibrate
    fps = clock.fps
    frame_size = (first_frame.shape[1], first_frame.shape[0])
    calibration, ratio, base_speed_limit = settings.calibration, settings.ratio, settings.base_speed_limit
    plan = None
    if not visual and not args.full_analysis:
        plan = make_simulation_plan(frame_size, calibration.points, fps,
                                    target_fps=SIMULATION_TARGET_FPS, padding=SIMULATION_ROAD_PADDING,
                                    maximum_image_size=DETECTION_IMAGE_SIZE, fallback_step=FRAME_SKIP)
    frame_step = plan.frame_step if plan is not None else FRAME_SKIP
    image_size = plan.image_size if plan is not None else DETECTION_IMAGE_SIZE
    if plan is not None:
        print(f"Fast analysis: detect every {frame_step} source frames; road crop {plan.bounds}; inference size {image_size}.", flush=True)
    tracker = SpeedTracker(calibration.points, calibration.road_width_m, calibration.road_length_m)
    model_path = Path(YOLO_MODEL_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {model_path}")
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    controls = VideoCalibrationControls(calibration, ratio, frame_size) if args.calibrate else None
    if controls is not None:
        controls.attach()
        print("Video calibration: drag corners or use sliders; SPACE pauses; R resets; U undoes; S saves; Q quits.")
    elif visual:
        print("Playback: SPACE pauses; Q / Escape quits. Calibration is locked; use --calibrate to edit.")
    collector = FiveMinuteAverages()
    origin = None
    collection_offset = 0.0
    last_progress_minute = -1
    started = last_wall_progress = perf_counter()
    detection_frames = 0
    paused = save_requested = False
    tracked_boxes = []
    print(f"Collecting three five-minute windows of VIDEO time; speed multiplier: {ratio:g}.")
    frame, position_msec, frame_index = first_frame, first_timestamp_msec, 0
    while True:
        timestamp = clock.timestamp(frame_index, position_msec)
        if origin is None:
            origin = timestamp
        elapsed = timestamp - origin
        collection_elapsed = elapsed - collection_offset
        report_windows(collector.advance(collection_elapsed), ratio)
        if collector.complete:
            break
        minute = int(collection_elapsed // 60)
        wall_now = perf_counter()
        if minute > last_progress_minute or (not visual and wall_now-last_wall_progress >= 5):
            window_start = len(collector.windows) * WINDOW_SECONDS / 60
            seconds = int(collection_elapsed)
            duration = wall_now-started
            rate = elapsed/duration if duration > 0 else 0
            print(f"Collecting {window_start:g}-{window_start + WINDOW_SECONDS/60:g} minutes "
                  f"(video {seconds//60:02d}:{seconds%60:02d}/15:00; wall {duration:.1f}s; "
                  f"{rate:.1f}x video speed; {detection_frames} detection frames).", flush=True)
            last_progress_minute, last_wall_progress = minute, wall_now
        if frame_index % frame_step == 0:
            frame = resize_for_processing(frame)
            detection_frame = plan.crop(frame) if plan is not None else frame
            result = model.track(detection_frame, classes=VEHICLE_CLASSES, conf=DETECTION_CONFIDENCE,
                                 imgsz=image_size, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
            detection_frames += 1
            vehicles, tracked_boxes = [], []
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                if plan is not None:
                    boxes = plan.restore_boxes(boxes)
                ids = result.boxes.id.int().cpu().tolist()
                for box, track_id in zip(boxes, ids):
                    tracked_boxes.append((box, track_id))
                    x1, _, x2, y2 = map(float, box)
                    if not tracker.point_in_road((x1+x2)/2, y2):
                        tracker.forget(track_id)
                        continue
                    speed = tracker.update_vehicle_speed(track_id, box, timestamp)
                    vehicles.append((box, track_id, speed))
            tracker.remove_inactive_tracks(timestamp)
            collector.add_measurements(collection_elapsed, [(track_id, speed) for _, track_id, speed in vehicles])
        if visual:
            # Show each decoded frame; reuse the last detections on skipped
            # inference frames without adding speed samples or model calls.
            frame = resize_for_processing(frame)
            annotated = result.plot(img=frame, labels=False)
        redraw = visual
        while visual:
            calibration_changed = False
            if controls is not None:
                calibration_changed = controls.poll()
                redraw = redraw or controls.needs_redraw
            if calibration_changed:
                calibration, ratio = controls.calibration, controls.ratio
                tracker.recalibrate(calibration.points, calibration.road_width_m, calibration.road_length_m)
                collector = FiveMinuteAverages()
                collection_offset, collection_elapsed = elapsed, 0.0
                last_progress_minute = -1
                vehicles = [(box, track_id, tracker.current_speed(track_id, timestamp))
                            for box, track_id in tracked_boxes
                            if tracker.point_in_road((float(box[0])+float(box[2]))/2, float(box[3]))]
                redraw = True
            if controls is not None and save_requested:
                save_requested = False
                if controls.valid:
                    try:
                        store.save(video_path, frame_size, VideoSettings(calibration, ratio, base_speed_limit))
                        controls.message = "Settings saved for analysis and playback."
                    except OSError as error:
                        controls.message = f"Could not save: {error}"
                else:
                    controls.message = "Fix the road shape before saving."
                redraw = True
            if redraw:
                display = compose_playback(annotated, vehicles, base_speed_limit, collection_elapsed,
                                           len(collector.windows), ratio, controls=controls, paused=paused)
                cv2.imshow(WINDOW_NAME, display)
                redraw = False
            key = playback_key(paused)
            if key in (ord("q"), 27) or not window_is_open():
                print("Collection cancelled; no prediction was requested.")
                return
            if key == ord(" "):
                paused = not paused
                redraw = True
            elif controls is not None and key == ord("r"):
                controls.reset()
                continue
            elif controls is not None and key == ord("u"):
                controls.editor.undo()
                continue
            elif controls is not None and key == ord("s"):
                save_requested = True
                continue
            if not paused:
                break
        if plan is not None and (frame_index+1) % frame_step != 0:
            ok, frame = cap.grab(), None
        else:
            ok, frame = cap.read()
        if not ok:
            report_windows(collector.advance(clock.end_timestamp()-origin-collection_offset), ratio)
            break
        position_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_index += 1
    if collector.complete:
        print(f"Processed 15 minutes of source video in {perf_counter()-started:.2f}s; {detection_frames} detection frames.", flush=True)
        forecast = forecast_collected_averages(collector.averages, ratio)
        if args.weather:
            display_weather_advisory(forecast, base_speed_limit)
        return forecast
    print(f"Video ended before 15 minutes: {len(collector.windows)}/3 complete five-minute averages. "
          "Use a longer video; no forecast was generated.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual", action="store_true", help="Play with saved calibration locked")
    parser.add_argument("--calibrate", "--recalibrate", dest="calibrate", action="store_true",
                        help="Set up a video, then edit calibration during playback; S saves")
    parser.add_argument("--full-analysis", action="store_true", help="Use full-frame detection and FRAME_SKIP sampling (slower)")
    parser.add_argument("--ratio", type=float, help="Override the saved ratio for this run")
    parser.add_argument("--weather", action="store_true", help="Fetch weather and show an advisory after prediction")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path, help="Video path; skips the file picker")
    source.add_argument("--default-video", action="store_true", help="Use VIDEO_PATH from config.py")
    args = parser.parse_args(argv)
    visual = args.visual or args.calibrate
    video_path = args.video
    if video_path is None:
        video_path = choose_video() if visual and not args.default_video else Path(VIDEO_PATH)
    if video_path is None:
        print("No video selected. Cancelled.")
        return
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}. Choose an existing video file.")
    if not isinstance(FRAME_SKIP, int) or FRAME_SKIP < 1:
        raise ValueError("FRAME_SKIP must be a positive integer.")
    mode = "editable calibration" if args.calibrate else "locked playback" if visual else "fast analysis"
    print(f"{video_path.name}: {mode}.", flush=True)
    store = VideoSettingsStore(CALIBRATION_PATH)
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        fps = float(FPS_OVERRIDE) if FPS_OVERRIDE is not None else float(cap.get(cv2.CAP_PROP_FPS))
        clock = VideoClock(fps)
        first_frame, first_timestamp_msec = read_initial_frame(cap, skip_preroll=clock.source == "video")
        first_frame = resize_for_processing(first_frame)
        frame_size = (first_frame.shape[1], first_frame.shape[0])
        if args.calibrate:
            ratio = prompt_ratio() if args.ratio is None else validate_ratio(args.ratio)
            base_limit = prompt_base_speed_limit()
            calibration = run_calibration(first_frame)
            if calibration is None:
                return
            settings = VideoSettings(calibration, ratio, base_limit)
            store.save(video_path, frame_size, settings)
        else:
            settings = store.load(video_path, frame_size, args.ratio)
        print(f"Using {settings.source}; ratio {settings.ratio:g}; fixed speed limit {settings.base_speed_limit} mph.", flush=True)
        return process_video(cap, first_frame, first_timestamp_msec, clock, video_path, settings, store, args)
    finally:
        cap.release()
        if visual:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
