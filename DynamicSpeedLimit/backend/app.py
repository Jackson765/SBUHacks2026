"""Serve the HTML dashboard and run one background analysis at a time."""
import os
import base64
from hashlib import sha256
from pathlib import Path
import sys
from threading import Event, Lock, Thread
from time import monotonic
from types import SimpleNamespace
from uuid import uuid4

# Match the CLI's Windows runtime setup before importing NumPy/OpenCV.
if os.name == "nt":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "YOLO"))

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import HTTPException

import main as pipeline
from calibration import default_corners
from config import CALIBRATION_PATH, DEFAULT_SPEED_LIMIT, FPS_OVERRIDE, VIDEO_PATH, ADVISORY_LEGAL_MAX_SPEED_MPH
from video_input import VideoClock, read_initial_frame, resize_for_processing
from video_settings import WorkingVideoSettingsStore, validate_entry, scale_for_speed_limit
from backend.media import codec_info, h264_stream
from backend.counters import AreaVehicleCounter

EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
PROTOCOL_VERSION = 2


def source_revision():
    digest = sha256()
    for relative in ("backend/app.py", "backend/media.py", "backend/counters.py", "YOLO/main.py", "YOLO/config.py", "YOLO/decision_engine.py", "YOLO/realtime_playback.py", "YOLO/video_input.py", "YOLO/video_settings.py"):
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()[:16]


STARTUP_REVISION = source_revision()


def speed_limit_result(status, forecast, advisory, base_limit):
    """Expose a completed recommendation without inventing one from a speed forecast."""
    result = dict(available=False, speed_mph=None, status="unavailable", base_speed_mph=base_limit)
    if status == "cancelled":
        result["message"] = "Analysis was stopped before a recommendation was completed."
    elif status == "failed":
        result["message"] = "Analysis failed; no new speed limit was determined."
    elif forecast is None or status != "completed":
        result["message"] = "Complete three five-minute windows with valid in-area speeds to determine a recommendation."
    elif not advisory:
        result["message"] = "The traffic forecast is ready, but the speed-limit advisory is unavailable."
    else:
        decision_status = advisory.get("status")
        value = advisory.get("advisory_speed_mph")
        if (decision_status == "demo_advisory" and not isinstance(value, bool)
                and isinstance(value, (int, float)) and np.isfinite(value)
                and 0 < value <= ADVISORY_LEGAL_MAX_SPEED_MPH):
            result.update(available=True, speed_mph=value, status="recommended",
                          message="Recommendation based on analyzed traffic and the current weather scenario. Review before use.")
        elif decision_status == "manual_review":
            result.update(status="manual_review", message="Manual review is required for the configured road or weather; no automatic limit is available.")
        elif decision_status == "partial_data":
            result["message"] = "Weather data is unavailable; a combined speed-limit recommendation could not be determined."
        else:
            result["message"] = "The available data did not produce a valid speed-limit recommendation."
    return result


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Could not decode this video. Try an MP4 with H.264 encoding.")
        fps = float(FPS_OVERRIDE) if FPS_OVERRIDE is not None else float(cap.get(cv2.CAP_PROP_FPS))
        clock = VideoClock(fps)
        frame, timestamp = read_initial_frame(cap, skip_preroll=clock.source == "video")
        frame = resize_for_processing(frame)
        return cap, clock, frame, timestamp
    except Exception:
        cap.release()
        raise


def jpeg(frame):
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("Could not encode the video preview.")
    return encoded.tobytes()


def create_app(config=None):
    app = Flask(__name__, static_folder=str(ROOT / "frontend"), static_url_path="/static")
    data_dir = Path(os.environ.get("DATA_DIR", ROOT / "backend" / "data")).resolve()
    app.config.update(
        DATA_DIR=data_dir, VIDEO_DIR=ROOT / "traffic",
        SETTINGS_PATH=Path(os.environ.get("CALIBRATION_PATH", CALIBRATION_PATH)),
        TEMP_SETTINGS_PATH=os.environ.get("CALIBRATION_TEMP_PATH"),
        MAX_CONTENT_LENGTH=2 * 1024**3,
    )
    if config:
        app.config.update(config)
    upload_dir = Path(app.config["DATA_DIR"]) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    settings_path = Path(app.config["SETTINGS_PATH"])
    temp_settings_path = Path(app.config["TEMP_SETTINGS_PATH"]) if app.config["TEMP_SETTINGS_PATH"] else settings_path.with_name("calibrationtemp.json")
    if settings_path.resolve() == temp_settings_path.resolve():
        raise ValueError("Default and working calibration files must have different paths.")
    temp_settings_path.parent.mkdir(parents=True, exist_ok=True)
    lock = Lock()
    state = {"job": None, "frame": None, "preview": None, "cancel": Event()}
    app.extensions["analysis_state"] = state

    def videos():
        # Opaque IDs prevent clients from opening arbitrary server files.
        found = {}
        for prefix, directory in (("sample", Path(app.config["VIDEO_DIR"])), ("upload", upload_dir)):
            for path in sorted(directory.glob("*")):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    found[f"{prefix}:{path.name}"] = path
        return found

    def video_path(video_id):
        path = videos().get(video_id)
        if path is None:
            abort(404, description="Video not found. Refresh the video list.")
        return path

    def store():
        return WorkingVideoSettingsStore(settings_path, temp_settings_path)

    def payload():
        body = request.get_json()
        if not isinstance(body, dict):
            abort(400, description="Expected a JSON object.")
        return body

    def require_idle():
        if state["job"] and state["job"]["status"] in {"running", "cancelling"}:
            abort(409, description="An analysis is already running. Wait or stop it first.")

    @app.errorhandler(ValueError)
    def invalid(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    @app.after_request
    def headers(response):
        if request.path == "/" or request.path == "/static/index.html" or request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", protocol_version=PROTOCOL_VERSION, backend_revision=STARTUP_REVISION,
                       restart_required=source_revision() != STARTUP_REVISION, process_id=os.getpid())

    @app.get("/api/videos")
    def list_videos():
        available = videos()
        default_id = f"sample:{Path(VIDEO_PATH).name}"
        return jsonify(videos=[dict(id=key, name=path.name) for key, path in available.items()],
                       default_video_id=default_id if default_id in available else None)

    @app.post("/api/videos")
    def upload_video():
        file = request.files.get("video")
        if file is None or not file.filename:
            abort(400, description="Choose a video file.")
        suffix = Path(file.filename).suffix.lower()
        if suffix not in EXTENSIONS:
            abort(400, description="Use MP4, MOV, AVI, MKV, WebM, or M4V.")
        path = upload_dir / f"{uuid4().hex}{suffix}"
        try:
            file.save(path)
            cap, _, _, _ = open_video(path)
            cap.release()
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return jsonify(id=f"upload:{path.name}", name=path.name), 201

    @app.get("/api/videos/<video_id>")
    def video_detail(video_id):
        path = video_path(video_id)
        cap, clock, frame, _ = open_video(path)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        size = (frame.shape[1], frame.shape[0])
        with lock:
            settings_store = store()
            try:
                settings_store.load_default(path, size, automatic_ratio=True)
                can_revert = True
            except ValueError:
                can_revert = False
            try:
                settings = settings_store.load(path, size, automatic_ratio=True)
                calibration = settings.calibration
                entry = dict(points=calibration.points, road_width_m=calibration.road_width_m,
                             road_length_m=calibration.road_length_m, ratio=settings.ratio,
                             base_speed_limit=settings.base_speed_limit, source=settings.source)
                needs_calibration, message = False, ""
            except ValueError as error:
                entry = dict(points=default_corners(*size), road_width_m=None, road_length_m=None,
                             ratio=scale_for_speed_limit(DEFAULT_SPEED_LIMIT), base_speed_limit=DEFAULT_SPEED_LIMIT, source="needs calibration")
                needs_calibration, message = True, str(error)
        return jsonify(id=video_id, name=path.name, frame_size=list(size), settings=entry,
                       recovery_warning=("Working calibration could not be read. Save or revert to replace it; the original will be backed up."
                                         if settings_store.load_error else None),
                       **codec_info(path),
                       source_fps=clock.fps if np.isfinite(clock.fps) and clock.fps > 0 else None,
                       needs_calibration=needs_calibration, can_revert=can_revert, message=message,
                       duration_seconds=count / clock.fps if clock.fps > 0 and count > 0 else None)

    @app.get("/api/videos/<video_id>/frame")
    def first_frame(video_id):
        cap, _, frame, _ = open_video(video_path(video_id))
        cap.release()
        return Response(jpeg(frame), mimetype="image/jpeg")

    @app.get("/api/videos/<video_id>/media")
    def source_media(video_id):
        # Native video playback, including HTTP byte ranges for seeking/buffering,
        # preserves the encoded frame rate independently of detector throughput.
        path = video_path(video_id)
        if not codec_info(path)["media_transcoded"]:
            return send_file(path, conditional=True, max_age=0)
        return Response(h264_stream(path, request.args.get("start", 0)), mimetype="video/mp4")

    @app.put("/api/videos/<video_id>/settings")
    def save_settings(video_id):
        path, body = video_path(video_id), payload()
        cap, _, frame, _ = open_video(path)
        cap.release()
        size = (frame.shape[1], frame.shape[0])
        body["frame_size"] = list(size)
        body["ratio"] = scale_for_speed_limit(body.get("base_speed_limit"))
        try:
            settings = validate_entry(body, size)
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError("Provide four corners, measured dimensions, and a whole speed limit.") from error
        with lock:
            require_idle()
            store().save(path, size, settings)
        return jsonify(saved=True)

    @app.post("/api/videos/<video_id>/settings/revert")
    def revert_settings(video_id):
        path = video_path(video_id)
        cap, _, frame, _ = open_video(path)
        cap.release()
        with lock:
            require_idle()
            settings_store = store()
            size = (frame.shape[1], frame.shape[0])
            settings_store.save(path, size, settings_store.load_default(path, size, automatic_ratio=True))
        return jsonify(saved=True)

    def run_job(path, settings, quick_analysis, cancel):
        cap = None
        last_preview = 0
        preview_interval = .5 if quick_analysis else .1
        started = monotonic()
        last_vehicles = []
        detections_elapsed = None
        last_frame, last_frame_elapsed = None, 0
        last_frame_detections, last_frame_detections_elapsed = [], None
        preview_sequence = 0
        counter = AreaVehicleCounter(settings.calibration, settings.base_speed_limit)

        def detections():
            return [dict(box=[float(value) for value in box],
                         speed_mph=float(speed) if speed is not None else None,
                         over_limit=speed is not None and speed > settings.base_speed_limit)
                    for box, _, speed in last_vehicles]

        def preview_packet(frame, elapsed):
            nonlocal last_preview, preview_sequence
            encoded = jpeg(frame)
            last_preview = monotonic()
            preview_sequence += 1
            return encoded, dict(sequence=preview_sequence, elapsed_seconds=elapsed,
                                 vehicle_detections_elapsed_seconds=last_frame_detections_elapsed,
                                 vehicle_detections=last_frame_detections)

        def finish_preview():
            if last_frame is not None:
                encoded, packet = preview_packet(last_frame, last_frame_elapsed)
                with lock:
                    state["frame"], state["preview"] = encoded, packet
                    state["job"]["preview_elapsed_seconds"] = last_frame_elapsed

        def observe(elapsed, collector, frame, vehicles):
            nonlocal last_vehicles, detections_elapsed, last_frame, last_frame_elapsed
            nonlocal last_frame_detections, last_frame_detections_elapsed
            update = dict(elapsed_seconds=min(900, elapsed), averages=collector.averages,
                          wall_elapsed_seconds=monotonic()-started,
                          vehicle_counts=[window.vehicle_count for window in collector.windows])
            if vehicles is not None:
                last_vehicles = counter.update(vehicles)
                detections_elapsed = elapsed
            update.update(counter.snapshot())
            update["vehicle_detections_elapsed_seconds"] = detections_elapsed
            update["vehicle_detections"] = detections()
            encoded, packet = None, None
            if frame is not None:
                last_frame, last_frame_elapsed = frame, elapsed
                last_frame_detections = update["vehicle_detections"]
                last_frame_detections_elapsed = detections_elapsed
            if frame is not None and monotonic() - last_preview >= preview_interval:
                encoded, packet = preview_packet(frame, elapsed)
            with lock:
                state["job"].update(update)
                if encoded is not None:
                    state["frame"], state["preview"] = encoded, packet
                    state["job"]["preview_elapsed_seconds"] = elapsed

        try:
            cap, clock, frame, timestamp = open_video(path)
            args = SimpleNamespace(visual=False, calibrate=False, full_analysis=not quick_analysis, weather=False,
                                   real_time=not quick_analysis)
            processor = pipeline.process_video_quick if quick_analysis else pipeline.process_video_realtime
            forecast = processor(cap, frame, timestamp, clock, path, settings, None, args,
                                 observer=observe, should_stop=cancel.is_set)
            finish_preview()
            advisory = pipeline.display_weather_advisory(forecast, settings.base_speed_limit) if not cancel.is_set() else None
            with lock:
                status = "cancelled" if cancel.is_set() else "completed" if forecast is not None else "incomplete"
                state["job"].update(status=status, forecast=forecast, advisory=advisory,
                                    speed_limit_result=speed_limit_result(status, forecast, advisory, settings.base_speed_limit),
                                    wall_elapsed_seconds=monotonic()-started,
                                    message={"cancelled": "Analysis stopped.", "completed": "Forecast ready.",
                                             "incomplete": "A forecast requires three complete five-minute windows with valid speeds."}[status])
        except Exception as error:
            app.logger.exception("Video analysis failed")
            with lock:
                state["job"].update(status="failed", message=str(error),
                                    speed_limit_result=speed_limit_result("failed", None, None, settings.base_speed_limit))
        finally:
            if cap is not None:
                cap.release()

    @app.post("/api/jobs")
    def start_job():
        if source_revision() != STARTUP_REVISION:
            abort(409, description="Backend code changed. Restart start_backend.cmd before starting analysis.")
        body = payload()
        if (not isinstance(body.get("video_id"), str)
                or ("weather" in body and not isinstance(body["weather"], bool))
                or not isinstance(body.get("quick_analysis", False), bool)):
            abort(400, description="Provide a video_id and boolean quick_analysis option.")
        path = video_path(body["video_id"])
        cap, video_clock, frame, _ = open_video(path)
        cap.release()
        with lock:
            require_idle()
            settings = store().load(path, (frame.shape[1], frame.shape[0]), automatic_ratio=True)
            state["cancel"] = Event()
            state["frame"] = jpeg(frame)
            state["preview"] = dict(sequence=0, elapsed_seconds=0, vehicle_detections_elapsed_seconds=None,
                                    vehicle_detections=[])
            quick_analysis = body.get("quick_analysis", False)
            state["job"] = dict(id=uuid4().hex, video_id=body["video_id"], status="running",
                                processor="quick" if quick_analysis else "regular",
                                protocol_version=PROTOCOL_VERSION, backend_revision=STARTUP_REVISION,
                                source_fps=video_clock.fps if np.isfinite(video_clock.fps) and video_clock.fps > 0 else None,
                                quick_analysis=quick_analysis, weather_advisory_enabled=True,
                                elapsed_seconds=0, wall_elapsed_seconds=0, averages=[], vehicle_counts=[], current_average_mph=None,
                                visible_vehicles=0, detected_vehicle_count=0, speeding_vehicle_count=0,
                                vehicle_detections=[], vehicle_detections_elapsed_seconds=None,
                                preview_elapsed_seconds=0,
                                speeding_threshold_mph=settings.base_speed_limit+10,
                                base_speed_limit=settings.base_speed_limit,
                                ratio=settings.ratio, forecast=None, advisory=None, speed_limit_result=None,
                                message="Analyzing video…")
            result = dict(state["job"])
            Thread(target=run_job,
                   args=(path, settings, quick_analysis, state["cancel"]),
                   daemon=True).start()
        return jsonify(result), 202

    @app.get("/api/jobs/current")
    def current_job():
        with lock:
            return jsonify(job=state["job"])

    def require_job(job_id):
        if not state["job"] or state["job"]["id"] != job_id:
            abort(404, description="Analysis not found. It may have expired after a server restart or a new run.")

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id):
        with lock:
            require_job(job_id)
            return jsonify(state["job"])

    @app.get("/api/jobs/<job_id>/frame")
    def job_frame(job_id):
        with lock:
            require_job(job_id)
            frame = state["frame"]
            elapsed = state["job"].get("preview_elapsed_seconds")
        response = Response(frame, mimetype="image/jpeg")
        if elapsed is not None:
            response.headers["X-Frame-Elapsed"] = str(elapsed)
        return response

    @app.get("/api/jobs/<job_id>/preview")
    def paired_preview(job_id):
        with lock:
            require_job(job_id)
            packet = dict(state["preview"])
            frame = state["frame"]
        packet["image_url"] = "data:image/jpeg;base64," + base64.b64encode(frame).decode("ascii")
        return jsonify(packet)

    @app.post("/api/jobs/<job_id>/cancel")
    def cancel_job(job_id):
        with lock:
            require_job(job_id)
            if state["job"]["status"] == "running":
                state["cancel"].set()
                state["job"]["status"] = "cancelling"
            return jsonify(state["job"])

    return app


if __name__ == "__main__":
    from waitress import serve
    app = create_app()
    host, port = os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "8000"))
    print(f"Flow dashboard: http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}", flush=True)
    serve(app, host=host, port=port, threads=8, max_request_body_size=app.config["MAX_CONTENT_LENGTH"])
