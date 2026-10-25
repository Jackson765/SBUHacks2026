"""HTTP integration using real generated video, without downloading models."""
import io
import json
import base64
from pathlib import Path
import sys
import tempfile
from threading import Event
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import create_app, speed_limit_result
import cv2
import numpy as np
from speed_windows import FiveMinuteAverages
from video_settings import VideoSettingsStore, validate_entry


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video_dir = self.root / "traffic"
        self.video_dir.mkdir()
        self.video = self.video_dir / "test.avi"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
        self.assertTrue(writer.isOpened())
        for index in range(10):
            writer.write(np.full((120, 160, 3), 30 + index, dtype=np.uint8))
        writer.release()
        self.app = create_app(dict(TESTING=True, DATA_DIR=self.root / "data",
                                   VIDEO_DIR=self.video_dir, SETTINGS_PATH=self.root / "calibration.json"))
        self.client = self.app.test_client()
        self.video_id = "sample:test.avi"
        self.base = f"/api/videos/{self.video_id}"
        self.settings = dict(points=[[20, 20], [140, 20], [140, 100], [20, 100]],
                             road_width_m=12, road_length_m=20, ratio=30/45, base_speed_limit=30)

    def tearDown(self):
        self.app.extensions["analysis_state"]["cancel"].set()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            job = self.client.get("/api/jobs/current").get_json()["job"]
            if not job or job["status"] not in {"running", "cancelling"}:
                break
            time.sleep(.01)
        self.temp.cleanup()

    def save(self):
        response = self.client.put(self.base + "/settings", json=self.settings)
        self.assertEqual(response.status_code, 200, response.get_json())

    def seed_default(self):
        entry = dict(self.settings, frame_size=[160, 120])
        VideoSettingsStore(self.root / "calibration.json").save(
            self.video, (160, 120), validate_entry(entry, (160, 120)))
        return (self.root / "calibration.json").read_bytes()

    def start(self, **extra):
        body = dict(video_id=self.video_id, quick_analysis=True)
        body.update(extra)
        response = self.client.post("/api/jobs", json=body)
        self.assertEqual(response.status_code, 202, response.get_json())
        return response.get_json()["id"]

    def finish(self, job_id):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = self.client.get(f"/api/jobs/{job_id}").get_json()
            if job["status"] not in {"running", "cancelling"}:
                return job
            time.sleep(.01)
        self.fail("Background job did not finish")

    def complete_pipeline(self, *args, observer, should_stop):
        collector = FiveMinuteAverages()
        for second, speed in [(0, 20), (300, 25), (600, 30)]:
            collector.add_measurements(second, [(1, speed)])
        observer(600, collector, args[1], [])
        collector.advance(900)
        observer(900, collector, None, [])
        return dict(predicted_speed=27.5, ratio=2, scaled_predicted_speed=55,
                    scaled_average_speeds=[40, 50, 60])

    def test_html_health_catalog_and_real_first_frame(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<!doctype html>", response.data)
        self.assertIn(b'id="showVehicleBoxes"', response.data)
        self.assertIn(b"Vehicle boxes hidden while analysis catches up", response.data)
        self.assertNotIn(b'id="weather"', response.data)
        self.assertNotIn(b"Include weather advisory", response.data)
        self.assertIn(b"automatically attempted after a forecast", response.data)
        self.assertIn(b'<details class="card collection-panel">', response.data)
        self.assertNotIn(b"Current vehicle average", response.data)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        response.close()
        health = self.client.get("/api/health").get_json()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["protocol_version"], 2)
        self.assertFalse(health["restart_required"])
        self.assertTrue(health["backend_revision"])
        self.assertEqual(self.client.get("/api/videos").get_json()["videos"],
                         [dict(id=self.video_id, name="test.avi")])
        metadata = self.client.get(self.base).get_json()
        self.assertTrue(metadata["needs_calibration"])
        self.assertEqual(metadata["frame_size"], [160, 120])
        self.assertEqual(metadata["source_fps"], 10)
        response = self.client.get(self.base + "/frame")
        decoded = cv2.imdecode(np.frombuffer(response.data, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape[:2], (120, 160))
        self.assertEqual(response.headers["Cache-Control"], "no-store")

    def test_native_media_supports_byte_ranges_without_starting_analysis(self):
        with patch("backend.app.codec_info", return_value=dict(media_transcoded=False, source_codec="avc1")):
            response = self.client.get(self.base + "/media", headers={"Range": "bytes=0-127"})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.data, self.video.read_bytes()[:128])
        self.assertEqual(response.headers["Content-Range"], f"bytes 0-127/{self.video.stat().st_size}")
        response.close()
        self.assertIsNone(self.client.get("/api/jobs/current").get_json()["job"])

    def test_non_native_video_is_streamed_as_h264_mp4(self):
        metadata = self.client.get(self.base).get_json()
        self.assertTrue(metadata["media_transcoded"])
        response = self.client.get(self.base + "/media", buffered=False)
        try:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "video/mp4")
            block = next(iter(response.response))
            self.assertIn(b"ftyp", block[:32])
        finally:
            response.close()
        self.assertIsNone(self.client.get("/api/jobs/current").get_json()["job"])

    def test_invalid_stream_seek_is_rejected(self):
        for value in ("-1", "nan", "900", "bad"):
            self.assertEqual(self.client.get(self.base + "/media?start=" + value).status_code, 400)

    def test_changed_backend_requires_restart_before_starting_job(self):
        with patch("backend.app.source_revision", return_value="changed"):
            self.assertTrue(self.client.get("/api/health").get_json()["restart_required"])
            response = self.client.post("/api/jobs", json=dict(video_id=self.video_id, quick_analysis=False))
        self.assertEqual(response.status_code, 409)
        self.assertIn("Restart", response.get_json()["error"])

    def test_calibration_round_trip_keeps_geometry_and_inputs(self):
        self.save()
        saved = self.client.get(self.base).get_json()
        self.assertFalse(saved["needs_calibration"])
        for key, value in self.settings.items():
            self.assertEqual(saved["settings"][key], value)
        self.assertTrue((self.root / "calibrationtemp.json").is_file())
        self.assertFalse((self.root / "calibration.json").exists())

    def test_existing_baseline_loads_without_creating_or_changing_files(self):
        baseline = self.seed_default()
        result = self.client.get(self.base).get_json()
        self.assertTrue(result["can_revert"])
        self.assertFalse(result["needs_calibration"])
        self.assertEqual(result["settings"]["source"], "default calibration")
        self.assertEqual((self.root / "calibration.json").read_bytes(), baseline)
        self.assertFalse((self.root / "calibrationtemp.json").exists())

    @patch("backend.app.pipeline.process_video")
    def test_saves_and_analysis_use_temp_without_modifying_baseline(self, process):
        baseline = self.seed_default()
        self.settings.update(road_width_m=15, ratio=3, base_speed_limit=45)
        self.save()
        self.assertEqual(self.client.get(self.base).get_json()["settings"]["road_width_m"], 15)
        process.side_effect = self.complete_pipeline
        job = self.finish(self.start())
        settings = process.call_args.args[5]
        self.assertEqual((settings.calibration.road_width_m, settings.ratio, settings.base_speed_limit), (15, 1, 45))
        self.assertEqual(job["ratio"], 1)
        self.assertEqual(job["base_speed_limit"], 45)
        self.assertEqual((self.root / "calibration.json").read_bytes(), baseline)

    def test_revert_restores_baseline_in_temp_and_preserves_other_videos(self):
        baseline = self.seed_default()
        defaults = dict(self.settings)
        self.settings.update(points=[[25, 25], [135, 25], [135, 95], [25, 95]],
                             road_width_m=15, road_length_m=30, ratio=3, base_speed_limit=45)
        self.save()
        other = self.video_dir / "other.avi"
        other.write_bytes(self.video.read_bytes())
        temp_path = self.root / "calibrationtemp.json"
        active = VideoSettingsStore(temp_path)
        active.save(other, (160, 120), validate_entry(dict(self.settings, frame_size=[160, 120]), (160, 120)))
        other_key = active.key(other)
        other_entry = json.loads(temp_path.read_text())["profiles"][other_key]
        for _ in range(2):
            response = self.client.post(self.base + "/settings/revert")
            self.assertEqual(response.status_code, 200, response.get_json())
            restored = self.client.get(self.base).get_json()["settings"]
            for key, value in defaults.items():
                self.assertEqual(restored[key], value)
            self.assertEqual(json.loads(temp_path.read_text())["profiles"][other_key], other_entry)
            self.assertEqual((self.root / "calibration.json").read_bytes(), baseline)

    def test_revert_without_default_leaves_saved_working_profile_unchanged(self):
        self.save()
        before = (self.root / "calibrationtemp.json").read_bytes()
        self.assertFalse(self.client.get(self.base).get_json()["can_revert"])
        response = self.client.post(self.base + "/settings/revert")
        self.assertEqual(response.status_code, 400)
        self.assertIn("No valid default calibration", response.get_json()["error"])
        self.assertEqual((self.root / "calibrationtemp.json").read_bytes(), before)

    def test_revert_uses_original_filename_preset_when_baseline_has_no_profile(self):
        original = dict(self.settings)
        preset_path = self.root / "presets.json"
        preset_path.write_text(json.dumps({self.video.name: dict(original, frame_size=[160, 120])}))
        self.settings.update(ratio=4, road_width_m=18)
        with patch("video_settings.VIDEO_DEFAULTS_PATH", preset_path):
            self.save()
            self.assertTrue(self.client.get(self.base).get_json()["can_revert"])
            self.assertEqual(self.client.post(self.base + "/settings/revert").status_code, 200)
            restored = self.client.get(self.base).get_json()["settings"]
            self.assertEqual(restored["ratio"], original["ratio"])
            self.assertEqual(restored["road_width_m"], original["road_width_m"])
        self.assertFalse((self.root / "calibration.json").exists())

    def test_same_baseline_and_temp_path_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "different paths"):
            create_app(dict(DATA_DIR=self.root / "data", SETTINGS_PATH=self.root / "calibration.json",
                            TEMP_SETTINGS_PATH=self.root / "calibration.json"))

    def test_invalid_calibration_is_rejected_without_writing(self):
        cases = [dict(road_width_m=-2), dict(base_speed_limit=0), dict(base_speed_limit=30.5),
                 dict(points=[[20, 20], [140, 100], [140, 20], [20, 100]]),
                 dict(points=[[-1, 20], [140, 20], [140, 100], [20, 100]])]
        for changes in cases:
            with self.subTest(changes=changes):
                response = self.client.put(self.base + "/settings", json={**self.settings, **changes})
                self.assertEqual(response.status_code, 400, response.get_json())
        self.assertFalse((self.root / "calibration.json").exists())
        self.assertFalse((self.root / "calibrationtemp.json").exists())

    def test_scale_is_calculated_from_limit_and_ignores_submitted_ratio(self):
        for limit in (15, 30, 45, 60, 90):
            with self.subTest(limit=limit):
                response = self.client.put(self.base + "/settings", json={**self.settings, "base_speed_limit": limit, "ratio": 999})
                self.assertEqual(response.status_code, 200)
                saved = self.client.get(self.base).get_json()["settings"]
                self.assertAlmostEqual(saved["ratio"], limit/45)
                profiles = json.loads((self.root/"calibrationtemp.json").read_text())["profiles"]
                self.assertAlmostEqual(profiles[VideoSettingsStore.key(self.video)]["ratio"], limit/45)
        body = {key: value for key, value in self.settings.items() if key != "ratio"}
        self.assertEqual(self.client.put(self.base + "/settings", json=body).status_code, 200)

    def test_legacy_profile_ratio_is_replaced_when_loaded_for_browser(self):
        self.settings["ratio"] = 8
        baseline = self.seed_default()
        result = self.client.get(self.base).get_json()["settings"]
        self.assertAlmostEqual(result["ratio"], 30/45)
        self.assertEqual((self.root/"calibration.json").read_bytes(), baseline)
        self.client.post(self.base + "/settings/revert")
        working = VideoSettingsStore(self.root/"calibrationtemp.json").load(self.video,(160,120))
        self.assertAlmostEqual(working.ratio, 30/45)

    @patch("backend.app.pipeline.process_video", return_value=None)
    def test_invalid_or_missing_legacy_ratio_does_not_block_automatic_scale(self, process):
        self.seed_default()
        for value in (0, "bad", None, "missing"):
            with self.subTest(value=value):
                baseline_path = self.root/"calibration.json"
                data = json.loads(baseline_path.read_text())
                entry = data["profiles"][VideoSettingsStore.key(self.video)]
                if value == "missing":
                    entry.pop("ratio", None)
                else:
                    entry["ratio"] = value
                baseline_path.write_text(json.dumps(data))
                baseline_bytes = baseline_path.read_bytes()
                (self.root/"calibrationtemp.json").unlink(missing_ok=True)
                response = self.client.get(self.base)
                self.assertEqual(response.status_code, 200)
                self.assertFalse(response.get_json()["needs_calibration"])
                self.assertTrue(response.get_json()["can_revert"])
                self.assertAlmostEqual(response.get_json()["settings"]["ratio"], 30/45)
                job = self.finish(self.start())
                self.assertAlmostEqual(job["ratio"], 30/45)
                self.assertEqual(self.client.post(self.base+"/settings/revert").status_code, 200)
                self.assertEqual(baseline_path.read_bytes(), baseline_bytes)

    def test_malformed_working_calibration_loads_defaults_and_can_revert_with_backup(self):
        baseline = self.seed_default()
        path = self.root/"calibrationtemp.json"
        original = b'{malformed json'
        path.write_bytes(original)
        response = self.client.get(self.base)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["can_revert"])
        self.assertFalse(response.get_json()["needs_calibration"])
        self.assertTrue(response.get_json()["recovery_warning"])
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.client.post(self.base+"/settings/revert").status_code, 200)
        backups = list(self.root.glob("calibrationtemp.invalid-*.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)
        self.assertEqual((self.root/"calibration.json").read_bytes(), baseline)
        self.assertFalse(self.client.get(self.base).get_json()["needs_calibration"])

    def test_malformed_working_file_can_be_replaced_by_new_manual_calibration(self):
        path = self.root/"calibrationtemp.json"
        path.write_bytes(b'not json')
        self.assertTrue(self.client.get(self.base).get_json()["needs_calibration"])
        self.save()
        self.assertFalse(self.client.get(self.base).get_json()["needs_calibration"])
        self.assertEqual(next(self.root.glob("calibrationtemp.invalid-*.json")).read_bytes(), b'not json')

    @patch("backend.app.pipeline.process_video")
    def test_preview_pairs_frame_and_detections_and_retains_final_frame(self, process):
        self.save()
        stage, release = Event(), Event()
        inside = [40,30,50,60]

        def feed(*args, observer, should_stop):
            collector = FiveMinuteAverages()
            observer(0, collector, args[1], [(inside,1,20)])
            # A newer inference without an encoded JPEG must not change its boxes.
            with patch("backend.app.monotonic", return_value=0):
                observer(5, collector, args[1], [(inside,2,45)])
            stage.set()
            release.wait(3)

        process.side_effect = feed
        job_id = self.start()
        try:
            self.assertTrue(stage.wait(2))
            state = self.client.get(f"/api/jobs/{job_id}").get_json()
            self.assertEqual(state["vehicle_detections_elapsed_seconds"], 5)
            packet = self.client.get(f"/api/jobs/{job_id}/preview").get_json()
            self.assertEqual(packet["elapsed_seconds"], 0)
            self.assertEqual(packet["vehicle_detections_elapsed_seconds"], 0)
            self.assertEqual(packet["vehicle_detections"][0]["speed_mph"], 20)
        finally:
            release.set()
        self.finish(job_id)
        final = self.client.get(f"/api/jobs/{job_id}/preview").get_json()
        self.assertEqual(final["elapsed_seconds"], 5)
        self.assertEqual(final["vehicle_detections_elapsed_seconds"], 5)
        self.assertEqual(final["vehicle_detections"][0]["speed_mph"], 45)
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(final["image_url"].split(",",1)[1]),np.uint8),cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape[:2], (120,160))

    @patch("backend.app.pipeline.process_video")
    def test_area_counters_are_live_unique_and_reset_for_both_modes(self, process):
        self.save()
        inside = [40, 30, 50, 60]
        outside = [0, 0, 10, 10]
        edge = [10, 5, 30, 20]
        for quick in (True, False):
            stage = Event()
            release = Event()

            def feed(*args, observer, should_stop):
                collector = FiveMinuteAverages()
                observer(0, collector, args[1], [(inside, 1, None), (inside, 2, 40), (outside, 99, 200)])
                observer(.1, collector, args[1], None)
                stage.set()
                release.wait(3)
                collector.advance(300)
                observer(300, collector, args[1], [(inside, 1, 40), (inside, 2, 45), (inside, 2, 45), (edge, 3, 0), (outside, 99, 200)])
                observer(300, collector, None, None)

            process.side_effect = feed
            job_id = self.start(quick_analysis=quick)
            try:
                self.assertTrue(stage.wait(2))
                live = self.client.get(f"/api/jobs/{job_id}").get_json()
                self.assertEqual(live["status"], "running")
                self.assertEqual(live["visible_vehicles"], 2)
                self.assertEqual(live["detected_vehicle_count"], 2)
                self.assertEqual(live["speeding_vehicle_count"], 1)
                self.assertEqual(live["speeding_threshold_mph"], 40)
                self.assertAlmostEqual(live["ratio"], 30/45)
                self.assertEqual(len(live["vehicle_detections"]), 2)
                self.assertEqual(live["vehicle_detections"][0]["box"], [40, 30, 50, 60])
                self.assertIsNone(live["vehicle_detections"][0]["speed_mph"])
                self.assertTrue(live["vehicle_detections"][1]["over_limit"])
            finally:
                release.set()
            finished = self.finish(job_id)
            self.assertEqual(finished["detected_vehicle_count"], 3)
            self.assertEqual(finished["speeding_vehicle_count"], 2)
            self.assertEqual(finished["visible_vehicles"], 3)
            self.assertAlmostEqual(finished["current_average_mph"], 85/3)

    def test_upload_decodes_and_requires_its_own_calibration(self):
        response = self.client.post("/api/videos", data={"video": (io.BytesIO(self.video.read_bytes()), "clip.avi")})
        self.assertEqual(response.status_code, 201, response.get_json())
        video_id = response.get_json()["id"]
        self.assertTrue(video_id.startswith("upload:"))
        self.assertTrue(self.client.get(f"/api/videos/{video_id}").get_json()["needs_calibration"])

    def test_invalid_uploads_are_rejected_and_removed(self):
        for filename in ["attack.exe", "broken.mp4"]:
            with self.subTest(filename=filename):
                response = self.client.post("/api/videos", data={"video": (io.BytesIO(b"invalid data"), filename)})
                self.assertEqual(response.status_code, 400)
        self.assertEqual(list((self.root / "data" / "uploads").iterdir()), [])

    def test_unknown_files_jobs_and_malformed_requests(self):
        self.assertEqual(self.client.get("/api/videos/sample:secret.mp4").status_code, 404)
        self.assertEqual(self.client.get("/api/jobs/missing").status_code, 404)
        self.assertEqual(self.client.post("/api/jobs", json=[]).status_code, 400)
        self.assertEqual(self.client.post("/api/jobs", json=dict(video_id=self.video_id, weather="yes")).status_code, 400)
        self.assertEqual(self.client.post("/api/jobs", json=dict(video_id=self.video_id, quick_analysis="yes")).status_code, 400)
        self.assertEqual(self.client.put(self.base + "/settings", json={}).status_code, 400)
        self.assertEqual(self.client.post("/api/jobs", json=dict(video_id=self.video_id)).status_code, 400)

    def test_upload_size_limit_returns_json_error(self):
        self.app.config["MAX_CONTENT_LENGTH"] = 256
        response = self.client.post("/api/videos", data={"video": (io.BytesIO(b"a" * 1000), "clip.avi")})
        self.assertEqual(response.status_code, 413)
        self.assertIn("error", response.get_json())

    @patch("backend.app.pipeline.display_weather_advisory", return_value=None)
    @patch("backend.app.pipeline.process_video")
    def test_completed_job_publishes_averages_forecast_and_preview(self, process, weather):
        self.save()
        process.side_effect = self.complete_pipeline
        job_id = self.start()
        job = self.finish(job_id)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["averages"], [20, 25, 30])
        self.assertEqual(job["elapsed_seconds"], 900)
        self.assertEqual(job["vehicle_detections_elapsed_seconds"], 900)
        self.assertEqual(job["preview_elapsed_seconds"], 600)
        self.assertEqual(job["forecast"]["predicted_speed"], 27.5)
        self.assertEqual(job["base_speed_limit"], 30)
        self.assertTrue(job["quick_analysis"])
        self.assertFalse(process.call_args.args[7].real_time)
        self.assertFalse(process.call_args.args[7].full_analysis)
        weather.assert_called_once_with(job["forecast"], 30)
        self.assertTrue(job["weather_advisory_enabled"])
        frame_response = self.client.get(f"/api/jobs/{job_id}/frame")
        self.assertEqual(frame_response.mimetype, "image/jpeg")
        self.assertEqual(frame_response.headers["X-Frame-Elapsed"], "600")
        packet = self.client.get(f"/api/jobs/{job_id}/preview").get_json()
        self.assertEqual(packet["elapsed_seconds"], 600)
        self.assertEqual(packet["vehicle_detections_elapsed_seconds"], 600)
        self.assertFalse(job["speed_limit_result"]["available"])
        self.assertEqual(self.client.get("/api/jobs/current").get_json()["job"]["id"], job_id)

    @patch("backend.app.pipeline.display_weather_advisory", return_value=dict(status="manual_review"))
    @patch("backend.app.pipeline.process_video")
    def test_weather_advisory_is_always_enabled_even_for_legacy_false_option(self, process, weather):
        self.save()
        process.side_effect = self.complete_pipeline
        job = self.finish(self.start(weather=False))
        self.assertEqual(job["advisory"], dict(status="manual_review"))
        self.assertTrue(job["weather_advisory_enabled"])
        weather.assert_called_once_with(job["forecast"], 30)
        self.assertEqual(job["speed_limit_result"]["status"], "manual_review")

    @patch("backend.app.pipeline.display_weather_advisory", return_value=dict(status="demo_advisory",advisory_speed_mph=55))
    @patch("backend.app.pipeline.process_video")
    def test_successful_analysis_returns_a_separate_recommended_speed_limit(self, process, weather):
        self.save()
        process.side_effect = self.complete_pipeline
        job = self.finish(self.start())
        self.assertTrue(job["speed_limit_result"]["available"])
        self.assertEqual(job["speed_limit_result"]["speed_mph"], 55)
        self.assertEqual(job["base_speed_limit"], 30)

    def test_result_does_not_promote_forecast_or_partial_candidate_to_a_limit(self):
        forecast = dict(predicted_speed=55)
        for advisory in (None,dict(status="partial_data",candidate_speed_mph=55),dict(status="manual_review",advisory_speed_mph=55),dict(status="demo_advisory",advisory_speed_mph=100)):
            with self.subTest(advisory=advisory):
                result = speed_limit_result("completed",forecast,advisory,30)
                self.assertFalse(result["available"])
                self.assertIsNone(result["speed_mph"])
        for status in ("cancelled","incomplete","failed"):
            self.assertFalse(speed_limit_result(status,None,None,30)["available"])

    @patch("backend.app.pipeline.process_video_realtime")
    def test_real_time_mode_is_passed_to_pipeline(self, process):
        self.save()
        process.side_effect = self.complete_pipeline
        job_id = self.start(quick_analysis=False)
        job = self.finish(job_id)
        self.assertEqual(job["status"], "completed")
        self.assertFalse(job["quick_analysis"])
        self.assertTrue(process.call_args.args[7].real_time)
        self.assertTrue(process.call_args.args[7].full_analysis)

    @patch("backend.app.pipeline.process_video_realtime")
    def test_missing_quick_option_defaults_to_real_time(self, process):
        self.save()
        process.side_effect = self.complete_pipeline
        response = self.client.post("/api/jobs", json=dict(video_id=self.video_id))
        self.assertEqual(response.status_code, 202)
        job = self.finish(response.get_json()["id"])
        self.assertFalse(job["quick_analysis"])
        self.assertTrue(process.call_args.args[7].real_time)

    def test_real_time_pipeline_paces_a_real_clip_through_its_final_frame(self):
        self.save()
        detector = Mock()
        detector.track.return_value = [SimpleNamespace(boxes=None)]
        with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=Mock(return_value=detector))}), \
                patch("backend.app.pipeline.make_simulation_plan", side_effect=AssertionError("Fast plan used")), \
                patch("backend.app.pipeline.cv2.imshow", side_effect=AssertionError("Desktop UI opened")):
            started = time.monotonic()
            job_id = self.start(quick_analysis=False)
            # Before one wall second passes, collection must not reach a second.
            while time.monotonic() - started < .5:
                job = self.client.get(f"/api/jobs/{job_id}").get_json()
                self.assertEqual(job["status"], "running")
                self.assertLessEqual(job["elapsed_seconds"], time.monotonic() - started + .01)
                time.sleep(.02)
            job = self.finish(job_id)
            self.assertGreaterEqual(time.monotonic() - started, 1)
        self.assertEqual(job["status"], "incomplete")
        self.assertAlmostEqual(job["elapsed_seconds"], 1)
        self.assertGreaterEqual(job["wall_elapsed_seconds"], job["elapsed_seconds"])
        self.assertIsNone(job["forecast"])
        self.assertEqual(detector.track.call_count, 5)
        self.assertTrue(all(call.args[0].shape[:2] == (120, 160) for call in detector.track.call_args_list))

    def test_both_modes_on_actual_thirty_fps_clip_have_distinct_wall_timing(self):
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
        self.assertTrue(writer.isOpened())
        for index in range(90):
            writer.write(np.full((120, 160, 3), index, np.uint8))
        writer.release()
        self.save()
        detector = Mock()
        detector.track.return_value = [SimpleNamespace(boxes=None)]
        durations = {}
        with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=Mock(return_value=detector))}):
            for quick in (True, False):
                started = time.monotonic()
                job_id = self.start(quick_analysis=quick)
                while True:
                    job = self.client.get(f"/api/jobs/{job_id}").get_json()
                    if not quick:
                        self.assertLessEqual(job["elapsed_seconds"], time.monotonic() - started + .05)
                    if job["status"] not in {"running", "cancelling"}:
                        break
                    self.assertLess(time.monotonic()-started, 8)
                    time.sleep(.02)
                durations[quick] = time.monotonic()-started
                self.assertEqual(job["status"], "incomplete")
                self.assertAlmostEqual(job["elapsed_seconds"], 3)
                self.assertEqual(job["source_fps"], 30)
                self.assertEqual(job["processor"], "quick" if quick else "regular")
                self.assertEqual(job["protocol_version"], 2)
        self.assertGreaterEqual(durations[False], 3)
        self.assertLess(durations[True], durations[False] / 2)

    @patch("backend.app.pipeline.process_video")
    def test_single_job_guard_and_cooperative_cancellation(self, process):
        started = Event()

        def wait_for_cancel(*args, observer, should_stop):
            started.set()
            deadline = time.monotonic() + 5
            while not should_stop() and time.monotonic() < deadline:
                time.sleep(.01)

        self.save()
        process.side_effect = wait_for_cancel
        job_id = self.start()
        self.assertTrue(started.wait(2))
        self.assertEqual(self.client.post("/api/jobs", json=dict(video_id=self.video_id)).status_code, 409)
        self.assertEqual(self.client.put(self.base + "/settings", json=self.settings).status_code, 409)
        self.assertEqual(self.client.post(self.base + "/settings/revert").status_code, 409)
        self.client.post(f"/api/jobs/{job_id}/cancel")
        self.assertEqual(self.finish(job_id)["status"], "cancelled")

    @patch("backend.app.pipeline.process_video", side_effect=ValueError("Invalid timestamps"))
    def test_worker_failure_reports_error_and_allows_another_run(self, process):
        self.save()
        with self.assertLogs(self.app.logger, level="ERROR"):
            job = self.finish(self.start())
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["message"], "Invalid timestamps")
        process.side_effect = self.complete_pipeline
        self.assertEqual(self.finish(self.start())["status"], "completed")

    def test_real_pipeline_short_video_does_not_invent_forecast(self):
        self.save()
        detector = Mock()
        detector.track.return_value = [SimpleNamespace(boxes=None)]
        with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=Mock(return_value=detector))}), \
                patch("backend.app.pipeline.forecast_collected_averages") as forecast, \
                patch("backend.app.pipeline.cv2.imshow", side_effect=AssertionError("Desktop UI opened")):
            job = self.finish(self.start())
        self.assertEqual(job["status"], "incomplete")
        self.assertIsNone(job["forecast"])
        self.assertEqual(job["averages"], [])
        self.assertGreater(job["elapsed_seconds"], 0)
        self.assertGreater(detector.track.call_count, 0)
        forecast.assert_not_called()


if __name__ == "__main__":
    unittest.main()
