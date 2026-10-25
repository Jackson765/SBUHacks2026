# Flow — Dynamic Speed Limit

The browser UI is **`frontend/index.html`**, a plain HTML file with embedded CSS
and JavaScript. There is no Node.js dependency or frontend build step.
**`backend/app.py`** serves it and provides the video, calibration, and analysis
API. The backend reuses the existing Python modules in `YOLO/` and `model_b/`;
their command-line entry points continue to work.

## Run locally

On Windows, from the repository root in PowerShell:

```powershell
.\DynamicSpeedLimit\start_backend.cmd
```

The launcher uses an activated Python environment when available, otherwise the
`dynamic-speed` environment under Anaconda/Miniconda. It changes to the correct
project directory automatically. `FLOW_PYTHON` can specify another interpreter.
Keep the terminal open and visit **http://127.0.0.1:8000**. Stop an older server
with Ctrl+C before restarting to pick up backend changes. If dependencies are
missing, the launcher prints the install command using its selected Python.

Alternatively, from `DynamicSpeedLimit/` in PowerShell:

```powershell
conda activate dynamic-speed
python -m pip install -r backend/requirements.txt
python -m backend.app
```

Open **http://127.0.0.1:8000**. Use Python 3.11 for a fresh environment:

```powershell
conda create -n dynamic-speed python=3.11
```

Open the UI through the server so its `/api` requests reach Python. Opening the
HTML directly from disk does not run the backend.

1. The default video's first frame and calibration load when the page opens,
   before starting analysis. You can also select another video from `traffic/`
   or upload footage (up to 2 GiB). Finished runs do not replace the calibration
   view on page reload; an analysis still running is resumed.
2. Review the saved calibration. For a new video, drag the four corners to a
   **measured** road rectangle and enter its width/length in meters, scaling
   ratio, and fixed input speed limit. Save calibration. Number keys 1–4 select
   corners; arrow keys move them, with Shift for larger steps.
3. **Quick analysis starts unchecked**, selecting real-time playback by default.
   The browser plays at **1x using the source frame rate**. A 30 FPS recording
   displays about 30 frames each wall-clock second. H.264 MP4s use native
   byte-range playback; HEVC and other codecs stream as H.264 using the bundled
   FFmpeg runtime, retaining source timing. Conversion changes only browser
   playback, not the source used for measurement. Check Quick analysis to
   process as fast as possible with JPEG detection snapshots. The methods are
   `process_video_realtime` and `process_video_quick`. The page distinguishes
   **Media** time from **Analyzed** time because inference can lag smooth video
   playback. Averages and completed windows refer to analyzed footage. Stop
   cancels analysis and stops playback.
4. A forecast requires 15 minutes of video and valid measurements in every
   window. Short clips report incomplete collection. The forecast is for ten
   minutes after collection ends; it does not change the fixed input limit.

Calibration is fixed during each browser run. To change it, stop analysis, edit
the fields/corners, save, and start a new run. Current weather is optional and
applied as a scenario to recorded video. An unavailable advisory preserves the
traffic forecast.

`YOLO/calibration.json` is the read-only baseline. Browser saves and desktop
calibration saves write to **`YOLO/calibrationtemp.json`**, which takes precedence
for analysis and playback. If a video has no matching working profile, the app
uses its baseline profile, then the original `YOLO/video_defaults.json` preset.
Existing baseline profiles are preserved without migration or rewriting.

**Revert to default** restores the selected video's baseline settings (or its
original filename preset) into `calibrationtemp.json`, including corners,
dimensions, ratio, and fixed limit. It saves immediately, refreshes the form,
and clears old analysis results. Other videos' working profiles are retained.
The button is disabled during analysis or when a video has no valid default;
new uploads must be calibrated using measured dimensions.

Uploaded videos live in ignored `backend/data/uploads/`. By default, calibration
uses the same ignored `YOLO/calibrationtemp.json` as the CLI. `DATA_DIR` changes the
upload root, `CALIBRATION_PATH` selects the read-only baseline, and
`CALIBRATION_TEMP_PATH` optionally selects a different working file (otherwise
it is `calibrationtemp.json` beside the baseline). A server restart
clears analysis status/results, but retained videos and calibration remain. The
latest analysis can be recovered by reloading the page while the server runs.

The page checks the backend protocol and source revision before starting.
If it says **Restart backend**, an older Python process is serving the page or
Python files changed after startup. Stop that process with Ctrl+C, rerun
`start_backend.cmd`, and refresh. Editing HTML does not reload imported Python
modules. See [the playback diagnosis](PLAYBACK_DIAGNOSIS.md) for the observed
cause, measured 30 FPS verification, and remaining limits.

## How to host it

Host the HTML and Python API **together in one Docker web service**. A small Linux
VM or a Docker service on Render is a straightforward deployment. The included
container uses CPU inference; size CPU/RAM after timing your actual footage.
Static-only hosting cannot execute YOLO or the Python models.

Build and run from `DynamicSpeedLimit/`:

```powershell
docker build -t flow-traffic .
docker volume create flow-data
docker run --name flow-traffic --restart unless-stopped -p 8000:8000 --mount source=flow-data,target=/data flow-traffic
```

Open `http://localhost:8000`. The image includes the traffic videos and the
tracked YOLO and Model B weights. Keep `/data` on a persistent volume so uploads
and browser calibration survive container replacement. The container reads
`/data/calibration.json` as its optional baseline and saves edits to
`/data/calibrationtemp.json`. To retain your local baseline when deploying,
copy it to `/data/calibration.json`; otherwise the original filename presets
provide defaults. For a public domain,
put the service behind an HTTPS reverse proxy and restrict access to your team;
the API has no built-in user accounts.

For **Render**:

1. Create a **Web Service** from this repository using the **Docker** runtime.
   Set the root directory to `DynamicSpeedLimit`; use its `Dockerfile`.
2. Attach a persistent disk at `/data`. The Dockerfile already sets
   `DATA_DIR=/data`, `CALIBRATION_PATH=/data/calibration.json`, and
   `CALIBRATION_TEMP_PATH=/data/calibrationtemp.json`.
3. Set the health check path to `/api/health`. Keep a single service instance.
   The app binds to `0.0.0.0` and honors the platform's `PORT` environment variable.
4. After deployment, use the service's HTTPS URL for both HTML and API.

Render's filesystem is ephemeral unless you attach a disk; persistent disks
require a paid service. See [Render persistent disks](https://render.com/docs/disks)
and [Docker deployment](https://render.com/docs/docker).

The entry point uses [Waitress](https://flask.palletsprojects.com/en/stable/deploying/waitress/),
a production WSGI server that supports Windows and Linux. It runs eight HTTP
threads and one background analysis at a time. Job state is in memory: run
**one process / one instance**, and do not enable an auto-reloader. To support
multiple analysis workers later, move jobs/results to a shared database and
task queue. This deployment has no separate frontend build or CORS setup.

## API and verification

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | HTTP health check |
| `GET /api/videos` | List sample/uploaded videos by opaque ID |
| `POST /api/videos` | Upload multipart `video` |
| `GET /api/videos/<id>` | Processing dimensions and saved settings |
| `GET /api/videos/<id>/frame` | First frame as JPEG |
| `GET /api/videos/<id>/media` | Native media or H.264 stream; optional `start` for transcoded resume |
| `PUT /api/videos/<id>/settings` | Validate and save corners/dimensions/ratio/limit |
| `POST /api/videos/<id>/settings/revert` | Save this video's default settings to the working file |
| `POST /api/jobs` | Start with `{ "video_id": "…", "weather": false }` |
| `GET /api/jobs/current` | Recover the most recent job |
| `GET /api/jobs/<id>` | Read progress, averages, forecast, advisory |
| `GET /api/jobs/<id>/frame` | Latest annotated JPEG |
| `POST /api/jobs/<id>/cancel` | Request cooperative cancellation |

```powershell
python -m unittest discover -s backend/tests
python -m unittest discover -s YOLO/tests
```

The web tests use real short generated videos and mocked detection/forecasting;
they require no network or model downloads. For calibration methodology,
measurement limitations, and desktop/CLI modes, see [YOLO/README.md](YOLO/README.md).
