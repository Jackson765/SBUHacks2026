# Calibrated video speeds and prediction

Run from the project root in the existing `dynamic-speed` environment:

```powershell
conda activate dynamic-speed

# Analyze the default video as quickly as decoding and inference allow.
python YOLO/main.py

# Choose a video and watch detection with its saved settings.
python YOLO/main.py --visual
python YOLO/main.py --visual --video "traffic/traffic4.mp4"

# Play with corner dragging and calibration sliders enabled.
python YOLO/main.py --calibrate
python YOLO/main.py --calibrate --video "traffic/traffic4.mp4"

# Optionally fetch weather and print an advisory after the prediction.
python YOLO/main.py --weather

# Use full-frame inference and the original FRAME_SKIP sampling.
python YOLO/main.py --full-analysis
```

`--visual` locks calibration during playback. `--calibrate` enables calibration
controls on prerecorded video and implies visual playback; `--visual --calibrate`
is also accepted. `--recalibrate` is a compatibility alias. Space pauses/resumes;
Q/Escape or closing the playback window stops the run.

Playback uses nonblocking GUI event polling while running. Each decoded frame
is displayed; frames between detections reuse the previous boxes/speed labels
without extra inference or speed samples. Paused video redraws only after a
control/state change. Detection resolution, sampling, Model B and
simulation calculations are unchanged; processing speed still depends on hardware.

## Video settings and calibration

The single default is `VIDEO_PATH` in `config.py` (`traffic4.mp4`).
`--video <path>` selects a file explicitly; `--default-video` selects that same
default and bypasses the picker. Relative paths start at the terminal's current
directory. Visual and calibration modes open a file picker when no source is
specified.

`YOLO/video_defaults.json` contains the original filename presets, migrated
without changing their values. This file is tracked with the application.
`YOLO/calibration.json` is the local per-video settings store (version 2), ignored
by Git. Its `profiles` object holds saved settings keyed by absolute video path,
including file size/modification time and processing size. The previous saved
profile was retained, and older version-1 profiles also load without losing
geometry or run settings. Files without calibrated dimensions need `--calibrate`
before analysis. Each settings file is read once per run.

All modes use matching saved settings first, then the migrated filename defaults.
`--ratio 2` overrides the selected ratio for that run without changing the saved
profile. A profile for a replaced file is ignored; a resolution mismatch requires
recalibration. The ratio stays fixed in normal analysis and locked playback.
The input speed limit stays fixed in every mode.

Use a stationary recording of a flat road rectangle with **measured** dimensions.
Width runs across the road; length runs along the road. `--calibrate` starts
with setup, then opens editable playback:

1. Select the video and enter the ratio, speed limit, and measured dimensions.
2. Drag the first frame's four colored handles to the measured road corners.
3. Press Enter to save and start playback with calibration controls. R resets
   the handles, Backspace undoes a drag, and Q/Escape cancels setup.

| Handle | Color | Corner |
| --- | --- | --- |
| 1 | Red | Far-left |
| 2 | Green | Far-right |
| 3 | Blue | Near-right |
| 4 | Yellow | Near-left |

Far and near refer to distance from the recording position. Width is between
1-2 / 4-3; length is between 1-4 / 2-3. A crossed or degenerate polygon cannot be
accepted. Initial handle positions are positioning aids, not a measurement.

During `--calibrate` playback, drag the corners or use the Scale %, Width %,
Length %, and Ratio % sliders. Percentages apply to the starting settings. Space
pauses/resumes, U undoes a drag, and R resets the controls. Press **S** to save the
current settings for future analysis and playback. Ordinary `--visual` playback has
none of these editing controls.

Accepted calibration or ratio changes restart the three five-minute windows at
the current video time. This prevents measurements from different calibrations
entering one forecast. A further 15 minutes of valid video history is needed
after an edit to complete collection.

Setup and S save geometry, effective dimensions, ratio, and fixed input limit
atomically to `YOLO/calibration.json`, retaining other videos' profiles. The next
normal or visual run reuses the saved settings automatically. No source editing
or copying terminal output is needed.

Frames preserve their aspect ratio and are downscaled only when their longest
edge exceeds `MAX_PROCESSING_DIMENSION`. Display and setup coordinates are
mapped back to the processing frame before saving.

All prerecorded modes skip only leading finite negative decoder timestamps,
then retain the first usable frame's actual timestamp. Later missing,
nonfinite, negative, or non-increasing timestamps still fail validation.
There is no automatic FPS fallback or timestamp clamping. Explicit verified
constant-FPS timing continues to use frame indexes.

## Fast simulation and three averages

Default analysis opens no preview and does not wait for real time to pass. The
existing `simulation.py` plan samples approximately five frames per video second,
uses a padded crop around the calibrated road, and maps detections back to the
original processing coordinates. Skipped frames advance the decoder without
display conversion. `SIMULATION_TARGET_FPS` and `SIMULATION_ROAD_PADDING` in
`config.py` control the plan; sampling stays within the tracker's maximum gap.

`--full-analysis` uses full-frame inference at `DETECTION_IMAGE_SIZE` with
`FRAME_SKIP`; visual playback also uses those settings. Sampling and crop context
can affect detections and measured averages. Runtime depends on hardware and
video decoding; visual playback also advances at processing speed.

All collection boundaries use **video time**, independently of wall time:

| Average | Video time |
| --- | --- |
| First | 00:00 up to 05:00 |
| Second | 05:00 up to 10:00 |
| Third | 10:00 up to 15:00 |

Within each block, the collector averages each tracked vehicle's valid mph
measurements and then averages across vehicles. A vehicle crossing a boundary
contributes separately to each block. A measured stopped vehicle contributes
zero; a block with no valid measurements stays missing. Only measurements inside
the calibrated polygon are used. Tracking errors or fragmented IDs can affect
the weighting.

The terminal reports progress and each completed block, including the
`10-15 minutes` average. Collection ends after 15 minutes and closes visual playback.
In calibration mode, these elapsed times start again after each accepted edit.
Short visual clips play to EOF. A forecast requires all three completed blocks
with valid averages; short or cancelled runs do not invent missing history.

`ratio_predict.py` scales those three averages exactly once, calls the existing
Model B predictor, then divides its predicted speed by the same ratio:

```text
Local averages:   [25, 28, 30] mph
Ratio:           2
Model B inputs:  [50, 56, 60]
Model B output:  55
Local forecast:  55 / 2 = 27.5 mph
```

These numbers illustrate the arithmetic. The forecast predicts average speed
10 minutes after collection ends. It is separate from the fixed input speed
limit. Model B, its trained artifacts, and `simulation.py` are preserved.

For direct prediction from three existing averages:

```powershell
python YOLO/ratio_predict.py --ratio 2 --history 25 28 30
```

## Optional weather advisory

`--weather` fetches Open-Meteo weather only after a completed forecast and prints
a short weather category, advisory/status, and relevant reasons. Normal runs
neither import the optional weather modules nor make weather requests. Weather
failure preserves the existing forecast. The advisory uses its final local-mph
speed once and does not change the fixed input sign.

Current weather is applied as a scenario to the recorded video; it is not weather
measured when the footage was recorded. The rules are experimental, with settings
in `config.py`:

| Setting | Meaning |
| --- | --- |
| `ADVISORY_FREE_FLOW_SPEED_MPH` | 55 mph assumed free-flow speed; replace with a measured value. |
| `ADVISORY_ROAD_TYPE` | `unknown` by default, requiring manual review. |
| `ADVISORY_POLICY` | Traffic/weather rules, freshness limits, and rounding. |
| `WEATHER_LOCATION` | Stony Brook, NY area (40.91, -73.14). |
| `WEATHER_TIMEOUT` / `WEATHER_CACHE_SECONDS` | Bounded requests and a ten-minute process cache. |

Unknown/unsupported roads and severe weather require manual review; missing
weather does not count as clear conditions. Invalid predictions or missing
required data cannot produce a combined advisory.

The weather parser checks units, observation time, coordinates, and accumulation
interval. Units include Fahrenheit, mph, precipitation/rain mm, snowfall cm, and
visibility m. Current accumulation values cover the reported interval, not an
assumed hourly rainfall intensity. Cached data is checked again for freshness.
Full policy and weather validation live in `decision_engine.py` and `api_code.py`.

## Dependencies, timing, and verification

Use the existing environment with `numpy`, `opencv-python`, `ultralytics`,
`lap>=0.5.12`, and Model B's `joblib` / `lightgbm` dependencies.
The optional weather command also uses `requests`. The file picker uses Python's `tkinter`. Vehicle weights are `YOLO/yolov8n.pt`.

On Windows, `main.py` sets `MKL_THREADING_LAYER=SEQUENTIAL` before NumPy imports
unless that environment variable is already set. This prevents the observed
Conda MKL/PyTorch OpenMP conflict (`WinError 127` loading `torch/lib/shm.dll`).
If a conflicting value was previously set, remove it before a fresh process:

```powershell
Remove-Item Env:MKL_THREADING_LAYER -ErrorAction SilentlyContinue
python YOLO/main.py
```

Speeds use decoded video timestamps and a one-second position history. Tracks
need sufficient accepted samples before a speed appears. Timestamp-based EOF
handling includes the final duration only when supported by constant-FPS
metadata; unknown durations are not invented. Use `TIMESTAMP_SOURCE = "fps"`
only for verified constant-frame-rate footage, with `FPS_OVERRIDE` if needed.

Road slopes, lens distortion, occlusion, contact-point estimates, tracking, and
recording motion affect accuracy. Independently time vehicles over another
measured distance to validate mph; validate the ratio and transferred forecasts
against later local measurements as well.

```powershell
conda run -n dynamic-speed --no-capture-output python -m unittest discover -s YOLO/tests
```

Tests cover calibration geometry, speed measurement, video-time collection,
ratio scaling, run modes, simulation planning, and advisory/weather validation.
GUI interactions and long video timelines are simulated in automated tests.
