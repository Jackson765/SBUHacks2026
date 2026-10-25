# YOLO reliability, speed, and simplification plan

Reviewed and implemented October 25, 2026. Findings below describe the original
code; implementation and verification results follow at the end.

## Agreed default behavior

- `python YOLO/main.py`: process the configured video without a preview, then
  print three five-minute averages and the existing scaled/unscaled prediction.
- `--visual`: display playback using saved settings, with calibration locked.
- `--calibrate`: edit calibration during prerecorded playback and save with S.
- Weather/advisory becomes opt-in, for example `--weather`, as selected by the user.
- Keep Model B, its weights/data, the existing simulation algorithm, speed
  measurement, three-window aggregation, and ratio arithmetic.

## Findings from the current code

| Finding | Evidence | Consequence |
| --- | --- | --- |
| Default startup fails on the first timestamp | `main.py:389` passes `skip_preroll=visual`. Actual traffic4 timestamps begin -35.7667, 33.3333, 66.6667 ms. | Default analysis rejects frame zero; visual/calibration already skip that negative prefix. |
| Startup tests preserve the failing behavior | `test_video_input.py` explicitly expects the default negative timestamp to be passed through and rejected. All 111 tests still pass. | Passing the suite currently does not establish that the normal command succeeds. |
| Too many responsibilities in the entry point | `main.py` has 581 lines; `main()` alone has 226. It mixes settings, decoder policy, inference, collection, controls, rendering, and advisory output. | Small mode changes can break unrelated behavior. |
| Settings have inconsistent sources | Headless analysis uses `VIDEO_PRESETS`; playback prefers the single saved calibration JSON. `VIDEO_PATH` and `PRESET_VIDEO_NAME` also select different defaults. | Saving calibration does not automatically update the normal run, and profiles for different videos overwrite one another. |
| Advisory text is on the default path | Main prints the long policy label before inference, and the optional formatter prints more labels/reasons later. Weather/decision modules are imported unconditionally. | Noisy output and unnecessary default dependencies; not the principal per-frame CPU cost. |
| Some legacy API interfaces have no application callers | `generateWeather()`, `getWeather()`, and compatibility globals are unused by the actual forecast path. | Extra state and interfaces to maintain. |
| Inference dominates runtime | Recent native profile: approximately 49.4 ms per full-frame inference versus 1.0 ms for plotting. | Removing strings alone will not materially speed up video analysis. |

## Implementation order

### 1. Repair the normal command first

Use one initial-frame policy for prerecorded inputs: skip only leading finite
negative decoder timestamps, retain the first valid frame's actual timestamp,
then keep strict monotonic/nonfinite checks for the rest of the video. Reuse the
existing reader helper instead of introducing a second clock. Do not clamp
timestamps to zero or silently switch to FPS timing.

This is an input/startup correction in `main.py`, not a change to `simulation.py`
or Model B mathematics. Omitting a leading frame can change sampling alignment,
so record new actual outputs rather than promise the old numeric forecast.

Replace the regression that preserves the crash with coverage that the bare
command proceeds past valid preroll. Keep tests rejecting genuinely invalid
timing. Then run the actual default video through all three windows.

### 2. Make the default output small

Remove the repeated `UNVALIDATED DEMO` banner, pending-advisory display, and
advisory prose from the normal run. Print a short startup line, concise progress,
the three averages, and the prediction.

Add one explicit weather option controlling imports, retrieval, decision, and
display. Import `api_code`/`decision_engine` only on that optional path. Keep its
output to weather, advisory/status, and a short reason; keep genuine missing-data
and severe-condition handling. A failed weather lookup must preserve the forecast.

### 3. Consolidate video settings

Use one settings loader and a per-video saved store for geometry, dimensions,
ratio, and fixed limit. Migrate current preset values exactly. Define a single
default video and explicit precedence: command overrides, matching saved settings,
then migrated defaults. Announce the selected settings briefly.

Review migration precedence before applying saved profiles to headless analysis;
do not silently change the existing simulation's inputs. Eliminate manual source
copy/paste and the hidden `PROMPT_FOR_INPUTS` mode once the migration is verified.
Remove the duplicate JSON read in `load_visual_settings()`.

### 4. Simplify mode control without duplicating the pipeline

Make `main()` coordinate argument parsing, settings, and execution. Keep one
shared analysis loop using the existing tracker, `FiveMinuteAverages`, and ratio
adapter. Put display composition in `ui.py` and video settings/persistence in a
focused helper. Keep calibration-only edit handling behind that mode's controls.

Remove unused legacy weather wrappers/globals and obsolete tests for those
interfaces. Preserve useful validation and behavior-level regression assertions.
Prefer plain functions and small data objects over extra frameworks or background
worker/queue machinery.

### 5. Optimize measured costs

Retain the existing simulation's cropped five-sample/second inference and decoder
grabs. Retain nonblocking playback events, fresh preview frames, and dirty-only
paused redraws. Those optimizations already exist and should not be reimplemented.

Benchmark playback-only inference sizes/device options on representative footage
before changing them. A smaller inference image can lose distant vehicles or
change tracking/averages; treat it as an explicit speed/accuracy setting rather
than a supposedly neutral cleanup. Keep the protected simulation settings intact.

Render overlays at display resolution/cache static sign elements only if profiling
shows a useful improvement. Keep weather/network calls outside the frame loop.

## Completion checks

1. The bare command finishes the real configured video, prints 0-5, 5-10, 10-15
   averages, and calls Model B once. Missing windows still prevent a prediction.
2. Default execution makes no weather request and prints no advisory banner.
3. Weather opt-in is mocked/offline in tests; failure does not erase the forecast.
4. `--visual` remains locked; `--calibrate` supports play/pause/edit/save/reload.
5. Ratios are applied exactly once to the three inputs and divided out once from
   the result; valid-input numerical tests and protected-file hashes stay intact.
6. Compare native wall time, inference count, preview update count, and detection
   quality using the same videos/settings. Report measured gains, not assumptions.

Review baseline: 111 tests passed. The real bare command was reproduced failing
with exit 1 before any averages, due to its first negative timestamp. No new full
15-minute prediction was produced during this review.

## Implementation results ? October 25, 2026

- Fixed initial negative decoder preroll in every prerecorded mode, retaining
  actual timestamps and strict validation after startup. The real bare command
  now completes all three windows.
- Default output is averages/prediction plus collection progress. `--weather`
  opts into lazy imports, one post-forecast lookup/decision, and concise output.
  Removed the repeated banner, pending-advisory UI and unused weather wrappers.
- Migrated all five preset entries exactly to tracked `video_defaults.json` and
  retained the previous saved crashtest profile in the local version-2
  `calibration.json` per-video store. Version-1 profiles also load automatically. One
  configured default is traffic4. Normal analysis and playback share settings;
  saving no longer requires editing source. CLI ratio > matching saved profile
  > migrated default. Geometry-only legacy data retains its previous ratio/limit
  fallback (1 / 30 mph for the existing crashtest profile).
- `main.py` shrank from 581 to 290 lines (`main()` from 226 to 52); it
  coordinates modes and one shared processing loop; input/timing helpers
  live in `video_input.py`, settings in `video_settings.py`, and preview composition
  in `ui.py`. Calibration now returns geometry and the caller saves once together
  with ratio/limit. Removed the hidden input switch and source dictionary.
- Preserved inference sizes/cadence, road crop, decoder grabs, nonblocking playback
  events, fresh skipped-frame previews and dirty-only paused redraws. No new
  dependency or speculative resolution/accuracy change was introduced.
- Real default: 900 source seconds processed in **183.72s**, **4,500 inferences**;
  averages **50.953767281001056, 50.1170255526554, 47.23769850142366 mph**;
  ratio **1**; Model B forecast **48.808 mph**. New sampling starts after the
  invalid leading frame; this explains changes from old reference forecasts.
  This is about 4.9x real time, not a controlled speed comparison with older runs.
- Native video/detector smoke: paused calibration, Scale200 / Ratio150, save,
  then locked playback reused 21.376m x 66.81216m / ratio1.5 / fixed45mph.
  GUI events were driven programmatically and images captured; desktop mouse
  interaction remains a manual check. Profiles used temporary paths.
- Final suite: **114 tests pass**, including numerical regressions, shared saved
  settings in headless/visual modes, legacy-profile loading, default-video
  selection and optional dependency isolation.
- Verification artifacts: `runs/refactor_default_run.log`,
  `runs/refactor_mode_evaluation.json`, `runs/refactor_protected_files.json`.
  Numerical regressions and protected SHA-256 comparisons retain Model B,
  simulation, tracker, ratio arithmetic and five-minute aggregation unchanged.

Remaining: manually check the native picker and controls in the desktop session;
validate physical calibration and local forecast accuracy with measured footage.
