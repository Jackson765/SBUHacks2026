# Playback diagnosis — 2026-10-25

The repeated 5x playback report had a live-runtime cause, plus two distinct
browser-display problems. Earlier checks verified Python timing and mode flags,
but did not count actual presented browser frames.

| Finding | Evidence | Fix |
| --- | --- | --- |
| An old backend ignored the checkbox | Port 8000 belonged to Python PID 56600, started at 04:25:36. Its cancelled-job response lacked `quick_analysis` and `wall_elapsed_seconds`, although its HTML contained the new controls. | Protocol 2, source revision, acknowledged processor, and restart-required checks. Incompatible servers cannot silently start analysis. The confirmed idle old process was replaced by the updated backend. |
| JPEG snapshots could not display 30 FPS | Regular snapshots were limited to one per 100 ms, with another 100 ms client delay after fetch, displaying fewer than 10 frames/s. | Regular mode uses native video at fixed playback rate 1. Quick mode retains snapshots. |
| The default video was not browser-compatible in this environment | `traffic4.mp4` is HEVC, 1080x1920, 29.9995426 FPS. Direct Edge playback advanced its clock but reported videoWidth 0 and zero presented frames. | HEVC/other formats stream as H.264 fragmented MP4. Missing video tracks produce an error instead of silent audio-only playback. |

The retained decoded timestamps run approximately 33.333 ms apart. Detection
still skips the first negative preroll frame. Measurement uses the original
timestamps rather than replacing them with an invented exact-30-FPS clock.

- **Quick checked:** `process_video_quick` forces pacing off and uses the fast
  inference plan unless full analysis was explicitly selected. The browser
  displays detection snapshots.
- **Quick unchecked:** `process_video_realtime` forces paced full-frame analysis.
  The browser independently presents source video at 1x, preserving smooth
  playback when detection cannot process 30 frames each second.

Both methods share the measurement and forecast calculations. Regular analysis
cannot advance faster than wall time. Media time and analyzed time are displayed
separately; averages refer to analyzed frames. Regular video shows the road
calibration overlay with analyzed metrics beside it. Late detector boxes are
not stamped onto unrelated current video frames.

FFmpeg retains source frame timing and aspect ratio, encoding H.264/YUV420P
with a maximum dimension of 960 pixels. Conversion begins on demand, streams
without converting the whole file first, and stops when the viewer disconnects.
Original videos are unchanged. The binary comes from
[imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg); fragmented MP4 uses
the documented [FFmpeg format flags](https://ffmpeg.org/ffmpeg-formats.html).

## Verification

All **127 YOLO tests and 26 HTTP tests pass**, covering an actual 30-FPS
three-second clip in both modes, frame intervals, EOF duration, no catch-up
bursts, HTTP byte ranges, H.264 streaming, dispatch, and stale-revision rejection.

A real Edge test on the actual default recording measured playback after the
actual YOLO detector began processing frames:

| Measurement | Result |
| --- | --- |
| Actual wall time | 5.003900 s |
| Native video time advanced | 5.003984 s |
| Playback rate | 1.0x |
| Presented / decoded frames | 150 / 150 |
| Presented frame rate | 29.9766 FPS |
| Dropped frames | 0 |
| Regular-mode JPEG requests | 0 |
| Analyzed source / total job wall time | 4.566756 s / 8.984 s |
| Stale-server request blocked | Yes; no new job created |

The UI test also switches Quick analysis on, confirms quick-processor dispatch
and JPEG preview selection, and verifies accelerated source progress. That
quick UI routing segment uses a controlled processor fixture; HTTP and core
tests exercise the actual quick pipeline with mocked detections. Browser timing
uses real elapsed time, `requestVideoFrameCallback`, and playback-quality frame
counts, rather than virtual test time.

Playback can buffer on slow machines/connections, but is never intentionally
set to 5x. Analysis can lag media playback, as the measurements above show.
Full 15-minute inference throughput and physical speed accuracy remain separate
validation tasks. Local raw test output is ignored at
`backend/.qa/playback-diagnosis.json`.
