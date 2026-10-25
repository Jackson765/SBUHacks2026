"""Browser-compatible media transport; detector timing keeps the original source."""
import logging
from math import isfinite
import os
from pathlib import Path
import subprocess
import tempfile

import cv2


def codec_info(path):
    cap = cv2.VideoCapture(str(path))
    try:
        value = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((value >> (8 * index)) & 255) for index in range(4)).strip("\0").lower()
    finally:
        cap.release()
    native = Path(path).suffix.lower() in {".mp4", ".m4v", ".mov"} and codec in {"avc1", "h264"}
    return dict(source_codec=codec, media_transcoded=not native)


def h264_stream(path, start=0):
    """Encode fragmented MP4 as needed, keeping source frame timing and aspect ratio."""
    start = float(start)
    if not isfinite(start) or start < 0 or start >= 900:
        raise ValueError("Media start must be between 0 and 900 seconds.")
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError as error:
        raise ValueError("Install backend/requirements.txt to enable H.264 browser playback.") from error
    command = [get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin",
               "-threads", "2", "-ss", str(start), "-i", str(path), "-t", str(900-start),
               "-map", "0:v:0", "-an", "-sn", "-dn", "-vf",
               "scale=w='min(960,iw)':h='min(960,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
               "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
               "-threads", "2", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-g", "30",
               "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
               "-frag_duration", "1000000", "-f", "mp4", "pipe:1"]
    def chunks():
        log = tempfile.TemporaryFile()
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except Exception:
            log.close()
            raise
        try:
            while True:
                block = process.stdout.read1(65536)
                if not block:
                    break
                yield block
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
            log.seek(0)
            errors = log.read(4096).decode("utf-8", errors="replace").strip()
            log.close()
            if errors:
                logging.getLogger(__name__).warning("Media encoder: %s", errors)
    return chunks()
