"""Ensure a recording is a browser-playable, seekable H.264 MP4.

Real review software keeps footage in a format an operator can scrub frame-by
-frame. We do the same: at ingest, any non-mp4 source gets an .mp4 proxy created
alongside it (the original file is left untouched as evidence). Detection then
runs on the mp4 so the stored seek offsets line up exactly with what plays back.

Degrades gracefully: if ffmpeg isn't available or fails, we ingest the original.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

_MIN_OK_BYTES = 10_000


def ensure_mp4(src) -> Path:
    src = Path(src)
    if src.suffix.lower() == ".mp4":
        return src
    dst = src.with_suffix(".mp4")
    if dst.exists() and dst.stat().st_size > _MIN_OK_BYTES:
        return dst

    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        print(f"[transcode] imageio-ffmpeg unavailable; ingesting original {src.name}")
        return src

    # Downscale the proxy so the longer side is <= 1280 (never upscales) and use
    # a fast encoder. High-res phone footage (1080p/4K) then transcodes quickly
    # AND decodes far faster during analysis, with negligible search impact
    # (YOLO runs at 736px, CLIP at 224px). Seek offsets stay exact (derived from
    # this proxy). Audio is dropped.
    scale = ("scale='if(gt(iw,ih),min(1280,iw),-2)':"
             "'if(gt(iw,ih),-2,min(1280,ih))':force_divisible_by=2")
    cmd = [ffmpeg, "-y", "-i", str(src),
           "-vf", scale,
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0 and dst.exists() and dst.stat().st_size > _MIN_OK_BYTES:
            print(f"[transcode] {src.name} -> {dst.name} (H.264 playback proxy)")
            return dst
        print(f"[transcode] ffmpeg failed for {src.name}; ingesting original")
    except Exception as exc:  # noqa: BLE001
        print(f"[transcode] error on {src.name}: {exc}; ingesting original")
    return src
