"""Frame extraction.

Samples frames from a video at a target FPS and records, for every saved
frame: the camera id, the original frame number, the time offset from the
start of the clip, and a real-world ISO timestamp. These FrameInfo records
are what the rest of the ingestion pipeline consumes.

CLI:
    python -m app.ingestion.video_processor <video> --camera CAM-01 --fps 2
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

from .. import config


@dataclass
class FrameInfo:
    camera_id: str
    video_name: str
    frame_number: int        # index in the ORIGINAL video stream
    sampled_index: int       # 0,1,2,... among the frames we actually kept
    offset_seconds: float    # seconds elapsed from the start of the clip
    timestamp: str           # real-world ISO timestamp of this frame
    path: str                # where the JPEG was saved


def _parse_start_time(start_time) -> datetime:
    """Wall-clock time of the first frame. Defaults to 'now' (UTC)."""
    if start_time is None:
        return datetime.now(timezone.utc)
    if isinstance(start_time, datetime):
        return start_time
    return datetime.fromisoformat(str(start_time))


def extract_frames(
    video_path,
    camera_id: str,
    start_time=None,
    fps: float | None = None,
    frame_root=None,
    jpeg_quality: int = 90,
) -> list[FrameInfo]:
    """Extract frames from one video at `fps` frames per second."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    target_fps = fps or config.DEFAULT_FPS
    frame_root = Path(frame_root or config.FRAME_DIR)
    base_time = _parse_start_time(start_time)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if native_fps <= 0:
        native_fps = 30.0  # some streams don't report fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    interval = max(1, round(native_fps / target_fps))  # keep every Nth frame

    out_dir = frame_root / camera_id / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[FrameInfo] = []
    idx = 0
    sampled = 0
    while True:
        # grab() advances without decoding (cheap); we only decode the frames
        # we intend to keep via retrieve().
        if not cap.grab():
            break
        if idx % interval == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            offset = idx / native_fps
            ts = base_time + timedelta(seconds=offset)
            fname = out_dir / f"frame_{sampled:06d}.jpg"
            cv2.imwrite(str(fname), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            frames.append(FrameInfo(
                camera_id=camera_id,
                video_name=video_path.name,
                frame_number=idx,
                sampled_index=sampled,
                offset_seconds=round(offset, 3),
                timestamp=ts.isoformat(),
                path=str(fname),
            ))
            sampled += 1
            if sampled % 25 == 0:
                print(f"  ... {sampled} frames extracted")
        idx += 1

    cap.release()
    duration = (total_frames / native_fps) if total_frames else (idx / native_fps)
    print(f"[frames] {video_path.name}: kept {sampled} frames @ {target_fps}fps "
          f"(native {native_fps:.1f}fps, ~{duration:.1f}s, camera {camera_id})")
    return frames


def extract_from_directory(video_dir=None, fps: float | None = None, start_time=None):
    """Extract frames from every video in a directory.

    Camera id is inferred from a 'CAM-XX_...' filename prefix, else CAM-01.
    """
    video_dir = Path(video_dir or config.VIDEO_DIR)
    exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    videos = (
        sorted(p for p in video_dir.iterdir() if p.suffix.lower() in exts)
        if video_dir.exists() else []
    )
    results: dict[str, list[FrameInfo]] = {}
    if not videos:
        print(f"[frames] no videos found in {video_dir}")
        return results
    for v in videos:
        prefix = v.stem.split("_")[0]
        camera_id = prefix if prefix.upper().startswith("CAM") else "CAM-01"
        results[v.name] = extract_frames(v, camera_id=camera_id, fps=fps, start_time=start_time)
    return results


def _cli():
    ap = argparse.ArgumentParser(description="Extract frames from a video")
    ap.add_argument("video")
    ap.add_argument("--camera", default="CAM-01")
    ap.add_argument("--fps", type=float, default=config.DEFAULT_FPS)
    ap.add_argument("--start-time", default=None, help="ISO datetime of the first frame")
    args = ap.parse_args()
    frames = extract_frames(args.video, args.camera, start_time=args.start_time, fps=args.fps)
    print(f"Done. {len(frames)} frames saved.")


if __name__ == "__main__":
    _cli()
