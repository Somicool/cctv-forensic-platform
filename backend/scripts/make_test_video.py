"""Generate a short synthetic CCTV-style clip so the ingestion pipeline can
be tested end-to-end without real footage.

The clip has known, controlled content:
  - a RED rectangle ("vehicle") moving left  -> right
  - a BLUE rectangle ("person")  moving right -> left
  - a burnt-in timestamp per frame

    python scripts/make_test_video.py

Writes to backend/data/videos/CAM-01_synthetic.mp4
"""
import sys
from pathlib import Path

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import config  # noqa: E402


def make(path=None, seconds: int = 6, fps: int = 30, size=(640, 480)) -> Path:
    path = Path(path or (config.VIDEO_DIR / "CAM-01_synthetic.mp4"))
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open (codec issue)")

    n = seconds * fps
    for i in range(n):
        frame = np.full((h, w, 3), 30, dtype=np.uint8)          # dark backdrop
        # "vehicle" (red, BGR) moving left -> right
        x = int((i / n) * (w - 120))
        cv2.rectangle(frame, (x, 300), (x + 100, 350), (0, 0, 200), -1)
        # "person" (blue, BGR) moving right -> left
        px = int((1 - i / n) * (w - 60)) + 20
        cv2.rectangle(frame, (px - 12, 180), (px + 12, 260), (200, 120, 0), -1)
        cv2.putText(frame, f"t={i / fps:0.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()
    print(f"wrote {path} ({n} frames, {seconds}s @ {fps}fps)")
    return path


if __name__ == "__main__":
    make()
