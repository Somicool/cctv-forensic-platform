"""Create an 'unfamiliar' clip the system has never ingested, by transforming an
existing real clip (horizontal mirror + brightness/contrast shift). The result is
real, detectable footage from a genuinely different-looking scene - used to test
generalisation to unseen footage and dynamic camera registration.

    python scripts/make_unfamiliar_clip.py     # CAM-02_plaza.avi -> CAM-06_market.mp4
"""
import sys
from pathlib import Path

import cv2

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import config                                   # noqa: E402


def make_unfamiliar(src=None, dst=None, max_seconds: float = 8.0):
    src = Path(src) if src else (config.VIDEO_DIR / "CAM-02_plaza.avi")
    dst = Path(dst) if dst else (config.VIDEO_DIR / "CAM-06_market.mp4")
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source clip {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(fps * max_seconds)

    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError("cv2.VideoWriter failed to open (mp4v codec)")

    n = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                              # mirror
        frame = cv2.convertScaleAbs(frame, alpha=1.12, beta=18)  # brighter + contrast
        out.write(frame)
        n += 1

    cap.release()
    out.release()
    return dst, n


if __name__ == "__main__":
    path, frames = make_unfamiliar()
    print(f"wrote {path} ({frames} frames)")
