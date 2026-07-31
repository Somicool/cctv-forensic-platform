"""Benchmark + validate the secondary-detector INTEGRATION (Phases 2-3).

Measures the real cost of running a secondary detector alongside the primary
YOLOv10 and merging into one stream, and validates the merge logic on real
frames. Uses the base yolo11n.pt as a STAND-IN secondary detector (relabelling
COCO 'car' -> 'auto-rickshaw') purely to exercise the plumbing on real inference
- the trained detector shares the same nano architecture, so the timing/VRAM are
representative. Auto-rickshaw precision/recall come from evaluate.py on the
trained model (see README).

    python scripts/benchmark_integration.py
"""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
BACKEND = HERE.parent / "backend"
sys.path.insert(0, str(BACKEND))

import cv2  # noqa: E402
from app import config  # noqa: E402

# --- inject a stand-in secondary detector (base yolo11n) for the benchmark ---
STANDIN = HERE / "yolo11n.pt"
if not STANDIN.exists():
    STANDIN = "yolo11n.pt"          # ultralytics will fetch it
config.SECONDARY_DETECTOR_SPECS = [{
    "name": "standin_india_vehicles", "weights": str(STANDIN),
    # relabel a couple of COCO classes as India-specific just to exercise the
    # multi-class merge plumbing (car->auto-rickshaw, truck->tractor).
    "class_map": {2: (100, "auto-rickshaw"), 7: (101, "tractor")},
    "conf": 0.35, "imgsz": 640, "enabled": True,
}]

from app.ingestion import detector          # noqa: E402
from app.ingestion.detectors import plugins  # noqa: E402
plugins.reset()

import torch  # noqa: E402


def load_frames(n=30):
    fr = config.FRAME_DIR
    vids = ["Export__Chauta-Bazaar-003_Friday-July-10",
            "Export__Rly-Station-Towards-Bismillah-Re",
            "Export__Mahidharpura-Pipla-Sheri-Diamond"]
    paths = []
    for v in vids:
        fs = sorted(glob.glob(str(fr / v / "**" / "*.jpg"), recursive=True))
        if fs:
            paths += [fs[int(i)] for i in np.linspace(0, len(fs) - 1, num=min(n // 3 + 1, len(fs)))]
    return [cv2.imread(p) for p in paths[:n] if cv2.imread(p) is not None]


def main():
    frames = load_frames(30)
    if not frames:
        print("No frames found under backend/data/frames - run an ingest first.")
        return
    print(f"[bench] {len(frames)} real frames  | device={config.DEVICE}  imgsz={config.YOLO_IMGSZ}")
    print(f"[bench] plugins active: {plugins.active()}  ({[p['name'] for p in plugins.get_plugins()]})")

    primary = detector.get_model()
    if config.DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated() / 1024**2

    t_primary = t_secondary = t_merge = 0.0
    n_primary = n_secondary = n_suppressed = n_tracked = n_kept_sec = 0
    sec_tracker = plugins.IoUTracker()

    for i, frame in enumerate(frames):
        # primary detect (COCO)
        s = time.perf_counter()
        pr = primary.predict(frame, conf=config.DETECT_CONF, imgsz=config.YOLO_IMGSZ,
                             classes=list(config.PRIMARY_CLASSES), device=config.DEVICE, verbose=False)
        t_primary += time.perf_counter() - s
        prim = []
        for r in pr:
            if r.boxes is None:
                continue
            for b in r.boxes:
                prim.append({"cls_id": int(b.cls[0]), "conf": float(b.conf[0]),
                             "xyxy": tuple(float(v) for v in b.xyxy[0].tolist())})
        n_primary += len(prim)

        # secondary detect
        s = time.perf_counter()
        sec = plugins.detect_frame(frame)
        t_secondary += time.perf_counter() - s
        n_secondary += len(sec)

        # merge (class-aware NMS: specific Indian class wins, generics dedupe)
        s = time.perf_counter()
        kept_p, kept_s = plugins.merge_detections(prim, sec)
        tracked = sec_tracker.update(kept_s, i)
        t_merge += time.perf_counter() - s
        n_suppressed += len(prim) - len(kept_p)      # primary vehicles replaced
        n_tracked += sum(1 for t in tracked if "track_id" in t)
        n_kept_sec += len(kept_s)

    peak_mem = (torch.cuda.max_memory_allocated() / 1024**2 - base_mem) if config.DEVICE == "cuda" else 0.0
    N = len(frames)
    print("\n================ INTEGRATION BENCHMARK ================")
    print(f"  primary  YOLOv10b : {1000*t_primary/N:6.1f} ms/frame  ({n_primary} boxes)")
    print(f"  secondary  nano   : {1000*t_secondary/N:6.1f} ms/frame  ({n_secondary} boxes)")
    print(f"  merge (NMS+track) : {1000*t_merge/N:6.2f} ms/frame")
    print(f"  ADDED by secondary: {1000*(t_secondary+t_merge)/N:6.1f} ms/frame  "
          f"(+{100*(t_secondary+t_merge)/max(t_primary,1e-9):.0f}% of detect stage)")
    print(f"  extra GPU memory  : {peak_mem:6.0f} MB")
    print("  --- plumbing validation ---")
    print(f"  class-aware NMS: {n_suppressed} primary boxes replaced by specific/dedup")
    print(f"  secondary kept after merge: {n_kept_sec}  (all tracked: {n_tracked == n_kept_sec})")
    print("======================================================")
    # end-to-end estimate for a 300s clip @2fps (600 sampled frames)
    add_ms = 1000 * (t_secondary + t_merge) / N
    print(f"\nEstimated end-to-end: +{add_ms*600/1000:.1f}s on a 300s/2fps clip (600 frames).")


if __name__ == "__main__":
    main()
