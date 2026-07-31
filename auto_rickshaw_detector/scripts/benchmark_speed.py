"""Measure inference speed of the (base or trained) detector on THIS machine.

Reports mean latency + FPS so you know the real cost on your RTX 3050 before
deciding to run it as a second detection pass. Works with the base yolo11n.pt
even before training, so you can measure the architecture's speed up-front.

    python scripts/benchmark_speed.py                      # base yolo11n on GPU
    python scripts/benchmark_speed.py --weights weights/auto_rickshaw.pt
    python scripts/benchmark_speed.py --device cpu --imgsz 640
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="Benchmark detector inference speed")
    ap.add_argument("--weights", default="yolo11n.pt",
                    help="weights to time (default: base yolo11n.pt, auto-downloaded)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    dummy = (np.random.rand(args.imgsz, args.imgsz, 3) * 255).astype("uint8")

    for _ in range(args.warmup):
        model.predict(dummy, imgsz=args.imgsz, device=args.device, verbose=False)

    t = []
    for _ in range(args.runs):
        s = time.perf_counter()
        model.predict(dummy, imgsz=args.imgsz, device=args.device, verbose=False)
        t.append((time.perf_counter() - s) * 1000.0)
    t = np.array(t)

    print("\n=============== Inference speed ===============")
    print(f"  weights : {args.weights}")
    print(f"  device  : {args.device}   imgsz: {args.imgsz}")
    print(f"  mean    : {t.mean():.1f} ms/frame  (min {t.min():.1f}, p95 {np.percentile(t,95):.1f})")
    print(f"  FPS     : {1000.0 / t.mean():.1f}")
    print("===============================================")


if __name__ == "__main__":
    main()
