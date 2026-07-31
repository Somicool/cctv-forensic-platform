"""Export the trained auto-rickshaw detector for deployment.

Keeps the PyTorch .pt (what the main pipeline would load via ultralytics) and
optionally emits ONNX for a runtime-agnostic option.

    python scripts/export.py                       # verify + copy best.pt
    python scripts/export.py --onnx                # also export ONNX
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="Export the auto-rickshaw detector")
    ap.add_argument("--weights", default=str(HERE / "weights" / "india_vehicles.pt"))
    ap.add_argument("--onnx", action="store_true", help="also export an ONNX model")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    w = Path(args.weights)
    if not w.exists():
        raise SystemExit(f"weights not found: {w}\nTrain first: python scripts/train.py --data ...")

    from ultralytics import YOLO
    model = YOLO(str(w))
    print(f"[export] loaded {w}  classes={model.names}")

    (HERE / "weights").mkdir(exist_ok=True)
    stable = HERE / "weights" / "india_vehicles.pt"
    if w.resolve() != stable.resolve():
        shutil.copy2(w, stable)
    print(f"[export] PyTorch weights ready -> {stable}")

    if args.onnx:
        path = model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True)
        print(f"[export] ONNX -> {path}")


if __name__ == "__main__":
    main()
