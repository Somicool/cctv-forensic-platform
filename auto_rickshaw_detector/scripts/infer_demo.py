"""Quick visual check of the trained auto-rickshaw detector on images.

Runs detection on an image or a folder and saves annotated results to
./runs/predict/. Handy for eyeballing quality on your own CCTV frames before
any integration.

    python scripts/infer_demo.py --source path/to/image_or_folder
    python scripts/infer_demo.py --source ../backend/data/frames --conf 0.35
"""
from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="Run the auto-rickshaw detector on images")
    ap.add_argument("--source", required=True, help="image file or folder")
    ap.add_argument("--weights", default=str(HERE / "weights" / "india_vehicles.pt"))
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    w = Path(args.weights)
    if not w.exists():
        raise SystemExit(f"weights not found: {w}\nTrain first: python scripts/train.py --data ...")

    from ultralytics import YOLO
    model = YOLO(str(w))
    res = model.predict(source=args.source, conf=args.conf, imgsz=args.imgsz,
                        device=args.device, save=True,
                        project=str(HERE / "runs" / "predict"), name="auto_rickshaw",
                        exist_ok=True, verbose=False)
    total = sum(len(r.boxes) if r.boxes is not None else 0 for r in res)
    print(f"[infer] {len(res)} image(s), {total} auto-rickshaw detection(s)")
    if res:
        print(f"[infer] annotated output -> {res[0].save_dir}")


if __name__ == "__main__":
    main()
