"""Train the lightweight auto-rickshaw detector (YOLO11n / YOLOv8n).

Reads defaults from configs/train.yaml; any value can be overridden on the CLI.
Writes runs to ./runs/train/<name>/ and copies the best weights to
./weights/auto_rickshaw.pt on completion.

    python scripts/train.py --data data/auto_rickshaw/data.yaml
    python scripts/train.py --data ... --model yolov8n.pt --epochs 150 --imgsz 960
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent.parent
CFG = HERE / "configs" / "train.yaml"


def load_cfg():
    return yaml.safe_load(CFG.read_text(encoding="utf-8")) if CFG.exists() else {}


def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser(description="Train the India-specific vehicle detector")
    ap.add_argument("--data", default=str(HERE / "data" / "india_vehicles" / "data.yaml"),
                    help="path to data.yaml from prepare_dataset.py")
    ap.add_argument("--model", default=cfg.get("model", "yolo11n.pt"))
    ap.add_argument("--epochs", type=int, default=cfg.get("epochs", 100))
    ap.add_argument("--imgsz", type=int, default=cfg.get("imgsz", 640))
    ap.add_argument("--batch", type=int, default=cfg.get("batch", 16))
    ap.add_argument("--device", default=str(cfg.get("device", 0)))
    ap.add_argument("--name", default="india_vehicles_yolo")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)

    # hyper-params: config defaults, overridable above
    train_kwargs = dict(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=str(HERE / "runs" / "train"), name=args.name,
        patience=cfg.get("patience", 25), workers=cfg.get("workers", 8),
        cache=cfg.get("cache", True), optimizer=cfg.get("optimizer", "auto"),
        lr0=cfg.get("lr0", 0.01), seed=cfg.get("seed", 42),
        hsv_h=cfg.get("hsv_h", 0.015), hsv_s=cfg.get("hsv_s", 0.7), hsv_v=cfg.get("hsv_v", 0.4),
        degrees=cfg.get("degrees", 5.0), translate=cfg.get("translate", 0.1),
        scale=cfg.get("scale", 0.5), fliplr=cfg.get("fliplr", 0.5),
        mosaic=cfg.get("mosaic", 1.0), mixup=cfg.get("mixup", 0.1),
        resume=args.resume, exist_ok=True,
    )
    print(f"[train] model={args.model} epochs={args.epochs} imgsz={args.imgsz} "
          f"batch={args.batch} device={args.device}")
    results = model.train(**train_kwargs)

    # copy best weights to a stable location
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        (HERE / "weights").mkdir(exist_ok=True)
        dst = HERE / "weights" / "india_vehicles.pt"   # the name the backend plugin loads
        shutil.copy2(best, dst)
        print(f"\n[train] best weights -> {dst}")
    print(f"[train] run dir: {results.save_dir}")


if __name__ == "__main__":
    main()
