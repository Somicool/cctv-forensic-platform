"""Evaluate the trained India-specific vehicle detector (per-class + overall).

    python scripts/evaluate.py --data data/india_vehicles/data.yaml
    python scripts/evaluate.py --data ... --weights weights/india_vehicles.pt --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="Evaluate the India-specific vehicle detector")
    ap.add_argument("--data", default=str(HERE / "data" / "india_vehicles" / "data.yaml"))
    ap.add_argument("--weights", default=str(HERE / "weights" / "india_vehicles.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    m = model.val(data=args.data, imgsz=args.imgsz, device=args.device, split=args.split,
                  project=str(HERE / "runs" / "val"), name="india_vehicles", exist_ok=True)

    b = m.box
    names = m.names if hasattr(m, "names") else model.names
    print("\n============ India-Vehicle Detector - Evaluation ============")
    print(f"  split       : {args.split}")
    print(f"  Precision   : {float(b.mp):.3f}")
    print(f"  Recall      : {float(b.mr):.3f}")
    print(f"  mAP@0.50    : {float(b.map50):.3f}")
    print(f"  mAP@0.50:95 : {float(b.map):.3f}")
    print("  --- per class (AP@50 / AP@50-95) ---")
    try:
        for i, ci in enumerate(b.ap_class_index):
            nm = names[int(ci)] if isinstance(names, dict) else names[int(ci)]
            ap50 = float(b.ap50[i]) if b.ap50 is not None else float("nan")
            ap = float(b.ap[i].mean()) if b.ap is not None else float("nan")
            star = "  <== auto-rickshaw" if str(nm).lower() == "auto-rickshaw" else ""
            print(f"    {str(nm):16s} AP50={ap50:.3f}  AP={ap:.3f}{star}")
    except Exception as exc:  # noqa: BLE001
        print("    (per-class breakdown unavailable:", exc, ")")
    print("============================================================")


if __name__ == "__main__":
    main()
