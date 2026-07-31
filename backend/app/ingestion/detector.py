"""Object detection with YOLO.

Detects persons and vehicles in a frame, filters to the classes we care about
(config.DETECT_CLASSES) above a confidence threshold, and can save a cropped
JPEG for each detection (crops are what CLIP / OSNet / plate / face stages
consume downstream).

CLI (quick self-test on YOLO's bundled sample photo of people + a bus):
    python -m app.ingestion.detector
    python -m app.ingestion.detector path/to/image.jpg
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from .. import config

_model = None


def get_model():
    """Lazy-load the YOLO model once and reuse it."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(config.YOLO_MODEL)
    return _model


@dataclass
class Detection:
    class_id: int
    class_label: str
    confidence: float
    bbox: tuple            # (x, y, w, h) in pixels
    crop_path: str | None = None


def detect(image, conf: float | None = None, classes=None) -> list[Detection]:
    """Run YOLO on an image (file path, URL, or ndarray). Returns Detections
    filtered to the wanted classes."""
    model = get_model()
    conf = config.DETECT_CONF if conf is None else conf
    wanted = set(classes) if classes is not None else set(config.PRIMARY_CLASSES)

    results = model.predict(image, conf=conf, device=config.DEVICE,
                            imgsz=config.YOLO_IMGSZ, verbose=False)
    dets: list[Detection] = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            cls = int(b.cls[0])
            if cls not in wanted:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            dets.append(Detection(
                class_id=cls,
                class_label=config.DETECT_CLASSES.get(cls, str(cls)),
                confidence=float(b.conf[0]),
                bbox=(x1, y1, x2 - x1, y2 - y1),
            ))
    return dets


def crop_detection(image, det: Detection, out_path) -> str:
    """Crop a detection's bbox out of `image` (ndarray) and save it."""
    x, y, w, h = det.bbox
    x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
    x2, y2 = int(round(x + w)), int(round(y + h))
    crop = image[y1:y2, x1:x2]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    return str(out_path)


def detect_and_crop(image_path, out_dir, conf: float | None = None) -> list[Detection]:
    """Read an image file, detect, save a crop per detection, return them."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    dets = detect(img, conf=conf)
    out_dir = Path(out_dir)
    stem = Path(image_path).stem
    for i, d in enumerate(dets):
        d.crop_path = crop_detection(
            img, d, out_dir / f"{stem}_det{i:03d}_{d.class_label}.jpg"
        )
    return dets


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image = sys.argv[1]
    else:
        # YOLO ships a sample photo (people + bus) locally - no download needed.
        from ultralytics.utils import ASSETS
        image = str(ASSETS / "bus.jpg")

    dets = detect_and_crop(image, config.CROP_DIR / "test")
    print(f"{len(dets)} detections on {image}")
    for d in dets:
        bbox = tuple(round(v) for v in d.bbox)
        print(f"  {d.class_label:<10} conf={d.confidence:.2f} bbox={bbox} -> {d.crop_path}")
