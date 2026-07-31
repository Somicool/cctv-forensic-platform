"""Multi-class dataset preparation for the India-specific vehicle detector.

Builds a clean MULTI-CLASS YOLO dataset from any of several offline sources,
preserving every useful Indian vehicle class the source supports (auto-rickshaw,
car, bus, truck, motorcycle, bicycle, tractor, tempo/LCV, HCV, mini-truck/Tata
Ace, ...) - it does NOT collapse to a single class.

Automatically:
  * inspects the source and identifies its classes,
  * canonicalises class names (merges synonyms -> one label),
  * removes corrupted / unreadable images,
  * splits train / val / test,
  * writes a correct data.yaml with all class names.

Sources
-------
  roboflow : turnkey download from Roboflow Universe (needs ROBOFLOW_API_KEY).
  idd      : India Driving Dataset (IDD-Detection), Pascal-VOC XML.
  folder   : any local dataset - auto-detects YOLO (images/+labels/) or VOC XML.

Examples
--------
  python scripts/prepare_dataset.py --source idd    --raw raw/IDD_Detection
  python scripts/prepare_dataset.py --source folder --raw raw/indian_vehicles
  python scripts/prepare_dataset.py --source roboflow --rf-workspace <ws> --rf-project <proj> --rf-version 1
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import yaml

HERE = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Normalised source class name -> canonical output label. Mirrors the backend's
# config.VEHICLE_NAME_ALIASES so the trained model's class names line up with the
# integration's alias resolver. Synonyms merge into one canonical label.
CANONICAL = {
    "autorickshaw": "auto-rickshaw", "auto": "auto-rickshaw", "rickshaw": "auto-rickshaw",
    "tuktuk": "auto-rickshaw", "threewheeler": "auto-rickshaw", "3wheeler": "auto-rickshaw",
    "autorick": "auto-rickshaw",
    "tractor": "tractor",
    "tempo": "tempo", "matador": "tempo", "tempotraveller": "tempo",
    "minitruck": "mini-truck", "tataace": "mini-truck", "ace": "mini-truck",
    "chotahathi": "mini-truck", "chhotahathi": "mini-truck",
    "hcv": "hcv", "heavyvehicle": "hcv", "heavycommercialvehicle": "hcv",
    "trailer": "hcv", "multiaxle": "hcv", "trucktrailer": "hcv",
    "lcv": "lcv", "lightcommercialvehicle": "lcv",
    "scooter": "scooter", "moped": "scooter", "activa": "scooter",
    "pickup": "pickup", "pickuptruck": "pickup", "goodscarrier": "pickup",
    "car": "car", "sedan": "car", "hatchback": "car", "suv": "car", "jeep": "car",
    "taxi": "car", "van": "car",
    "bus": "bus", "minibus": "bus",
    "truck": "truck", "lorry": "truck",
    "motorcycle": "motorcycle", "motorbike": "motorcycle", "bike": "motorcycle",
    "twowheeler": "motorcycle",
    "bicycle": "bicycle", "cycle": "bicycle",
    "person": "person", "pedestrian": "person", "rider": "person",
}


def _norm(s: str) -> str:
    return "".join(c for c in (s or "").lower() if c.isalnum())


def canon(name: str, keep_all: bool):
    """Canonical label for a source class name, or None to drop it (unless keep_all)."""
    c = CANONICAL.get(_norm(name))
    if c:
        return c
    return name.strip().lower() if keep_all else None


# --------------------------------------------------------------- image checks
def _image_ok(path: Path) -> bool:
    """Reject corrupted / unreadable / truncated images before they enter the set."""
    try:
        img = cv2.imread(str(path))
        if img is None or img.size == 0 or img.shape[0] < 8 or img.shape[1] < 8:
            return False
    except Exception:
        return False
    return True


# --------------------------------------------------------------- VOC (IDD)
def _voc_objects(xml_path: Path):
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None, []
    size = root.find("size")
    if size is None:
        return None, []
    W = float(size.findtext("width") or 0)
    H = float(size.findtext("height") or 0)
    if W <= 0 or H <= 0:
        return None, []
    objs = []
    for obj in root.findall("object"):
        name = obj.findtext("name") or ""
        b = obj.find("bndbox")
        if b is None:
            continue
        x1 = float(b.findtext("xmin")); y1 = float(b.findtext("ymin"))
        x2 = float(b.findtext("xmax")); y2 = float(b.findtext("ymax"))
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        bw, bh = (x2 - x1) / W, (y2 - y1) / H
        if bw > 0 and bh > 0:
            objs.append((name, cx, cy, bw, bh))
    return root.findtext("filename"), objs


def _find_image(xml_path: Path, filename, roots):
    cands = ([filename] if filename else []) + [xml_path.stem]
    for root in roots:
        for c in cands:
            for ext in IMG_EXTS:
                p = root / f"{Path(c).stem}{ext}"
                if p.exists():
                    return p
    for root in roots:
        for ext in IMG_EXTS:
            hits = list(root.rglob(f"{xml_path.stem}{ext}"))
            if hits:
                return hits[0]
    return None


def collect_voc(raw: Path, keep_all: bool):
    ann = raw / "Annotations" if (raw / "Annotations").exists() else raw
    roots = [p for p in (raw / "JPEGImages", raw / "images", raw / "leftImg8bit", raw) if p.exists()]
    items = []                                   # (img_path, [(label, cx,cy,bw,bh)])
    for xml in ann.rglob("*.xml"):
        filename, objs = _voc_objects(xml)
        kept = [(canon(n, keep_all), cx, cy, bw, bh) for (n, cx, cy, bw, bh) in objs]
        kept = [o for o in kept if o[0]]
        img = _find_image(xml, filename, roots)
        if img and (kept or keep_all):
            items.append((img, kept))
    return items


# --------------------------------------------------------------- YOLO / RF
def _read_names(raw: Path):
    for y in list(raw.rglob("data.yaml")) + list(raw.rglob("*.yaml")):
        try:
            d = yaml.safe_load(y.read_text(encoding="utf-8"))
        except Exception:
            continue
        names = d.get("names")
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
        if isinstance(names, list):
            return {i: n for i, n in enumerate(names)}
    return {}


def _sibling_image(label_file: Path):
    stem = label_file.stem
    if label_file.parent.name == "labels":
        for ext in IMG_EXTS:
            p = label_file.parent.parent / "images" / f"{stem}{ext}"
            if p.exists():
                return p
    for ext in IMG_EXTS:
        p = label_file.with_suffix(ext)
        if p.exists():
            return p
    return None


def collect_yolo(raw: Path, keep_all: bool):
    names = _read_names(raw)
    items = []
    for lf in raw.rglob("*.txt"):
        if lf.name in {"classes.txt", "requirements.txt", "readme.txt"}:
            continue
        img = _sibling_image(lf)
        if img is None:
            continue
        kept = []
        for ln in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = ln.split()
            if len(parts) < 5:
                continue
            src_name = names.get(int(float(parts[0])), str(parts[0]))
            lbl = canon(src_name, keep_all)
            if not lbl:
                continue
            kept.append((lbl, float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        if kept or keep_all:
            items.append((img, kept))
    return items


def collect_folder(raw: Path, keep_all: bool):
    if list(raw.rglob("*.xml")):
        print("[folder] detected Pascal-VOC (*.xml)")
        return collect_voc(raw, keep_all)
    if list(raw.rglob("*.txt")):
        print("[folder] detected YOLO (*.txt)")
        return collect_yolo(raw, keep_all)
    raise SystemExit("No YOLO or VOC annotations under " + str(raw))


def collect_roboflow(args, keep_all: bool):
    key = os.environ.get("ROBOFLOW_API_KEY", args.rf_key)
    if not (key and args.rf_workspace and args.rf_project):
        raise SystemExit("roboflow needs ROBOFLOW_API_KEY + --rf-workspace + --rf-project")
    from roboflow import Roboflow
    dl = HERE / "raw" / f"roboflow_{args.rf_project}_v{args.rf_version}"
    Roboflow(api_key=key).workspace(args.rf_workspace).project(args.rf_project) \
        .version(args.rf_version).download("yolov8", location=str(dl))
    return collect_yolo(dl, keep_all)


# --------------------------------------------------------------- write out
def build(items, out: Path, splits):
    # drop corrupted images + assign a stable class index per canonical label
    clean, dropped = [], 0
    for img, objs in items:
        if not _image_ok(Path(img)):
            dropped += 1
            continue
        clean.append((img, objs))
    if not clean:
        raise SystemExit("No usable images after corruption + class filtering.")

    labels = sorted({o[0] for _img, objs in clean for o in objs})
    if not labels:
        raise SystemExit("No mapped classes found - check the dataset / aliases.")
    idx = {lbl: i for i, lbl in enumerate(labels)}
    print(f"[build] classes ({len(labels)}): {labels}")
    print(f"[build] usable images: {len(clean)}  (dropped {dropped} corrupted)")

    random.seed(42)
    random.shuffle(clean)
    n = len(clean)
    n_test = int(n * splits[2])
    n_val = int(n * splits[1])
    parts = {"test": clean[:n_test], "val": clean[n_test:n_test + n_val],
             "train": clean[n_test + n_val:]}

    for sub in ("images", "labels"):
        for sp in parts:
            (out / sub / sp).mkdir(parents=True, exist_ok=True)

    counts = {}
    for sp, rows in parts.items():
        for k, (img, objs) in enumerate(rows):
            img = Path(img)
            stem = f"{img.stem}_{k:06d}"
            shutil.copy2(img, out / "images" / sp / f"{stem}{img.suffix.lower()}")
            lines = [f"{idx[o[0]]} {o[1]:.6f} {o[2]:.6f} {o[3]:.6f} {o[4]:.6f}" for o in objs]
            (out / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        counts[sp] = len(rows)

    data = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
            "test": "images/test", "names": {i: l for l, i in idx.items()}}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"[build] split -> {counts}")
    print(f"[data.yaml] -> {out / 'data.yaml'}")


def main():
    ap = argparse.ArgumentParser(description="Prepare a MULTI-CLASS India-vehicle YOLO dataset")
    ap.add_argument("--source", required=True, choices=["roboflow", "idd", "folder"])
    ap.add_argument("--raw", type=Path, help="raw dataset dir (idd/folder)")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "india_vehicles")
    ap.add_argument("--split", default="0.7,0.2,0.1", help="train,val,test fractions")
    ap.add_argument("--keep-all", action="store_true",
                    help="keep every source class (default: keep only recognised vehicle/person classes)")
    ap.add_argument("--rf-key", default=None)
    ap.add_argument("--rf-workspace", default=None)
    ap.add_argument("--rf-project", default=None)
    ap.add_argument("--rf-version", type=int, default=1)
    args = ap.parse_args()

    splits = tuple(float(x) for x in args.split.split(","))
    if len(splits) != 3 or abs(sum(splits) - 1.0) > 1e-6:
        raise SystemExit("--split must be three fractions summing to 1, e.g. 0.7,0.2,0.1")

    if args.source == "roboflow":
        items = collect_roboflow(args, args.keep_all)
    else:
        if not args.raw or not args.raw.exists():
            raise SystemExit(f"--raw required and must exist for source={args.source}")
        items = (collect_voc if args.source == "idd" else collect_folder)(args.raw, args.keep_all)

    build(items, args.out, splits)
    print("\nDone. Train with:")
    print(f"  python scripts/train.py --data {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
