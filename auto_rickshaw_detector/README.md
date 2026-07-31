# India-Specific Vehicle Detector (offline extension)

A lightweight, **multi-class** India-specific vehicle detector, built as a
completely isolated, **fully offline** training pipeline and wired into VigilSense
as an optional **secondary detector**. The primary YOLOv10 detector and the whole
downstream pipeline (ByteTrack, OCR, Person/Vehicle ReID, OpenCLIP, FAISS,
natural-language search, export) are **unchanged** — they just receive more,
correctly-labelled boxes and never know which detector produced them.

Why: the primary YOLOv10b is trained on **COCO**, which has **no auto-rickshaw /
tractor / tempo / mini-truck / HCV / LCV** classes. This detector recognises them
directly; class-aware NMS merges the two streams into one.

---

## Layout
```
auto_rickshaw_detector/
├── configs/train.yaml               # RTX 3050-tuned hyper-parameters (YOLO11n)
├── scripts/
│   ├── prepare_dataset.py            # MULTI-CLASS: inspect + canonicalise + de-corrupt + split + YAML
│   ├── train.py                      # train YOLO11n -> weights/india_vehicles.pt
│   ├── evaluate.py                   # per-class P/R/mAP (+ auto-rickshaw highlighted)
│   ├── export.py                     # copy best.pt (+ optional ONNX)
│   ├── infer_demo.py                 # visual check on your own CCTV frames
│   ├── benchmark_speed.py            # raw model speed on this machine
│   └── benchmark_integration.py      # secondary-pass overhead + merge validation
├── requirements.txt · README.md · .gitignore
# data/, raw/, runs/, weights/ are created at runtime and git-ignored
```
Uses the project's existing `.venv` (ultralytics already installed). Fully offline
once a dataset is present locally and the base `yolo11n.pt` is cached.

---

## Model & classes
**YOLO11n** (nano) — tiny + fast, runs as a cheap second pass next to YOLOv10b.

Canonical output classes (whatever subset the dataset provides is preserved; it
is **never** reduced to a single class):

| India-specific (new global ids) | COCO-equivalent (merged onto existing ids) |
|---|---|
| auto-rickshaw (100), tractor (101), tempo/LCV (102), mini-truck/Tata Ace (103), HCV (104), LCV (105) | car (2), bus (5), truck (7), motorcycle (3), bicycle (1), person (0) |

`prepare_dataset.py` canonicalises synonyms (e.g. `auto`, `tuk-tuk`, `3-wheeler`
→ `auto-rickshaw`; `tata ace`, `chhota hathi` → `mini-truck`) via a name-alias map
that mirrors the backend's `config.VEHICLE_NAME_ALIASES`, so the trained model's
class names line up 1:1 with the integration's resolver.

---

## Dataset sources (offline)
| Source | Images | Access |
|---|---|---|
| **IDD — India Driving Dataset (Detection)** *(recommended)* | 10,004 imgs / 34 classes (up to ~47k in newer releases) | free registration, Pascal-VOC XML |
| **Roboflow Universe** Indian-vehicle projects | varies | free API key, YOLO export |
| any local **YOLO or VOC** folder | — | offline |

```bash
python scripts/prepare_dataset.py --source idd    --raw raw/IDD_Detection
python scripts/prepare_dataset.py --source folder --raw raw/indian_vehicles
python scripts/prepare_dataset.py --source roboflow --rf-workspace <ws> --rf-project <proj> --rf-version 1
```
`prepare_dataset.py` automatically: inspects classes → canonicalises → **removes
corrupted/unreadable images** → **train/val/test split (0.7/0.2/0.1)** → writes
`data.yaml` with all class names.

## Train → Evaluate → Export
```bash
python scripts/train.py                       # -> weights/india_vehicles.pt
python scripts/evaluate.py --split test       # per-class + overall P/R/mAP
python scripts/export.py --onnx
```
The backend plugin auto-loads `weights/india_vehicles.pt` the moment it exists.

---

## Integration (already wired, in the backend)
- Config: `SECONDARY_DETECTOR_SPECS` + `INDIA_VEHICLE_CLASSES` + `VEHICLE_NAME_ALIASES`
  (`backend/app/config.py`). New classes registered into `DETECT_CLASSES` +
  `VEHICLE_CLASSES` so attributes, plate OCR, filters and search treat them natively.
- Plugin: `backend/app/ingestion/detectors/plugins.py` — auto-builds the class map
  from the model's names, runs per frame, and `merge_detections()` does class-aware
  NMS (specific Indian class wins; generic duplicates dropped; persons preserved).
- Tracker: `iter_track_chunks` merges primary + secondary into one stream; primary
  keeps ByteTrack ids, secondary gets a lightweight per-class IoU tracker.
- **Safe/optional:** with no weights present, `plugins.active()` is False and the
  pipeline behaves exactly as before (verified).

## Searchable (Phase 10)
Once detected, the new classes are first-class searchable via CLIP + filters:
`auto-rickshaw`, `yellow auto-rickshaw`, `green auto-rickshaw`, `tractor`, `tempo`,
`mini truck`, `Tata Ace`, `HCV`, `LCV`, and compositions like `auto-rickshaw near
Camera 3`. Plate compositions (`plate ending 6419`) use the fuzzy Plate search.

## Extensibility
Add another India-specific detector later with **zero downstream changes**: train
it, drop the `.pt` in, and append one entry to `SECONDARY_DETECTOR_SPECS`.
