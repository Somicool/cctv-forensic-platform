"""Secondary-detector plugin registry + class-aware merge + light tracker.

A "plugin" is a specialized YOLO model (a YOLO11n trained on an Indian driving
dataset) declared in config.SECONDARY_DETECTOR_SPECS. Plugins run on the same
frame as the primary YOLOv10 detector; their boxes are merged into a single
stream with class-aware NMS that PREFERS the more specific Indian class
(auto-rickshaw / tractor / tempo / mini-truck / HCV / LCV) and dedupes generic
overlaps (a secondary "car" never double-counts the primary "car").

Design goals
------------
* Extensible   : add a detector = append one spec + drop in a .pt (no code here).
                 The class map is built AUTOMATICALLY from the model's own class
                 names via config.VEHICLE_NAME_ALIASES, so any Indian dataset's
                 label set is supported without edits here.
* Safe/optional: a spec activates only if its weights exist; otherwise the whole
                 module is inert and the pipeline behaves exactly as before.
* Downstream-agnostic: produces the same detection records as the primary path,
                 so ByteTrack/OCR/ReID/CLIP/FAISS/search/export are untouched.
"""
from __future__ import annotations

from pathlib import Path

from ... import config

_plugins = None          # lazily-loaded list of {name, model, class_map, conf, imgsz}


def reset() -> None:
    """Force plugins to reload on next use (e.g. after training + placing weights)."""
    global _plugins
    _plugins = None


def _norm(s: str) -> str:
    return "".join(c for c in (s or "").lower() if c.isalnum())


def _build_class_map(model, spec) -> dict:
    """Map the model's own class ids -> (global_id, label, india_specific).

    Uses an explicit spec['class_map'] if given; otherwise resolves each of the
    model's class NAMES through config.VEHICLE_NAME_ALIASES. Unknown names are
    skipped (logged), so a detector only ever contributes classes we understand."""
    if spec.get("class_map"):
        out = {}
        for mc, val in spec["class_map"].items():
            gid, lbl = val[0], val[1]
            out[int(mc)] = (gid, lbl, gid in config.INDIA_SPECIFIC_IDS)
        return out

    names = getattr(model, "names", {}) or {}
    out, skipped = {}, []
    for mc, nm in names.items():
        gid = config.VEHICLE_NAME_ALIASES.get(_norm(nm))
        if gid is None:
            skipped.append(nm)
            continue
        label = config.DETECT_CLASSES.get(gid, str(gid))
        out[int(mc)] = (gid, label, gid in config.INDIA_SPECIFIC_IDS)
    if skipped:
        print(f"[plugins] '{spec['name']}': skipped unmapped classes {skipped}")
    return out


def get_plugins() -> list[dict]:
    """Load (once) every enabled secondary detector whose weights file exists."""
    global _plugins
    if _plugins is not None:
        return _plugins
    _plugins = []
    for spec in getattr(config, "SECONDARY_DETECTOR_SPECS", []):
        if not spec.get("enabled", True):
            continue
        w = Path(spec["weights"])
        if not w.exists():
            continue                      # not trained yet -> silently skip (no-op)
        try:
            from ultralytics import YOLO
            model = YOLO(str(w))
        except Exception as exc:          # noqa: BLE001
            print(f"[plugins] could not load secondary detector '{spec['name']}': {exc}")
            continue
        class_map = _build_class_map(model, spec)
        if not class_map:
            print(f"[plugins] '{spec['name']}' has no mappable classes - skipping")
            continue
        _plugins.append({
            "name": spec["name"], "model": model, "class_map": class_map,
            "conf": float(spec.get("conf", 0.35)), "imgsz": int(spec.get("imgsz", 640)),
        })
        labels = sorted({l for _g, l, _s in class_map.values()})
        print(f"[plugins] loaded '{spec['name']}' <- {w.name}  classes={labels}")
    return _plugins


def active() -> bool:
    return len(get_plugins()) > 0


def detect_frame(frame_bgr) -> list[dict]:
    """Run all secondary detectors on one BGR frame.

    Returns {cls_id, label, conf, xyxy, india_specific} in the SAME full-frame
    pixel coordinates as the primary detector, ready to merge."""
    out: list[dict] = []
    for p in get_plugins():
        res = p["model"].predict(frame_bgr, conf=p["conf"], imgsz=p["imgsz"],
                                 device=config.DEVICE, verbose=False)
        for r in res:
            if r.boxes is None:
                continue
            for b in r.boxes:
                mapped = p["class_map"].get(int(b.cls[0]))
                if mapped is None:
                    continue
                gid, lbl, spec = mapped
                out.append({
                    "cls_id": gid, "label": lbl, "conf": float(b.conf[0]),
                    "xyxy": tuple(float(v) for v in b.xyxy[0].tolist()),
                    "india_specific": spec,
                })
    return out


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _specificity(cls_id: int, india: bool) -> int:
    """Priority when two boxes describe the same object: India-specific (2) beats
    generic vehicle (1); non-vehicles (0) are never contested here."""
    if india:
        return 2
    return 1 if cls_id in config.VEHICLE_CLASSES else 0


def merge_detections(primary: list[dict], secondary: list[dict], iou_thr=None):
    """Class-aware NMS across the primary (COCO) + secondary (Indian) streams.

    Rules:
      * non-vehicle primary boxes (person, bag, ...) are always kept.
      * among VEHICLE boxes, overlapping detections of the SAME object are
        resolved by (specificity, confidence): the more specific Indian class
        wins, and a generic secondary duplicate of a primary vehicle is dropped.
    Returns (kept_primary, kept_secondary) so the caller can preserve ByteTrack
    ids on primary boxes and assign tracker ids to surviving secondary boxes.
    """
    thr = config.SECONDARY_NMS_IOU if iou_thr is None else iou_thr

    kept_primary = [d for d in primary if d["cls_id"] not in config.VEHICLE_CLASSES]

    cand = []                                   # contested vehicle boxes
    for d in primary:
        if d["cls_id"] in config.VEHICLE_CLASSES:
            cand.append({"box": d, "src": "p",
                         "spec": _specificity(d["cls_id"], False), "conf": d["conf"]})
    for d in secondary:
        cand.append({"box": d, "src": "s",
                     "spec": _specificity(d["cls_id"], d.get("india_specific", False)),
                     "conf": d["conf"]})

    cand.sort(key=lambda c: (c["spec"], c["conf"]), reverse=True)
    accepted = []
    for c in cand:
        if any(_iou(c["box"]["xyxy"], a["box"]["xyxy"]) >= thr for a in accepted):
            continue                            # suppressed by a higher-priority box
        accepted.append(c)

    kept_primary += [c["box"] for c in accepted if c["src"] == "p"]
    kept_secondary = [c["box"] for c in accepted if c["src"] == "s"]
    return kept_primary, kept_secondary


class IoUTracker:
    """Greedy per-class IoU tracker for secondary detections across sampled frames.

    ByteTrack is coupled inside the primary model.track() call and can't easily
    ingest external boxes, so specialized-detector boxes get their own lightweight
    tracker. Track ids live in a private high range so they never collide with
    ByteTrack ids, and downstream sees identical TrackedDetection records.
    """

    def __init__(self, id_offset: int = 1_000_000, iou_thr: float = 0.2, max_gap: int = 4):
        self.id_offset = id_offset
        self.iou_thr = iou_thr
        self.max_gap = max_gap
        self._tracks: list[dict] = []     # {id, cls_id, xyxy, last_frame}
        self._next = 0

    def update(self, boxes: list[dict], frame_idx: int) -> list[dict]:
        """Assign a persistent track_id (in place) to each box for this frame."""
        self._tracks = [t for t in self._tracks if frame_idx - t["last_frame"] <= self.max_gap]
        used = set()
        for bx in boxes:
            best_i, best_iou = None, 0.0
            for ti, t in enumerate(self._tracks):
                if ti in used or t["cls_id"] != bx["cls_id"]:
                    continue
                i = _iou(bx["xyxy"], t["xyxy"])
                if i > best_iou:
                    best_iou, best_i = i, ti
            if best_i is not None and best_iou >= self.iou_thr:
                t = self._tracks[best_i]
                t["xyxy"], t["last_frame"] = bx["xyxy"], frame_idx
                bx["track_id"] = t["id"]
                used.add(best_i)
            else:
                tid = self.id_offset + self._next
                self._next += 1
                self._tracks.append({"id": tid, "cls_id": bx["cls_id"],
                                     "xyxy": bx["xyxy"], "last_frame": frame_idx})
                bx["track_id"] = tid
        return boxes
