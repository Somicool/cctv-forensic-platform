"""High-accuracy ANPR (Automatic Number Plate Recognition) pipeline.

Layered ON TOP of the existing OCR (plate_reader + ocr_engines + gemini_plate) -
it does NOT replace them. The old path stays fully intact and is used when
config.ANPR_ENABLED is False; this module is only a smarter orchestration:

  1. Dedicated plate DETECTION - localise the number-plate region inside each
     vehicle crop (a YOLO plate detector when weights are present, otherwise the
     classical morphological proposer) and OCR the PLATE, not the whole vehicle.
  2. Plate-crop ENHANCEMENT - edge-preserving denoise + super-resolution for small
     plates (perspective-correction / CLAHE / unsharp are inherited from
     plate_reader.read_plates, which we run on the detected region).
  3. Multi-frame OCR - keep only the SHARPEST frames of a tracked vehicle (blur
     gating), OCR each detected plate region, aggregate with temporal voting.
  4. Confidence + support - returns the final plate, its confidence AND the number
     of supporting frames.
  5. Indian validation - reuses plate_reader._candidate (strict Indian format with
     OCR-confusion repair + a tightened general fallback); partial plates preserved.
  6. AI fallback - Gemini Vision is queried ONCE, and only if the best OCR
     confidence stays below the threshold (never for high-confidence plates).

Fully offline by default (super-resolution falls back to LANCZOS; Gemini is
optional + key-gated).
"""
from __future__ import annotations

from pathlib import Path

import cv2

from .. import config
from . import plate_reader, gemini_plate

# --------------------------------------------------------------- plate detector
_plate_model = None
_plate_model_tried = False


def _get_plate_model():
    """Lazily load an optional dedicated YOLO plate detector (offline, weights-gated)."""
    global _plate_model, _plate_model_tried
    if _plate_model_tried:
        return _plate_model
    _plate_model_tried = True
    w = Path(getattr(config, "PLATE_DETECTOR_WEIGHTS", "") or "")
    if w and w.exists():
        try:
            from ultralytics import YOLO
            _plate_model = YOLO(str(w))
            print(f"[anpr] dedicated plate detector loaded <- {w.name}")
        except Exception as exc:              # noqa: BLE001
            print(f"[anpr] plate detector load failed ({exc}); using morphological fallback")
            _plate_model = None
    return _plate_model


def detect_plate_regions(vehicle_bgr) -> list:
    """Tight plate-region crops from a vehicle crop.

    Uses the dedicated YOLO plate detector if available, else the classical
    morphological plate-region proposer (plate_reader._refine_regions). Either way
    OCR then runs on the PLATE region rather than the whole vehicle. Empty list =>
    the caller OCRs the (enhanced) whole crop as a last resort."""
    if vehicle_bgr is None or not getattr(vehicle_bgr, "size", 0):
        return []
    model = _get_plate_model()
    if model is not None:
        try:
            H, W = vehicle_bgr.shape[:2]
            res = model.predict(vehicle_bgr, conf=config.PLATE_DETECTOR_CONF,
                                device=config.DEVICE, verbose=False)
            rois = []
            for r in res:
                if r.boxes is None:
                    continue
                for b in sorted(r.boxes, key=lambda z: float(z.conf[0]), reverse=True):
                    x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                    px, py = int((x2 - x1) * 0.06), int((y2 - y1) * 0.22)   # small context pad
                    x1, y1 = max(0, x1 - px), max(0, y1 - py)
                    x2, y2 = min(W, x2 + px), min(H, y2 + py)
                    roi = vehicle_bgr[y1:y2, x1:x2]
                    if roi.size:
                        rois.append(roi)
            if rois:
                return rois
        except Exception:                     # noqa: BLE001 - fall back on any error
            pass
    return plate_reader._refine_regions(vehicle_bgr)


# --------------------------------------------------------------- enhancement
def _sharpness(img) -> float:
    """Variance of the Laplacian - higher = sharper (used for blur gating)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


_sr = None
_sr_tried = False


def _get_sr():
    """Optional OpenCV dnn_superres model (FSRCNN/EDSR). None -> LANCZOS fallback."""
    global _sr, _sr_tried
    if _sr_tried:
        return _sr
    _sr_tried = True
    mp = getattr(config, "ANPR_SR_MODEL", "") or ""
    if config.ANPR_SR_ENABLED and mp and Path(mp).exists():
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(mp)
            arch = Path(mp).stem.split("_")[0].lower()      # e.g. fsrcnn_x2 -> fsrcnn
            sr.setModel(arch, int(config.ANPR_SR_SCALE))
            _sr = sr
            print(f"[anpr] super-resolution model: {Path(mp).name}")
        except Exception as exc:              # noqa: BLE001
            print(f"[anpr] SR model load failed ({exc}); using LANCZOS")
            _sr = None
    return _sr


def _super_resolve(img):
    sr = _get_sr()
    if sr is not None:
        try:
            return sr.upsample(img)
        except Exception:                     # noqa: BLE001
            pass
    h, w = img.shape[:2]
    f = int(config.ANPR_SR_SCALE)
    return cv2.resize(img, (max(1, w * f), max(1, h * f)), interpolation=cv2.INTER_LANCZOS4)


def enhance_plate(roi):
    """Denoise (edge-preserving) + super-resolve small plate crops. CLAHE / unsharp
    / deskew are applied downstream by plate_reader.read_plates on this ROI."""
    if roi is None or not getattr(roi, "size", 0):
        return roi
    out = roi
    if config.ANPR_DENOISE:
        out = cv2.bilateralFilter(out, 5, 45, 45)
    if config.ANPR_SR_ENABLED and min(out.shape[:2]) < config.ANPR_PLATE_MIN_SIDE * 3:
        out = _super_resolve(out)
    return out


# --------------------------------------------------------------- read a track
def _load(im):
    if isinstance(im, str):
        return cv2.imread(im)
    return im


def _vote_over_frames(vehicle_crops, floor):
    """Detect the plate region in each vehicle crop, enhance, OCR and temporally
    vote. Returns (tally, frames_used, best_roi) where best_roi is the enhanced
    plate crop that produced the single highest-confidence read."""
    tally: dict[str, dict] = {}
    frames_used = 0
    best_roi, best_roi_conf = None, -1.0
    for img in vehicle_crops:
        if img is None or not getattr(img, "size", 0):
            continue
        regions = detect_plate_regions(img) or [img]     # plate region, else whole crop
        frame_hit = False
        for roi in regions:
            eroi = enhance_plate(roi)
            for p in plate_reader.read_plates(eroi, min_conf=floor):
                t = tally.setdefault(p["text"], {"text": p["text"], "conf": 0.0,
                                                 "votes": 0, "score": 0.0, "source": "paddle"})
                t["votes"] += 1
                t["score"] += p["conf"]
                t["conf"] = max(t["conf"], p["conf"])
                frame_hit = True
                if p["conf"] > best_roi_conf:
                    best_roi_conf, best_roi = p["conf"], eroi
        if frame_hit:
            frames_used += 1
    return tally, frames_used, best_roi


def _save_plate(roi, save_dir, tag):
    try:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        p = Path(save_dir) / f"plate_{tag}.jpg"
        cv2.imwrite(str(p), roi)
        return str(p)
    except Exception:                        # noqa: BLE001
        return None


def _finalize(tally, frames_used, best_roi, gemini_srcs, save_dir, tag):
    """Rank votes, apply the low-confidence Gemini fallback, save the best plate
    crop, and stamp frames/plate_crop onto every candidate."""
    results = sorted(tally.values(), key=lambda x: (x["votes"], x["score"]), reverse=True)
    best_conf = results[0]["conf"] if results else 0.0
    if (config.GEMINI_ENABLED and best_conf < config.PLATE_GEMINI_CONF
            and gemini_srcs and gemini_plate.available()):
        g = gemini_plate.read_plate(gemini_srcs[0])
        if g:
            plate = plate_reader._candidate(g[0])
            if plate:
                t = tally.setdefault(plate, {"text": plate, "conf": 0.0, "votes": 0,
                                             "score": 0.0, "source": "gemini"})
                t["votes"] += 1
                t["score"] += g[1]
                t["conf"] = max(t["conf"], g[1])
                t["source"] = "gemini"
                results = sorted(tally.values(), key=lambda x: (x["votes"], x["score"]), reverse=True)
    plate_crop = _save_plate(best_roi, save_dir, tag) if (save_dir and best_roi is not None and results) else None
    for r in results:
        r["frames"] = frames_used
        r["plate_crop"] = plate_crop
    return results


def read_plate_track(images, max_frames: int | None = None, min_conf: float | None = None,
                     save_dir=None, tag: str = "track") -> list[dict]:
    """ANPR read for ONE tracked vehicle from several saved crops (blur-gated
    multi-frame voting). Returns candidates sorted by (votes, summed-confidence),
    each {text, conf, votes, score, source, frames, plate_crop}."""
    max_frames = max_frames or config.ANPR_MAX_FRAMES
    floor = config.PLATE_MIN_CONF if min_conf is None else min_conf

    loaded = []
    for im in images:
        img = _load(im)
        if img is None or not getattr(img, "size", 0):
            continue
        loaded.append((_sharpness(img), img))
    if not loaded:
        return []
    loaded.sort(key=lambda t: t[0], reverse=True)          # blur gating: sharpest first
    kept = [im for s, im in loaded if s >= config.ANPR_BLUR_MIN][:max_frames]
    if not kept:
        kept = [im for _s, im in loaded[:max(1, max_frames // 2)]]

    tally, frames_used, best_roi = _vote_over_frames(kept, floor)
    return _finalize(tally, frames_used, best_roi, kept, save_dir, tag)


# --------------------------------------------------------------- adaptive re-sample
def _padded_crop(frame, bbox, pad=0.12):
    H, W = frame.shape[:2]
    x, y, w, h = bbox
    px, py = w * pad, h * pad
    x1, y1 = max(0, int(x - px)), max(0, int(y - py))
    x2, y2 = min(W, int(x + w + px)), min(H, int(y + h + py))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _bbox_at(frame_idx, sdets):
    """Interpolate the vehicle bbox (+conf) at an arbitrary native frame from the
    sparse sampled detections (sdets = sorted [(frame, (x,y,w,h), conf), ...])."""
    if frame_idx <= sdets[0][0]:
        return sdets[0][1], sdets[0][2]
    if frame_idx >= sdets[-1][0]:
        return sdets[-1][1], sdets[-1][2]
    for a, b in zip(sdets, sdets[1:]):
        if a[0] <= frame_idx <= b[0]:
            span = (b[0] - a[0]) or 1
            r = (frame_idx - a[0]) / span
            bb = tuple(a[1][k] + (b[1][k] - a[1][k]) * r for k in range(4))
            return bb, a[2] + (b[2] - a[2]) * r
    return sdets[-1][1], sdets[-1][2]


def read_plate_track_adaptive(video_path, dets, native_fps, min_conf=None,
                              save_dir=None, tag: str = "track") -> list[dict]:
    """Adaptive high-FPS ANPR for a two-wheeler / auto track.

    dets: [{frame_number, bbox:(x,y,w,h), confidence}] - the sparse 2-FPS samples
    where the vehicle was tracked. We re-open the source video, re-sample DENSELY
    across the track's active window, score every candidate frame (blur + size +
    detection confidence), then OCR only the sharpest/largest plate crops and vote.
    This recovers small/blurred bike & auto plates the sparse sampling misses."""
    floor = config.PLATE_MIN_CONF if min_conf is None else min_conf
    sdets = sorted(((int(d["frame_number"]), tuple(float(v) for v in d["bbox"]),
                     float(d.get("confidence") or 0.5)) for d in dets if d.get("bbox")),
                   key=lambda z: z[0])
    if not sdets or not video_path or not Path(video_path).exists():
        return []
    native_fps = native_fps or 30.0
    pad = int(native_fps * 0.3)
    f0, f1 = max(0, sdets[0][0] - pad), sdets[-1][0] + pad
    stride = max(1, round(native_fps / max(1, config.ANPR_ADAPTIVE_FPS)))
    cand = list(range(f0, f1 + 1, stride))
    if len(cand) > config.ANPR_ADAPTIVE_MAX_FRAMES:
        import numpy as np
        cand = [cand[i] for i in np.linspace(0, len(cand) - 1, config.ANPR_ADAPTIVE_MAX_FRAMES).astype(int)]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return []
    scored = []
    for fi in cand:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        bbox, conf = _bbox_at(fi, sdets)
        crop = _padded_crop(frame, bbox, 0.12)
        if crop is None or not crop.size:
            continue
        scored.append({"crop": crop, "blur": _sharpness(crop),
                       "area": crop.shape[0] * crop.shape[1], "conf": conf})
    cap.release()
    if not scored:
        return []

    # combined frame-quality score (blur + plate/vehicle size + detection conf),
    # normalised within this track's candidates; OCR only the best few.
    bmax = max(s["blur"] for s in scored) or 1.0
    amax = max(s["area"] for s in scored) or 1.0
    for s in scored:
        s["q"] = (config.ANPR_SCORE_W_BLUR * (s["blur"] / bmax)
                  + config.ANPR_SCORE_W_SIZE * (s["area"] / amax)
                  + config.ANPR_SCORE_W_CONF * s["conf"])
    scored.sort(key=lambda s: s["q"], reverse=True)
    topk = [s["crop"] for s in scored[:config.ANPR_ADAPTIVE_TOPK]]

    tally, frames_used, best_roi = _vote_over_frames(topk, floor)
    return _finalize(tally, frames_used, best_roi, topk, save_dir, tag)


if __name__ == "__main__":
    import sys
    from .. import database
    vids = database.query_detections(class_labels=["car", "truck", "bus", "motorcycle"], limit=200)
    by = {}
    for d in vids:
        by.setdefault((d["video_id"], d["track_id"]), []).append(d)
    grp = max(by.values(), key=len) if by else []
    paths = [d["crop_path"] for d in grp][:config.ANPR_MAX_FRAMES]
    print(f"ANPR self-test on a {len(paths)}-crop vehicle track")
    for r in read_plate_track(paths):
        print(f"  {r['text']:14s} conf={r['conf']:.2f} votes={r['votes']} frames={r['frames']} src={r['source']}")
    if not paths:
        print("no vehicle crops found; run ingestion first")
