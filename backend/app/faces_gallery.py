"""Face Gallery - save the best face of a person and find the same individual
across all indexed footage.

Additive module. It reuses the faces already detected during ingestion (the
`faces` table + the InsightFace `face` FAISS index) and the existing
result/playback helpers - it does NOT re-run person detection or redesign any
backend. Saved faces are stored permanently in the `saved_faces` table (with a
copy of the embedding) and only removed on explicit delete.
"""
from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from . import config, database
from .search import vector_store
from .search.text_search import media_url, _camera_names, _video_index, playback_fields, to_result_item

_PERSON_LABELS = {config.DETECT_CLASSES[c] for c in config.PERSON_CLASSES}


# --------------------------------------------------------------- helpers
def _sharpness(path) -> float:
    img = cv2.imread(str(path)) if path else None
    if img is None or not img.size:
        return 0.0
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _face_rows_for_track(video_id, track_id) -> list[dict]:
    if video_id is None or track_id is None:
        return []
    with database.get_conn() as conn:
        rows = conn.execute(
            "SELECT f.* FROM faces f JOIN detections d ON d.detection_id = f.detection_id "
            "WHERE d.video_id=? AND d.track_id=?", (video_id, track_id)).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------- original frame + expanded box
def _frame_path(video_id, frame_number) -> str | None:
    """Original full frame for a (video, frame). Ingestion stores every sampled
    frame as a 'scene' row whose crop_path IS the frame path - so we can recover
    the ORIGINAL frame for any detection instead of relying on its tight crop."""
    if video_id is None or frame_number is None:
        return None
    with database.get_conn() as conn:
        row = conn.execute(
            "SELECT crop_path FROM detections WHERE video_id=? AND class_label='scene' "
            "AND frame_number=? LIMIT 1", (video_id, frame_number)).fetchone()
    p = row["crop_path"] if row else None
    return p if (p and Path(p).exists()) else None


def _expand_box(bbox, W, H, frac=None):
    """Expand a person bbox by ~frac (default config.FACE_BOX_EXPAND), clamped to
    the image. Gives the face the context a tight body crop is missing."""
    frac = config.FACE_BOX_EXPAND if frac is None else frac
    x, y, w, h = (float(v) for v in bbox)
    px, py = w * frac, h * frac
    x1, y1 = max(0, int(x - px)), max(0, int(y - py))
    x2, y2 = min(int(W), int(x + w + px)), min(int(H), int(y + h + py))
    return (x1, y1, x2, y2) if (x2 > x1 and y2 > y1) else None


def _expanded_from_frame(det: dict):
    """(expanded person crop ndarray, frame) for a detection, from the ORIGINAL
    frame. Falls back to the stored tight crop when the frame isn't available."""
    fp = _frame_path(det.get("video_id"), det.get("frame_number"))
    if fp:
        frame = cv2.imread(fp)
        if frame is not None and frame.size and det.get("bbox_x") is not None:
            H, W = frame.shape[:2]
            box = _expand_box((det["bbox_x"], det["bbox_y"], det["bbox_w"], det["bbox_h"]), W, H)
            if box:
                x1, y1, x2, y2 = box
                crop = frame[y1:y2, x1:x2]
                if crop is not None and crop.size:
                    return crop, frame
    cp = det.get("crop_path")
    img = cv2.imread(str(cp)) if cp and Path(cp).exists() else None
    return img, None


def expanded_crop_url(detection_id: int) -> str | None:
    """Cache + return a /media URL for the EXPANDED person crop (display).
    Cheap: pure image crop, no AI."""
    refs = database.get_detections([detection_id])
    if not refs:
        return None
    out = config.EXPANDED_CROP_DIR / f"exp_{detection_id}.jpg"
    if out.exists():
        return media_url(str(out))
    crop, _f = _expanded_from_frame(refs[0])
    if crop is None or not crop.size:
        return media_url(refs[0].get("crop_path"))
    try:
        cv2.imwrite(str(out), crop)
        return media_url(str(out))
    except Exception:
        return media_url(refs[0].get("crop_path"))


# ------------------------------------------------- face quality scoring
def _detect_faces_kps(img):
    """InsightFace faces WITH landmarks (kps), in the ORIGINAL `img` coordinate
    space. Small crops are upscaled for detection, so bbox/kps are scaled back -
    otherwise the face region would slice out of bounds. Face detection only;
    person detection is never re-run."""
    try:
        from .ingestion import face_recognizer
        if img is None or not img.size:
            return []
        up = face_recognizer._upscale(img, 320)
        faces = list(face_recognizer.get_face_app().get(up))
        if not faces:
            return []
        s = up.shape[1] / float(img.shape[1] or 1)          # upscale factor
        if abs(s - 1.0) > 1e-6:
            for f in faces:                                  # map back to `img` space
                try:
                    f.bbox = np.asarray(f.bbox, dtype="float32") / s
                    if getattr(f, "kps", None) is not None:
                        f.kps = np.asarray(f.kps, dtype="float32") / s
                except Exception:
                    pass
        return faces
    except Exception:
        return []


def _frontal_score(f) -> float:
    """0-1 frontal-ness from the 5 landmarks: eye/nose horizontal symmetry plus
    both eyes being visible. A profile face scores low."""
    try:
        kps = np.asarray(f.kps, dtype="float32")
        if kps.shape[0] < 3:
            return 0.5
        le, re, nose = kps[0], kps[1], kps[2]
        eye_dx = abs(float(re[0] - le[0]))
        if eye_dx < 1e-3:
            return 0.0
        # nose should sit near the midpoint of the eyes when frontal
        mid = (le[0] + re[0]) / 2.0
        off = abs(float(nose[0]) - mid) / eye_dx           # 0 = perfectly centred
        return float(max(0.0, min(1.0, 1.0 - off * 1.6)))
    except Exception:
        return 0.5


def _eyes_score(f, fw) -> float:
    """0-1: are both eyes clearly visible/separated (vs squashed by profile)?"""
    try:
        kps = np.asarray(f.kps, dtype="float32")
        if kps.shape[0] < 2 or fw <= 0:
            return 0.5
        eye_dx = abs(float(kps[1][0] - kps[0][0]))
        # a frontal face has eye separation ~35-45% of face width
        return float(max(0.0, min(1.0, (eye_dx / fw) / 0.38)))
    except Exception:
        return 0.5


def _brightness_score(reg) -> float:
    """0-1: well-exposed face (peaks around mid brightness, penalises dark/blown)."""
    try:
        g = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY) if reg.ndim == 3 else reg
        m = float(g.mean()) / 255.0
        return float(max(0.0, 1.0 - abs(m - 0.55) / 0.45))
    except Exception:
        return 0.5


def _occlusion_score(f, crop_shape) -> float:
    """0-1 visibility: how much of the face box lies inside the frame (a face cut
    off by the crop/frame edge is likely occluded or truncated)."""
    try:
        H, W = crop_shape[:2]
        x1, y1, x2, y2 = (float(v) for v in f.bbox.tolist())
        area = max(1.0, (x2 - x1) * (y2 - y1))
        vis_w = max(0.0, min(x2, W) - max(x1, 0.0))
        vis_h = max(0.0, min(y2, H) - max(y1, 0.0))
        return float(max(0.0, min(1.0, (vis_w * vis_h) / area)))
    except Exception:
        return 0.5


def _noise_score(reg) -> float:
    """0-1: low sensor/compression noise scores high (median-residual estimate)."""
    try:
        g = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY) if reg.ndim == 3 else reg
        if g.size < 64:
            return 0.5
        resid = cv2.absdiff(g, cv2.medianBlur(g, 3))
        return float(max(0.0, min(1.0, 1.0 - (float(resid.std()) / 18.0))))
    except Exception:
        return 0.5


def _face_quality(f, crop) -> dict:
    """Composite forensic face quality over 9 factors: detection confidence,
    sharpness/blur, resolution, face size, frontal pose, brightness, eyes visible,
    occlusion and image noise."""
    x1, y1, x2, y2 = (float(v) for v in f.bbox.tolist())
    fw, fh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    det = float(f.det_score)
    size = min(1.0, (min(fw, fh) / 90.0))                  # ~90px face = full marks
    resolution = int(fw * fh)                              # face pixel area

    reg = None
    sharp = 0.0
    try:
        H, W = crop.shape[:2]
        fx1, fy1 = max(0, int(x1)), max(0, int(y1))
        fx2, fy2 = min(W, int(x2)), min(H, int(y2))
        reg = crop[fy1:fy2, fx1:fx2]
        if reg is not None and reg.size:
            g = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY) if reg.ndim == 3 else reg
            sharp = min(1.0, float(cv2.Laplacian(g, cv2.CV_64F).var()) / 220.0)
        else:
            reg = None
    except Exception:
        reg = None

    pose = _frontal_score(f)
    eyes = _eyes_score(f, fw)
    bright = _brightness_score(reg) if reg is not None else 0.5
    occl = _occlusion_score(f, crop.shape)
    noise = _noise_score(reg) if reg is not None else 0.5

    q = (config.FACE_Q_W_DET * det + config.FACE_Q_W_SIZE * size
         + config.FACE_Q_W_SHARP * sharp + config.FACE_Q_W_POSE * pose
         + config.FACE_Q_W_EYES * eyes + config.FACE_Q_W_BRIGHT * bright
         + config.FACE_Q_W_OCCL * occl + config.FACE_Q_W_NOISE * noise)
    return {"quality": round(float(q), 4), "det_score": round(det, 4),
            "face_size": int(min(fw, fh)), "resolution": resolution,
            "sharpness": round(sharp, 3), "frontal": round(pose, 3),
            "eyes": round(eyes, 3), "brightness": round(bright, 3),
            "occlusion": round(occl, 3), "noise": round(noise, 3),
            "bbox": [x1, y1, x2, y2]}


def _scan_order(dets: list[dict], max_frames: int) -> list[dict]:
    """Frames to examine: the ENTIRE track. Short tracks are scanned fully; long
    tracks are sampled EVENLY across their whole span (never front-biased), with
    the largest boxes always included since they carry the best face detail."""
    by_frame = sorted(dets, key=lambda d: (d.get("frame_number") or 0))
    if len(by_frame) <= max_frames:
        return by_frame
    idx = np.linspace(0, len(by_frame) - 1, max_frames).astype(int)
    out, seen = [], set()
    for i in idx:                                          # even coverage of the full track
        d = by_frame[int(i)]
        if d["detection_id"] not in seen:
            seen.add(d["detection_id"]); out.append(d)
    big = sorted(dets, key=lambda d: (d.get("bbox_w") or 0) * (d.get("bbox_h") or 0), reverse=True)
    for d in big[:max(4, max_frames // 4)]:
        if d["detection_id"] not in seen:
            seen.add(d["detection_id"]); out.append(d)
    return out


def rank_faces_in_track(video_id, track_id, max_frames=None) -> dict:
    """Inspect the WHOLE person track, score every detected face on the 9 quality
    factors and RANK them. Returns:
        {"best": <highest-quality face or None>, "ranked": [...metrics...],
         "frames_seen": n, "faces_seen": n}
    The running best is replaced whenever a later frame yields a better face, so
    the winner is the best representative face of the entire track - never the
    first one detected."""
    empty = {"best": None, "ranked": [], "frames_seen": 0, "faces_seen": 0}
    if video_id is None or track_id is None:
        return empty
    max_frames = max_frames or config.FACE_TRACK_SCAN_FRAMES
    with database.get_conn() as conn:
        dets = [dict(r) for r in conn.execute(
            "SELECT * FROM detections WHERE video_id=? AND track_id=? AND class_label!='scene'",
            (video_id, track_id)).fetchall()]
    if not dets:
        return empty

    best, ranked, frames_seen, faces_seen = None, [], 0, 0
    for d in _scan_order(dets, max_frames):
        crop, _frame = _expanded_from_frame(d)
        if crop is None or not crop.size:
            continue
        frames_seen += 1
        for f in _detect_faces_kps(crop):
            if float(f.det_score) < config.FACE_MIN_DET_SCORE:
                continue
            faces_seen += 1
            m = _face_quality(f, crop)
            ranked.append({**{k: v for k, v in m.items() if k != "bbox"},
                           "detection_id": d["detection_id"],
                           "frame_number": d.get("frame_number")})
            # progressive replacement: a better face later in the track wins
            if best is None or m["quality"] > best["quality"]:
                best = {**m, "detection_id": d["detection_id"], "camera_id": d.get("camera_id"),
                        "timestamp": d.get("timestamp"), "frame_number": d.get("frame_number"),
                        "crop": crop, "insight": f}
    ranked.sort(key=lambda x: x["quality"], reverse=True)
    if best is not None:
        best["low_quality"] = bool(best["quality"] < config.FACE_LOW_QUALITY_THRESHOLD)
    return {"best": best, "ranked": ranked, "frames_seen": frames_seen, "faces_seen": faces_seen}


def best_face_in_track(video_id, track_id, ref_frame=None, max_frames=None) -> dict | None:
    """Best representative face of the whole track (compatibility wrapper)."""
    return rank_faces_in_track(video_id, track_id, max_frames=max_frames)["best"]


# ------------------------------------------------- artifacts + per-track cache
def _write_face_artifacts(found: dict, tag: str) -> dict:
    """Write the three face images for a winning face:
      face  - tight best-face crop (Face Gallery image)
      prev  - normalised 256px square preview
      prof  - expanded person crop (person profile image)
    Returns absolute paths (any may be None on failure)."""
    out = {"face": None, "preview": None, "profile": None}
    try:
        x1, y1, x2, y2 = (int(v) for v in found["bbox"])
        H, W = found["crop"].shape[:2]
        px, py = int((x2 - x1) * 0.25), int((y2 - y1) * 0.35)
        reg = found["crop"][max(0, y1 - py):min(H, y2 + py), max(0, x1 - px):min(W, x2 + px)]
        if reg is not None and reg.size:
            fp = config.SAVED_FACE_DIR / f"face_{tag}.jpg"
            cv2.imwrite(str(fp), reg)
            out["face"] = str(fp)
            side = 256
            prev = cv2.resize(reg, (side, side), interpolation=cv2.INTER_CUBIC)
            pp = config.SAVED_FACE_DIR / f"preview_{tag}.jpg"
            cv2.imwrite(str(pp), prev)
            out["preview"] = str(pp)
    except Exception:
        pass
    try:                                                    # person profile image
        prof = config.SAVED_FACE_DIR / f"profile_{tag}.jpg"
        cv2.imwrite(str(prof), found["crop"])
        out["profile"] = str(prof)
    except Exception:
        pass
    return out


def _cache_get(video_id, track_id) -> dict | None:
    if video_id is None or track_id is None:
        return None
    with database.get_conn() as conn:
        row = conn.execute("SELECT * FROM track_best_face WHERE video_id=? AND track_id=?",
                           (video_id, track_id)).fetchone()
    return dict(row) if row else None


def _cache_put(video_id, track_id, found, arts, scan) -> None:
    import json
    metrics = {k: found.get(k) for k in
               ("det_score", "face_size", "resolution", "sharpness", "frontal",
                "eyes", "brightness", "occlusion", "noise")}
    with database.get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO track_best_face (video_id, track_id, detection_id, quality, "
            " low_quality, face_crop, preview_crop, person_crop, metrics, frames_seen, faces_seen, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (video_id, track_id, found["detection_id"], found["quality"],
             1 if found.get("low_quality") else 0, arts.get("face"), arts.get("preview"),
             arts.get("profile"), json.dumps(metrics), scan.get("frames_seen"),
             scan.get("faces_seen"), database._now()))


def _preview_from_cache(cached: dict, detection_id: int) -> dict:
    import json
    try:
        metrics = json.loads(cached.get("metrics") or "{}")
    except Exception:
        metrics = {}
    return {"available": True, "source": "track-best (cached)",
            "face_id": None, "detection_id": cached.get("detection_id"),
            "camera_id": None, "timestamp": None,
            "person_crop_url": media_url(cached.get("person_crop")) or expanded_crop_url(detection_id),
            "face_crop_url": media_url(cached.get("face_crop")),
            "preview_crop_url": media_url(cached.get("preview_crop")),
            "quality": cached.get("quality"),
            "low_quality": bool(cached.get("low_quality")),
            "frames_seen": cached.get("frames_seen"), "faces_seen": cached.get("faces_seen"),
            **metrics}


def best_face_for_detection(detection_id: int, deep: bool = True) -> dict | None:
    """Best face for the PERSON that `detection_id` belongs to.

    deep=True  -> scan the whole track from the ORIGINAL frames (recovers faces the
                  tight stored crops miss).
    deep=False -> fast preview only from faces already stored at ingest time.
    Returns a preview dict, or None when the entire track has no usable face."""
    refs = database.get_detections([detection_id])
    if not refs:
        return None
    ref = refs[0]
    vid, tid = ref.get("video_id"), ref.get("track_id")

    # fast path: a face already stored for this track at ingest time
    stored = _face_rows_for_track(vid, tid)
    if not stored:
        with database.get_conn() as conn:
            stored = [dict(r) for r in conn.execute(
                "SELECT * FROM faces WHERE detection_id=?", (detection_id,)).fetchall()]
    # Best representative face already computed for this track -> serve instantly
    # (this is what lets search results show the BEST face, not the first one).
    cached = _cache_get(vid, tid)
    if cached and cached.get("face_crop"):
        return _preview_from_cache(cached, detection_id)

    if not deep:
        # preview only: never scan. No stored face -> report unavailable.
        if not stored:
            return None
        best = max(stored, key=lambda r: _sharpness(r.get("crop_path")))
        return {"available": True, "source": "stored", "face_id": best["face_id"],
                "detection_id": best["detection_id"], "camera_id": best.get("camera_id"),
                "timestamp": best.get("timestamp"), "gender": best.get("gender"),
                "age": best.get("age"),
                "person_crop_url": expanded_crop_url(detection_id),
                "face_crop_url": media_url(best.get("crop_path")),
                "quality": round(min(1.0, _sharpness(best.get("crop_path")) / 300.0), 3)}

    # deep: rank every face across the WHOLE track and take the best
    scan = rank_faces_in_track(vid, tid)
    found = scan["best"]
    if found is None:
        if stored:                                          # fall back to the ingest face
            best = max(stored, key=lambda r: _sharpness(r.get("crop_path")))
            return {"available": True, "source": "stored", "face_id": best["face_id"],
                    "detection_id": best["detection_id"], "camera_id": best.get("camera_id"),
                    "timestamp": best.get("timestamp"), "gender": best.get("gender"),
                    "age": best.get("age"),
                    "person_crop_url": expanded_crop_url(detection_id),
                    "face_crop_url": media_url(best.get("crop_path")),
                    "quality": round(min(1.0, _sharpness(best.get("crop_path")) / 300.0), 3)}
        return None

    # write the face / preview / profile images and cache the winner for this track
    arts = _write_face_artifacts(found, f"t{vid}_{tid}")
    try:
        _cache_put(vid, tid, found, arts, scan)
    except Exception:
        pass

    f = found["insight"]
    return {"available": True, "source": "track-best",
            "face_id": None, "detection_id": found["detection_id"],
            "camera_id": found.get("camera_id"), "timestamp": found.get("timestamp"),
            "gender": None, "age": int(f.age) if getattr(f, "age", None) is not None else None,
            "person_crop_url": media_url(arts.get("profile")) or expanded_crop_url(found["detection_id"]),
            "face_crop_url": media_url(arts.get("face")),
            "preview_crop_url": media_url(arts.get("preview")),
            "quality": found["quality"], "low_quality": bool(found.get("low_quality")),
            "det_score": found["det_score"], "face_size": found["face_size"],
            "resolution": found["resolution"], "sharpness": found["sharpness"],
            "frontal": found["frontal"], "eyes": found["eyes"],
            "brightness": found["brightness"], "occlusion": found["occlusion"],
            "noise": found["noise"], "frame_number": found.get("frame_number"),
            "frames_seen": scan["frames_seen"], "faces_seen": scan["faces_seen"],
            "candidates": scan["ranked"][:8]}


def _tight_face_crop(person_crop_path, saved_id) -> str | None:
    """Crop the tight face region out of a person crop (InsightFace) and save it.
    Face DETECTION only (never person detection). Falls back to None on failure."""
    try:
        img = cv2.imread(str(person_crop_path))
        if img is None or not img.size:
            return None
        from .ingestion import face_recognizer
        faces = face_recognizer.get_face_app().get(face_recognizer._upscale(img))
        if not faces:
            return None
        f = max(faces, key=lambda x: float(x.det_score))
        x1, y1, x2, y2 = (int(v) for v in f.bbox.tolist())
        H, W = img.shape[:2]
        # NOTE: bbox is in the upscaled space; re-run on the original for correct coords
        faces2 = face_recognizer.get_face_app().get(img)
        if faces2:
            f = max(faces2, key=lambda x: float(x.det_score))
            x1, y1, x2, y2 = (int(v) for v in f.bbox.tolist())
        pad_x, pad_y = int((x2 - x1) * 0.25), int((y2 - y1) * 0.35)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(W, x2 + pad_x), min(H, y2 + pad_y)
        crop = img[y1:y2, x1:x2]
        if crop is None or not crop.size:
            return None
        out = config.SAVED_FACE_DIR / f"face_{saved_id}.jpg"
        cv2.imwrite(str(out), crop)
        return str(out)
    except Exception:
        return None


def _encode_emb(v: np.ndarray) -> str:
    return base64.b64encode(np.asarray(v, dtype="float32").tobytes()).decode("ascii")


def _decode_emb(s: str) -> np.ndarray | None:
    try:
        return np.frombuffer(base64.b64decode(s), dtype="float32")
    except Exception:
        return None


def _row_urls(row: dict) -> dict:
    import json
    d = dict(row)
    d["face_crop_url"] = media_url(row.get("face_crop"))
    d["person_crop_url"] = media_url(row.get("person_crop"))
    d["preview_crop_url"] = media_url(row.get("preview_crop"))
    d["low_quality"] = bool(row.get("low_quality"))
    if row.get("quality_metrics"):
        try:
            d["metrics"] = json.loads(row["quality_metrics"])
        except Exception:
            pass
    d.pop("embedding", None)                    # don't ship the raw vector to the UI
    d.pop("quality_metrics", None)
    return d


# --------------------------------------------------------------- public API
def save_face(detection_id: int, investigation: str | None = None) -> dict | None:
    """Permanently save the BEST face found anywhere in the person's track.

    Recovers the face from the original frames + expanded boxes (scanning the whole
    track), so tight body-only search crops no longer cause a failure. Returns None
    only when the entire track genuinely has no usable face."""
    refs = database.get_detections([detection_id])
    if not refs:
        return None
    import json
    ref = refs[0]
    vid, tid = ref.get("video_id"), ref.get("track_id")
    # Rank EVERY face in the track and take the best representative one.
    scan = rank_faces_in_track(vid, tid)
    found = scan["best"]

    if found is not None:
        f = found["insight"]
        src_det = found["detection_id"]
        # embedding straight from the winning face (no re-detection elsewhere)
        emb_b64 = _encode_emb(np.asarray(f.normed_embedding, dtype="float32"))
        arts = _write_face_artifacts(found, f"t{vid}_{tid}")
        try:
            _cache_put(vid, tid, found, arts, scan)
        except Exception:
            pass
        expanded_crop_url(src_det)                       # ensure the expanded crop exists
        profile = arts.get("profile") or str(config.EXPANDED_CROP_DIR / f"exp_{src_det}.jpg")
        if not Path(profile).exists():
            profile = (database.get_detections([src_det]) or [{}])[0].get("crop_path")
        metrics = {k: found.get(k) for k in
                   ("det_score", "face_size", "resolution", "sharpness", "frontal",
                    "eyes", "brightness", "occlusion", "noise")}
        metrics.update(frames_seen=scan["frames_seen"], faces_seen=scan["faces_seen"])
        with database.get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO saved_faces (face_id, detection_id, investigation, camera_id, "
                " timestamp, confidence, face_crop, person_crop, embedding, gender, age, "
                " created_at, preview_crop, low_quality, quality_metrics) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (None, src_det, investigation, found.get("camera_id"), found.get("timestamp"),
                 found["quality"], arts.get("face") or profile, profile, emb_b64, None,
                 int(f.age) if getattr(f, "age", None) is not None else None, database._now(),
                 arts.get("preview"), 1 if found.get("low_quality") else 0, json.dumps(metrics)))
            saved_id = cur.lastrowid
            row = dict(conn.execute("SELECT * FROM saved_faces WHERE saved_id=?", (saved_id,)).fetchone())
        return _row_urls(row)

    # nothing recovered from the frames -> fall back to a face stored at ingest
    best = best_face_for_detection(detection_id, deep=False)
    if not best or not best.get("face_id"):
        return None
    face_id = best["face_id"]
    frow = (database.get_faces([face_id]) or [{}])[0]
    person_crop = frow.get("crop_path")
    emb = vector_store.get_vector("face", face_id)
    with database.get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO saved_faces (face_id, detection_id, investigation, camera_id, "
            " timestamp, confidence, face_crop, person_crop, embedding, gender, age, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (face_id, best["detection_id"], investigation, best.get("camera_id"),
             best.get("timestamp"), best.get("quality"), None, person_crop,
             _encode_emb(emb) if emb is not None else None,
             best.get("gender"), best.get("age"), database._now()))
        saved_id = cur.lastrowid
    face_crop = _tight_face_crop(person_crop, saved_id) or person_crop
    with database.get_conn() as conn:
        conn.execute("UPDATE saved_faces SET face_crop=? WHERE saved_id=?", (face_crop, saved_id))
        row = dict(conn.execute("SELECT * FROM saved_faces WHERE saved_id=?", (saved_id,)).fetchone())
    return _row_urls(row)


def list_saved() -> list[dict]:
    with database.get_conn() as conn:
        rows = conn.execute("SELECT * FROM saved_faces ORDER BY created_at DESC").fetchall()
    return [_row_urls(dict(r)) for r in rows]


def get_saved(saved_id: int) -> dict | None:
    with database.get_conn() as conn:
        row = conn.execute("SELECT * FROM saved_faces WHERE saved_id=?", (saved_id,)).fetchone()
    return _row_urls(dict(row)) if row else None


def delete_saved(saved_id: int) -> dict:
    with database.get_conn() as conn:
        row = conn.execute("SELECT face_crop FROM saved_faces WHERE saved_id=?", (saved_id,)).fetchone()
        conn.execute("DELETE FROM saved_faces WHERE saved_id=?", (saved_id,))
    # remove the saved tight-crop file (leave the shared person crop alone)
    try:
        if row and row["face_crop"] and str(config.SAVED_FACE_DIR) in str(row["face_crop"]):
            Path(row["face_crop"]).unlink(missing_ok=True)
    except Exception:
        pass
    return {"deleted": saved_id}


def find_similar(saved_id: int, top_k: int | None = None) -> dict:
    """Find the same individual across all indexed footage using the STORED face
    embedding (no person detection re-run). Returns matches sorted by similarity."""
    top_k = top_k or config.DEFAULT_TOP_K
    with database.get_conn() as conn:
        row = conn.execute("SELECT * FROM saved_faces WHERE saved_id=?", (saved_id,)).fetchone()
    if not row:
        return {"saved_id": saved_id, "results": []}
    row = dict(row)
    emb = _decode_emb(row.get("embedding")) if row.get("embedding") else None
    if emb is None and row.get("face_id") is not None:
        emb = vector_store.get_vector("face", row["face_id"])
    if emb is None:
        return {"saved_id": saved_id, "results": []}

    ids, scores = vector_store.search("face", emb, top_k=top_k * 3)
    score_by_face = dict(zip(ids, scores))
    face_rows = database.get_faces(ids)
    dets = {d["detection_id"]: d for d in database.get_detections([r["detection_id"] for r in face_rows])}
    cam_names = _camera_names()
    vindex = _video_index()

    results = []
    for r in face_rows:
        sim = float(score_by_face.get(r["face_id"], 0.0))
        if sim < config.FACE_SIMILAR_MIN:
            continue
        d = dets.get(r["detection_id"])
        if not d:
            continue
        item = to_result_item(d, sim, cam_names, vindex).model_dump()
        pb = playback_fields(d, vindex)
        results.append({
            "face_id": r["face_id"],
            "detection_id": r["detection_id"],
            "similarity": round(sim, 4),
            "camera_id": r.get("camera_id"),
            "camera_name": cam_names.get(r.get("camera_id")),
            "timestamp": r.get("timestamp"),
            "gender": r.get("gender"), "age": r.get("age"),
            "face_crop_url": media_url(r.get("crop_path")),
            "person_crop_url": media_url(d.get("crop_path")),
            "video_url": pb.get("video_url"),
            "offset_seconds": pb.get("offset_seconds"),
            "bbox": item.get("bbox"),
            "frame_width": item.get("frame_width"),
            "frame_height": item.get("frame_height"),
            "video_id": pb.get("video_id"),
        })
        if len(results) >= top_k:
            break
    results.sort(key=lambda x: x["similarity"], reverse=True)
    database.log_audit("search", query_type="face-similar", result_count=len(results))
    return {"saved_id": saved_id, "reference": _row_urls(row), "total": len(results), "results": results}
