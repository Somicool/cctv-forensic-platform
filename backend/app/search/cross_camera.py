"""Cross-camera tracking.

Given a reference detection, find the SAME entity elsewhere:
  - persons          -> OSNet re-ID index
  - vehicles / other -> CLIP visual index

Four filtering layers keep matches trustworthy (a raw re-ID search returns many
look-alikes):
  1. Similarity threshold - 0.82 for persons (re-ID), 0.80 for vehicles (CLIP).
  2. Appearance consistency - reject a person match whose BOTH upper AND lower
     clothing colours differ from the suspect's.
  3. Spatio-temporal plausibility - using camera GPS, reject a match that would
     require impossible travel (> 60 km/h) from the suspect in the time gap.
  4. Track-level de-dup - one appearance per (camera, track) + a 5s same-camera
     window.

Returns a time-sorted TrackResponse (route + journey summary).

    python -m app.search.cross_camera   # self-test on an ingested person
"""
from __future__ import annotations

import math
from datetime import datetime

from .. import config, database
from ..models.schemas import TrackAppearance, TrackResponse, TrackSummary
from ..ingestion import embedder, reid_embedder
from . import vector_store
from .text_search import media_url, _video_index, playback_fields

_PERSON_LABELS = {config.DETECT_CLASSES[c] for c in config.PERSON_CLASSES}
_DEDUP_WINDOW_S = 5.0            # same camera within this many seconds = one sighting
_PERSON_MIN_SIM = 0.82          # raised from 0.75 to cut re-ID look-alikes
_VEHICLE_MIN_SIM = 0.80
_MAX_KMH = 60.0                 # plausible travel speed between cameras


def _camera_geo() -> dict:
    return {c["camera_id"]: c for c in database.list_cameras()}


def _parse_ts(ts):
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, a)))


def _colors_consistent(ref_upper, ref_lower, attrs) -> bool:
    """Reject only when BOTH upper and lower colours are known on both sides and
    both differ (a strong signal it's a different person)."""
    mu = (attrs or {}).get("upper_color")
    ml = (attrs or {}).get("lower_color")
    if ref_upper and ref_lower and mu and ml and ref_upper != mu and ref_lower != ml:
        return False
    return True


def _plausible(ref_ll, ref_ts, cam, m_ts) -> bool:
    """False if travelling from the suspect's camera to this one in the time gap
    would exceed _MAX_KMH. Unknown GPS/time -> can't judge, allow."""
    if not ref_ll or not cam or cam.get("lat") is None or cam.get("lon") is None:
        return True
    if ref_ts is None or m_ts is None:
        return True
    dist = _haversine_km(ref_ll[0], ref_ll[1], cam["lat"], cam["lon"])
    if dist < 0.05:                     # effectively the same location
        return True
    dt_h = abs((m_ts - ref_ts).total_seconds()) / 3600.0
    if dt_h <= 1e-6:                    # different place at the same instant
        return False
    return (dist / dt_h) <= _MAX_KMH


def _summarize(appearances: list[TrackAppearance]) -> TrackSummary | None:
    if not appearances:
        return None
    cams = {a.camera_id for a in appearances}
    times = sorted(a.timestamp for a in appearances if a.timestamp)
    span = None
    if len(times) >= 2:
        t0, t1 = _parse_ts(times[0]), _parse_ts(times[-1])
        if t0 and t1:
            span = (t1 - t0).total_seconds()
    return TrackSummary(
        total_appearances=len(appearances), unique_cameras=len(cams),
        first_seen=times[0] if times else None, last_seen=times[-1] if times else None,
        span_seconds=round(span, 1) if span is not None else None,
    )


def track_across_cameras(detection_id, top_k: int = 500, threshold: float | None = None,
                         max_results: int | None = None) -> TrackResponse:
    top_k = int(max_results or top_k)
    refs = database.get_detections([detection_id])
    if not refs:
        return TrackResponse(reference_detection_id=detection_id, appearances=[])
    ref = refs[0]

    is_person = ref.get("class_label") in _PERSON_LABELS
    index = "reid" if is_person else "clip"
    thr = threshold if threshold is not None else (_PERSON_MIN_SIM if is_person else _VEHICLE_MIN_SIM)

    ref_attrs = ref.get("attributes") or {}
    ref_upper, ref_lower = ref_attrs.get("upper_color"), ref_attrs.get("lower_color")
    geo = _camera_geo()
    ref_cam = geo.get(ref.get("camera_id"), {})
    ref_ll = ((ref_cam.get("lat"), ref_cam.get("lon"))
              if ref_cam.get("lat") is not None and ref_cam.get("lon") is not None else None)
    ref_ts = _parse_ts(ref.get("timestamp"))

    vec = vector_store.get_vector(index, detection_id)
    if vec is None:
        vec = (reid_embedder.embed_person(ref["crop_path"]) if is_person
               else embedder.embed_image(ref["crop_path"]))

    ids, scores = vector_store.search(index, vec, top_k=top_k)
    dets = {d["detection_id"]: d for d in database.get_detections(ids)}
    vindex = _video_index()

    # Layers 1-3: threshold, colour consistency, spatio-temporal plausibility.
    # Then layer 4a: best-scoring detection per (camera, track).
    best: dict = {}
    for did, sc in zip(ids, scores):
        if sc < thr:                                   # layer 1
            continue
        d = dets.get(did)
        if not d or d.get("class_label") == "scene":
            continue
        if is_person and not _colors_consistent(ref_upper, ref_lower, d.get("attributes")):
            continue                                   # layer 2
        if not _plausible(ref_ll, ref_ts, geo.get(d.get("camera_id"), {}),
                          _parse_ts(d.get("timestamp"))):
            continue                                   # layer 3
        key = (d.get("camera_id"), d.get("track_id"))
        if key not in best or sc > best[key][1]:
            best[key] = (d, sc)

    # Layer 4b: collapse near-simultaneous sightings in the SAME camera (<=5s).
    by_cam: dict = {}
    for (cam_id, _t), (d, sc) in best.items():
        by_cam.setdefault(cam_id, []).append((d, sc, _parse_ts(d.get("timestamp"))))
    merged: list = []
    for _cam_id, items in by_cam.items():
        items.sort(key=lambda x: x[2] or datetime.min)
        kept: list = []
        for d, sc, ts in items:
            if (kept and ts is not None and kept[-1][2] is not None
                    and abs((ts - kept[-1][2]).total_seconds()) <= _DEDUP_WINDOW_S):
                if sc > kept[-1][1]:
                    kept[-1] = (d, sc, ts)
                continue
            kept.append((d, sc, ts))
        merged.extend(kept)

    appearances = []
    for d, sc, _ts in merged:
        c = geo.get(d.get("camera_id"), {})
        pb = playback_fields(d, vindex)
        appearances.append(TrackAppearance(
            camera_id=d.get("camera_id"), camera_name=c.get("name"),
            timestamp=d.get("timestamp"), detection_id=d["detection_id"],
            similarity=float(sc), crop_url=media_url(d.get("crop_path")),
            lat=c.get("lat"), lon=c.get("lon"),
            video_url=pb.get("video_url"), offset_seconds=pb.get("offset_seconds"),
        ))
    appearances.sort(key=lambda a: a.timestamp or "")

    return TrackResponse(
        reference_detection_id=detection_id,
        reference_class=ref.get("class_label"),
        appearances=appearances,
        summary=_summarize(appearances),
    )


if __name__ == "__main__":
    persons = database.query_detections(class_labels=["person"], limit=20)
    if not persons:
        print("No person detections; run scripts/ingest_all.py first.")
    else:
        ref = persons[0]
        resp = track_across_cameras(ref["detection_id"], top_k=500)
        cams = sorted({a.camera_id for a in resp.appearances})
        print(f"tracking det {resp.reference_detection_id} (from {ref['camera_id']}) -> "
              f"{len(resp.appearances)} appearances across cameras {cams}")
        if resp.summary:
            print(f"  summary: {resp.summary.total_appearances} appearances, "
                  f"{resp.summary.unique_cameras} cameras, span={resp.summary.span_seconds}s")
