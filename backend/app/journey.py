"""Journey Reconstruction - probable movement of the SAME person across cameras.

Purely additive: it reuses the existing Person Re-ID / CLIP / face indexes, the
cameras table (GPS) and the stored detections. Nothing in OCR, search, tracking,
ReID, the database schema of other tables, the APIs or the frontend is changed.

Identity fusion
---------------
A person is matched by a WEIGHTED FUSION of several signals, never by clothing
alone. Weights are renormalised over whichever signals are available, so a match
still works when (say) no face was captured:

    face   - InsightFace embedding  (highest priority when available)
    reid   - OSNet person embedding (body/gait appearance)
    body   - CLIP visual embedding  (overall appearance)
    attrs  - clothing colours + accessories agreement

Vehicle information is deliberately EXCLUDED from identity, so the same person
walking in one camera and riding a scooter in another still matches.

Extensibility
-------------
`SIGNAL_WEIGHTS` and `TRANSITION_EVIDENCE` are open registries. Future sources -
real GPS tracks, ANPR plate hits, mobile-location data - can be added as another
signal / evidence provider without touching the reconstruction logic: add a key
to the registry and supply a `score(ref, cand) -> float | None` (identity) or an
`evidence(a, b) -> dict` (transition) callable.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime

import numpy as np

from . import (config, database, camera_registry, journey_engine, routing,
               track_identity, track_match)
from .search import vector_store
from .search.text_search import media_url, _camera_names, _video_index, playback_fields

# ---------------------------------------------------------------- tunables
SIGNAL_WEIGHTS = {"face": 0.50, "reid": 0.30, "body": 0.12, "attrs": 0.08}
# Forensic operating point, chosen from a threshold sweep on this dataset
# (same/different pairs derived from ByteTrack ground truth):
#   0.65 -> recall 90.0%, false-match 10.5%   (too many false matches)
#   0.78 -> recall 69.0%, false-match  0.5%,  precision 99.3%   <- default
#   0.82 -> recall 63.5%, false-match  0.0%,  precision 100%
# A fabricated journey is far more damaging than a missed camera, so precision is
# preferred. Raise to 0.82 for zero false matches, lower to 0.72 for more recall.
IDENTITY_MIN = 0.78           # fused identity score required to accept an appearance
# Body-appearance + clothing alone are weak identity evidence (look-alikes in
# similar clothes). When NEITHER a face nor a re-ID embedding is available for a
# candidate, demand a much higher fused score before claiming it is the same person.
IDENTITY_MIN_WEAK = 0.90
FACE_STRONG = 0.55             # a face similarity at/above this alone confirms identity
POOL = 400                     # candidates pulled per index
MAX_KMH = 60.0                 # faster than this between cameras = impossible transition
WALK_KMH = 7.0                 # <= walking
CYCLE_KMH = 25.0               # <= two-wheeler / cycle; above = motor vehicle
SAME_CAM_GAP_S = 20.0          # merge sightings in one camera within this window
# Future transition-evidence providers (gps track, anpr hit, mobile ping, ...).
TRANSITION_EVIDENCE: dict = {}

_PERSON_LABELS = {config.DETECT_CLASSES[c] for c in config.PERSON_CLASSES}


# ---------------------------------------------------------------- helpers
def _parse_ts(ts):
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, a)))


def _attach_media(rows: list[dict], vindex: dict | None = None) -> list[dict]:
    """Add the person's crop image and its source-clip playback fields to any row
    that carries a detection_id.

    The legacy detection-level path already did this inline, but the track-level
    matcher did not, so track-level journey nodes and the per-camera candidates
    reached the UI with no crop and no video_url - which is why "Jump to Video"
    reported "No source recording linked to this detection". Resolved from the
    SAME stored detection rows and video index the rest of the app uses; nothing is
    recomputed and existing values are never overwritten."""
    ids = [r.get("detection_id") for r in rows if r.get("detection_id") is not None]
    if not ids:
        return rows
    dets = {d["detection_id"]: d for d in database.get_detections(ids)}
    vindex = _video_index() if vindex is None else vindex
    for r in rows:
        d = dets.get(r.get("detection_id"))
        if not d:
            continue
        pb = playback_fields(d, vindex)
        if not r.get("crop_url"):
            r["crop_url"] = media_url(d.get("crop_path"))
        if not r.get("video_url"):
            r["video_url"] = pb.get("video_url")
        if r.get("offset_seconds") is None:
            r["offset_seconds"] = pb.get("offset_seconds")
        if not r.get("frame_width"):
            r["frame_width"] = pb.get("frame_width")
        if not r.get("frame_height"):
            r["frame_height"] = pb.get("frame_height")
        if not r.get("bbox") and d.get("bbox_x") is not None:
            r["bbox"] = [d["bbox_x"], d["bbox_y"], d["bbox_w"], d["bbox_h"]]
    return rows


def _cam_geo() -> dict:
    return {c["camera_id"]: c for c in database.list_cameras()}


def _cos(a, b) -> float | None:
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype="float32").ravel()
    b = np.asarray(b, dtype="float32").ravel()
    if a.size != b.size or not a.size:
        return None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return None
    return float(np.dot(a, b) / (na * nb))


def _attr_score(ref_attrs: dict, cand_attrs: dict) -> float | None:
    """Clothing + accessories agreement (a supporting signal only, never alone).
    Vehicle fields are ignored so the mode of travel cannot change identity."""
    ref_attrs, cand_attrs = ref_attrs or {}, cand_attrs or {}
    pts, tot = 0.0, 0.0
    for k in ("upper_color", "lower_color"):
        a, b = ref_attrs.get(k), cand_attrs.get(k)
        if a and b:
            tot += 1.0
            pts += 1.0 if a == b else 0.0
    ra = {x.lower() for x in (ref_attrs.get("accessories") or [])}
    ca = {x.lower() for x in (cand_attrs.get("accessories") or [])}
    if ra or ca:
        tot += 1.0
        pts += (len(ra & ca) / len(ra | ca)) if (ra | ca) else 0.0
    return (pts / tot) if tot else None


def _face_emb_for_track(video_id, track_id):
    """Best stored face embedding for a person track (None if the person has no face)."""
    if video_id is None or track_id is None:
        return None
    with database.get_conn() as conn:
        rows = conn.execute(
            "SELECT f.face_id FROM faces f JOIN detections d ON d.detection_id=f.detection_id "
            "WHERE d.video_id=? AND d.track_id=?", (video_id, track_id)).fetchall()
    for r in rows:
        v = vector_store.get_vector("face", r["face_id"])
        if v is not None:
            return v
    return None


def _mode_from_speed(kmh) -> str:
    """Backward-compatible shim - the Journey Engine owns mode inference now."""
    return journey_engine.infer_travel_mode(kmh)["mode"]


# ---------------------------------------------------------------- matching
def _candidate_appearances(ref: dict, cameras: list[str] | None) -> list[dict]:
    """Find candidate sightings of the SAME person across cameras using fused
    identity signals. Returns one entry per (camera, track), best score kept."""
    ref_det = ref["detection_id"]
    ref_reid = vector_store.get_vector("reid", ref_det)
    ref_body = vector_store.get_vector("clip", ref_det)
    ref_face = _face_emb_for_track(ref.get("video_id"), ref.get("track_id"))
    ref_attrs = ref.get("attributes") or {}

    # gather a candidate pool from every available index
    pool: dict[int, dict] = {}
    def _collect(name, vec):
        if vec is None:
            return
        ids, scores = vector_store.search(name, vec, top_k=POOL)
        for i, s in zip(ids, scores):
            pool.setdefault(int(i), {})[name] = float(s)
    _collect("reid", ref_reid)
    _collect("clip", ref_body)

    face_by_det: dict[int, float] = {}
    if ref_face is not None:
        fids, fscores = vector_store.search("face", ref_face, top_k=POOL)
        frows = database.get_faces(fids)
        fs = dict(zip(fids, fscores))
        for fr in frows:
            did = fr.get("detection_id")
            if did is not None:
                sim = float(fs.get(fr["face_id"], 0.0))
                if sim > face_by_det.get(did, -1):
                    face_by_det[did] = sim
                pool.setdefault(int(did), {})
        # face similarity also reaches detections of the same track
        for did, sim in list(face_by_det.items()):
            pool.setdefault(int(did), {})["face"] = sim

    if not pool:
        return []

    dets = {d["detection_id"]: d for d in database.get_detections(list(pool.keys()))}
    best: dict[tuple, dict] = {}
    vindex = _video_index()
    cam_names = _camera_names()
    cam_filter = set(cameras) if cameras else None

    for did, sig in pool.items():
        d = dets.get(did)
        if not d:
            continue
        # identity is only asserted on PERSON detections (vehicles never define it)
        if d.get("class_label") not in _PERSON_LABELS:
            continue
        cam = d.get("camera_id")
        if cam_filter is not None and cam not in cam_filter:
            continue

        parts = {}
        if "face" in sig:
            parts["face"] = sig["face"]
        if "reid" in sig:
            parts["reid"] = sig["reid"]
        if "body" not in parts and "clip" in sig:
            parts["body"] = sig["clip"]
        a = _attr_score(ref_attrs, d.get("attributes"))
        if a is not None:
            parts["attrs"] = a
        if not parts:
            continue
        wsum = sum(SIGNAL_WEIGHTS.get(k, 0.0) for k in parts)
        if wsum <= 0:
            continue
        fused = sum(SIGNAL_WEIGHTS.get(k, 0.0) * v for k, v in parts.items()) / wsum
        # a strong face match confirms identity on its own (highest priority)
        if parts.get("face", 0.0) >= FACE_STRONG:
            fused = max(fused, parts["face"])
        strong = ("face" in parts) or ("reid" in parts)
        if fused < (IDENTITY_MIN if strong else IDENTITY_MIN_WEAK):
            continue

        pb = playback_fields(d, vindex)
        item = {
            "detection_id": did, "camera_id": cam,
            "camera_name": cam_names.get(cam), "timestamp": d.get("timestamp"),
            "track_id": d.get("track_id"), "video_id": d.get("video_id"),
            "identity_score": round(fused, 4), "signals": {k: round(v, 4) for k, v in parts.items()},
            "evidence_strength": ("face" if "face" in parts else
                                  ("reid" if "reid" in parts else "appearance-only")),
            "crop_url": media_url(d.get("crop_path")),
            "video_url": pb.get("video_url"), "offset_seconds": pb.get("offset_seconds"),
            "attributes": d.get("attributes") or {},
        }
        key = (cam, d.get("track_id"))
        if key not in best or fused > best[key]["identity_score"]:
            best[key] = item
    return list(best.values())


def _per_camera(appearances: list[dict]) -> list[dict]:
    """Collapse to one node per camera: best appearance + first/last seen there."""
    by_cam: dict[str, list[dict]] = {}
    for a in appearances:
        by_cam.setdefault(a["camera_id"], []).append(a)
    nodes = []
    for cam, items in by_cam.items():
        items.sort(key=lambda x: x["identity_score"], reverse=True)
        top = dict(items[0])
        times = sorted(t for t in (_parse_ts(i["timestamp"]) for i in items) if t)
        top["first_seen"] = times[0].isoformat() if times else top["timestamp"]
        top["last_seen"] = times[-1].isoformat() if times else top["timestamp"]
        top["dwell_seconds"] = round((times[-1] - times[0]).total_seconds(), 1) if len(times) > 1 else 0.0
        top["sightings"] = len(items)
        top["alternatives"] = items[1:3]          # runner-up appearances in this camera
        nodes.append(top)
    nodes.sort(key=lambda n: (_parse_ts(n["first_seen"]) or datetime.min))
    return nodes


def _build_legs(nodes: list[dict], geo: dict) -> tuple[list[dict], list[str]]:
    """Delegates to the Journey Engine, which computes distance, travel time,
    estimated speed, camera direction and travel mode, and rejects impossible
    transitions. Kept as a wrapper so existing callers are unaffected."""
    return journey_engine.build_legs(nodes, geo, TRANSITION_EVIDENCE)


def _score_journey(nodes: list[dict], legs: list[dict]) -> float:
    return journey_engine.score(nodes, legs)


def _stats(nodes: list[dict], legs: list[dict]) -> dict:
    return journey_engine.stats(nodes, legs)


UNAVAILABLE_MSG = ("Journey reconstruction unavailable until valid camera "
                   "locations are configured.")


def _route_points(nodes: list[dict], geo: dict,
                  candidates: list[dict] | None = None) -> tuple[list[dict], list[str], list[str]]:
    """Ordered, de-duplicated camera coordinates to route through.

    Includes the ASSERTED journey nodes AND the matched-but-unconfirmed cameras
    (probable / possible / weak / ambiguous). Routing eligibility is a question of
    "was this person seen here and do we know where here is", which is separate
    from "is this identity confirmed". Restricting the route to nodes that cleared
    IDENTITY_ACCEPT meant a person matched in four located cameras still reported
    "only one matched camera has coordinates".

    Identity semantics are untouched: nodes, legs and confidence are unchanged, and
    every routed point carries its tier so an unconfirmed sighting is never drawn as
    a confirmed one. Proximity is never a filter - two cameras metres apart are two
    real locations.

    Returns (points, skipped_no_location, unconfirmed_camera_ids)."""
    seen: dict[str, dict] = {}
    order: list[tuple] = []

    def add(cam_id, ts, tier, identity, confirmed):
        if not cam_id or cam_id in seen:
            return
        seen[cam_id] = {"camera_id": cam_id, "tier": tier, "identity": identity,
                        "confirmed": confirmed, "first_seen": ts,
                        "_ts": _parse_ts(ts)}
        order.append(cam_id)

    for n in nodes:
        add(n.get("camera_id"), n.get("first_seen") or n.get("timestamp"),
            "reference" if n.get("is_reference") else (n.get("tier") or "confirmed"),
            n.get("identity_score"), True)
    for c in (candidates or []):
        add(c.get("camera_id"), c.get("first_seen") or c.get("timestamp"),
            c.get("tier") or track_identity.tier(c.get("identity") or 0.0),
            c.get("identity"), False)

    located, skipped, unconfirmed = [], [], []
    for cam_id in order:
        meta = seen[cam_id]
        g = geo.get(cam_id) or {}
        lat, lon = g.get("lat"), g.get("lon")
        if lat is None or lon is None:
            skipped.append(cam_id)
            continue
        try:
            meta = {**meta, "lat": float(lat), "lon": float(lon)}
        except (TypeError, ValueError):          # stored as unparsable text
            skipped.append(cam_id)
            continue
        located.append(meta)
        if not meta["confirmed"]:
            unconfirmed.append(cam_id)

    # chronological, which is the order the person actually passed the cameras
    located.sort(key=lambda p: (p["_ts"] or datetime.min))
    for p in located:
        p.pop("_ts", None)
    return located, skipped, unconfirmed


def _route_for(nodes: list[dict], geo: dict, candidates: list[dict] | None = None) -> dict:
    """Real road route between the matched cameras, via OSRM (OpenStreetMap).

    Cameras the person was matched at are routed in time order, and only those with
    stored coordinates. Cameras without coordinates are skipped and named in
    `skipped_no_location` so the omission is visible rather than silent.

    Fewer than two located cameras produces no route at all. A routing failure
    produces an explicit "Road route unavailable." - a straight line between
    cameras is never substituted, because it would assert a path through buildings
    that the evidence does not support."""
    pts, skipped, unconfirmed = _route_points(nodes, geo, candidates)
    if len(pts) < 2:
        matched = len(pts) + len(skipped)
        if not pts and not skipped:
            reason = UNAVAILABLE_MSG
        elif not pts:
            reason = (f"None of the {matched} matched camera(s) has stored "
                      "coordinates. Add them in the Camera Registry.")
        else:
            reason = (f"Only 1 of the {matched} matched cameras has stored "
                      "coordinates - at least two are required to reconstruct a "
                      "route. Missing: " + ", ".join(skipped) + ".")
        return {"available": False, "provider": None, "geometry": [],
                "road_route": False, "reason": reason,
                "cameras_with_gps": len(pts), "cameras_matched": matched,
                "cameras_needed": 2, "skipped_no_location": skipped}

    res = routing.cached_route(pts, profile="foot", alternatives=True)
    res["cameras_with_gps"] = len(pts)
    res["cameras_matched"] = len(pts) + len(skipped)
    res["skipped_no_location"] = skipped
    # full per-camera detail (id + tier + score) so the map can mark an
    # unconfirmed waypoint differently from a confirmed one
    res["routed_cameras"] = [p["camera_id"] for p in pts]
    res["routed_detail"] = [
        {k: p.get(k) for k in ("camera_id", "tier", "identity", "confirmed", "first_seen")}
        for p in pts]
    res["includes_unconfirmed"] = bool(unconfirmed)
    res["unconfirmed_cameras"] = unconfirmed
    if unconfirmed:
        res["notice"] = (f"Route spans {len(unconfirmed)} unconfirmed sighting(s) "
                         f"({', '.join(unconfirmed)}). These are probable or possible "
                         "matches, not confirmed identities.")
    if not res.get("available"):
        res["reason"] = res.get("reason") or "Road route unavailable."
    else:
        # score the alternatives so none is presented as certain (Part 7)
        res["alternatives"] = _score_alternatives(res)
    return res


def _score_alternatives(res: dict) -> list[dict]:
    """Label each road route returned by OSRM with a relative confidence.

    OSRM offers several plausible ways to drive/walk between the same points. The
    evidence only fixes the cameras, not the roads taken between them, so the
    shortest-duration route is the most likely rather than the certain one.
    Confidence is its share of inverse travel time across the candidates."""
    routes = [{"label": "Route A", "distance_m": res.get("distance_m"),
               "duration_s": res.get("duration_s"), "geometry": res.get("geometry") or [],
               "primary": True}]
    for i, alt in enumerate(res.get("alternatives") or []):
        routes.append({"label": f"Route {chr(ord('B') + i)}",
                       "distance_m": alt.get("distance_m"),
                       "duration_s": alt.get("duration_s"),
                       "geometry": alt.get("geometry") or [], "primary": False})
    weights = [1.0 / max(float(r["duration_s"] or 1.0), 1.0) for r in routes]
    total = sum(weights) or 1.0
    for r, w in zip(routes, weights):
        r["confidence"] = round(w / total, 4)
        r["note"] = ("most direct road route consistent with the sightings"
                     if r["primary"] else "alternative road route, also consistent")
    return routes


def _journey(nodes: list[dict], geo: dict, label: str,
             candidates: list[dict] | None = None) -> dict:
    """`candidates` are the matched-but-unconfirmed cameras. They take no part in
    legs, timeline or confidence - only in routing, where a located sighting is
    routable regardless of whether its identity is asserted."""
    legs, rejects = _build_legs(nodes, geo)
    t0 = time.perf_counter()
    route = _route_for(nodes, geo, candidates)
    route_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {"label": label, "nodes": nodes, "legs": legs,
            # per-camera timeline with the mode used to reach the next camera and
            # an explicit end state on the final sighting
            "timeline": journey_engine.build_timeline(nodes, legs),
            "rejected_transitions": rejects,
            "confidence": _score_journey(nodes, legs),
            "stats": _stats(nodes, legs),
            "route": route, "route_ms": route_ms,
            # the map must not draw a path unless a real route exists
            "map_ready": bool(route.get("available")),
            "map_notice": None if route.get("available") else route.get("reason")}


# ---------------------------------------------------------------- public API
def _nodes_from_candidates(match: dict, accept: float) -> list[dict]:
    """Turn track-level candidates into journey nodes (one per camera)."""
    nodes = []
    ref = match["reference"]
    cam_names = _camera_names()
    nodes.append({
        "detection_id": ref.get("detection_id"), "camera_id": ref.get("camera_id"),
        "camera_name": cam_names.get(ref.get("camera_id")),
        "timestamp": ref.get("first_seen"),
        "track_id": ref.get("track_id"), "video_id": ref.get("video_id"),
        "identity_score": 1.0, "confidence": 1.0, "signals": {"reference": 1.0},
        "evidence_strength": "reference", "is_reference": True,
        "first_seen": ref.get("first_seen"), "last_seen": ref.get("last_seen"),
        "dwell_seconds": 0.0, "sightings": ref.get("n_detections") or 1,
        "attributes": {"upper_color": ref.get("upper_color"),
                       "lower_color": ref.get("lower_color"),
                       "accessories": ref.get("accessories") or []},
        # vehicle observed WITH the person - travel-mode evidence, never identity
        "vehicle_context": ref.get("vehicle_context") or [],
        "travel_method": None, "reasons": ["Reference track"], "alternatives": [],
    })
    for c in match.get("best_per_camera", []):
        # Gate on the IDENTITY score, which is what the threshold sweep in
        # track_identity calibrated. `confidence` additionally folds in
        # spatio-temporal plausibility and is used for ranking/display only, so
        # gating on it would silently shift the calibrated operating point.
        if c["identity"] < accept or c.get("camera_id") == ref.get("camera_id"):
            continue                                  # reference camera already added
        if c.get("ambiguous"):
            # Two candidates in this camera are inseparable on the evidence. Naming
            # one as the journey would be a confidently wrong identity; it stays in
            # the candidate list for the investigator to judge instead.
            continue
        sig = c.get("signals") or {}
        nodes.append({
            "detection_id": c.get("detection_id"), "camera_id": c.get("camera_id"),
            "camera_name": cam_names.get(c.get("camera_id")),
            "timestamp": c.get("first_seen"),
            "track_id": c.get("track_id"), "video_id": c.get("video_id"),
            "identity_score": c["identity"], "confidence": c["confidence"],
            "tier": c.get("tier"), "signals": sig,
            # per-source contribution breakdown shown for every confirmed match
            "fusion": c.get("fusion"), "context": c.get("context"),
            "evidence_strength": ("face" if "face" in sig else
                                  ("reid" if "reid" in sig else "appearance-only")),
            "first_seen": c.get("first_seen"), "last_seen": c.get("last_seen"),
            "dwell_seconds": 0.0, "sightings": c.get("n_detections") or 1,
            "attributes": {"upper_color": c.get("upper_color"),
                           "lower_color": c.get("lower_color"),
                           "accessories": c.get("accessories") or []},
            "vehicle_context": c.get("vehicle_context") or [],
            "travel_method": c.get("travel_method"), "transition": c.get("transition"),
            "face_pct": c.get("face_pct"), "reid_pct": c.get("reid_pct"),
            "clothing_pct": c.get("clothing_pct"), "accessories_pct": c.get("accessories_pct"),
            "body_pct": c.get("body_pct"), "reasons": c.get("reasons") or [],
            "alternatives": c.get("camera_alternatives") or [],
        })
    nodes.sort(key=lambda n: (_parse_ts(n["first_seen"]) or datetime.min))
    _attach_media(nodes)          # crop + source clip for the timeline / jump-to-video
    return nodes


def _alternative_node(cand: dict) -> dict:
    """Normalise a runner-up sighting into a journey node.

    Runner-ups arrive in two shapes: track-level candidates from track_match
    (`identity`, `first_seen`) and legacy detection-level appearances
    (`identity_score`, `timestamp`). Both are accepted so the alternative-journey
    scoring never depends on which matcher produced the candidate."""
    r = dict(cand)
    ts = r.get("first_seen") or r.get("timestamp")
    r["timestamp"] = r.get("timestamp") or ts
    r["first_seen"] = r.get("first_seen") or ts
    r["last_seen"] = r.get("last_seen") or ts
    if r.get("identity_score") is None:
        r["identity_score"] = r.get("identity") or r.get("confidence") or 0.0
    r.setdefault("dwell_seconds", 0.0)
    r.setdefault("sightings", r.get("n_detections") or 1)
    r.setdefault("vehicle_context", [])
    r["alternatives"] = []
    return r


def reconstruct(detection_id: int, cameras: list[str] | None = None,
                investigation: str | None = None, persist: bool = True,
                accept: float | None = None, top_k: int = 5) -> dict:
    """Reconstruct the probable journey of the person in `detection_id`.

    Uses TRACK-LEVEL identity: the whole ByteTrack track of the reference person
    is compared, descriptor-to-descriptor, against every candidate track in the
    selected cameras - so posture/vehicle changes (motorcycle -> walking) no longer
    break the match. Always returns the top-K candidates, never a bare
    'no match found'.

    `accept` is the fused-identity score required before a camera is ASSERTED as
    part of the journey; it defaults to the swept operating point in
    track_identity.IDENTITY_ACCEPT. Candidates below it are still returned under
    `matching`, tiered probable / possible / weak, so nothing is hidden."""
    if accept is None:
        accept = track_identity.IDENTITY_ACCEPT
    refs = database.get_detections([detection_id])
    if not refs:
        return {"error": "unknown detection"}
    ref = refs[0]
    if ref.get("class_label") not in _PERSON_LABELS:
        return {"error": "Journey reconstruction applies to a person result."}

    geo = _cam_geo()
    vid, tid = ref.get("video_id"), ref.get("track_id")
    match = None
    if vid is not None and tid is not None:
        track_identity.build_all(min_dets=2)                 # idempotent, cheap after first run
        match = track_match.find_candidates(vid, tid, cameras=cameras, top_k=top_k)

    track_level = bool(match) and not match.get("error")
    if track_level:
        # Every candidate shown under "Probable matches per camera" gets its crop
        # and its source clip, so an unconfirmed sighting can be reviewed in the
        # footage exactly like a confirmed one.
        vindex = _video_index()
        _attach_media(match.get("best_per_camera") or [], vindex)
        _attach_media(match.get("candidates") or [], vindex)
    nodes = _nodes_from_candidates(match, accept) if track_level else []
    # If the track-level path asserted no second camera, fall back to the original
    # single-detection matcher before giving up - it uses a different (older) route
    # to the same evidence and occasionally clears its own threshold.
    if len(nodes) <= 1:
        legacy = _per_camera(_candidate_appearances(ref, cameras))
        if len(legacy) > len(nodes):
            nodes, track_level = legacy, False
    if not nodes:
        return {"error": "No matching appearances found for this person."}
    # A single node means "only the reference camera is confirmed". That is NOT an
    # error: the probable candidates below the accept threshold are still returned
    # under `matching` so the investigator sees the near-misses.
    confirmed_cameras = len({n.get("camera_id") for n in nodes})

    # Cameras the person was matched at but which were NOT asserted as journey
    # nodes (below IDENTITY_ACCEPT, or ambiguous). They are routable waypoints -
    # each one is still a place this person probably passed, and it is labelled
    # with its tier - but they change nothing about identity, legs or confidence.
    asserted = {n.get("camera_id") for n in nodes}
    unconfirmed = [c for c in ((match or {}).get("best_per_camera") or [])
                   if c.get("camera_id") and c["camera_id"] not in asserted]

    primary = _journey(nodes, geo, "Primary journey", unconfirmed)

    # Alternatives (honest, minimal): drop the weakest camera, and swap the
    # weakest camera for its runner-up appearance. Each is scored independently.
    alts = []
    if len(nodes) > 2:
        weakest = min(nodes, key=lambda n: n.get("identity_score") or 0.0)
        pruned = [n for n in nodes if n is not weakest]
        alt = _journey(pruned, geo, f"Without {weakest['camera_id']} (weakest match)",
                       unconfirmed)
        if alt["confidence"] > 0:
            alts.append(alt)
    weakest = min(nodes, key=lambda n: n.get("identity_score") or 0.0)
    if weakest.get("alternatives"):
        swapped = []
        for n in nodes:
            if n is weakest:
                swapped.append(_alternative_node(weakest["alternatives"][0]))
            else:
                swapped.append(n)
        swapped.sort(key=lambda n: (_parse_ts(n["first_seen"]) or datetime.min))
        alt = _journey(swapped, geo, f"Alternative sighting in {weakest['camera_id']}",
                       unconfirmed)
        if alt["confidence"] > 0:
            alts.append(alt)
    alts.sort(key=lambda j: j["confidence"], reverse=True)

    result = {
        "reference_detection_id": detection_id,
        "reference": {"camera_id": ref.get("camera_id"), "timestamp": ref.get("timestamp"),
                      "crop_url": media_url(ref.get("crop_path")),
                      "attributes": ref.get("attributes") or {}},
        "scope": "selected" if cameras else "all",
        "cameras_searched": sorted(cameras) if cameras else "all",
        "investigation": investigation,
        "signal_weights": SIGNAL_WEIGHTS,
        "primary": primary, "alternatives": alts[:3],
        # Siting record per camera, including the viewing cone, so the map renders
        # from the SAME stored geometry the reconstruction used.
        "camera_geo": {k: {"lat": v.get("lat"), "lon": v.get("lon"), "name": v.get("name"),
                           "address": v.get("address"), "road_name": v.get("road_name"),
                           "facing_deg": v.get("facing_deg"), "fov_deg": v.get("fov_deg"),
                           "coverage_m": v.get("coverage_m"),
                           "facing": camera_registry.compass_name(v.get("facing_deg")),
                           "coverage_cone": camera_registry.coverage_cone(v),
                           "cone_estimated": bool(v.get("facing_deg") is not None
                                                  and (v.get("fov_deg") is None
                                                       or v.get("coverage_m") is None))}
                       for k, v in geo.items()},
        "registry": camera_registry.registry_status(),
        "route_engine": {"active": routing.get_engine().name,
                         "providers": routing.providers()},
        # every probable candidate track, with its full reason breakdown, so the
        # investigator sees near-misses instead of a bare "no match found"
        "matching": {"mode": "track-level" if track_level else "detection-level",
                     "accept_threshold": accept,
                     "probable_threshold": track_identity.IDENTITY_PROBABLE,
                     "confirmed_cameras": confirmed_cameras,
                     "status": ("confirmed" if confirmed_cameras > 1 else
                                "unconfirmed - showing probable candidates only"),
                     "searched_tracks": (match or {}).get("searched_tracks"),
                     "compared_tracks": (match or {}).get("compared_tracks"),
                     "signal_weights": (match or {}).get("signal_weights"),
                     "candidates": (match or {}).get("candidates", []),
                     "best_per_camera": (match or {}).get("best_per_camera", [])},
    }
    if persist:
        try:
            result["journey_id"] = _save(result)
        except Exception:
            pass
    return result


def _save(result: dict) -> int:
    with database.get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO journeys (investigation, reference_detection_id, scope, confidence, "
            " camera_count, distance_km, span_seconds, data, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (result.get("investigation"), result["reference_detection_id"], result["scope"],
             result["primary"]["confidence"], result["primary"]["stats"]["cameras"],
             result["primary"]["stats"]["distance_km"], result["primary"]["stats"]["span_seconds"],
             json.dumps(result), database._now()))
        return cur.lastrowid


def list_journeys(investigation: str | None = None) -> list[dict]:
    q = ("SELECT journey_id, investigation, reference_detection_id, scope, confidence, "
         " camera_count, distance_km, span_seconds, created_at FROM journeys")
    params = []
    if investigation:
        q += " WHERE investigation=?"
        params.append(investigation)
    q += " ORDER BY created_at DESC"
    with database.get_conn() as conn:
        return [dict(r) for r in conn.execute(q, params).fetchall()]


def get_journey(journey_id: int) -> dict | None:
    with database.get_conn() as conn:
        row = conn.execute("SELECT data FROM journeys WHERE journey_id=?", (journey_id,)).fetchone()
    if not row:
        return None
    try:
        d = json.loads(row["data"])
        d["journey_id"] = journey_id
        return d
    except Exception:
        return None


def delete_journey(journey_id: int) -> dict:
    with database.get_conn() as conn:
        conn.execute("DELETE FROM journeys WHERE journey_id=?", (journey_id,))
    return {"deleted": journey_id}


# ---------------------------------------------------------------- export
def export_journey(journey_id: int, fmt: str = "json") -> dict | None:
    """Export a stored journey. Returns None when the journey does not exist."""
    j = get_journey(journey_id)
    if j is None:
        return None
    primary = j.get("primary") or {}
    stats = primary.get("stats") or {}

    if fmt == "json":
        return {"format": "json", "journey_id": journey_id, "journey": j}

    if fmt == "summary":
        return {
            "format": "summary", "journey_id": journey_id,
            "investigation": j.get("investigation"),
            "reference": j.get("reference"),
            "confidence": primary.get("confidence"),
            "statistics": {
                "total_distance_km": stats.get("distance_km"),
                "total_time_seconds": stats.get("travel_seconds"),
                "cameras_visited": stats.get("cameras_visited") or stats.get("cameras"),
                "average_speed_kmh": stats.get("avg_speed_kmh"),
                "estimated_transport": stats.get("estimated_transport"),
                "span_seconds": stats.get("span_seconds"),
                "gps_available": stats.get("gps_available"),
                "rejected_transitions": stats.get("rejected_transitions"),
                "unverified_legs": stats.get("unverified_legs"),
            },
            "timeline": primary.get("timeline") or [],
            "rejected_transitions": primary.get("rejected_transitions") or [],
            "route": {"available": (primary.get("route") or {}).get("available"),
                      "provider": (primary.get("route") or {}).get("provider"),
                      "notice": primary.get("map_notice")},
            "route_engine": j.get("route_engine"),
        }

    # geojson: camera markers always; the road path ONLY if a routing engine
    # produced one. A straight line between cameras is never emitted.
    geo = j.get("camera_geo") or {}
    features = []
    for row in (primary.get("timeline") or []):
        g = geo.get(row["camera_id"]) or {}
        if g.get("lat") is None or g.get("lon") is None:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [g["lon"], g["lat"]]},
            "properties": {"camera_id": row["camera_id"], "camera_name": g.get("name"),
                           "sequence": row["index"] + 1, "timestamp": row["timestamp"],
                           "confidence": row.get("confidence"),
                           "travel_to_next": row.get("next_mode_label"),
                           "end_state": row.get("end_state"),
                           "address": g.get("address"), "road_name": g.get("road_name"),
                           "facing_deg": g.get("facing_deg"), "fov_deg": g.get("fov_deg")},
        })
    route = primary.get("route") or {}
    notice = None
    if route.get("available") and route.get("geometry"):
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [[lon, lat] for lat, lon in route["geometry"]]},
            "properties": {"kind": "road_route", "provider": route.get("provider"),
                           "distance_m": route.get("distance_m"),
                           "duration_s": route.get("duration_s")},
        })
    else:
        notice = primary.get("map_notice") or UNAVAILABLE_MSG
    return {"format": "geojson", "journey_id": journey_id,
            "geojson": {"type": "FeatureCollection", "features": features},
            "road_route_included": bool(route.get("available") and route.get("geometry")),
            "notice": notice}


def save_to_case_file(journey_id: int, investigation: str | None = None,
                      note: str | None = None) -> dict:
    """Seal a journey into the case file, reusing the existing SHA-256 chain of
    custody: the sighting at every camera becomes an evidence item."""
    j = get_journey(journey_id)
    if j is None:
        return {"error": "journey not found"}
    primary = j.get("primary") or {}
    det_ids = [n["detection_id"] for n in (primary.get("nodes") or [])
               if n.get("detection_id") is not None]
    if not det_ids:
        return {"error": "journey has no exportable sightings"}
    stats = primary.get("stats") or {}
    case = investigation or j.get("investigation") or "unassigned"
    summary = (f"Journey #{journey_id}: {stats.get('cameras_visited') or stats.get('cameras')} "
               f"cameras, {stats.get('distance_km') or 'unknown'} km, "
               f"transport {stats.get('estimated_transport')}, "
               f"confidence {primary.get('confidence')}. "
               f"Modes: {' -> '.join(stats.get('mode_sequence') or []) or 'n/a'}.")
    from .forensics import create_export
    from .models.schemas import ExportRequest
    res = create_export(ExportRequest(
        detection_ids=det_ids, case_number=case, officer="Journey Engine",
        notes=" ".join(x for x in (summary, note) if x)))
    out = res.model_dump() if hasattr(res, "model_dump") else dict(res)
    out.update({"journey_id": journey_id, "investigation": case,
                "sightings_sealed": len(det_ids), "summary": summary})
    database.log_audit("journey_case_file", query_type="journey",
                       result_count=len(det_ids),
                       details={"journey_id": journey_id, "export_id": out.get("export_id")})
    return out
