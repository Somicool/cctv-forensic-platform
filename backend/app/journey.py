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
from datetime import datetime

import numpy as np

from . import config, database
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
    if kmh is None:
        return "unknown"
    if kmh <= WALK_KMH:
        return "walking"
    if kmh <= CYCLE_KMH:
        return "two-wheeler"
    return "vehicle"


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
    """Transitions between consecutive cameras: travel time, GPS distance, speed,
    inferred mode. Impossible transitions (too fast for the distance) are flagged."""
    legs, rejects = [], []
    for a, b in zip(nodes, nodes[1:]):
        t0, t1 = _parse_ts(a["last_seen"]), _parse_ts(b["first_seen"])
        dt_s = (t1 - t0).total_seconds() if (t0 and t1) else None
        ca, cb = geo.get(a["camera_id"]) or {}, geo.get(b["camera_id"]) or {}
        dist = None
        if all(v is not None for v in (ca.get("lat"), ca.get("lon"), cb.get("lat"), cb.get("lon"))):
            dist = round(_haversine_km(ca["lat"], ca["lon"], cb["lat"], cb["lon"]), 4)
        speed = None
        if dist is not None and dt_s and dt_s > 0:
            speed = round(dist / (dt_s / 3600.0), 2)
        plausible, why = True, "plausible"
        overlap = dt_s is not None and dt_s < 0
        if overlap:
            # seen in the next camera before leaving this one: either overlapping
            # camera coverage, or (if the cameras are far apart) impossible.
            if dist is not None and dist > 0.15:
                plausible = False
                why = (f"impossible: seen at both cameras simultaneously "
                       f"({abs(round(dt_s,1))}s overlap, {dist} km apart)")
            else:
                why = f"overlapping coverage - simultaneous sighting ({abs(round(dt_s,1))}s overlap)"
            speed = None
        elif dist is not None and dt_s is not None:
            if dt_s == 0 and dist > 0.05:
                plausible, why = False, "same instant at different locations"
            elif speed is not None and speed > MAX_KMH:
                plausible = False
                why = f"impossible: {dist} km in {round(dt_s/60,1)} min = {speed} km/h (> {MAX_KMH})"
        elif dist is None:
            why = "no camera GPS - distance/speed unavailable"
        leg = {"from_camera": a["camera_id"], "to_camera": b["camera_id"],
               "from_time": a["last_seen"], "to_time": b["first_seen"],
               "travel_seconds": round(dt_s, 1) if dt_s is not None else None,
               "distance_km": dist, "avg_speed_kmh": speed,
               "mode": ("overlap" if overlap else _mode_from_speed(speed)),
               "plausible": plausible, "note": why,
               "evidence": ["reid/face identity"]}
        # future providers (gps track, anpr, mobile) can enrich each leg here
        for name, fn in TRANSITION_EVIDENCE.items():
            try:
                extra = fn(a, b)
                if extra:
                    leg.setdefault("extra", {})[name] = extra
                    leg["evidence"].append(name)
            except Exception:
                pass
        legs.append(leg)
        if not plausible:
            rejects.append(f"{a['camera_id']} -> {b['camera_id']}: {why}")
    return legs, rejects


def _score_journey(nodes: list[dict], legs: list[dict]) -> float:
    """Confidence = mean identity strength penalised by implausible transitions
    and by legs we could not verify (no GPS)."""
    if not nodes:
        return 0.0
    ident = sum(n["identity_score"] for n in nodes) / len(nodes)
    if not legs:
        return round(ident, 4)
    ok = sum(1 for l in legs if l["plausible"]) / len(legs)
    unverified = sum(1 for l in legs if l["distance_km"] is None) / len(legs)
    return round(max(0.0, ident * (0.35 + 0.65 * ok) * (1.0 - 0.10 * unverified)), 4)


def _stats(nodes: list[dict], legs: list[dict]) -> dict:
    dists = [l["distance_km"] for l in legs if l["distance_km"] is not None]
    # overlapping (negative) legs are not travel time - exclude from totals
    times = [l["travel_seconds"] for l in legs
             if l["travel_seconds"] is not None and l["travel_seconds"] > 0]
    total_km = round(sum(dists), 3) if dists else None
    total_s = round(sum(times), 1) if times else None
    avg = round(total_km / (total_s / 3600.0), 2) if (total_km and total_s and total_s > 0) else None
    t0 = _parse_ts(nodes[0]["first_seen"]) if nodes else None
    t1 = _parse_ts(nodes[-1]["last_seen"]) if nodes else None
    return {"cameras": len(nodes), "legs": len(legs),
            "distance_km": total_km, "travel_seconds": total_s,
            "avg_speed_kmh": avg,
            "dwell_seconds": round(sum(n.get("dwell_seconds") or 0 for n in nodes), 1),
            "span_seconds": round((t1 - t0).total_seconds(), 1) if (t0 and t1) else None,
            "first_seen": nodes[0]["first_seen"] if nodes else None,
            "last_seen": nodes[-1]["last_seen"] if nodes else None,
            "gps_available": bool(dists)}


def _journey(nodes: list[dict], geo: dict, label: str) -> dict:
    legs, rejects = _build_legs(nodes, geo)
    return {"label": label, "nodes": nodes, "legs": legs,
            "rejected_transitions": rejects,
            "confidence": _score_journey(nodes, legs),
            "stats": _stats(nodes, legs)}


# ---------------------------------------------------------------- public API
def reconstruct(detection_id: int, cameras: list[str] | None = None,
                investigation: str | None = None, persist: bool = True) -> dict:
    """Reconstruct the probable journey of the person in `detection_id`.

    cameras=None -> all cameras; otherwise only the given camera ids.
    Returns the primary journey plus alternatives, each with a confidence."""
    refs = database.get_detections([detection_id])
    if not refs:
        return {"error": "unknown detection"}
    ref = refs[0]
    if ref.get("class_label") not in _PERSON_LABELS:
        return {"error": "Journey reconstruction applies to a person result."}

    apps = _candidate_appearances(ref, cameras)
    geo = _cam_geo()
    nodes = _per_camera(apps)
    if not nodes:
        return {"error": "No matching appearances found for this person."}

    primary = _journey(nodes, geo, "Primary journey")

    # Alternatives (honest, minimal): drop the weakest camera, and swap the
    # weakest camera for its runner-up appearance. Each is scored independently.
    alts = []
    if len(nodes) > 2:
        weakest = min(nodes, key=lambda n: n["identity_score"])
        pruned = [n for n in nodes if n is not weakest]
        alt = _journey(pruned, geo, f"Without {weakest['camera_id']} (weakest match)")
        if alt["confidence"] > 0:
            alts.append(alt)
    weakest = min(nodes, key=lambda n: n["identity_score"])
    if weakest.get("alternatives"):
        swapped = []
        for n in nodes:
            if n is weakest:
                r = dict(weakest["alternatives"][0])
                r.setdefault("first_seen", r["timestamp"]); r.setdefault("last_seen", r["timestamp"])
                r.setdefault("dwell_seconds", 0.0); r.setdefault("sightings", 1)
                r["alternatives"] = []
                swapped.append(r)
            else:
                swapped.append(n)
        swapped.sort(key=lambda n: (_parse_ts(n["first_seen"]) or datetime.min))
        alt = _journey(swapped, geo, f"Alternative sighting in {weakest['camera_id']}")
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
        "camera_geo": {k: {"lat": v.get("lat"), "lon": v.get("lon"), "name": v.get("name")}
                       for k, v in geo.items()},
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
