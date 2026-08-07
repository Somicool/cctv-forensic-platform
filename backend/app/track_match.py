"""Cross-camera track matching with spatio-temporal prioritisation + re-ranking.

Pipeline (replaces "one crop -> one embedding -> reject"):

    reference Track Identity Descriptor
        -> SPATIO-TEMPORAL PREFILTER   cheap: rank cameras by |dt| and GPS distance
        -> compare descriptors         expensive: set-to-set ReID / face / clothing
        -> RE-RANK                     identity x timeline x travel plausibility
        -> Top-K candidates            never "no match" - always the best guesses

Nothing here rejects a candidate outright: the investigator always receives the
top-K probable tracks with a full reason breakdown, and decides.
"""
from __future__ import annotations

import math
from datetime import datetime

from . import config, database, track_identity

# spatio-temporal search window
MAX_GAP_S = 3600.0            # cameras more than this apart in time are deprioritised
MAX_KMH = 60.0                # faster than this between two cameras is implausible
WALK_KMH, CYCLE_KMH = 7.0, 25.0
TOP_K = 5                     # candidate tracks returned per camera
# final confidence = identity x (base + span * spatio-temporal plausibility)
ST_BASE, ST_SPAN = 0.72, 0.28


def _ts(v):
    if not v:
        return None
    try:
        d = datetime.fromisoformat(str(v))
    except (TypeError, ValueError):
        return None
    return d.replace(tzinfo=None) if d.tzinfo else d


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, a)))


def _cams() -> dict:
    return {c["camera_id"]: c for c in database.list_cameras()}


def travel_mode(dist_km, dt_s, ref_veh=None, cand_veh=None) -> str:
    """Estimated travel method between two sightings.

    Uses speed when GPS is available; otherwise falls back to the vehicle CONTEXT
    observed on each track (which is never part of the identity itself). This is
    what expresses walking -> motorcycle, scooter -> walking, and so on."""
    if dist_km is not None and dt_s and dt_s > 0:
        kmh = dist_km / (dt_s / 3600.0)
        if kmh <= WALK_KMH:
            return "walking"
        if kmh <= CYCLE_KMH:
            return "two-wheeler"
        return "vehicle"
    ctx = set(cand_veh or []) or set(ref_veh or [])
    if {"motorcycle"} & ctx:
        return "motorcycle (observed)"
    if {"scooter"} & ctx:
        return "scooter (observed)"
    if {"bicycle"} & ctx:
        return "bicycle (observed)"
    if {"car", "auto-rickshaw", "truck", "bus"} & ctx:
        return "vehicle (observed)"
    return "walking (assumed)"


def transition_label(ref_desc, cand_desc, mode) -> str:
    """Human-readable posture/vehicle transition, e.g. 'motorcycle -> walking'."""
    def side(d):
        v = set(d.get("vehicle_context") or [])
        for k in ("motorcycle", "scooter", "bicycle", "car", "auto-rickshaw", "truck", "bus"):
            if k in v:
                return k
        return "walking"
    return f"{side(ref_desc)} -> {side(cand_desc)}"


def spatiotemporal_score(ref, cand, cams) -> dict:
    """Cheap plausibility BEFORE any embedding maths.

    Scores how reachable a candidate camera/time is from the reference. Cameras
    that are close in time (and, when GPS exists, close in space) score highest,
    so the expensive comparison runs on the most promising candidates first."""
    t_ref = _ts(ref.get("last_seen")) or _ts(ref.get("first_seen"))
    t_can = _ts(cand.get("first_seen")) or _ts(cand.get("last_seen"))
    dt = (t_can - t_ref).total_seconds() if (t_ref and t_can) else None

    a, b = cams.get(ref.get("camera_id")) or {}, cams.get(cand.get("camera_id")) or {}
    dist = None
    if all(v is not None for v in (a.get("lat"), a.get("lon"), b.get("lat"), b.get("lon"))):
        dist = _haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])

    # time proximity: sooner after the reference = more likely the same journey
    if dt is None:
        t_score, timeline = 0.5, "unknown"
    else:
        adt = abs(dt)
        t_score = max(0.0, 1.0 - adt / MAX_GAP_S)
        timeline = "valid" if dt >= -5 else "before reference"

    speed = None
    s_score, verdict = 0.5, "no camera GPS - distance unverified"
    if dist is not None and dt is not None:
        if dt <= 0:
            s_score = 0.35 if dist > 0.15 else 0.6
            verdict = "overlapping sighting" if dist <= 0.15 else "simultaneous but distant"
        else:
            speed = dist / (dt / 3600.0)
            if speed > MAX_KMH:
                s_score, verdict = 0.0, f"implausible speed {speed:.1f} km/h"
            else:
                s_score = max(0.15, 1.0 - speed / MAX_KMH)
                verdict = f"plausible ({speed:.1f} km/h)"
    combined = 0.55 * t_score + 0.45 * s_score
    return {"score": round(combined, 4), "dt_s": None if dt is None else round(dt, 1),
            "distance_km": None if dist is None else round(dist, 4),
            "speed_kmh": None if speed is None else round(speed, 2),
            "timeline": timeline, "travel_verdict": verdict,
            "time_score": round(t_score, 3), "speed_score": round(s_score, 3)}


def _bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p = math.pi / 180
    dl = (lon2 - lon1) * p
    y = math.sin(dl) * math.cos(lat2 * p)
    x = (math.cos(lat1 * p) * math.sin(lat2 * p)
         - math.sin(lat1 * p) * math.cos(lat2 * p) * math.cos(dl))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def context_evidence(ref: dict, cand: dict, st: dict, cams: dict) -> dict:
    """Spatio-temporal corroboration for the Identity Fusion Engine.

    Each value is a 0..1 plausibility, or absent when the deployment cannot supply
    it (no camera GPS, no facing angle). None of these can identify a person on
    their own - the engine only lets them corroborate or contradict."""
    ev: dict = {}
    dt, dist = st.get("dt_s"), st.get("distance_km")

    # timeline consistency: the candidate should be at or after the reference, and
    # sooner is more consistent with one continuous journey
    if dt is not None:
        if dt < -5:
            ev["timeline"] = 0.1                      # sighting precedes the reference
        else:
            ev["timeline"] = max(0.15, 1.0 - abs(dt) / MAX_GAP_S)

    # journey continuity: one person cannot be in two cameras at the same instant,
    # so overlapping sighting intervals in DIFFERENT cameras contradict the match
    r_end, c_start = _ts(ref.get("last_seen")), _ts(cand.get("first_seen"))
    r_start, c_end = _ts(ref.get("first_seen")), _ts(cand.get("last_seen"))
    if all(v is not None for v in (r_start, r_end, c_start, c_end)):
        overlap = (min(r_end, c_end) - max(r_start, c_start)).total_seconds()
        same_cam = ref.get("camera_id") == cand.get("camera_id")
        if overlap > 0 and not same_cam:
            span = max(1.0, (c_end - c_start).total_seconds())
            ev["continuity"] = max(0.05, 0.5 - 0.5 * min(1.0, overlap / span))
        else:
            ev["continuity"] = 1.0 if not same_cam else 0.6

    # the remaining three need camera geometry, which many deployments lack
    if dist is not None:
        ev["gps_proximity"] = max(0.1, 1.0 - min(1.0, dist / 2.0))
    speed = st.get("speed_kmh")
    if speed is not None:
        ev["travel_time"] = 0.05 if speed > MAX_KMH else max(0.2, 1.0 - speed / MAX_KMH)
    a, b = cams.get(ref.get("camera_id")) or {}, cams.get(cand.get("camera_id")) or {}
    if (a.get("facing_deg") is not None and dist
            and all(v is not None for v in (a.get("lat"), a.get("lon"),
                                            b.get("lat"), b.get("lon")))):
        brg = _bearing_deg(a["lat"], a["lon"], b["lat"], b["lon"])
        diff = abs((brg - a["facing_deg"] + 180.0) % 360.0 - 180.0)
        # leaving through the camera's field of view is consistent travel
        ev["direction"] = max(0.1, 1.0 - diff / 180.0)
    return ev


def _reasons(cmp, st) -> list[str]:
    """Short investigator-facing reasons, strongest first."""
    sig = cmp.get("signals") or {}
    fusion = cmp.get("fusion") or {}
    out = []
    corr = fusion.get("corroboration") or {}
    if corr.get("strong", 0) >= 2:
        out.append(f"{corr['strong']}/{corr['groups']} independent evidence groups agree")
    if "face" in sig:
        out.append(f"Face {round(sig['face'] * 100)}%")
    if "reid" in sig:
        out.append(f"ReID {round(sig['reid'] * 100)}%")
    if "clothing" in sig:
        out.append(f"Clothing {round(sig['clothing'] * 100)}%")
    if sig.get("upper_color", 0) >= 0.99:
        out.append("Upper clothing colour matches")
    if sig.get("lower_color", 0) >= 0.99:
        out.append("Lower clothing colour matches")
    if sig.get("bag", 0) >= 0.99:
        out.append("Bag match")
    if sig.get("headwear", 0) >= 0.99:
        out.append("Helmet/cap match")
    if sig.get("body", 0) >= 0.85:
        out.append("Body shape match")
    if st.get("timeline") == "valid":
        out.append("Timeline valid")
    if st.get("speed_kmh") is not None:
        out.append(f"Travel speed {st['travel_verdict']}")
    if corr.get("contradicting"):
        out.append(f"{corr['contradicting']} evidence group(s) disagree")
    return out


def find_candidates(video_id: int, track_id: int, cameras=None, top_k: int = TOP_K,
                    prefilter: int = 60) -> dict:
    """Top-K candidate tracks per camera for one reference person track.

    Never returns 'no match': candidates are ranked and returned with their full
    reason breakdown so the investigator can judge borderline cases."""
    ref = track_identity.load_descriptor(video_id, track_id)
    if ref is None:
        return {"error": "reference track has no identity descriptor"}

    cams = _cams()
    pool = track_identity.list_descriptors()
    # exclude the reference track and (by default) its own camera
    pool = [d for d in pool if not (d["video_id"] == video_id and d["track_id"] == track_id)]
    if cameras:
        want = set(cameras)
        pool = [d for d in pool if d.get("camera_id") in want]

    # ---- 1) SPATIO-TEMPORAL PREFILTER (cheap) --------------------------------
    scored = []
    for d in pool:
        st = spatiotemporal_score(ref, d, cams)
        scored.append((st["score"], st, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    considered = scored[:max(prefilter, top_k * 4)]

    # ---- 2) EXPENSIVE DESCRIPTOR COMPARISON on the shortlist -----------------
    results = []
    for _s, st, d in considered:
        ctx = context_evidence(ref, d, st, cams)
        cmp = track_identity.compare(ref, d, context=ctx)
        ident = cmp["identity"]
        # the fusion engine already folded the spatio-temporal evidence in, so the
        # old multiplicative discount would double-count it
        conf = ident
        mode = travel_mode(st["distance_km"], st["dt_s"],
                           ref.get("vehicle_context"), d.get("vehicle_context"))
        results.append({
            "video_id": d["video_id"], "track_id": d["track_id"],
            "camera_id": d.get("camera_id"),
            "first_seen": d.get("first_seen"), "last_seen": d.get("last_seen"),
            "detection_id": d.get("rep_detection_id"),
            "n_detections": d.get("n_detections"), "n_views": d.get("n_views"),
            "identity": ident, "confidence": round(conf, 4),
            "tier": cmp["tier"], "signals": cmp["signals"],
            "face_pct": round(cmp["signals"].get("face", 0) * 100),
            "reid_pct": round(cmp["signals"].get("reid", 0) * 100),
            "clothing_pct": round(cmp["signals"].get("clothing", 0) * 100),
            "accessories_pct": round(cmp["signals"].get("accessories", 0) * 100),
            "body_pct": round(cmp["signals"].get("body", 0) * 100),
            "fusion": cmp.get("fusion"), "context": ctx,
            "spatiotemporal": st, "travel_method": mode,
            "transition": transition_label(ref, d, mode),
            "upper_color": d.get("upper_color"), "lower_color": d.get("lower_color"),
            "accessories": d.get("accessories"), "vehicle_context": d.get("vehicle_context"),
            "reasons": _reasons(cmp, st),
        })

    # ---- 3) RE-RANK and keep the best per camera ----------------------------
    results.sort(key=lambda r: r["confidence"], reverse=True)
    per_cam: dict = {}
    for r in results:
        per_cam.setdefault(r["camera_id"], []).append(r)
    best_per_camera = []
    for cam, items in per_cam.items():
        top = dict(items[0])
        top["camera_alternatives"] = items[1:top_k]
        best_per_camera.append(top)
    best_per_camera.sort(key=lambda r: r["confidence"], reverse=True)

    return {
        "reference": {"video_id": video_id, "track_id": track_id,
                      "camera_id": ref.get("camera_id"),
                      "first_seen": ref.get("first_seen"), "last_seen": ref.get("last_seen"),
                      "upper_color": ref.get("upper_color"), "lower_color": ref.get("lower_color"),
                      "accessories": ref.get("accessories"), "has_face": ref.get("has_face"),
                      "n_views": ref.get("n_views"), "n_detections": ref.get("n_detections"),
                      "vehicle_context": ref.get("vehicle_context"),
                      "detection_id": ref.get("rep_detection_id")},
        "searched_tracks": len(pool), "compared_tracks": len(considered),
        "signal_weights": track_identity.WEIGHTS,
        "context_weights": {k: v["weight"] for k, v in
                            track_identity.identity_fusion.CONTEXT_SIGNALS.items()},
        "identity_accept": track_identity.IDENTITY_ACCEPT,
        "identity_probable": track_identity.IDENTITY_PROBABLE,
        "candidates": results[:max(top_k * 4, 20)],
        "best_per_camera": best_per_camera,
    }


def track_for_detection(detection_id: int):
    """(video_id, track_id) of a detection - entry point from a search result."""
    rows = database.get_detections([detection_id])
    if not rows:
        return None, None
    return rows[0].get("video_id"), rows[0].get("track_id")
