"""Journey Engine - turns confirmed cross-camera sightings into a movement story.

Separated out of `journey.py` so identity matching and movement reconstruction are
independent concerns: `journey.py` decides WHO was seen where, this module decides
HOW they moved. Nothing here touches identity, OCR, search, tracking or the
database schema.

Given sightings sorted by time it computes, for every camera-to-camera transition:

    distance          great-circle distance from the Camera Registry coordinates
    travel time       gap between leaving one camera and entering the next
    estimated speed   distance / travel time
    camera direction  bearing of travel, checked against the camera's facing angle
                      and field of view (did the person leave through the view?)
    travel mode       walking / scooter / motorcycle / car / unknown
    plausibility      transitions that are physically impossible are REJECTED

Two rules the engine never breaks:

1. Camera coverage is respected. Two cameras whose coverage circles overlap can
   legitimately see the same person at the same instant; that is not an
   impossible transition. Coverage radius comes from the registry, not a constant.
2. Missing data is reported, never invented. No GPS means distance, speed and
   mode are `None` and the leg is marked unverified - it is not guessed.
"""
from __future__ import annotations

import math
from datetime import datetime

# ---------------------------------------------------------------- travel modes
WALKING, SCOOTER, MOTORCYCLE, CAR, UNKNOWN = (
    "walking", "scooter", "motorcycle", "car", "unknown")
OVERLAP = "overlap"

# Speed bands (km/h) used only when no vehicle was actually observed with the
# person. Deliberately conservative: a band boundary is a guess, an observation
# is evidence.
WALK_MAX = 7.0
SCOOTER_MAX = 25.0
MOTORCYCLE_MAX = 45.0
CAR_MAX = 60.0                 # above this, the transition is impossible on foot/road
MAX_KMH = CAR_MAX

# How the detector's vehicle labels map onto the reported travel modes.
VEHICLE_MODE = {
    "motorcycle": MOTORCYCLE, "scooter": SCOOTER, "moped": SCOOTER,
    "car": CAR, "truck": CAR, "bus": CAR, "van": CAR,
    "auto-rickshaw": CAR, "tempo": CAR, "mini-truck": CAR, "pickup": CAR,
}
MODE_LABEL = {WALKING: "Walking", SCOOTER: "Scooter", MOTORCYCLE: "Motorcycle",
              CAR: "Car", UNKNOWN: "Unknown", OVERLAP: "Overlapping coverage"}
# Default coverage radius when a camera has none recorded, in metres.
DEFAULT_COVERAGE_M = 60.0
COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")


# ---------------------------------------------------------------- geometry
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, a)))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """Initial compass bearing from point A to point B, degrees clockwise of north."""
    p = math.pi / 180
    dl = (lon2 - lon1) * p
    y = math.sin(dl) * math.cos(lat2 * p)
    x = (math.cos(lat1 * p) * math.sin(lat2 * p)
         - math.sin(lat1 * p) * math.cos(lat2 * p) * math.cos(dl))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def compass(deg) -> str | None:
    return None if deg is None else COMPASS[int((deg % 360) / 22.5 + 0.5) % 16]


def _angle_diff(a, b) -> float:
    """Smallest absolute difference between two bearings, 0..180."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _parse_ts(ts):
    if not ts:
        return None
    try:
        d = datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None
    return d.replace(tzinfo=None) if d.tzinfo else d


def _coverage_km(cam: dict) -> float:
    try:
        m = float(cam.get("coverage_m") or DEFAULT_COVERAGE_M)
    except (TypeError, ValueError):
        m = DEFAULT_COVERAGE_M
    return max(0.0, m) / 1000.0


# ---------------------------------------------------------------- travel mode
def infer_travel_mode(speed_kmh, from_vehicles=None, to_vehicles=None) -> dict:
    """Estimate how the person travelled between two cameras.

    An observed vehicle beats a speed estimate: the pipeline detects the vehicle a
    person is physically on (spatial overlap in the same frame), which is direct
    evidence, whereas a speed band is an inference. Speed is then used to
    corroborate or contradict that observation."""
    observed = None
    for label in list(to_vehicles or []) + list(from_vehicles or []):
        if label in VEHICLE_MODE:
            observed = label
            break

    if speed_kmh is None:
        if observed:
            return {"mode": VEHICLE_MODE[observed], "confidence": 0.75,
                    "basis": f"{observed} observed with the person (no GPS to verify speed)",
                    "observed_vehicle": observed, "speed_kmh": None}
        return {"mode": UNKNOWN, "confidence": 0.0,
                "basis": "no camera GPS and no vehicle observed - mode cannot be estimated",
                "observed_vehicle": None, "speed_kmh": None}

    if speed_kmh <= WALK_MAX:
        band = WALKING
    elif speed_kmh <= SCOOTER_MAX:
        band = SCOOTER
    elif speed_kmh <= MOTORCYCLE_MAX:
        band = MOTORCYCLE
    elif speed_kmh <= CAR_MAX:
        band = CAR
    else:
        return {"mode": UNKNOWN, "confidence": 0.0,
                "basis": f"{speed_kmh} km/h exceeds the plausible maximum of {CAR_MAX} km/h",
                "observed_vehicle": observed, "speed_kmh": speed_kmh}

    if not observed:
        return {"mode": band, "confidence": 0.6,
                "basis": f"estimated from {speed_kmh} km/h over the measured distance",
                "observed_vehicle": None, "speed_kmh": speed_kmh}

    mode = VEHICLE_MODE[observed]
    if mode == band:
        return {"mode": mode, "confidence": 0.95,
                "basis": f"{observed} observed with the person and {speed_kmh} km/h agrees",
                "observed_vehicle": observed, "speed_kmh": speed_kmh}
    # observation wins, but the disagreement is reported rather than hidden
    return {"mode": mode, "confidence": 0.7,
            "basis": (f"{observed} observed with the person; measured {speed_kmh} km/h "
                      f"suggests {MODE_LABEL[band].lower()} - the person may have "
                      "travelled part of the way differently"),
            "observed_vehicle": observed, "speed_kmh": speed_kmh}


def dominant_mode(legs: list[dict]) -> dict:
    """Overall estimated transport for the journey, weighted by distance covered."""
    weight: dict = {}
    for leg in legs:
        m = (leg.get("travel") or {}).get("mode")
        if not m or m in (UNKNOWN, OVERLAP):
            continue
        w = leg.get("distance_km") or 0.0
        weight[m] = weight.get(m, 0.0) + max(w, 0.001)
    if not weight:
        return {"mode": UNKNOWN, "label": MODE_LABEL[UNKNOWN], "share": None}
    best = max(weight, key=weight.get)
    return {"mode": best, "label": MODE_LABEL[best],
            "share": round(weight[best] / sum(weight.values()), 3)}


# ---------------------------------------------------------------- direction
def direction_for(from_cam: dict, to_cam: dict) -> dict:
    """Travel bearing plus whether it agrees with the cameras' installed geometry.

    A person walking out of camera A toward camera B should leave through A's field
    of view. When it does not, the transition is still possible (the person may
    have doubled back out of shot) but it is worth flagging to the investigator."""
    out: dict = {"bearing_deg": None, "compass": None,
                 "left_through_view": None, "note": None}
    if not all(v is not None for v in (from_cam.get("lat"), from_cam.get("lon"),
                                       to_cam.get("lat"), to_cam.get("lon"))):
        out["note"] = "camera coordinates missing - direction not computed"
        return out
    brg = bearing_deg(from_cam["lat"], from_cam["lon"], to_cam["lat"], to_cam["lon"])
    out["bearing_deg"] = round(brg, 1)
    out["compass"] = compass(brg)
    facing, fov = from_cam.get("facing_deg"), from_cam.get("fov_deg")
    if facing is None:
        out["note"] = f"travelled {out['compass']}; source camera has no facing angle recorded"
        return out
    half = (float(fov) / 2.0) if fov else 45.0
    diff = _angle_diff(brg, float(facing))
    out["left_through_view"] = diff <= half
    out["note"] = (f"travelled {out['compass']}, "
                   + ("consistent with leaving through the camera's field of view"
                      if diff <= half else
                      f"{round(diff)}deg outside the camera's {round(half * 2)}deg view - "
                      "the person left out of shot"))
    return out


# ---------------------------------------------------------------- legs
def build_legs(nodes: list[dict], geo: dict, evidence_providers: dict | None = None):
    """One leg per consecutive camera pair, with rejection of the impossible."""
    legs, rejects = [], []
    for a, b in zip(nodes, nodes[1:]):
        t0, t1 = _parse_ts(a.get("last_seen")), _parse_ts(b.get("first_seen"))
        dt_s = (t1 - t0).total_seconds() if (t0 and t1) else None
        ca = geo.get(a["camera_id"]) or {}
        cb = geo.get(b["camera_id"]) or {}
        has_gps = all(v is not None for v in (ca.get("lat"), ca.get("lon"),
                                              cb.get("lat"), cb.get("lon")))
        dist = round(haversine_km(ca["lat"], ca["lon"], cb["lat"], cb["lon"]), 4) if has_gps else None
        speed = round(dist / (dt_s / 3600.0), 2) if (dist is not None and dt_s and dt_s > 0) else None

        plausible, why = True, "plausible"
        overlap = dt_s is not None and dt_s < 0
        if overlap:
            # coverage circles decide this, not a magic constant
            reach = _coverage_km(ca) + _coverage_km(cb)
            if dist is not None and dist > reach:
                plausible = False
                why = (f"impossible: seen at both cameras simultaneously "
                       f"({abs(round(dt_s, 1))}s overlap) but they are {dist} km apart "
                       f"with only {round(reach, 3)} km of combined coverage")
            else:
                why = (f"overlapping camera coverage - simultaneous sighting "
                       f"({abs(round(dt_s, 1))}s overlap)")
            speed = None
        elif dist is not None and dt_s is not None:
            if dt_s == 0 and dist > _coverage_km(ca) + _coverage_km(cb):
                plausible, why = False, "same instant at two locations outside camera coverage"
            elif speed is not None and speed > MAX_KMH:
                plausible = False
                why = (f"impossible: {dist} km in {round(dt_s / 60, 1)} min "
                       f"= {speed} km/h (limit {MAX_KMH})")
        elif dist is None:
            why = "no camera GPS - distance and speed unavailable"

        travel = (({"mode": OVERLAP, "confidence": None, "basis": why,
                    "observed_vehicle": None, "speed_kmh": None}) if overlap
                  else infer_travel_mode(speed, a.get("vehicle_context"),
                                         b.get("vehicle_context")))
        leg = {
            "from_camera": a["camera_id"], "to_camera": b["camera_id"],
            "from_time": a.get("last_seen"), "to_time": b.get("first_seen"),
            "travel_seconds": round(dt_s, 1) if dt_s is not None else None,
            "distance_km": dist, "avg_speed_kmh": speed,
            "travel": travel,
            "mode": travel["mode"],                    # kept flat for existing callers
            "mode_label": MODE_LABEL.get(travel["mode"], travel["mode"]),
            "direction": direction_for(ca, cb),
            "plausible": plausible, "note": why,
            "verified": dist is not None,
            "evidence": ["reid/face identity"],
        }
        for name, fn in (evidence_providers or {}).items():
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


# ---------------------------------------------------------------- timeline
def build_timeline(nodes: list[dict], legs: list[dict]) -> list[dict]:
    """Investigator-facing timeline: a row per camera, with the mode used to reach
    the next one, and an explicit end state for the final sighting."""
    by_from = {l["from_camera"]: l for l in legs}
    rows = []
    for i, n in enumerate(nodes):
        leg = by_from.get(n["camera_id"]) if i < len(nodes) - 1 else None
        last = i == len(nodes) - 1
        rows.append({
            "index": i,
            "camera_id": n["camera_id"], "camera_name": n.get("camera_name"),
            "timestamp": n.get("first_seen") or n.get("timestamp"),
            "last_seen": n.get("last_seen"),
            "dwell_seconds": n.get("dwell_seconds"),
            "detection_id": n.get("detection_id"),
            "video_id": n.get("video_id"), "track_id": n.get("track_id"),
            "confidence": n.get("confidence") or n.get("identity_score"),
            "tier": n.get("tier"),
            "is_reference": bool(n.get("is_reference")),
            "next_mode": (leg or {}).get("mode"),
            "next_mode_label": (leg or {}).get("mode_label"),
            "next_travel_seconds": (leg or {}).get("travel_seconds"),
            "next_distance_km": (leg or {}).get("distance_km"),
            "next_plausible": (leg or {}).get("plausible"),
            # the last camera is where the trail ends in the searched footage
            "end_state": "Exited Area" if last else None,
            "end_note": ("No further sighting in the cameras searched - the person "
                         "left the covered area or was not re-identified"
                         if last else None),
        })
    return rows


# ---------------------------------------------------------------- stats/score
def score(nodes: list[dict], legs: list[dict]) -> float:
    """Confidence = mean identity strength, penalised by implausible transitions
    and by legs that could not be verified for lack of camera GPS."""
    if not nodes:
        return 0.0
    ident = sum(n.get("identity_score") or 0.0 for n in nodes) / len(nodes)
    if not legs:
        return round(ident, 4)
    ok = sum(1 for l in legs if l["plausible"]) / len(legs)
    unverified = sum(1 for l in legs if l["distance_km"] is None) / len(legs)
    return round(max(0.0, ident * (0.35 + 0.65 * ok) * (1.0 - 0.10 * unverified)), 4)


def stats(nodes: list[dict], legs: list[dict]) -> dict:
    dists = [l["distance_km"] for l in legs if l["distance_km"] is not None]
    # negative (overlapping) gaps are not travel time
    times = [l["travel_seconds"] for l in legs
             if l["travel_seconds"] is not None and l["travel_seconds"] > 0]
    total_km = round(sum(dists), 3) if dists else None
    total_s = round(sum(times), 1) if times else None
    avg = round(total_km / (total_s / 3600.0), 2) if (total_km and total_s and total_s > 0) else None
    t0 = _parse_ts(nodes[0].get("first_seen")) if nodes else None
    t1 = _parse_ts(nodes[-1].get("last_seen")) if nodes else None
    transport = dominant_mode(legs)
    return {
        "cameras": len(nodes), "cameras_visited": len(nodes), "legs": len(legs),
        "distance_km": total_km, "travel_seconds": total_s, "avg_speed_kmh": avg,
        "estimated_transport": transport["label"],
        "estimated_transport_mode": transport["mode"],
        "estimated_transport_share": transport["share"],
        "mode_sequence": [l["mode_label"] for l in legs],
        "dwell_seconds": round(sum(n.get("dwell_seconds") or 0 for n in nodes), 1),
        "span_seconds": round((t1 - t0).total_seconds(), 1) if (t0 and t1) else None,
        "first_seen": nodes[0].get("first_seen") if nodes else None,
        "last_seen": nodes[-1].get("last_seen") if nodes else None,
        "gps_available": bool(dists),
        "rejected_transitions": sum(1 for l in legs if not l["plausible"]),
        "unverified_legs": sum(1 for l in legs if l["distance_km"] is None),
    }
