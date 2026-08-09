"""Track Identity Descriptors - forensic track-level person re-identification.

Why this exists
---------------
Comparing ONE crop against ONE crop fails the moment posture or context changes
(riding a motorcycle vs walking, backpack on vs off, helmet on vs off). A single
OSNet embedding of a rider is far from the same person's embedding while walking,
so a single-crop comparison rejects the true match.

This module instead builds ONE descriptor per completed ByteTrack track from
MANY representative frames spread across the whole track (different distances,
poses, viewing angles and lighting), and matches descriptor-to-descriptor using
SET-to-SET similarity (best-matching view pair, not average-to-average). That is
what makes identity survive a change of posture or vehicle.

The vehicle is never part of the identity: descriptors are built only from PERSON
detections. A co-travelling vehicle is recorded separately as context for travel
-mode reasoning, and never contributes to the identity score.

Descriptors are stored permanently in the `track_identity` table.
"""
from __future__ import annotations

import base64
import json

import numpy as np

from . import config, database, identity_fusion
from .search import vector_store

# how many representative views to keep per track
MAX_VIEWS = 10

# ---------------------------------------------------------------- operating point
# Identity weighting now lives in the Identity Fusion Engine, which treats ReID as
# ONE component of eight appearance sources plus five context sources. Exposed here
# for the API/UI, which report the weight of each evidence source.
WEIGHTS = {k: v["weight"] for k, v in identity_fusion.APPEARANCE_SIGNALS.items()}

# Fused-identity thresholds, measured not guessed. Scores now come from the
# Identity Fusion Engine, whose output is scale-aligned so this threshold keeps the
# exact meaning it was calibrated with (identity_fusion._align):
#     0.78 -> recall  94.9%, FMR   6.0%, precision 71.2%
#     0.86 -> recall  87.2%, FMR   2.0%, precision 85.0%, accuracy 96.8%  <- accept
#     0.92 -> recall  59.0%, FMR   1.0%, precision 88.5%
# A fabricated journey is more damaging than a missed camera, so IDENTITY_ACCEPT
# is set where the false-match rate falls to ~2%. Candidates between PROBABLE and
# ACCEPT are still shown to the investigator, just not asserted as the journey.
# The threshold is UNCHANGED from the ReID-dominated version; the recall gain
# (79.5% -> 87.2% at the same 2% false-match rate) comes from corroborating
# evidence, not from relaxing the bar.
IDENTITY_ACCEPT = 0.86         # asserted as the same person (journey node)
IDENTITY_PROBABLE = 0.78       # shown as a probable candidate, not asserted
IDENTITY_POSSIBLE = 0.68       # shown as a weak lead

ACCESSORY_KEYS = ("backpack", "helmet", "cap", "handbag", "sunglasses", "face mask")
# split into independent families so the fusion engine can weigh "carried a bag"
# separately from "wore a helmet" - a rider removing a helmet must not look like
# the same evidence as dropping a backpack
BAG_KEYS = ("backpack", "handbag")
HEADWEAR_KEYS = ("helmet", "cap")
# Colour labels a CCTV colour classifier legitimately confuses between cameras
# (exposure, white balance, shadow). Treated as a partial match so a lighting
# change cannot by itself reject a true identity.
COLOUR_NEIGHBOURS = (
    {"black", "grey", "gray", "dark", "navy"},
    {"white", "grey", "gray", "silver", "beige", "cream"},
    {"blue", "navy", "purple"},
    {"red", "maroon", "orange", "pink"},
    {"green", "olive", "teal"},
    {"brown", "beige", "orange", "maroon", "tan"},
    {"yellow", "orange", "beige", "gold"},
)
NEAR_COLOUR_SCORE = 0.6
# A voted colour label is only evidence if the vote was reasonably consistent.
# Audited on the four-camera footage: voted confidences of 0.375-0.5 are common
# (the label was the majority in under half the frames), and one track flipped
# orange -> white mid-track. Feeding a coin-flip label in as hard 0.0/1.0 evidence
# actively drove correct candidates DOWN, which is the light-clothing failure. Below
# this confidence the colour is reported as UNAVAILABLE and the fusion engine
# renormalises over the signals it can actually trust.
COLOUR_MIN_CONF = 0.45


def tier(identity: float) -> str:
    """Evidence tier for an identity score - what the investigator may rely on."""
    if identity >= IDENTITY_ACCEPT:
        return "confirmed"
    if identity >= IDENTITY_PROBABLE:
        return "probable"
    if identity >= IDENTITY_POSSIBLE:
        return "possible"
    return "weak"
_PERSON_LABELS = {config.DETECT_CLASSES[c] for c in config.PERSON_CLASSES}
_VEHICLE_LABELS = {config.DETECT_CLASSES[c] for c in config.VEHICLE_CLASSES}


# ---------------------------------------------------------------- utils
def _b64(v) -> str:
    return base64.b64encode(np.asarray(v, dtype="float32").tobytes()).decode("ascii")


def _unb64(s, dim) -> np.ndarray | None:
    try:
        a = np.frombuffer(base64.b64decode(s), dtype="float32")
        return a.reshape(-1, dim) if a.size and a.size % dim == 0 else None
    except Exception:
        return None


def _norm_rows(a):
    if a is None or not len(a):
        return None
    a = np.asarray(a, dtype="float32")
    if a.ndim == 1:
        a = a[None, :]
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n < 1e-6] = 1.0
    return a / n


def set_similarity(A, B, top: int = 3) -> float | None:
    """Set-to-set similarity between two groups of embeddings.

    Uses the BEST matching view pair, softened by the mean of the top-k pairs.
    Robust to posture/context change: if the person was seen from a comparable
    angle in both cameras, that pair carries the match even when other views
    (e.g. seated on a motorcycle) do not."""
    A, B = _norm_rows(A), _norm_rows(B)
    if A is None or B is None:
        return None
    M = A @ B.T                                        # cosine matrix
    flat = np.sort(M.ravel())[::-1]
    best = float(flat[0])
    k = min(top, flat.size)
    topmean = float(flat[:k].mean())
    return max(0.0, min(1.0, 0.65 * best + 0.35 * topmean))


# ---------------------------------------------------------------- build
def _pick_views(dets: list[dict], k: int = MAX_VIEWS) -> list[dict]:
    """Representative frames spanning the WHOLE track: sampled across time and
    across bbox size, so front/side/rear, near/far and varied lighting are all
    covered rather than one moment."""
    if len(dets) <= k:
        return dets
    by_time = sorted(dets, key=lambda d: (d.get("frame_number") or 0))
    picks, seen = [], set()
    # half the budget: even sweep over time (captures changing viewing angle)
    for i in np.linspace(0, len(by_time) - 1, max(2, k // 2)).astype(int):
        d = by_time[int(i)]
        if d["detection_id"] not in seen:
            seen.add(d["detection_id"]); picks.append(d)
    # other half: size extremes + middle (captures distance / resolution variety)
    by_size = sorted(dets, key=lambda d: (d.get("bbox_w") or 0) * (d.get("bbox_h") or 0))
    for i in np.linspace(0, len(by_size) - 1, max(2, k - len(picks))).astype(int):
        d = by_size[int(i)]
        if d["detection_id"] not in seen:
            seen.add(d["detection_id"]); picks.append(d)
    return picks[:k]


def _vehicle_context(video_id: int, people: list[dict], sample: int = 12) -> list[str]:
    """Vehicle the person appears to be ON, found by SPATIAL OVERLAP in the same
    frames (a motorcycle carries its own ByteTrack id, so it can't be found by
    track id). This is CONTEXT ONLY - used for travel-mode reasoning and never
    mixed into the identity score, so a vehicle can never become the identity."""
    if not people:
        return []
    picks = people if len(people) <= sample else [
        people[int(i)] for i in np.linspace(0, len(people) - 1, sample).astype(int)]
    frames = sorted({p.get("frame_number") for p in picks if p.get("frame_number") is not None})
    if not frames:
        return []
    ph = ",".join("?" * len(frames))
    with database.get_conn() as conn:
        vrows = [database._row_to_detection(r) for r in conn.execute(
            f"SELECT * FROM detections WHERE video_id=? AND frame_number IN ({ph}) "
            "AND class_label != 'scene'", (video_id, *frames)).fetchall()]
    vehicles = [v for v in vrows if v.get("class_label") in _VEHICLE_LABELS]
    if not vehicles:
        return []
    by_frame: dict = {}
    for v in vehicles:
        by_frame.setdefault(v.get("frame_number"), []).append(v)

    hits: dict = {}
    checked = 0
    for p in picks:
        f = p.get("frame_number")
        if f not in by_frame or p.get("bbox_x") is None:
            continue
        checked += 1
        px, py, pw, phh = p["bbox_x"], p["bbox_y"], p["bbox_w"], p["bbox_h"]
        p_area = max(1.0, pw * phh)
        # a rider's lower body sits inside the vehicle box -> test the mid/lower centre
        cx, cy = px + pw / 2.0, py + phh * 0.75
        best_lbl, best_ov = None, 0.0
        for v in by_frame[f]:
            if v.get("bbox_x") is None:
                continue
            vx, vy, vw, vh = v["bbox_x"], v["bbox_y"], v["bbox_w"], v["bbox_h"]
            if vw * vh < 0.45 * p_area:               # too small to be ridden
                continue
            pad_x, pad_y = vw * 0.20, vh * 0.30
            if not ((vx - pad_x <= cx <= vx + vw + pad_x)
                    and (vy - pad_y <= cy <= vy + vh + pad_y)):
                continue
            # horizontal overlap of the person with the vehicle (rider is astride it)
            ov = max(0.0, min(px + pw, vx + vw) - max(px, vx)) / max(1.0, pw)
            if ov > best_ov:
                best_ov, best_lbl = ov, v["class_label"]
        if best_lbl and best_ov >= 0.5:
            hits[best_lbl] = hits.get(best_lbl, 0) + 1
    if not checked or not hits:
        return []
    # keep only the single best-supported vehicle, seen in >=40% of sampled frames
    lbl = max(hits, key=hits.get)
    return [lbl] if hits[lbl] >= max(2, 0.40 * checked) else []


def _vote(values):
    vals = [v for v in values if v]
    if not vals:
        return None, 0.0
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts, key=counts.get)
    return best, round(counts[best] / len(vals), 3)


def build_descriptor(video_id: int, track_id: int, persist: bool = True) -> dict | None:
    """Build (and store) the Track Identity Descriptor for one person track."""
    with database.get_conn() as conn:
        rows = [database._row_to_detection(r) for r in conn.execute(
            "SELECT * FROM detections WHERE video_id=? AND track_id=? AND class_label!='scene'",
            (video_id, track_id)).fetchall()]
    people = [d for d in rows if d.get("class_label") in _PERSON_LABELS]
    if not people:
        return None

    views = _pick_views(people)
    reid_vecs, clip_vecs = [], []
    for d in views:
        v = vector_store.get_vector("reid", d["detection_id"])
        if v is not None:
            reid_vecs.append(v)
        v = vector_store.get_vector("clip", d["detection_id"])
        if v is not None:
            clip_vecs.append(v)

    # face embeddings for this track (highest-priority signal when present)
    face_vecs = []
    with database.get_conn() as conn:
        frows = conn.execute(
            "SELECT f.face_id FROM faces f JOIN detections d ON d.detection_id=f.detection_id "
            "WHERE d.video_id=? AND d.track_id=?", (video_id, track_id)).fetchall()
    for fr in frows:
        v = vector_store.get_vector("face", fr["face_id"])
        if v is not None:
            face_vecs.append(v)

    # appearance attributes voted across the whole track (stable than one frame)
    upper, up_conf = _vote([(d.get("attributes") or {}).get("upper_color") for d in people])
    lower, lo_conf = _vote([(d.get("attributes") or {}).get("lower_color") for d in people])
    acc_counts = {}
    for d in people:
        for a in ((d.get("attributes") or {}).get("accessories") or []):
            a = str(a).lower()
            acc_counts[a] = acc_counts.get(a, 0) + 1
    # an accessory seen in >=25% of frames is considered carried on this track
    accessories = sorted([a for a, n in acc_counts.items() if n >= max(1, 0.25 * len(people))])

    # body proportions (height/width) - survives clothing change
    ratios = [(d["bbox_h"] / d["bbox_w"]) for d in people
              if d.get("bbox_w") and d.get("bbox_h") and d["bbox_w"] > 0]
    body_ratio = float(np.median(ratios)) if ratios else None

    times = sorted(t for t in (d.get("timestamp") for d in people) if t)
    frames = [d.get("frame_number") or 0 for d in people]
    # co-travelling vehicle = CONTEXT ONLY (never part of identity)
    veh = _vehicle_context(video_id, people)

    desc = {
        "video_id": video_id, "track_id": track_id,
        "camera_id": people[0].get("camera_id"),
        "n_detections": len(people), "n_views": len(views),
        "first_seen": times[0] if times else None,
        "last_seen": times[-1] if times else None,
        "duration_s": None,
        "start_frame": min(frames) if frames else None,
        "end_frame": max(frames) if frames else None,
        "upper_color": upper, "upper_conf": up_conf,
        "lower_color": lower, "lower_conf": lo_conf,
        "accessories": accessories,
        "body_ratio": round(body_ratio, 4) if body_ratio else None,
        "avg_box": [round(float(np.mean([d["bbox_w"] for d in people])), 1),
                    round(float(np.mean([d["bbox_h"] for d in people])), 1)],
        "vehicle_context": veh,
        "has_face": bool(face_vecs), "n_faces": len(face_vecs),
        "rep_detection_id": max(people, key=lambda d: (d.get("bbox_w") or 0) * (d.get("bbox_h") or 0))["detection_id"],
    }
    if desc["first_seen"] and desc["last_seen"]:
        from datetime import datetime
        try:
            a = datetime.fromisoformat(desc["first_seen"]).replace(tzinfo=None)
            b = datetime.fromisoformat(desc["last_seen"]).replace(tzinfo=None)
            desc["duration_s"] = round((b - a).total_seconds(), 1)
        except Exception:
            pass

    if persist:
        with database.get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO track_identity (video_id, track_id, camera_id, "
                " n_detections, n_views, first_seen, last_seen, meta, reid_vecs, clip_vecs, "
                " face_vecs, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (video_id, track_id, desc["camera_id"], desc["n_detections"], desc["n_views"],
                 desc["first_seen"], desc["last_seen"], json.dumps(desc),
                 _b64(np.vstack(reid_vecs)) if reid_vecs else None,
                 _b64(np.vstack(clip_vecs)) if clip_vecs else None,
                 _b64(np.vstack(face_vecs)) if face_vecs else None,
                 database._now()))
    desc["_reid"] = np.vstack(reid_vecs) if reid_vecs else None
    desc["_clip"] = np.vstack(clip_vecs) if clip_vecs else None
    desc["_face"] = np.vstack(face_vecs) if face_vecs else None
    return desc


def load_descriptor(video_id: int, track_id: int, build_if_missing: bool = True) -> dict | None:
    with database.get_conn() as conn:
        row = conn.execute("SELECT * FROM track_identity WHERE video_id=? AND track_id=?",
                           (video_id, track_id)).fetchone()
    if row is None:
        return build_descriptor(video_id, track_id) if build_if_missing else None
    try:
        d = json.loads(row["meta"])
    except Exception:
        return build_descriptor(video_id, track_id) if build_if_missing else None
    d["_reid"] = _unb64(row["reid_vecs"], config.REID_DIM) if row["reid_vecs"] else None
    d["_clip"] = _unb64(row["clip_vecs"], config.CLIP_DIM) if row["clip_vecs"] else None
    d["_face"] = _unb64(row["face_vecs"], config.FACE_DIM) if row["face_vecs"] else None
    return d


def build_all(camera_ids=None, min_dets: int = 2, rebuild: bool = False) -> dict:
    """Build descriptors for every completed person track (idempotent)."""
    q = ("SELECT video_id, track_id, COUNT(1) n FROM detections WHERE class_label='person' "
         "AND track_id IS NOT NULL")
    params = []
    if camera_ids:
        q += f" AND camera_id IN ({','.join('?' * len(camera_ids))})"
        params += list(camera_ids)
    q += " GROUP BY video_id, track_id HAVING n >= ?"
    params.append(min_dets)
    with database.get_conn() as conn:
        rows = [dict(r) for r in conn.execute(q, params).fetchall()]
        have = set()
        if not rebuild:
            have = {(r["video_id"], r["track_id"]) for r in
                    conn.execute("SELECT video_id, track_id FROM track_identity").fetchall()}
    built = 0
    for r in rows:
        if not rebuild and (r["video_id"], r["track_id"]) in have:
            continue
        if build_descriptor(r["video_id"], r["track_id"]):
            built += 1
    return {"tracks": len(rows), "built": built, "existing": len(have)}


def list_descriptors(camera_ids=None) -> list[dict]:
    q = "SELECT * FROM track_identity"
    params = []
    if camera_ids:
        q += f" WHERE camera_id IN ({','.join('?' * len(camera_ids))})"
        params += list(camera_ids)
    with database.get_conn() as conn:
        rows = conn.execute(q, params).fetchall()
    out = []
    for row in rows:
        try:
            d = json.loads(row["meta"])
        except Exception:
            continue
        d["_reid"] = _unb64(row["reid_vecs"], config.REID_DIM) if row["reid_vecs"] else None
        d["_clip"] = _unb64(row["clip_vecs"], config.CLIP_DIM) if row["clip_vecs"] else None
        d["_face"] = _unb64(row["face_vecs"], config.FACE_DIM) if row["face_vecs"] else None
        out.append(d)
    return out


# ---------------------------------------------------------------- compare
def _colour_pair(x: str, y: str) -> float:
    if x == y:
        return 1.0
    for group in COLOUR_NEIGHBOURS:
        if x in group and y in group:
            return NEAR_COLOUR_SCORE          # same garment under different lighting
    return 0.0


_CONF_KEY = {"upper_color": "upper_conf", "lower_color": "lower_conf"}


def _garment_sim(a, b, key: str) -> float | None:
    """Graded clothing-colour agreement for ONE garment (upper or lower).

    Not a strict equality test: black/grey and white/beige are routinely swapped
    between cameras with different exposure, so near colours count as a partial
    match instead of a hard mismatch.

    Returns None - meaning "no evidence", not "mismatch" - when either side's voted
    colour was not consistent enough to rely on (see COLOUR_MIN_CONF). That is the
    important part: an unreliable label must not be able to veto a correct match."""
    x, y = a.get(key), b.get(key)
    if not x or not y:
        return None
    ck = _CONF_KEY.get(key)
    if ck:
        ca, cb = a.get(ck), b.get(ck)
        if (ca is not None and ca < COLOUR_MIN_CONF) or (cb is not None and cb < COLOUR_MIN_CONF):
            return None
    return _colour_pair(str(x).lower(), str(y).lower())


def _colour_sim(a, b) -> float | None:
    """Combined upper+lower colour agreement (legacy single-signal view)."""
    vals = [v for v in (_garment_sim(a, b, "upper_color"),
                        _garment_sim(a, b, "lower_color")) if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _accessory_group_sim(a, b, keys) -> float | None:
    """Overlap coefficient over ONE accessory family, NOT Jaccard: a person may
    remove a backpack or take off a helmet between cameras, so a missing accessory
    must not be punished - shared accessories only add evidence."""
    A = {x for x in (a.get("accessories") or []) if x in keys}
    B = {x for x in (b.get("accessories") or []) if x in keys}
    if not A and not B:
        return None
    if not A or not B:
        return 0.5                                  # unknown, neutral (no penalty)
    return len(A & B) / min(len(A), len(B))


def _bag_sim(a, b) -> float | None:
    return _accessory_group_sim(a, b, BAG_KEYS)


def _headwear_sim(a, b) -> float | None:
    return _accessory_group_sim(a, b, HEADWEAR_KEYS)


def _accessory_sim(a, b) -> float | None:
    """Combined accessory agreement (legacy single-signal view)."""
    return _accessory_group_sim(a, b, ACCESSORY_KEYS)


def _body_sim(a, b) -> float | None:
    ra, rb = a.get("body_ratio"), b.get("body_ratio")
    if not ra or not rb:
        return None
    return max(0.0, 1.0 - abs(ra - rb) / 1.2)


def appearance_evidence(a: dict, b: dict) -> dict:
    """Every appearance/biometric similarity between two Track Identity Descriptors.

    One entry per evidence source the fusion engine knows about. Missing sources
    are simply absent - the engine renormalises rather than assuming zero."""
    ev = {}
    for name, key in (("face", "_face"), ("reid", "_reid"), ("clothing", "_clip")):
        s = set_similarity(a.get(key), b.get(key))
        if s is not None:
            ev[name] = s
    for name, fn in (("upper_color", lambda x, y: _garment_sim(x, y, "upper_color")),
                     ("lower_color", lambda x, y: _garment_sim(x, y, "lower_color")),
                     ("bag", _bag_sim), ("headwear", _headwear_sim), ("body", _body_sim)):
        s = fn(a, b)
        if s is not None:
            ev[name] = s

    # De-duplicate the clothing colours. Audited on real tracks, the attribute
    # extractor returns the SAME dominant colour for upper and lower on almost every
    # person (white/white, orange/orange, green/green...), so it is one measurement
    # reported twice. Counting it twice gave colour 0.18 of the appearance weight
    # while supplying a single noisy feature, and let one bad label outvote ReID.
    # When both descriptors show upper == lower, keep it as ONE signal.
    if "upper_color" in ev and "lower_color" in ev:
        same_a = a.get("upper_color") and a.get("upper_color") == a.get("lower_color")
        same_b = b.get("upper_color") and b.get("upper_color") == b.get("lower_color")
        if same_a and same_b:
            ev.pop("lower_color")
    return ev


def compare(a: dict, b: dict, context: dict | None = None) -> dict:
    """Full identity comparison between two Track Identity Descriptors.

    Delegates the decision to the Identity Fusion Engine: ReID is only one
    component, and several independent sources agreeing can confirm identity even
    when the raw ReID score is moderate. `context` carries the spatio-temporal
    evidence (timeline, GPS proximity, travel time, camera direction, journey
    continuity), which corroborates but never creates identity."""
    ev = appearance_evidence(a, b)
    if not ev:
        return {"identity": 0.0, "tier": tier(0.0), "signals": {}, "fusion": None}

    fused = identity_fusion.fuse(ev, context)
    score = fused["score"]
    # a very strong face match confirms identity regardless of posture/context
    if ev.get("face", 0.0) >= 0.55:
        score = max(score, ev["face"])
    score = float(score)

    # `signals` keeps the legacy signal names so existing callers/UI keep working
    legacy = dict(ev)
    colour = _colour_sim(a, b)
    if colour is not None:
        legacy["colour"] = colour
    acc = _accessory_sim(a, b)
    if acc is not None:
        legacy["accessories"] = acc
    return {"identity": round(score, 4), "tier": tier(score),
            "signals": {k: round(float(v), 4) for k, v in legacy.items()},
            "fusion": fused}
