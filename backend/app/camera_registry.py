"""Camera Registry - permanent siting record for every CCTV camera.

Extends the existing `cameras` table (additive columns only) with everything the
Journey Engine needs to reconstruct real routes: GPS, address, road name, facing
direction, field of view, coverage distance, description and active status.

Registry-managed cameras are marked source='registry' and are NEVER auto-removed
by the camera_config.json sync, so an operator can register a camera before any
footage exists and it persists.

Also provides best-effort GPS extraction from video metadata (ffprobe), so a clip
that carries coordinates can auto-match / auto-create its camera; CCTV exports
usually do not, in which case the UI prompts once and reuses the camera forever.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from . import config, database

_FIELDS = ("name", "location", "lat", "lon", "address", "road_name",
           "facing_deg", "fov_deg", "coverage_m", "description", "active")

_COMPASS = {"n": 0, "north": 0, "ne": 45, "northeast": 45, "e": 90, "east": 90,
            "se": 135, "southeast": 135, "s": 180, "south": 180,
            "sw": 225, "southwest": 225, "w": 270, "west": 270,
            "nw": 315, "northwest": 315}


def parse_facing(v):
    """'North' | 'NE' | 137 | '137' -> degrees 0-359, or None."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v) % 360
    s = str(v).strip().lower()
    if s in _COMPASS:
        return float(_COMPASS[s])
    try:
        return float(s) % 360
    except ValueError:
        return None


def compass_name(deg):
    if deg is None:
        return None
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[int((float(deg) % 360 + 22.5) // 45) % 8]


def _valid_latlon(lat, lon) -> bool:
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    return -90 <= lat <= 90 and -180 <= lon <= 180 and not (lat == 0 and lon == 0)


_UNICODE_MARKS = {"\u00ba": "\u00b0", "\u2032": "'", "\u2019": "'", "\u02b9": "'",
                  "\u2033": '"', "\u201d": '"', "\u02ba": '"', "\u2018": "'",
                  "\u201c": '"', "\u2212": "-", "\u2013": "-"}
# degrees[ minutes[ seconds]] with optional °, ', " markers and a hemisphere letter
_DMS_RE = re.compile(r"""^\s*(?P<h1>[NSEWnsew])?\s*(?P<sign>[-+])?\s*
    (?P<d>\d+(?:[.,]\d+)?)\s*(?:\u00b0|deg\b|d\b)?\s*
    (?:(?P<m>\d+(?:[.,]\d+)?)\s*(?:'|\u2032|min\b|m\b)\s*
       (?:(?P<s>\d+(?:[.,]\d+)?)\s*(?:"|''|\u2033|sec\b|s\b)?\s*)?
     |(?P<m2>\d+(?:[.,]\d+)?)\s+(?P<s2>\d+(?:[.,]\d+)?)\s*
    )?
    (?P<h2>[NSEWnsew])?\s*$""", re.X)


def _norm_marks(s: str) -> str:
    for a, b in _UNICODE_MARKS.items():
        s = s.replace(a, b)
    return s


def _parse_single(s: str, which: str) -> float:
    """One coordinate in decimal degrees, DMS or DM notation."""
    m = _DMS_RE.match(s)
    if not m:
        raise ValueError(
            f"{which} could not be read: {s!r}. Use decimal degrees (21.1959) or "
            f"degrees-minutes-seconds (21\u00b011'45.2\"N)")
    num = lambda x: float(str(x).replace(",", "."))          # noqa: E731
    deg = num(m.group("d"))
    minutes = m.group("m") or m.group("m2")
    seconds = m.group("s") or m.group("s2")
    val = deg + (num(minutes) / 60.0 if minutes else 0.0) \
              + (num(seconds) / 3600.0 if seconds else 0.0)
    hemi = (m.group("h1") or m.group("h2") or "").upper()
    if m.group("sign") == "-" or hemi in ("S", "W"):
        val = -val
    return val


def parse_coord(v, which: str):
    """Parse one coordinate the way a human actually writes it.

    Operators copy coordinates out of Google Maps, phones, GPS units and survey
    reports, so the field legitimately arrives in several notations:

        21.1959                 decimal degrees
        21,1959                 comma decimal separator
        22\u00b032'54.6"             degrees-minutes-seconds  <- what Maps shows
        22\u00b0 32' 54.6" N         DMS with spaces and hemisphere
        22 32 54.6              DMS with plain spaces
        21.1959, 72.8302        the whole pair pasted into one box

    DMS is the important one: it is what "copy coordinates" gives you in Google
    Maps. An earlier version stripped the \u00b0 ' " characters as punctuation, which
    silently turned 22\u00b032'54.6" into 223254.6 and rejected it as out of range.
    Degrees, minutes and seconds are now converted properly.

    Returns (value_or_None, other_or_None): `other` is set when a full pair was
    pasted into one field, so the caller can fill the partner coordinate.
    """
    if v is None or (isinstance(v, str) and not v.strip()):
        return None, None
    if isinstance(v, (int, float)):
        return float(v), None
    s = _norm_marks(str(v).strip())

    # Split a pasted PAIR. A comma is ambiguous - "21.1959, 72.8302" is two values
    # while "21,1959" is one value with a comma decimal separator - so only split
    # when both halves parse AND land in valid latitude/longitude ranges.
    if "," in s:
        head, _, tail = s.partition(",")
        if tail.strip():
            try:
                a, b = _parse_single(head.strip(), which), _parse_single(tail.strip(), which)
                if abs(a) <= 90 and abs(b) <= 180:
                    return a, b
            except ValueError:
                pass                                  # not a pair; fall through
    return _parse_single(s, which), None


# ------------------------------------------------------------------ CRUD
DEFAULT_FOV_DEG = 70.0
DEFAULT_COVERAGE_M = 60.0


def coverage_cone(cam: dict, points: int = 18) -> list | None:
    """Approximate viewing cone as a closed [[lat, lon], ...] polygon.

    Built from the stored facing direction, field of view and coverage distance so
    the map can show what each camera can actually observe. Returns None without
    coordinates - the cone is never guessed from nothing. Facing/FOV fall back to
    documented defaults, which is stated in the payload via `cone_estimated`."""
    import math
    if not _valid_latlon(cam.get("lat"), cam.get("lon")):
        return None
    lat, lon = float(cam["lat"]), float(cam["lon"])
    facing = cam.get("facing_deg")
    if facing is None:
        return None                                   # direction unknown -> no cone
    fov = float(cam.get("fov_deg") or DEFAULT_FOV_DEG)
    reach_m = float(cam.get("coverage_m") or DEFAULT_COVERAGE_M)
    half = max(1.0, min(180.0, fov / 2.0))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * max(0.05, math.cos(math.radians(lat)))
    ring = [[lat, lon]]
    for i in range(points + 1):
        brg = math.radians(float(facing) - half + (2 * half) * i / points)
        ring.append([lat + (reach_m * math.cos(brg)) / m_per_deg_lat,
                     lon + (reach_m * math.sin(brg)) / m_per_deg_lon])
    ring.append([lat, lon])
    return ring


def _row(r) -> dict:
    d = dict(r)
    d["has_gps"] = _valid_latlon(d.get("lat"), d.get("lon"))
    d["facing"] = compass_name(d.get("facing_deg"))
    d["active"] = bool(d.get("active", 1))
    # viewing cone for the map (Part 8). Absent when location or direction is unknown.
    d["coverage_cone"] = coverage_cone(d)
    d["cone_estimated"] = bool(d["coverage_cone"]) and (
        d.get("fov_deg") is None or d.get("coverage_m") is None)
    d["status"] = ("offline" if not d["active"] else
                   "located" if d["has_gps"] else "no-location")
    return d


def list_cameras(include_inactive: bool = True) -> list[dict]:
    q = "SELECT * FROM cameras"
    if not include_inactive:
        q += " WHERE COALESCE(active,1)=1"
    q += " ORDER BY camera_id"
    with database.get_conn() as conn:
        rows = conn.execute(q).fetchall()
    out = [_row(r) for r in rows]
    # attach linked counts so the UI can show what depends on each camera
    with database.get_conn() as conn:
        vids = {r["camera_id"]: r["n"] for r in conn.execute(
            "SELECT camera_id, COUNT(1) n FROM videos GROUP BY camera_id").fetchall()}
        dets = {r["camera_id"]: r["n"] for r in conn.execute(
            "SELECT camera_id, COUNT(1) n FROM detections WHERE class_label!='scene' "
            "GROUP BY camera_id").fetchall()}
        # investigations that touched each camera, via the journeys they contain
        invs: dict = {}
        try:
            for r in conn.execute("SELECT investigation, data FROM journeys").fetchall():
                try:
                    nodes = (json.loads(r["data"]).get("primary") or {}).get("nodes") or []
                except Exception:
                    continue
                for cam in {n.get("camera_id") for n in nodes if n.get("camera_id")}:
                    invs.setdefault(cam, set()).add(r["investigation"] or "unassigned")
        except Exception:
            invs = {}
    for c in out:
        c["video_count"] = vids.get(c["camera_id"], 0)
        c["detection_count"] = dets.get(c["camera_id"], 0)
        c["investigation_count"] = len(invs.get(c["camera_id"], ()))
    return out


def get_camera(camera_id: str) -> dict | None:
    with database.get_conn() as conn:
        r = conn.execute("SELECT * FROM cameras WHERE camera_id=?", (camera_id,)).fetchone()
    return _row(r) if r else None


_NUMERIC = ("lat", "lon", "fov_deg", "coverage_m")
_LABEL = {"lat": "Latitude", "lon": "Longitude", "fov_deg": "Field of view",
          "coverage_m": "Coverage distance"}


def upsert_camera(data: dict) -> dict:
    """Create or update a registry camera.

    Updates are PARTIAL: only the keys actually present in `data` are written.
    This matters because several forms submit a subset of the record - the
    camera-assign dialog on upload sends direction and coverage but no
    coordinates, and the old full-row UPDATE therefore overwrote previously saved
    latitude/longitude with NULL. That is what made a camera revert to
    "No Location" after being saved a second time.

    A key present but blank ("" or None) is an explicit clear and is honoured; a
    key that is absent is left untouched.
    """
    cid = (data.get("camera_id") or "").strip()
    if not cid:
        raise ValueError("camera_id is required")

    # accept the "facing" alias without letting it mask an absent facing_deg
    incoming = dict(data)
    if "facing" in incoming and "facing_deg" not in incoming:
        incoming["facing_deg"] = incoming.get("facing")

    # Coordinates are parsed leniently first, because a whole pair pasted into the
    # latitude box should fill both rather than fail.
    if "lat" in incoming or "lon" in incoming:
        lat_v, spill = parse_coord(incoming.get("lat"), "Latitude")
        lon_v, _ = parse_coord(incoming.get("lon"), "Longitude")
        if lon_v is None and spill is not None:
            lon_v = spill                             # "21.19, 72.83" typed into Latitude
        if "lat" in incoming:
            incoming["lat"] = lat_v
        if "lon" in incoming or spill is not None:
            incoming["lon"] = lon_v

    vals: dict = {}
    for key in _FIELDS:
        if key not in incoming:
            continue                                  # absent -> preserve stored value
        v = incoming[key]
        if key == "facing_deg":
            vals[key] = parse_facing(v)
        elif key == "active":
            vals[key] = 0 if v in (False, 0, "0", "false", "False", None) else 1
        elif key in _NUMERIC:
            if v in ("", None):
                vals[key] = None
            else:
                try:
                    vals[key] = float(str(v).strip().replace(",", "."))
                except (TypeError, ValueError):
                    raise ValueError(
                        f"{_LABEL.get(key, key)} must be a number, got {v!r}") from None
        else:
            vals[key] = v if v not in ("",) else None

    # Coordinates are the field everything downstream depends on, so a bad pair is
    # rejected loudly instead of being silently stored as NULL.
    if "lat" in vals or "lon" in vals:
        existing = get_camera(cid) or {}
        lat = vals["lat"] if "lat" in vals else existing.get("lat")
        lon = vals["lon"] if "lon" in vals else existing.get("lon")
        if (lat is None) != (lon is None):
            raise ValueError(
                "Latitude and Longitude must be filled in together - "
                f"got Latitude={'(blank)' if lat is None else lat}, "
                f"Longitude={'(blank)' if lon is None else lon}")
        if lat is not None and not _valid_latlon(lat, lon):
            why = ("latitude must be between -90 and 90" if not (-90 <= float(lat) <= 90)
                   else "longitude must be between -180 and 180"
                   if not (-180 <= float(lon) <= 180)
                   else "0, 0 is not a real camera location")
            raise ValueError(f"Coordinates rejected ({why}): {lat}, {lon}")

    now = database._now()
    with database.get_conn() as conn:
        exists = conn.execute("SELECT 1 FROM cameras WHERE camera_id=?", (cid,)).fetchone()
        if exists:
            if vals:
                sets = ", ".join(f"{k}=?" for k in vals)
                conn.execute(f"UPDATE cameras SET {sets}, source='registry', updated_at=? "
                             "WHERE camera_id=?", (*vals.values(), now, cid))
            else:
                conn.execute("UPDATE cameras SET source='registry', updated_at=? "
                             "WHERE camera_id=?", (now, cid))
        else:
            vals.setdefault("name", incoming.get("name") or cid)
            vals.setdefault("active", 1)
            cols = ", ".join(vals)
            ph = ",".join("?" * len(vals))
            conn.execute(f"INSERT INTO cameras (camera_id, {cols}, source, created_at, updated_at) "
                         f"VALUES (?, {ph}, 'registry', ?, ?)", (cid, *vals.values(), now, now))
    return get_camera(cid)


def delete_camera(camera_id: str, force: bool = False) -> dict:
    """Delete a registry camera. Refuses when footage/detections still reference it
    unless force=True (the linked media itself is never touched here)."""
    cam = get_camera(camera_id)
    if not cam:
        return {"deleted": None, "error": "camera not found"}
    with database.get_conn() as conn:
        nv = conn.execute("SELECT COUNT(1) n FROM videos WHERE camera_id=?", (camera_id,)).fetchone()["n"]
    if nv and not force:
        return {"deleted": None, "error": f"{nv} video(s) are linked to this camera",
                "video_count": nv}
    with database.get_conn() as conn:
        conn.execute("DELETE FROM cameras WHERE camera_id=?", (camera_id,))
    return {"deleted": camera_id}


# ------------------------------------------------------------------ import/export
def export_cameras() -> list[dict]:
    """Registry as a portable JSON list (same shape import accepts)."""
    out = []
    for c in list_cameras():
        out.append({k: c.get(k) for k in ("camera_id", *_FIELDS)})
    return out


def import_cameras(items: list[dict], replace: bool = False) -> dict:
    """Bulk create/update cameras from an exported list."""
    if replace:
        with database.get_conn() as conn:
            conn.execute("DELETE FROM cameras WHERE COALESCE(source,'')='registry' "
                         "AND camera_id NOT IN (SELECT DISTINCT camera_id FROM videos "
                         "WHERE camera_id IS NOT NULL)")
    ok, failed = 0, []
    for it in (items or []):
        try:
            upsert_camera(it)
            ok += 1
        except Exception as exc:                      # noqa: BLE001
            failed.append({"camera_id": it.get("camera_id"), "error": str(exc)})
    return {"imported": ok, "failed": failed, "total": len(items or [])}


# ------------------------------------------------------------------ video GPS
_ISO6709 = re.compile(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)")


def _ffprobe_tags(path) -> dict:
    """All format/stream tags of a video via ffprobe (empty dict when unavailable)."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        ffprobe = str(Path(ffmpeg).with_name(Path(ffmpeg).name.replace("ffmpeg", "ffprobe")))
        if not Path(ffprobe).exists():
            ffprobe = "ffprobe"
        out = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format",
             "-show_streams", str(path)],
            capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            return {}
        data = json.loads(out.stdout or "{}")
        tags = dict((data.get("format") or {}).get("tags") or {})
        for s in data.get("streams") or []:
            tags.update(s.get("tags") or {})
        return {str(k).lower(): v for k, v in tags.items()}
    except Exception:
        return {}


def probe_video_gps(path) -> dict:
    """Best-effort GPS from video metadata.

    Returns {"available": bool, "lat":, "lon":, "source": tag name}. Most CCTV
    exports carry no location tag - that is expected and handled by prompting the
    operator once, after which the camera is reused from the registry."""
    p = Path(path)
    if not p.exists():
        return {"available": False, "reason": "file not found"}
    tags = _ffprobe_tags(p)
    for key in ("com.apple.quicktime.location.iso6709", "location",
                "location-eng", "gps_coordinates", "gpscoordinates"):
        val = tags.get(key)
        if not val:
            continue
        m = _ISO6709.search(str(val))
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            if _valid_latlon(lat, lon):
                return {"available": True, "lat": lat, "lon": lon, "source": key}
    lat, lon = tags.get("gps_latitude"), tags.get("gps_longitude")
    if lat and lon:
        try:
            if _valid_latlon(float(lat), float(lon)):
                return {"available": True, "lat": float(lat), "lon": float(lon),
                        "source": "gps_latitude/longitude"}
        except (TypeError, ValueError):
            pass
    return {"available": False, "reason": "no GPS metadata in this video"}


def match_or_create_from_gps(lat, lon, name_hint=None, radius_m: float = 40.0) -> dict | None:
    """Find an existing registry camera within `radius_m` of the coordinates, else
    create one. Used when a video actually carries GPS metadata."""
    if not _valid_latlon(lat, lon):
        return None
    import math
    best, best_d = None, None
    for c in list_cameras():
        if not c["has_gps"]:
            continue
        p = math.pi / 180
        a = (math.sin((float(c["lat"]) - lat) * p / 2) ** 2
             + math.cos(lat * p) * math.cos(float(c["lat"]) * p)
             * math.sin((float(c["lon"]) - lon) * p / 2) ** 2)
        d_m = 2 * 6371000.0 * math.asin(math.sqrt(min(1.0, a)))
        if best_d is None or d_m < best_d:
            best, best_d = c, d_m
    if best is not None and best_d is not None and best_d <= radius_m:
        return best
    cid = (name_hint or f"CAM-{abs(int(lat * 1000))}{abs(int(lon * 1000))}")[:64]
    return upsert_camera({"camera_id": cid, "name": name_hint or cid,
                          "lat": lat, "lon": lon,
                          "description": "auto-created from video GPS metadata"})


def registry_status() -> dict:
    """Summary used by the Journey Engine + UI to decide what is possible."""
    cams = list_cameras()
    with_gps = [c for c in cams if c["has_gps"]]
    return {"cameras": len(cams), "with_gps": len(with_gps),
            "without_gps": [c["camera_id"] for c in cams if not c["has_gps"]],
            "ready_for_journey": len(with_gps) >= 2}
