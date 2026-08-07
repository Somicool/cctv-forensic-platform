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


# ------------------------------------------------------------------ CRUD
def _row(r) -> dict:
    d = dict(r)
    d["has_gps"] = _valid_latlon(d.get("lat"), d.get("lon"))
    d["facing"] = compass_name(d.get("facing_deg"))
    d["active"] = bool(d.get("active", 1))
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
    for c in out:
        c["video_count"] = vids.get(c["camera_id"], 0)
        c["detection_count"] = dets.get(c["camera_id"], 0)
    return out


def get_camera(camera_id: str) -> dict | None:
    with database.get_conn() as conn:
        r = conn.execute("SELECT * FROM cameras WHERE camera_id=?", (camera_id,)).fetchone()
    return _row(r) if r else None


def upsert_camera(data: dict) -> dict:
    """Create or update a registry camera. Only the registry fields are touched."""
    cid = (data.get("camera_id") or "").strip()
    if not cid:
        raise ValueError("camera_id is required")
    vals = {
        "name": data.get("name") or cid,
        "location": data.get("location"),
        "lat": data.get("lat"), "lon": data.get("lon"),
        "address": data.get("address"), "road_name": data.get("road_name"),
        "facing_deg": parse_facing(data.get("facing_deg") if "facing_deg" in data else data.get("facing")),
        "fov_deg": data.get("fov_deg"), "coverage_m": data.get("coverage_m"),
        "description": data.get("description"),
        "active": 1 if data.get("active", True) else 0,
    }
    for k in ("lat", "lon", "fov_deg", "coverage_m"):
        if vals[k] in ("", None):
            vals[k] = None
        else:
            try:
                vals[k] = float(vals[k])
            except (TypeError, ValueError):
                vals[k] = None
    now = database._now()
    with database.get_conn() as conn:
        exists = conn.execute("SELECT 1 FROM cameras WHERE camera_id=?", (cid,)).fetchone()
        if exists:
            sets = ", ".join(f"{k}=?" for k in vals)
            conn.execute(f"UPDATE cameras SET {sets}, source='registry', updated_at=? "
                         "WHERE camera_id=?", (*vals.values(), now, cid))
        else:
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
