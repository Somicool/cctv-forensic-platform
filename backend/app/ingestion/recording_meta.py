"""Derive a recording's real start time + camera id from the file itself.

For a forensic tool over PRE-RECORDED footage, the real wall-clock time matters
(so you can search "between 8 and 10 PM"). We read it, in order of preference:
  1. an explicit value passed by the caller,
  2. a timestamp embedded in the filename (common CCTV/NVR export convention),
  3. the file's last-modified time (a real timestamp, if the name has none).

Camera id comes from a per-camera sub-folder or a 'CAM-XX_' filename prefix.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

# Matches 2026-02-14_20-00-00 / 2026-02-14 20.00.00 / 20260214_200000 /
# 2026-02-14T20:00:00 / 20260214200000, etc.
_TS_RE = re.compile(
    r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})[ _T-]?(\d{2})[-_.:]?(\d{2})[-_.:]?(\d{2})"
)
_CAM_RE = re.compile(r"^([A-Za-z]{2,}[-_]?\d+)")


def parse_start_time(path, explicit=None) -> datetime:
    if explicit is not None:
        if isinstance(explicit, datetime):
            return explicit
        try:
            return datetime.fromisoformat(str(explicit))
        except ValueError:
            pass

    name = Path(path).name
    m = _TS_RE.search(name)
    if m:
        try:
            return datetime(*(int(x) for x in m.groups()))
        except ValueError:
            pass  # matched digits weren't a valid date -> fall through

    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime)
    except OSError:
        return datetime.now(timezone.utc)


def parse_camera_id(path, default: str = "CAM-01") -> str:
    p = Path(path)
    # per-camera folder layout: recordings/<camera>/<file>
    parent = p.parent.name
    if parent and parent.lower() not in ("videos", "recordings", "data", ""):
        return parent
    # 'CAM-01_...' / 'CAM01_...' prefix
    m = _CAM_RE.match(p.stem)
    if m:
        return m.group(1).upper().replace("_", "-")
    # No camera prefix: treat the file as its own recording so distinct uploads
    # don't all collapse into one camera.
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", p.stem).strip("-")
    return slug[:40] or default
