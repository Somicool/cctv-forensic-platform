"""Derive a recording's real start time + camera id from the file itself.

For a forensic tool over PRE-RECORDED footage, the real wall-clock time matters
(so you can search "between 8 and 10 PM", and so cross-camera journeys line up).
We read it, in order of preference:
  1. an explicit value passed by the caller,
  2. a timestamp embedded in the filename (common CCTV/NVR export convention),
  3. the recording time stored INSIDE the container (mp4/mov `mvhd` atom) - what
     the camera itself wrote when it captured the footage,
  4. the file's last-modified time (a real timestamp, if nothing else has one).

Camera id comes from a per-camera sub-folder or a 'CAM-XX_' filename prefix.
"""
from __future__ import annotations

import re
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Matches 2026-02-14_20-00-00 / 2026-02-14 20.00.00 / 20260214_200000 /
# 2026-02-14T20:00:00 / 20260214200000, etc.
_TS_RE = re.compile(
    r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})[ _T-]?(\d{2})[-_.:]?(\d{2})[-_.:]?(\d{2})"
)
_CAM_RE = re.compile(r"^([A-Za-z]{2,}[-_]?\d+)")

# QuickTime/MP4 epoch. mvhd stores creation time as seconds since this date, UTC.
_QT_EPOCH = datetime(1904, 1, 1, tzinfo=timezone.utc)
_MVHD_SCAN_BYTES = 4_000_000        # moov sits near the start on faststart files


def container_creation_time(path) -> datetime | None:
    """Recording time written INSIDE an mp4/mov by the capturing device.

    Read straight from the `mvhd` atom, so it needs no ffprobe (the bundled
    imageio-ffmpeg ships ffmpeg but not ffprobe). Returned in LOCAL time so it is
    directly comparable with filename- and mtime-derived values. None when the
    field is absent or implausible - many NVR exports leave it zeroed.
    """
    try:
        p = Path(path)
        if p.suffix.lower() not in (".mp4", ".mov", ".m4v"):
            return None
        with open(p, "rb") as f:
            head = f.read(_MVHD_SCAN_BYTES)
        i = head.find(b"mvhd")
        if i < 0:                                    # moov may be at the end
            size = p.stat().st_size
            if size > _MVHD_SCAN_BYTES:
                with open(p, "rb") as f:
                    f.seek(max(0, size - _MVHD_SCAN_BYTES))
                    head = f.read(_MVHD_SCAN_BYTES)
                i = head.find(b"mvhd")
            if i < 0:
                return None
        version = head[i + 4]
        if version == 1:
            secs = struct.unpack(">Q", head[i + 8:i + 16])[0]
        else:
            secs = struct.unpack(">I", head[i + 8:i + 12])[0]
        if not secs:
            return None
        utc = _QT_EPOCH + timedelta(seconds=int(secs))
        # sanity: reject obviously bogus clocks (some encoders write garbage)
        now = datetime.now(timezone.utc)
        if not (datetime(1990, 1, 1, tzinfo=timezone.utc) <= utc <= now + timedelta(days=1)):
            return None
        return utc.astimezone()                        # aware, local clock
    except Exception:
        return None


def _local(value: datetime) -> datetime:
    """Attach the local timezone to a naive datetime, leaving aware ones alone.

    Every start time is returned timezone-AWARE on the local clock. Aware means
    arithmetic against other stored timestamps never raises; local means the ISO
    string the UI slices ("2026-08-06T18:57:39+05:30") reads as the wall-clock time
    the footage was actually recorded, not a UTC value 5.5 hours off.
    """
    return value.astimezone() if value.tzinfo is None else value


def parse_start_time(path, explicit=None) -> datetime:
    if explicit is not None:
        if isinstance(explicit, datetime):
            return _local(explicit)
        try:
            return _local(datetime.fromisoformat(str(explicit)))
        except ValueError:
            pass

    name = Path(path).name
    m = _TS_RE.search(name)
    if m:
        try:
            return _local(datetime(*(int(x) for x in m.groups())))
        except ValueError:
            pass  # matched digits weren't a valid date -> fall through

    # what the camera recorded, straight out of the container
    embedded = container_creation_time(path)
    if embedded is not None:
        return _local(embedded)

    try:
        return _local(datetime.fromtimestamp(Path(path).stat().st_mtime))
    except OSError:
        return datetime.now(timezone.utc).astimezone()


def start_time_source(path, explicit=None) -> str:
    """Which of the four sources supplied the start time (for logging/reporting)."""
    if explicit is not None:
        return "explicit"
    if _TS_RE.search(Path(path).name):
        return "filename"
    if container_creation_time(path) is not None:
        return "container-metadata"
    return "file-mtime"


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
