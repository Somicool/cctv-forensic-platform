"""Re-base stored timestamps onto each clip's REAL recording time.

Why this exists
---------------
`ingest_video` used to leave `videos.start_time` NULL, and the tracker's fallback
for a missing start time is `datetime.now()`. So every detection, track, face,
plate and saved-face row was stamped with the moment the clip was INGESTED rather
than when it was recorded. Two consequences:

  * evidence and Face Gallery entries showed the wrong date/time entirely;
  * clips ingested on different days sat hours apart on the journey timeline even
    when they were recorded minutes apart, which makes all cross-camera temporal
    evidence meaningless.

The ingest path is fixed (see app/ingestion/pipeline.py), but footage already in
the database needs re-basing. This shifts each clip's timestamps by a single
constant offset, so every RELATIVE time inside a clip - frame spacing, track
durations, dwell times - is preserved exactly. Only the absolute clock moves.

Usage (from the backend/ directory):
    python scripts/backfill_recording_times.py            # dry run, changes nothing
    python scripts/backfill_recording_times.py --apply    # write the changes
    python scripts/backfill_recording_times.py --apply --video-id 112
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, database                                    # noqa: E402
from app.ingestion import recording_meta                            # noqa: E402

# timestamp columns that are derived from the recording clock
TS_COLUMNS = [
    ("detections", "timestamp", "video_id"),
    ("tracks", "start_time", "video_id"),
    ("tracks", "end_time", "video_id"),
    ("faces", "timestamp", None),          # joined via detections
    ("plates", "timestamp", None),
    ("saved_faces", "timestamp", None),
    ("activity_history", "timestamp", None),
]


def _parse(ts):
    try:
        return datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None


def _shift(ts, delta):
    """Shift one stored timestamp, re-emitting it on the LOCAL clock so the UI
    (which slices the raw ISO string) shows real local recording time."""
    d = _parse(ts)
    if d is None:
        return None
    return (d + delta).astimezone().isoformat()


def plan(video_id=None) -> list[dict]:
    out = []
    with database.get_conn() as conn:
        vids = conn.execute(
            "SELECT video_id, filename, start_time, duration FROM videos"
            + (" WHERE video_id=?" if video_id else ""),
            (video_id,) if video_id else ()).fetchall()
        for v in vids:
            path = config.VIDEO_DIR / (v["filename"] or "")
            if not path.exists():
                out.append({"video_id": v["video_id"], "filename": v["filename"],
                            "skip": "source file missing"})
                continue
            base_row = conn.execute(
                "SELECT MIN(timestamp) mn, MAX(timestamp) mx, COUNT(*) n "
                "FROM detections WHERE video_id=?", (v["video_id"],)).fetchone()
            old_base = _parse(base_row["mn"])
            real = recording_meta.parse_start_time(path)
            source = recording_meta.start_time_source(path)
            if old_base is None:
                out.append({"video_id": v["video_id"], "filename": v["filename"],
                            "real_start": real, "source": source, "delta": None,
                            "n_detections": 0, "skip": "no detections to re-base"})
                continue
            out.append({
                "video_id": v["video_id"], "filename": v["filename"],
                "old_base": old_base, "old_max": _parse(base_row["mx"]),
                "real_start": real, "source": source,
                "delta": real - old_base, "n_detections": base_row["n"],
                "duration": v["duration"], "db_start_time": v["start_time"],
            })
    return out


def apply(items) -> dict:
    counts = {}
    with database.get_conn() as conn:
        for it in items:
            if it.get("skip") or it.get("delta") is None:
                continue
            vid, delta = it["video_id"], it["delta"]
            det_ids = [r["detection_id"] for r in conn.execute(
                "SELECT detection_id FROM detections WHERE video_id=?", (vid,))]
            ph = ",".join("?" * len(det_ids)) if det_ids else None

            # NOTE: rowid is aliased to _rid. Selecting bare `rowid` from a table
            # whose primary key is an INTEGER PRIMARY KEY makes SQLite report the
            # column under the DECLARED name (detection_id, saved_id, ...), so
            # row["rowid"] raises. The alias keeps this table-agnostic.
            for table, col, direct in TS_COLUMNS:
                if direct:
                    rows = conn.execute(
                        f"SELECT rowid AS _rid, {col} FROM {table} "
                        f"WHERE video_id=? AND {col} IS NOT NULL", (vid,)).fetchall()
                elif table == "activity_history":
                    # history stores the detection id in `ref`
                    if not det_ids:
                        continue
                    refs = [str(d) for d in det_ids]
                    rph = ",".join("?" * len(refs))
                    rows = conn.execute(
                        f"SELECT rowid AS _rid, {col} FROM activity_history "
                        f"WHERE ref IN ({rph}) AND {col} IS NOT NULL", refs).fetchall()
                else:
                    if not det_ids:
                        continue
                    rows = conn.execute(
                        f"SELECT rowid AS _rid, {col} FROM {table} "
                        f"WHERE detection_id IN ({ph}) AND {col} IS NOT NULL", det_ids).fetchall()
                n = 0
                for r in rows:
                    new = _shift(r[col], delta)
                    if new and new != r[col]:
                        conn.execute(f"UPDATE {table} SET {col}=? WHERE rowid=?", (new, r["_rid"]))
                        n += 1
                counts[f"{table}.{col}"] = counts.get(f"{table}.{col}", 0) + n

            # the clip's own timeline
            real = it["real_start"]
            end = None
            if it.get("duration"):
                from datetime import timedelta
                end = (real + timedelta(seconds=float(it["duration"]))).isoformat()
            conn.execute("UPDATE videos SET start_time=?, end_time=? WHERE video_id=?",
                         (real.isoformat(), end, vid))
            counts["videos.start_time"] = counts.get("videos.start_time", 0) + 1

            # evidence snapshots keep their own copy of the timestamp
            for r in conn.execute("SELECT rowid AS _rid, snapshot FROM case_evidence").fetchall():
                try:
                    snap = json.loads(r["snapshot"] or "{}")
                except Exception:
                    continue
                if snap.get("video_id") != vid or not snap.get("timestamp"):
                    continue
                snap["timestamp"] = _shift(snap["timestamp"], delta) or snap["timestamp"]
                conn.execute("UPDATE case_evidence SET snapshot=? WHERE rowid=?",
                             (json.dumps(snap), r["_rid"]))
                counts["case_evidence.snapshot"] = counts.get("case_evidence.snapshot", 0) + 1
    return counts


def refresh_snapshot_windows() -> int:
    """Recompute `visible_from` / `visible_until` on stored evidence snapshots.

    These describe when a track was on screen. They are RECOMPUTED from the
    detections rather than shifted, so this step is idempotent and safe to re-run:
    a snapshot saved before the re-base would otherwise keep a track window on the
    old clock while its own timestamp is on the new one.
    """
    n = 0
    with database.get_conn() as conn:
        rows = conn.execute("SELECT rowid AS _rid, snapshot FROM case_evidence").fetchall()
        for r in rows:
            try:
                snap = json.loads(r["snapshot"] or "{}")
            except Exception:
                continue
            vid, tid = snap.get("video_id"), snap.get("track_id")
            if vid is None or tid is None:
                continue
            if "visible_from" not in snap and "visible_until" not in snap:
                continue
            w = conn.execute(
                "SELECT MIN(timestamp) mn, MAX(timestamp) mx FROM detections "
                "WHERE video_id=? AND track_id=? AND class_label!='scene'", (vid, tid)).fetchone()
            if not w or not w["mn"]:
                continue
            if snap.get("visible_from") == w["mn"] and snap.get("visible_until") == w["mx"]:
                continue
            snap["visible_from"], snap["visible_until"] = w["mn"], w["mx"]
            conn.execute("UPDATE case_evidence SET snapshot=? WHERE rowid=?",
                         (json.dumps(snap), r["_rid"]))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    ap.add_argument("--video-id", type=int, default=None)
    args = ap.parse_args()

    items = plan(args.video_id)
    print("=" * 78)
    print("RE-BASE PLAN" if not args.apply else "RE-BASING")
    print("=" * 78)
    for it in items:
        print(f"\nvideo {it['video_id']:>3}  {(it['filename'] or '')[:52]}")
        if it.get("skip"):
            print(f"    SKIP: {it['skip']}")
            continue
        print(f"    stored base (ingest time) : {it['old_base'].isoformat()}")
        print(f"    real recording start      : {it['real_start'].isoformat()}   "
              f"[{it['source']}]")
        secs = it["delta"].total_seconds()
        print(f"    shift                     : {secs / 3600:+.2f} h  "
              f"({it['n_detections']} detections)")

    good = [i for i in items if not i.get("skip") and i.get("delta") is not None]
    if good:
        starts = sorted(i["real_start"] for i in good)
        print(f"\nreal recording window: {starts[0].isoformat()} .. {starts[-1].isoformat()}")
        print(f"spread across clips  : "
              f"{(starts[-1] - starts[0]).total_seconds() / 3600:.2f} h")

    if not args.apply:
        print("\nDry run - nothing written. Re-run with --apply.")
        return

    backup = config.DATA_DIR / f"backup_ts_{datetime.now():%Y%m%d-%H%M%S}"
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.DB_PATH, backup / config.DB_PATH.name)
    print(f"\nbackup: {backup}")

    counts = apply(items)
    counts["case_evidence.track_window"] = refresh_snapshot_windows()
    print("\nrows updated:")
    for k, v in sorted(counts.items()):
        print(f"    {k:32s} {v}")


if __name__ == "__main__":
    main()
