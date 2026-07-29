"""SQLite metadata store.

Holds the *facts* about everything detected (timestamps, cameras, colours,
plate text, ...). Vector fingerprints live in FAISS; each row here keeps the
FAISS row-id so the two can be joined at search time.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    camera_id   TEXT PRIMARY KEY,
    name        TEXT,
    location    TEXT,
    lat         REAL,
    lon         REAL,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS videos (
    video_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT,
    filename    TEXT,
    fps         REAL,          -- sampled (analysis) fps
    start_time  TEXT,          -- real-world time of first frame (ISO)
    duration    REAL,          -- seconds
    end_time    TEXT,          -- real-world time of last frame (ISO)
    native_fps  REAL,          -- source video fps (for exact seek offset)
    width       INTEGER,       -- native frame width  (for bbox overlay)
    height      INTEGER,       -- native frame height
    status      TEXT DEFAULT 'pending',
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id     INTEGER,
    camera_id    TEXT,
    track_id     INTEGER,
    frame_number INTEGER,
    timestamp    TEXT,         -- real-world time of this detection (ISO)
    class_label  TEXT,
    confidence   REAL,
    bbox_x       REAL,
    bbox_y       REAL,
    bbox_w       REAL,
    bbox_h       REAL,
    crop_path    TEXT,
    clip_vec_id  INTEGER,      -- row id in the CLIP FAISS index
    reid_vec_id  INTEGER,      -- row id in the re-ID FAISS index (persons)
    attributes   TEXT          -- JSON: colour, type, clothing, accessories...
);

CREATE TABLE IF NOT EXISTS tracks (
    track_key    TEXT PRIMARY KEY,   -- "{video_id}:{track_id}"
    video_id     INTEGER,
    camera_id    TEXT,
    track_id     INTEGER,
    class_label  TEXT,
    start_frame  INTEGER,
    end_frame    INTEGER,
    start_time   TEXT,
    end_time     TEXT,
    direction    TEXT
);

CREATE TABLE IF NOT EXISTS faces (
    face_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER,
    camera_id    TEXT,
    timestamp    TEXT,
    face_vec_id  INTEGER,      -- row id in the face FAISS index
    age          INTEGER,
    gender       TEXT,
    crop_path    TEXT
);

CREATE TABLE IF NOT EXISTS plates (
    plate_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER,
    camera_id    TEXT,
    timestamp    TEXT,
    plate_text   TEXT,
    confidence   REAL,
    crop_path    TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    log_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT,
    action       TEXT,
    query_text   TEXT,
    query_type   TEXT,
    result_count INTEGER,
    user         TEXT,
    details      TEXT
);

CREATE TABLE IF NOT EXISTS exports (
    export_id     TEXT PRIMARY KEY,
    case_number   TEXT,
    officer       TEXT,
    created_at    TEXT,
    manifest_hash TEXT,
    file_path     TEXT,
    detection_ids TEXT
);

CREATE INDEX IF NOT EXISTS idx_det_camera ON detections(camera_id);
CREATE INDEX IF NOT EXISTS idx_det_time   ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_det_class  ON detections(class_label);
CREATE INDEX IF NOT EXISTS idx_plate_text ON plates(plate_text);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn():
    """Context-managed connection with dict-like rows."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _migrate(conn) -> None:
    """Add columns introduced after a DB was first created (safe, idempotent)."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(videos)").fetchall()}
    for col, decl in (("duration", "REAL"), ("end_time", "TEXT"),
                      ("native_fps", "REAL"), ("width", "INTEGER"), ("height", "INTEGER")):
        if col not in cols:
            conn.execute(f"ALTER TABLE videos ADD COLUMN {col} {decl}")


def init_db() -> None:
    """Create tables and seed cameras from camera_config.json (once)."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)
        # Sync configured cameras from camera_config.json on every init, so the
        # camera list is user-owned (edit the file -> names/locations update).
        # Cameras that self-register from footage (not in the file) are untouched.
        if config.CAMERA_CONFIG_PATH.exists():
            try:
                cams = json.loads(config.CAMERA_CONFIG_PATH.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                cams = []
            for c in cams:
                conn.execute(
                    "INSERT OR IGNORE INTO cameras "
                    "(camera_id, name, location, lat, lon, created_at) VALUES (?,?,?,?,?,?)",
                    (c["camera_id"], c.get("name"), c.get("location"),
                     c.get("lat"), c.get("lon"), _now()),
                )
                conn.execute(
                    "UPDATE cameras SET name=?, location=?, lat=?, lon=? WHERE camera_id=?",
                    (c.get("name"), c.get("location"), c.get("lat"), c.get("lon"), c["camera_id"]),
                )
            # Drop stale cameras that aren't configured and have no footage
            # (e.g. removed demo entries), so the UI never shows empty cameras.
            config_ids = [c["camera_id"] for c in cams]
            if config_ids:
                ph = ",".join("?" * len(config_ids))
                conn.execute(
                    f"DELETE FROM cameras WHERE camera_id NOT IN ({ph}) AND camera_id NOT IN "
                    "(SELECT DISTINCT camera_id FROM videos WHERE camera_id IS NOT NULL)",
                    config_ids,
                )


def register_camera(camera_id, name=None, location=None, lat=None, lon=None) -> None:
    """Add a camera on the fly (used when the judge uploads unfamiliar footage)."""
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO cameras "
            "(camera_id, name, location, lat, lon, created_at) VALUES (?,?,?,?,?,?)",
            (camera_id, name or camera_id, location, lat, lon, _now()),
        )


def list_cameras() -> list[dict]:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM cameras").fetchall()]


def list_videos() -> list[dict]:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT video_id, camera_id, filename, fps, start_time, duration, end_time, "
            " native_fps, width, height, status FROM videos ORDER BY camera_id, start_time"
        ).fetchall()]


def video_index() -> dict:
    """{video_id: {filename, camera_id, start_time, duration, native_fps, width, height}}
    - used to resolve a detection to its recording clip + seek offset."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT video_id, camera_id, filename, start_time, duration, native_fps, width, height "
            "FROM videos"
        ).fetchall()
    return {r["video_id"]: dict(r) for r in rows}


def log_audit(action, query_text=None, query_type=None,
              result_count=None, user="officer", details=None) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO audit_log "
            "(timestamp, action, query_text, query_type, result_count, user, details) "
            "VALUES (?,?,?,?,?,?,?)",
            (_now(), action, query_text, query_type, result_count, user,
             json.dumps(details) if details is not None else None),
        )


def add_video(camera_id, filename, fps=None, start_time=None, duration=None, status="pending",
              end_time=None, native_fps=None, width=None, height=None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO videos (camera_id, filename, fps, start_time, duration, end_time, "
            " native_fps, width, height, status, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (camera_id, filename, fps, start_time, duration, end_time,
             native_fps, width, height, status, _now()),
        )
        return cur.lastrowid


def set_video_status(video_id, status) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE videos SET status=? WHERE video_id=?", (status, video_id))


_DET_COLS = ("video_id, camera_id, track_id, frame_number, timestamp, class_label, confidence, "
             "bbox_x, bbox_y, bbox_w, bbox_h, crop_path, clip_vec_id, reid_vec_id, attributes")
_DET_QS = "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"


def _det_values(d: dict) -> tuple:
    return (d.get("video_id"), d.get("camera_id"), d.get("track_id"), d.get("frame_number"),
            d.get("timestamp"), d.get("class_label"), d.get("confidence"),
            d.get("bbox_x"), d.get("bbox_y"), d.get("bbox_w"), d.get("bbox_h"),
            d.get("crop_path"), d.get("clip_vec_id"), d.get("reid_vec_id"),
            json.dumps(d["attributes"]) if d.get("attributes") is not None else None)


def insert_detection(d: dict) -> int:
    """Insert one detection row (dict of column values). Returns detection_id."""
    with get_conn() as conn:
        cur = conn.execute(
            f"INSERT INTO detections ({_DET_COLS}) VALUES {_DET_QS}", _det_values(d))
        return cur.lastrowid


def insert_detections_bulk(rows: list[dict]) -> list[int]:
    """Insert many detection rows in ONE connection/transaction and return their
    detection_ids in order. Identical rows to insert_detection, but avoids the
    per-row connect+commit overhead - a big speed-up on video ingest."""
    if not rows:
        return []
    ids = []
    with get_conn() as conn:
        cur = conn.cursor()
        for d in rows:
            cur.execute(f"INSERT INTO detections ({_DET_COLS}) VALUES {_DET_QS}", _det_values(d))
            ids.append(cur.lastrowid)
    return ids


def upsert_track(track_key, video_id, camera_id, track_id, class_label,
                 start_frame, end_frame, start_time, end_time, direction=None) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO tracks "
            "(track_key, video_id, camera_id, track_id, class_label, start_frame, end_frame, "
            " start_time, end_time, direction) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (track_key, video_id, camera_id, track_id, class_label,
             start_frame, end_frame, start_time, end_time, direction),
        )


def _row_to_detection(row) -> dict:
    d = dict(row)
    if d.get("attributes"):
        try:
            d["attributes"] = json.loads(d["attributes"])
        except (TypeError, ValueError):
            d["attributes"] = {}
    else:
        d["attributes"] = {}
    return d


def get_detections(ids) -> list[dict]:
    """Fetch detections by id, preserving the given order (for ranked results)."""
    ids = list(ids)
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM detections WHERE detection_id IN ({placeholders})", ids
        ).fetchall()
    by_id = {r["detection_id"]: _row_to_detection(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def get_track_detections(video_id, track_id) -> list[dict]:
    """All detection rows for ONE ByteTrack track within a video, time-ordered.

    Used by the single-camera tracking viewer to replay a track's per-frame
    bounding boxes. Reads stored metadata only - no detection/tracking is re-run.
    Scene (whole-frame) rows are excluded."""
    if video_id is None or track_id is None:
        return []
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM detections WHERE video_id=? AND track_id=? "
            "AND class_label != 'scene' ORDER BY frame_number",
            (video_id, track_id),
        ).fetchall()
    return [_row_to_detection(r) for r in rows]


def query_detections(camera_ids=None, start_time=None, end_time=None,
                     class_labels=None, limit=2000) -> list[dict]:
    """Metadata-only filtered query (time / camera / class)."""
    clauses, params = [], []
    if camera_ids:
        clauses.append(f"camera_id IN ({','.join('?' * len(camera_ids))})")
        params += list(camera_ids)
    if start_time:
        clauses.append("timestamp >= ?")
        params.append(start_time)
    if end_time:
        clauses.append("timestamp <= ?")
        params.append(end_time)
    if class_labels:
        clauses.append(f"class_label IN ({','.join('?' * len(class_labels))})")
        params += list(class_labels)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM detections {where} ORDER BY timestamp LIMIT ?", params + [limit]
        ).fetchall()
    return [_row_to_detection(r) for r in rows]


def count_detections() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM detections").fetchone()["n"]


def insert_face(f: dict) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO faces (detection_id, camera_id, timestamp, face_vec_id, age, gender, crop_path) "
            "VALUES (?,?,?,?,?,?,?)",
            (f.get("detection_id"), f.get("camera_id"), f.get("timestamp"),
             f.get("face_vec_id"), f.get("age"), f.get("gender"), f.get("crop_path")),
        )
        return cur.lastrowid


def get_faces(ids) -> list[dict]:
    ids = list(ids)
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM faces WHERE face_id IN ({placeholders})", ids
        ).fetchall()
    by_id = {r["face_id"]: dict(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def count_faces() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM faces").fetchone()["n"]


def insert_plate(p: dict) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO plates (detection_id, camera_id, timestamp, plate_text, confidence, crop_path) "
            "VALUES (?,?,?,?,?,?)",
            (p.get("detection_id"), p.get("camera_id"), p.get("timestamp"),
             p.get("plate_text"), p.get("confidence"), p.get("crop_path")),
        )
        return cur.lastrowid


def get_plates(ids) -> list[dict]:
    ids = list(ids)
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM plates WHERE plate_id IN ({placeholders})", ids
        ).fetchall()
    by_id = {r["plate_id"]: dict(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def search_plates(text, camera_ids=None, start_time=None, end_time=None, limit=200) -> list[dict]:
    """Partial/full match on plate_text. Both sides are normalised (uppercased,
    spaces/hyphens stripped) so 'AB1234' matches a stored 'GJ 05 AB 1234'."""
    needle = "%" + "".join(ch for ch in (text or "").upper() if ch.isalnum()) + "%"
    clauses = ["REPLACE(REPLACE(UPPER(plate_text), ' ', ''), '-', '') LIKE ?"]
    params = [needle]
    if camera_ids:
        clauses.append(f"camera_id IN ({','.join('?' * len(camera_ids))})")
        params += list(camera_ids)
    if start_time:
        clauses.append("timestamp >= ?")
        params.append(start_time)
    if end_time:
        clauses.append("timestamp <= ?")
        params.append(end_time)
    where = " AND ".join(clauses)
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM plates WHERE {where} ORDER BY confidence DESC LIMIT ?",
            params + [limit],
        ).fetchall()
    return [dict(r) for r in rows]


def count_plates() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM plates").fetchone()["n"]


def insert_export(export_id, case_number, officer, created_at, manifest_hash,
                  file_path, detection_ids) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO exports "
            "(export_id, case_number, officer, created_at, manifest_hash, file_path, detection_ids) "
            "VALUES (?,?,?,?,?,?,?)",
            (export_id, case_number, officer, created_at, manifest_hash, file_path,
             json.dumps(list(detection_ids))),
        )


def list_exports() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM exports ORDER BY created_at DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["detection_ids"] = json.loads(d["detection_ids"]) if d.get("detection_ids") else []
        except (TypeError, ValueError):
            d["detection_ids"] = []
        out.append(d)
    return out


if __name__ == "__main__":
    init_db()
    print(f"Initialised DB at {config.DB_PATH}")
    print(f"Cameras: {len(list_cameras())}")
    print(f"Detections: {count_detections()}")
    print(f"Faces: {count_faces()}")
    print(f"Plates: {count_plates()}")
