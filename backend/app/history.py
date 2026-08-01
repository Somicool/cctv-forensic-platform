"""Investigation activity history (persistent).

Records the persons AND vehicles an investigator searched, found, and tracked, so
the Dashboard shows a durable history that survives reloads/restarts. Written by
the frontend on key actions (search / view / track); the Dashboard reads it back.

De-dup: a new entry with the same (kind, action, ref) replaces the old one and
refreshes its time, so the history stays clean (one row per entity+action) while
still reflecting the latest interaction.
"""
from __future__ import annotations

from . import database

_ALLOWED = ("kind", "action", "ref", "label", "camera_id", "timestamp",
            "crop_url", "plate", "query")


def add(entry: dict) -> None:
    e = {k: entry.get(k) for k in _ALLOWED}
    kind, action, ref = e.get("kind"), e.get("action"), str(e.get("ref") or "")
    if not action:
        return
    with database.get_conn() as conn:
        if ref:
            conn.execute("DELETE FROM activity_history WHERE kind IS ? AND action=? AND ref=?",
                         (kind, action, ref))
        conn.execute(
            "INSERT INTO activity_history "
            "(kind, action, ref, label, camera_id, timestamp, crop_url, plate, query, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (kind, action, ref, e.get("label"), e.get("camera_id"), e.get("timestamp"),
             e.get("crop_url"), e.get("plate"), e.get("query"), database._now()))


def list_recent(limit: int = 300) -> list[dict]:
    with database.get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM activity_history ORDER BY created_at DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [dict(r) for r in rows]


def clear() -> dict:
    with database.get_conn() as conn:
        n = conn.execute("SELECT COUNT(1) AS c FROM activity_history").fetchone()["c"]
        conn.execute("DELETE FROM activity_history")
    return {"cleared": int(n)}
