"""Case File persistence - the bookmarked evidence set + the case metadata.

Both of these used to live ONLY in React state (`context/investigation.jsx`), so
everything an investigator saved disappeared on a page refresh or a backend
restart. Nothing else about the investigation workflow changes: the frontend
keeps the same in-memory context API, this module just gives it a durable home
in SQLite alongside exports, saved faces and journeys.

Design notes
------------
* A full SNAPSHOT of each evidence item is stored next to its detection_id.
  The Case File must keep showing exactly what the officer saved even if that
  clip is later re-ingested (which renumbers detections) - forensic evidence
  should never silently change underneath the person who collected it.
* `position` preserves the order items were added, so the case reads the same
  way after a restart.
* One case is "active" by default; the case_key column leaves room for named
  cases later without another migration.
"""
from __future__ import annotations

import json

from . import database

DEFAULT_CASE = "active"

_INFO_FIELDS = ("title", "caseNumber", "officer", "notes")


def _key(case_key: str | None) -> str:
    return (case_key or "").strip() or DEFAULT_CASE


def _snapshot(row) -> dict:
    """Re-hydrate a stored evidence item, guaranteeing detection_id is present."""
    try:
        item = json.loads(row["snapshot"]) if row["snapshot"] else {}
    except (TypeError, ValueError):
        item = {}
    if not isinstance(item, dict):
        item = {}
    item["detection_id"] = row["detection_id"]
    return item


# --------------------------------------------------------------- evidence
def list_evidence(case_key: str | None = None) -> list[dict]:
    """Saved evidence for a case, in the order it was added."""
    ck = _key(case_key)
    with database.get_conn() as conn:
        rows = conn.execute(
            "SELECT detection_id, snapshot FROM case_evidence WHERE case_key=? "
            "ORDER BY COALESCE(position, 0), rowid", (ck,)).fetchall()
    return [_snapshot(r) for r in rows]


def set_evidence(items: list[dict], case_key: str | None = None) -> dict:
    """Replace the whole evidence set for a case (idempotent write-through).

    The frontend context produces a new array on every add/remove, so storing the
    resulting set is both the smallest change and self-healing: the database can
    never drift out of step with what the officer is looking at.
    """
    ck = _key(case_key)
    now = database._now()
    clean = []
    for i, it in enumerate(items or []):
        if not isinstance(it, dict):
            continue
        did = it.get("detection_id")
        if did is None:
            continue
        try:
            did = int(did)
        except (TypeError, ValueError):
            continue
        clean.append((ck, did, i, json.dumps(it), now))

    with database.get_conn() as conn:
        # Keep the ORIGINAL added_at for items already in the case, so the chain
        # of custody records when evidence was really collected.
        prior = {r["detection_id"]: r["added_at"] for r in conn.execute(
            "SELECT detection_id, added_at FROM case_evidence WHERE case_key=?", (ck,)).fetchall()}
        conn.execute("DELETE FROM case_evidence WHERE case_key=?", (ck,))
        conn.executemany(
            "INSERT OR REPLACE INTO case_evidence "
            "(case_key, detection_id, position, snapshot, added_at) VALUES (?,?,?,?,?)",
            [(c, d, p, s, prior.get(d) or a) for (c, d, p, s, a) in clean])
    return {"ok": True, "case_key": ck, "count": len(clean)}


def add_evidence(item: dict, case_key: str | None = None) -> dict:
    """Append one item (no-op if that detection is already in the case)."""
    ck = _key(case_key)
    did = (item or {}).get("detection_id")
    if did is None:
        return {"ok": False, "reason": "missing detection_id"}
    with database.get_conn() as conn:
        nxt = conn.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 AS n FROM case_evidence WHERE case_key=?",
            (ck,)).fetchone()["n"]
        conn.execute(
            "INSERT OR REPLACE INTO case_evidence "
            "(case_key, detection_id, position, snapshot, added_at) VALUES (?,?,?,?,?)",
            (ck, int(did), nxt, json.dumps(item), database._now()))
    return {"ok": True, "case_key": ck, "detection_id": int(did)}


def remove_evidence(detection_id: int, case_key: str | None = None) -> dict:
    ck = _key(case_key)
    with database.get_conn() as conn:
        cur = conn.execute("DELETE FROM case_evidence WHERE case_key=? AND detection_id=?",
                           (ck, int(detection_id)))
    return {"ok": True, "removed": cur.rowcount}


def clear_evidence(case_key: str | None = None) -> dict:
    ck = _key(case_key)
    with database.get_conn() as conn:
        cur = conn.execute("DELETE FROM case_evidence WHERE case_key=?", (ck,))
    return {"ok": True, "removed": cur.rowcount}


# --------------------------------------------------------------- case metadata
def get_case_info(case_key: str | None = None) -> dict:
    """Case metadata in the exact shape the frontend context uses."""
    ck = _key(case_key)
    with database.get_conn() as conn:
        row = conn.execute("SELECT * FROM case_meta WHERE case_key=?", (ck,)).fetchone()
    if not row:
        return {"title": "", "caseNumber": "", "officer": "", "notes": ""}
    return {"title": row["title"] or "", "caseNumber": row["case_number"] or "",
            "officer": row["officer"] or "", "notes": row["notes"] or ""}


def save_case_info(info: dict, case_key: str | None = None) -> dict:
    ck = _key(case_key)
    info = info or {}
    vals = {k: (info.get(k) or "") for k in _INFO_FIELDS}
    with database.get_conn() as conn:
        conn.execute(
            "INSERT INTO case_meta (case_key, title, case_number, officer, notes, updated_at) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(case_key) DO UPDATE SET "
            "title=excluded.title, case_number=excluded.case_number, "
            "officer=excluded.officer, notes=excluded.notes, updated_at=excluded.updated_at",
            (ck, vals["title"], vals["caseNumber"], vals["officer"], vals["notes"],
             database._now()))
    return {"ok": True, "case_key": ck}


# --------------------------------------------------------------- combined load
def load_case(case_key: str | None = None) -> dict:
    """Everything the frontend needs to restore an investigation on startup."""
    ck = _key(case_key)
    return {"case_key": ck, "evidence": list_evidence(ck), "case_info": get_case_info(ck)}
