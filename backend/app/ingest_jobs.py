"""In-memory ingestion job registry for streaming progress over WebSocket.

Prototype-scale: a process-local, thread-safe dict. Good enough for a single
uvicorn worker demo; a multi-worker deployment would swap this for Redis or a
DB table (same tiny interface).
"""
from __future__ import annotations

import uuid
from threading import Event, Lock

_JOBS: dict[str, dict] = {}
_lock = Lock()

# Global "stop the running ingest" signal. Only one batch ingest runs at a time
# (ingest_all), so a single process-wide Event is enough. The tracker checks it
# per frame (responsive mid-video) and the batch loop checks it between files.
_stop_event = Event()


def request_stop() -> None:
    _stop_event.set()


def clear_stop() -> None:
    _stop_event.clear()


def stop_requested() -> bool:
    return _stop_event.is_set()


def has_running_job() -> bool:
    """True if any job is currently 'processing'. Used to refuse starting a
    second GPU-heavy job (ingest / colour-recompute) concurrently - two at once
    exhausts a small GPU and corrupts the CUDA context."""
    with _lock:
        return any(j.get("status") == "processing" for j in _JOBS.values())


def new_job(video: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _JOBS[job_id] = {"job_id": job_id, "video": video, "status": "queued",
                         "stage": None, "pct": 0, "message": "", "stats": None}
    return job_id


def update(job_id: str, **fields) -> None:
    with _lock:
        if job_id in _JOBS:
            _JOBS[job_id].update(fields)


def get(job_id: str) -> dict | None:
    with _lock:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def set_job(job_id: str, data: dict) -> None:
    """Seed a job directly (used by tests / manual seeding)."""
    with _lock:
        _JOBS[job_id] = dict(data)
