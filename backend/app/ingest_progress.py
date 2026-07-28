"""Live per-video ingest progress for the clip currently being processed.

Tracks the WHOLE pipeline, not just the detect+track pass, so the per-video bar
keeps advancing to 100% instead of freezing after tracking:

  * the tracker calls set_progress(frame, total) per frame -> fills the
    _TRACK_LO.._TRACK_HI slice of the bar (detect+track is only the first part
    of the work), and
  * the pipeline calls set_stage(pct, stage, message) at each later stage
    (clip / reid / store / faces / plates / done) -> fills the rest.

pct is forward-only within a video (never jumps backwards); reset() clears it
between videos. Thread-safe, process-global - only one ingest runs at a time, so
a single global is enough. The job-status endpoint reads get() so the frontend's
existing poll renders a live per-video bar with a stage label.
"""
from __future__ import annotations

from threading import Lock

_lock = Lock()
_state = {"frame": 0, "total": 0, "pct": 0, "stage": "", "message": "",
          # progressive-pipeline fields (searchable-while-processing):
          "indexed": 0, "searchable": False, "video_id": None}

# detect+track fills this slice of the per-video bar; the later stages
# (clip/reid/store/faces/plates/done) fill the rest via set_stage().
_TRACK_LO, _TRACK_HI = 3, 40


def set_progress(frame: int, total: int) -> None:
    """Per-frame update from the tracker during the detect+track pass."""
    with _lock:
        frac = (frame / total) if total else 0.0
        pct = _TRACK_LO + int(frac * (_TRACK_HI - _TRACK_LO))
        pct = max(_TRACK_LO, min(pct, _TRACK_HI))
        _state.update({"frame": int(frame), "total": int(total),
                       "pct": max(_state.get("pct", 0), pct),
                       "stage": "detect+track", "message": ""})


def set_stage(pct: int, stage: str = "", message: str = "") -> None:
    """Stage-boundary update from the pipeline (clip/reid/store/faces/plates/done).
    Forward-only so an earlier stage's lower pct can't pull the bar backwards."""
    with _lock:
        target = max(0, min(int(pct), 100))
        _state.update({"pct": max(_state.get("pct", 0), target),
                       "stage": stage or _state.get("stage", ""),
                       "message": message})


def set_meta(**kw) -> None:
    """Update progressive-pipeline fields (indexed count, searchable flag,
    video_id) shown live in the UI while the rest of the clip keeps processing."""
    with _lock:
        for k in ("indexed", "searchable", "video_id", "stage", "message"):
            if k in kw and kw[k] is not None:
                _state[k] = kw[k]


def reset() -> None:
    with _lock:
        _state.update({"frame": 0, "total": 0, "pct": 0, "stage": "", "message": "",
                       "indexed": 0, "searchable": False, "video_id": None})


def get() -> dict:
    with _lock:
        return dict(_state)
