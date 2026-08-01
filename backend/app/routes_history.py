"""Investigation activity-history endpoints (persistent dashboard history)."""
from __future__ import annotations

from fastapi import APIRouter, Body

from . import history

router = APIRouter()


@router.get("/history")
def get_history(limit: int = 300):
    """Recent activity (persons + vehicles searched / found / tracked), newest first."""
    return history.list_recent(limit)


@router.post("/history")
def add_history(entry: dict = Body(...)):
    """Log one activity entry (called by the frontend on search / view / track)."""
    history.add(entry)
    return {"ok": True}


@router.delete("/history")
def clear_history():
    """Wipe the activity history - the dashboard then records fresh from now."""
    return history.clear()
