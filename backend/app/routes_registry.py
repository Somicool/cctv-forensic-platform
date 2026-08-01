"""Demo Vehicle Registry endpoints (offline, synthetic).

Additive router - does not touch OCR / tracking / search. The frontend calls
these by plate; the underlying provider (demo now, real police API later) is
swappable in registry.get_provider() without changing these routes or the UI.
"""
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from . import registry

router = APIRouter()


@router.get("/vehicle-registry")
def list_vehicle_registry():
    """All stored registry records (demo)."""
    return registry.get_provider().list_all()


@router.post("/vehicle-registry/backfill")
def backfill_vehicle_registry():
    """Seed registry records for every plate already recognised in prior ingests."""
    return registry.backfill_from_db()


@router.get("/vehicle-registry/{plate}")
def get_vehicle_registry(plate: str):
    """Fetch (or lazily create) the permanent registry record for a plate."""
    rec = registry.get_provider().get_or_create(plate)
    if rec is None:
        raise HTTPException(status_code=404, detail="invalid or empty plate")
    return rec


@router.put("/vehicle-registry/{plate}")
def update_vehicle_registry(plate: str, updates: dict = Body(...)):
    """Manually overwrite fields of a stored record (edits are permanent)."""
    rec = registry.get_provider().update(plate, updates)
    if rec is None:
        raise HTTPException(status_code=404, detail="record not found")
    return rec
