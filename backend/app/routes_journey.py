"""Journey Reconstruction endpoints (additive)."""
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from . import journey

router = APIRouter()


@router.post("/journey/reconstruct")
def reconstruct_journey(payload: dict = Body(...)):
    """Reconstruct the probable cross-camera movement of a person.

    body: {detection_id, cameras?: [camera_id] | null (null = all), investigation?}
    """
    det = payload.get("detection_id")
    if det is None:
        raise HTTPException(status_code=400, detail="detection_id required")
    cams = payload.get("cameras")
    if cams is not None and not isinstance(cams, list):
        raise HTTPException(status_code=400, detail="cameras must be a list or null")
    res = journey.reconstruct(int(det), cameras=cams or None,
                             investigation=payload.get("investigation"))
    if res.get("error"):
        raise HTTPException(status_code=404, detail=res["error"])
    return res


@router.get("/journeys")
def list_journeys(investigation: str | None = None):
    return journey.list_journeys(investigation)


@router.get("/journeys/{journey_id}")
def get_journey(journey_id: int):
    j = journey.get_journey(journey_id)
    if j is None:
        raise HTTPException(status_code=404, detail="journey not found")
    return j


@router.delete("/journeys/{journey_id}")
def delete_journey(journey_id: int):
    return journey.delete_journey(journey_id)
