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


@router.get("/journey/route-providers")
def route_providers():
    """Routing backends the Journey Engine can use, and how to configure them."""
    from . import routing
    return {"active": routing.get_engine().name, "providers": routing.providers()}


@router.get("/journeys/{journey_id}/export")
def export_journey(journey_id: int, fmt: str = "json"):
    """Court-ready export of a stored journey.

    `json`    full reconstruction including per-leg evidence and rejections
    `summary` timeline + statistics only, for a written report
    `geojson` cameras and (when a routing engine is configured) the road path
    """
    fmt = (fmt or "json").lower()
    if fmt not in ("json", "summary", "geojson"):
        raise HTTPException(status_code=400, detail="fmt must be json, summary or geojson")
    out = journey.export_journey(journey_id, fmt)
    if out is None:
        raise HTTPException(status_code=404, detail="journey not found")
    return out


@router.post("/journeys/{journey_id}/case-file")
def journey_to_case_file(journey_id: int, payload: dict = Body(default={})):
    """Seal a journey into the case file as an evidence item."""
    res = journey.save_to_case_file(journey_id, payload.get("investigation"),
                                    payload.get("note"))
    if res.get("error"):
        raise HTTPException(status_code=404, detail=res["error"])
    return res
