"""Camera Registry + Journey/Route engine endpoints (additive).

The pre-existing GET/POST /cameras endpoints are left untouched; these live under
/camera-registry so nothing that already works changes.
"""
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from . import camera_registry, config, database, routing

router = APIRouter()


@router.get("/camera-registry")
def list_registry(include_inactive: bool = True):
    return camera_registry.list_cameras(include_inactive=include_inactive)


@router.get("/camera-registry/status")
def registry_status():
    """Whether the registry has enough located cameras for journey reconstruction."""
    st = camera_registry.registry_status()
    st["route_engine"] = {"active": routing.get_engine().name,
                          "providers": routing.providers()}
    if not st["ready_for_journey"]:
        st["notice"] = ("Journey reconstruction unavailable until valid camera "
                        "locations are configured.")
    return st


@router.get("/camera-registry/export")
def export_registry():
    return camera_registry.export_cameras()


@router.post("/camera-registry/import")
def import_registry(payload: dict = Body(...)):
    items = payload.get("cameras") if isinstance(payload, dict) else None
    if items is None and isinstance(payload, list):
        items = payload
    if not isinstance(items, list):
        raise HTTPException(status_code=400, detail="expected {cameras: [...]}")
    return camera_registry.import_cameras(items, replace=bool(payload.get("replace")))


@router.get("/camera-registry/{camera_id}")
def get_registry_camera(camera_id: str):
    cam = camera_registry.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="camera not found")
    with database.get_conn() as conn:
        cam["videos"] = [dict(r) for r in conn.execute(
            "SELECT video_id, filename, status, duration FROM videos WHERE camera_id=? "
            "ORDER BY video_id DESC", (camera_id,)).fetchall()]
    return cam


@router.post("/camera-registry")
def upsert_registry_camera(payload: dict = Body(...)):
    """Create or update a camera. Stored permanently (survives config sync)."""
    try:
        return camera_registry.upsert_camera(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/camera-registry/{camera_id}")
def delete_registry_camera(camera_id: str, force: bool = False):
    res = camera_registry.delete_camera(camera_id, force=force)
    if res.get("error"):
        raise HTTPException(status_code=409, detail=res["error"])
    return res


@router.get("/video-gps/{filename}")
def video_gps(filename: str):
    """Probe a video in the library for GPS metadata (most CCTV exports have none)."""
    path = config.VIDEO_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="video not found")
    res = camera_registry.probe_video_gps(path)
    if res.get("available"):
        cam = camera_registry.match_or_create_from_gps(res["lat"], res["lon"])
        res["camera"] = cam
        res["matched"] = bool(cam)
    return res


@router.get("/route-engine")
def route_engine():
    """Active route provider + all integration targets (none implemented yet)."""
    return {"active": routing.get_engine().name, "providers": routing.providers()}


@router.post("/route-engine/active")
def set_route_engine(payload: dict = Body(...)):
    name = (payload or {}).get("name")
    if not routing.set_active(str(name)):
        raise HTTPException(status_code=400, detail=f"unknown or unregistered engine: {name}")
    return {"active": routing.get_engine().name}
