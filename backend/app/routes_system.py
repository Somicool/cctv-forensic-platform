"""System info + runtime settings endpoints (for the Settings page).

Additive and read-mostly. GET /system/info reports the real device, models,
feature flags, thresholds and storage usage so the UI never hardcodes values.
POST /system/settings changes only an ALLOWLIST of existing runtime config flags
(the same ones the pipeline already reads), so nothing else is affected.
"""
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from . import config, database

router = APIRouter()

# Only these config flags may be changed at runtime from the UI.
_ALLOWED = {
    "PROCESSING_MODE": ("mode", str),                 # default mode for new ingests
    "FACE_RECOGNITION_ENABLED": ("bool", bool),
    "PLATE_RECOGNITION_ENABLED": ("bool", bool),
    "ANPR_ENABLED": ("bool", bool),
    "ANPR_ADAPTIVE_ENABLED": ("bool", bool),
    "GEMINI_ENABLED": ("bool", bool),
    "FACE_DIAG_LOG": ("bool", bool),
}


def _dir_stats(path):
    """(file count, megabytes) for a data directory - cheap, non-recursive-safe."""
    try:
        files = [p for p in path.rglob("*") if p.is_file()]
        return {"files": len(files),
                "mb": round(sum(p.stat().st_size for p in files) / 1048576, 1)}
    except Exception:
        return {"files": 0, "mb": 0.0}


@router.get("/system/info")
def system_info():
    """Everything the Settings page displays: device, models, flags, storage."""
    info = {"device": config.DEVICE, "low_vram": config.LOW_VRAM}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
            info["torch"] = torch.__version__
            info["cuda"] = torch.version.cuda
    except Exception:
        pass

    # India-specific secondary detectors (active only when weights exist)
    try:
        from .ingestion.detectors import plugins
        secondary = [{"name": p["name"],
                      "classes": sorted({l for _g, l, _s in p["class_map"].values()})}
                     for p in plugins.get_plugins()]
    except Exception:
        secondary = []

    info["models"] = {
        "detector": config.YOLO_MODEL,
        "detector_imgsz": config.YOLO_IMGSZ,
        "tracker": config.TRACKER_CFG,
        "clip": f"{config.CLIP_MODEL} ({config.CLIP_PRETRAINED})",
        "reid": config.REID_MODEL,
        "face": config.FACE_MODEL,
        "ocr_engine": config.OCR_ENGINE,
        "paddle_gpu": config.PADDLE_USE_GPU,
        "gemini_model": config.GEMINI_MODEL,
        "india_detectors": secondary,
    }
    info["processing"] = {
        "mode": config.PROCESSING_MODE,
        "modes": sorted(config.MODE_PRESETS.keys()),
        "preset": config.MODE_PRESETS.get(config.PROCESSING_MODE, {}),
        "detect_conf": config.DETECT_CONF,
        "progressive_chunk_frames": config.PROGRESSIVE_CHUNK_FRAMES,
    }
    info["flags"] = {
        "FACE_RECOGNITION_ENABLED": config.FACE_RECOGNITION_ENABLED,
        "PLATE_RECOGNITION_ENABLED": config.PLATE_RECOGNITION_ENABLED,
        "ANPR_ENABLED": config.ANPR_ENABLED,
        "ANPR_ADAPTIVE_ENABLED": config.ANPR_ADAPTIVE_ENABLED,
        "GEMINI_ENABLED": config.GEMINI_ENABLED,
        "FACE_DIAG_LOG": config.FACE_DIAG_LOG,
    }
    info["gemini_key_present"] = bool(__import__("os").environ.get(config.GEMINI_API_KEY_ENV))
    info["thresholds"] = {
        "face_accept_quality": config.FACE_ACCEPT_QUALITY,
        "face_similar_min": config.FACE_SIMILAR_MIN,
        "plate_fuzzy_threshold": config.PLATE_FUZZY_THRESHOLD,
        "plate_single_conf": config.PLATE_SINGLE_CONF,
        "reid_sim_threshold": config.REID_SIM_THRESHOLD,
    }

    # counts + storage
    try:
        with database.get_conn() as conn:
            q = lambda s: conn.execute(s).fetchone()[0]        # noqa: E731
            info["counts"] = {
                "videos": q("SELECT COUNT(1) FROM videos"),
                "cameras": q("SELECT COUNT(1) FROM cameras"),
                "detections": q("SELECT COUNT(1) FROM detections WHERE class_label!='scene'"),
                "faces": q("SELECT COUNT(1) FROM faces"),
                "saved_faces": q("SELECT COUNT(1) FROM saved_faces"),
                "plates": q("SELECT COUNT(1) FROM plates"),
                "vehicle_registry": q("SELECT COUNT(1) FROM vehicle_registry"),
                "exports": q("SELECT COUNT(1) FROM exports"),
                "activity": q("SELECT COUNT(1) FROM activity_history"),
            }
    except Exception:
        info["counts"] = {}
    try:
        from .search import vector_store
        info["faiss"] = vector_store.stats()
    except Exception:
        info["faiss"] = {}
    info["storage"] = {
        "videos": _dir_stats(config.VIDEO_DIR),
        "frames": _dir_stats(config.FRAME_DIR),
        "crops": _dir_stats(config.CROP_DIR),
        "saved_faces": _dir_stats(config.SAVED_FACE_DIR),
        "exports": _dir_stats(config.EXPORT_DIR),
        "db_mb": round(config.DB_PATH.stat().st_size / 1048576, 1) if config.DB_PATH.exists() else 0.0,
    }
    return info


@router.post("/system/settings")
def update_settings(payload: dict = Body(...)):
    """Change an allowlisted runtime flag. Returns the updated flag set."""
    changed = {}
    for key, val in (payload or {}).items():
        k = key.upper()
        if k not in _ALLOWED:
            raise HTTPException(status_code=400, detail=f"setting not allowed: {key}")
        kind = _ALLOWED[k][0]
        if kind == "mode":
            v = str(val).lower()
            if v not in config.MODE_PRESETS:
                raise HTTPException(status_code=400, detail=f"unknown mode: {val}")
        else:
            v = bool(val)
        setattr(config, k, v)
        changed[k] = v
    return {"updated": changed,
            "flags": {k: getattr(config, k) for k in _ALLOWED if k != "PROCESSING_MODE"},
            "processing_mode": config.PROCESSING_MODE}
