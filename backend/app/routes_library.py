"""Library + batch-ingest + describe-search + camera + recompute endpoints.

Kept in their own router (separate from routes.py). All additive - they list the
videos folder, report processed status, run/stop the EXISTING ingest, expose live
progress, run the describe-and-filter search, register cameras (GPS), and
recompute clothing colours / licence plates. Nothing here changes the pipeline.

Only ONE GPU-heavy job (ingest / colour-recompute / plate-recompute) may run at a
time: on a small GPU, two at once exhaust VRAM and corrupt the CUDA context, so
every start-point refuses if a job is already running.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from . import config, database, ingest_jobs, ingest_progress
from .models.schemas import Camera, TextSearchRequest, TrackResponse

router = APIRouter()

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv"}


def _safe_video_name(name: str | None) -> str:
    """Sanitise an uploaded filename to a bare, safe basename with a video ext."""
    base = Path(name or "").name                       # strip any path components
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip() or "upload.mp4"
    if Path(base).suffix.lower() not in _VIDEO_EXTS:
        base += ".mp4"
    return base


def _unique_dest(name: str) -> Path:
    """A non-clobbering destination path inside the videos dir."""
    dest = config.VIDEO_DIR / name
    if not dest.exists():
        return dest
    stem, suf, i = dest.stem, dest.suffix, 1
    while (config.VIDEO_DIR / f"{stem}_{i}{suf}").exists():
        i += 1
    return config.VIDEO_DIR / f"{stem}_{i}{suf}"


def _has_mp4_twin(p) -> bool:
    """True if a non-mp4 source has a transcoded .mp4 sibling. Auto-transcode
    (pipeline.ensure_mp4) leaves the original on disk as evidence, so we surface
    only the playable .mp4 - otherwise the source would show as a duplicate,
    unprocessed entry and get re-ingested."""
    return p.suffix.lower() != ".mp4" and p.with_suffix(".mp4").exists()


@router.get("/library")
def library_route():
    """Every video FILE in the library folder, flagged processed / not-processed."""
    by_name = {v["filename"]: v for v in database.list_videos()}
    items = []
    if config.VIDEO_DIR.exists():
        for p in sorted(config.VIDEO_DIR.iterdir()):
            if p.suffix.lower() not in _VIDEO_EXTS:
                continue
            if _has_mp4_twin(p):                 # hide the source; list only the .mp4
                continue
            v = by_name.get(p.name)
            items.append({
                "filename": p.name,
                "processed": bool(v) and v.get("status") == "done",
                "status": v.get("status") if v else "not_processed",
                "video_id": v.get("video_id") if v else None,
                "camera_id": v.get("camera_id") if v else None,
                "duration": v.get("duration") if v else None,
                "url": f"/media/videos/{p.name}",
                "size_mb": round(p.stat().st_size / 1048576, 1),
            })
    return items


@router.post("/ingest/all")
def ingest_all_route(mode: str | None = None):
    """Process every unprocessed video in the folder (one sequential background job).
    Optional `mode` ("fast" default | "accurate"); None -> config.PROCESSING_MODE."""
    if ingest_jobs.has_running_job():
        return {"job_id": None, "total": 0, "busy": True,
                "message": "Another job is already running - wait for it to finish."}
    from .ingestion import pipeline
    done = {v["filename"] for v in database.list_videos() if v.get("status") == "done"}
    files = ([p for p in sorted(config.VIDEO_DIR.iterdir())
              if p.suffix.lower() in _VIDEO_EXTS and p.name not in done
              and not _has_mp4_twin(p)]         # don't re-ingest a source whose .mp4 exists
             if config.VIDEO_DIR.exists() else [])
    if not files:
        return {"job_id": None, "total": 0, "message": "All videos are already processed."}

    job_id = ingest_jobs.new_job(f"{len(files)} video(s)")
    ingest_jobs.update(job_id, status="processing", total=len(files), done=0, current=files[0].name)
    ingest_jobs.clear_stop()                     # fresh run - clear any old stop signal

    def run():
        stopped = False
        i = 0
        for i, f in enumerate(files):
            if ingest_jobs.stop_requested():
                stopped = True
                break
            ingest_progress.reset()               # new video -> reset the per-video bar
            ingest_jobs.update(job_id, status="processing", current=f.name, done=i)
            try:
                pipeline.ingest_video(f, mode=mode)
            except Exception as exc:  # noqa: BLE001 - keep going, report per file
                if ingest_jobs.stop_requested():   # the exception was our stop signal
                    stopped = True
                    break
                ingest_jobs.update(job_id, last_error=f"{f.name}: {exc}")
        ingest_jobs.clear_stop()
        ingest_progress.reset()
        ingest_jobs.update(job_id, status="stopped" if stopped else "done",
                           done=(i if stopped else len(files)), current=None)

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id, "total": len(files), "files": [f.name for f in files]}


@router.post("/ingest/upload")
async def ingest_upload_route(file: UploadFile = File(...),
                              camera_id: str | None = Form(None),
                              mode: str | None = Form(None)):
    """Accept a video uploaded from the user's device, save it into the videos
    folder, and kick off the EXISTING ingestion pipeline in a background job.
    Progress streams over the same GET /ingest/job/{job_id} the batch ingest uses.
    """
    if ingest_jobs.has_running_job():
        return {"job_id": None, "busy": True,
                "message": "Another job is already running - wait for it to finish."}

    config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    dest = _unique_dest(_safe_video_name(file.filename))

    # Stream the upload to disk in chunks (handles large clips without buffering
    # the whole file in memory).
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    finally:
        await file.close()

    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    from .ingestion import pipeline
    fname = dest.name
    cam = (camera_id or "").strip() or None
    job_id = ingest_jobs.new_job(fname)
    ingest_jobs.update(job_id, status="processing", total=1, done=0, current=fname)
    ingest_jobs.clear_stop()
    ingest_progress.reset()

    run_mode = (mode or config.PROCESSING_MODE)

    def run():
        try:
            # Fast mode (default): adaptive sampling, gated faces/plates, incremental
            # index -> the uploaded clip becomes searchable quickly. "accurate" runs
            # the full forensic pipeline.
            pipeline.ingest_video(dest, camera_id=cam, mode=run_mode)
            ingest_jobs.update(job_id, status="done", done=1, current=None)
        except Exception as exc:  # noqa: BLE001 - report failure to the client
            ingest_jobs.update(job_id, status="error", last_error=str(exc), current=None)
        finally:
            ingest_progress.reset()

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id, "filename": fname, "total": 1, "mode": run_mode}


@router.post("/videos/delete")
def delete_video_route(payload: dict):
    """Permanently delete a video: its file(s) on disk, its DB rows (detections,
    tracks, faces, plates, videos), its FAISS vectors, and its crops/frames.
    Body: {"filename": "<name in videos dir>"}. Refuses while a job is running."""
    if ingest_jobs.has_running_job():
        raise HTTPException(status_code=409, detail="A processing job is running - try again once it finishes.")

    fname = Path((payload or {}).get("filename") or "").name   # strip any path
    if not fname:
        raise HTTPException(status_code=400, detail="filename is required")
    vdir = config.VIDEO_DIR.resolve()

    from .search import vector_store
    removed = {"clip": 0, "reid": 0, "face": 0}
    video_ids, cameras = [], set()

    with database.get_conn() as conn:
        rows = conn.execute("SELECT video_id, camera_id FROM videos WHERE filename=?", (fname,)).fetchall()
        for r in rows:
            video_ids.append(r["video_id"])
            if r["camera_id"]:
                cameras.add(r["camera_id"])
        for vid in video_ids:
            det_ids = [x["detection_id"] for x in
                       conn.execute("SELECT detection_id FROM detections WHERE video_id=?", (vid,)).fetchall()]
            face_ids = [x["face_id"] for x in conn.execute(
                "SELECT face_id FROM faces WHERE detection_id IN "
                "(SELECT detection_id FROM detections WHERE video_id=?)", (vid,)).fetchall()]
            if det_ids:
                removed["clip"] += vector_store.remove("clip", det_ids)
                removed["reid"] += vector_store.remove("reid", det_ids)
            if face_ids:
                removed["face"] += vector_store.remove("face", face_ids)
            conn.execute("DELETE FROM faces WHERE detection_id IN "
                         "(SELECT detection_id FROM detections WHERE video_id=?)", (vid,))
            conn.execute("DELETE FROM plates WHERE detection_id IN "
                         "(SELECT detection_id FROM detections WHERE video_id=?)", (vid,))
            conn.execute("DELETE FROM detections WHERE video_id=?", (vid,))
            conn.execute("DELETE FROM tracks WHERE video_id=?", (vid,))
            conn.execute("DELETE FROM videos WHERE video_id=?", (vid,))
    if video_ids:
        vector_store.save()

    # ---- disk cleanup (guarded to the data dirs) ----
    import shutil
    stem = Path(fname).stem
    deleted_files = []
    # the requested file + any same-stem source of another extension (transcode leftovers)
    for p in vdir.glob(stem + ".*"):
        if p.suffix.lower() in _VIDEO_EXTS and p.resolve().parent == vdir:
            try: p.unlink(); deleted_files.append(p.name)
            except OSError: pass
    # crops + frames for this clip (per camera/<stem>)
    for cam in (cameras or [None]):
        for root in (config.CROP_DIR, config.FRAME_DIR):
            d = (root / cam / stem) if cam else None
            if d and d.exists():
                shutil.rmtree(d, ignore_errors=True)

    return {"deleted": True, "filename": fname, "video_ids": video_ids,
            "files_removed": deleted_files, "vectors_removed": removed}


@router.post("/ingest/stop")
def ingest_stop_route():
    """Ask the running batch ingest to stop. Takes effect within a few seconds
    (the tracker checks per frame); the current video is abandoned, not saved."""
    ingest_jobs.request_stop()
    return {"stopping": True, "message": "Stopping - halting the current video now."}


@router.get("/ingest/job/{job_id}")
def ingest_job_route(job_id: str):
    job = ingest_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown job")
    job["video_progress"] = ingest_progress.get()   # live per-video bar data
    return job


@router.post("/recompute-colors")
def recompute_colors_route(video_id: int | None = None):
    """Recompute person clothing colours (upper/lower split + HSV) for stored
    detections - fixes colour attributes without re-running the full pipeline."""
    if ingest_jobs.has_running_job():
        return {"job_id": None, "busy": True,
                "message": "Another job is already running - wait for it to finish."}
    from .ingestion import recompute_colors
    job_id = ingest_jobs.new_job("recompute colours")
    ingest_jobs.update(job_id, status="processing", total=0, done=0, current="recomputing colours")
    ingest_jobs.clear_stop()

    def run():
        try:
            recompute_colors.recompute_colors(video_id=video_id, job_id=job_id)
        except Exception as exc:  # noqa: BLE001
            ingest_jobs.update(job_id, status="error", last_error=str(exc))

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id, "video_id": video_id}


@router.post("/recompute-plates")
def recompute_plates_route(video_id: int | None = None):
    """Re-run licence-plate OCR (enhanced + multi-frame voting) over stored vehicle
    crops and populate the plates table - fixes plate search without a full re-run."""
    if ingest_jobs.has_running_job():
        return {"job_id": None, "busy": True,
                "message": "Another job is already running - wait for it to finish."}
    from .ingestion import recompute_plates
    job_id = ingest_jobs.new_job("recompute plates")
    ingest_jobs.update(job_id, status="processing", total=0, done=0, current="reading plates")
    ingest_jobs.clear_stop()

    def run():
        try:
            recompute_plates.recompute_plates(video_id=video_id, job_id=job_id)
        except Exception as exc:  # noqa: BLE001
            ingest_jobs.update(job_id, status="error", last_error=str(exc))

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id, "video_id": video_id}


@router.post("/search/describe")
def describe_search_route(req: TextSearchRequest):
    """Describe-and-filter search: parse the plain-language query into structured
    constraints, run CLIP visual search, keep only detections that satisfy every
    HARD constraint, and annotate each result with what it matched."""
    from .search import describe_search
    return describe_search.search(req.query, filters_obj=req.filters, top_k=req.top_k)


@router.post("/track/{detection_id}", response_model=TrackResponse)
def track_person_route(detection_id: int, threshold: float | None = None, max_results: int = 500):
    """Cross-camera tracking: trace one detection across every camera.

    Parameterized POST variant (the plain GET /track/{id} uses defaults). Returns
    the entity's time-sorted appearances + a journey summary, and logs the search.
    """
    from .search import cross_camera
    resp = cross_camera.track_across_cameras(
        detection_id, threshold=threshold, max_results=max_results)
    database.log_audit(
        "track", query_text=str(detection_id), query_type="cross_camera",
        result_count=len(resp.appearances),
        details={"threshold": threshold, "max_results": max_results,
                 "unique_cameras": resp.summary.unique_cameras if resp.summary else 0},
    )
    return resp


def _upsert_camera_config(cam: Camera) -> None:
    """Persist a camera into camera_config.json (the registry init_db syncs from
    on startup), so a UI-added camera survives a restart."""
    path = config.CAMERA_CONFIG_PATH
    try:
        cams = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except (ValueError, OSError):
        cams = []
    entry = {"camera_id": cam.camera_id, "name": cam.name, "location": cam.location,
             "lat": cam.lat, "lon": cam.lon}
    for i, c in enumerate(cams):
        if c.get("camera_id") == cam.camera_id:
            cams[i] = entry
            break
    else:
        cams.append(entry)
    try:
        path.write_text(json.dumps(cams, indent=2), encoding="utf-8")
    except OSError:
        pass


@router.post("/cameras", response_model=list[Camera])
def upsert_camera_route(cam: Camera):
    """Add or update a camera (id, name, location, GPS lat/lon). Writes the DB
    (shows on the map immediately) AND camera_config.json (survives restart),
    then returns the full camera list."""
    cid = (cam.camera_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="camera_id is required")
    cam.camera_id = cid
    with database.get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO cameras (camera_id, name, location, lat, lon, created_at) "
            "VALUES (?,?,?,?,?, datetime('now'))",
            (cid, cam.name, cam.location, cam.lat, cam.lon))
        conn.execute(
            "UPDATE cameras SET name=?, location=?, lat=?, lon=? WHERE camera_id=?",
            (cam.name, cam.location, cam.lat, cam.lon, cid))
    _upsert_camera_config(cam)
    return database.list_cameras()
