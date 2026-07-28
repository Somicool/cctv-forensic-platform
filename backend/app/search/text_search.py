"""Descriptive text search engine.

query -> CLIP text embedding -> FAISS 'clip' search -> metadata filters ->
track-level de-dup -> relative-ranked ResultItem list (the API-ready
SearchResponse). This is the formalised version of the search_demo proof.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .. import config, database
from ..models.schemas import ResultItem, SearchFilters, SearchResponse
from ..ingestion import embedder
from . import vector_store, filters, result_utils

# Words that make a query unambiguously about a vehicle or a person, used to bias
# results to the right class (so "a motorcycle" never returns people).
_VEHICLE_WORDS = {"car", "cars", "truck", "trucks", "van", "vans", "suv", "suvs",
                  "hatchback", "sedan", "bus", "buses", "motorcycle", "motorbike",
                  "motorcycles", "bike", "bikes", "bicycle", "bicycles", "cycle",
                  "scooter", "vehicle", "vehicles", "auto", "rickshaw", "lorry",
                  "pickup", "jeep", "taxi", "cab", "ambulance"}
_PERSON_WORDS = {"man", "men", "woman", "women", "person", "persons", "people",
                 "child", "children", "boy", "girl", "pedestrian", "pedestrians",
                 "guy", "lady", "human", "someone", "kid", "worker", "officer",
                 "she", "he"}


def calibrate_relevance(sim: float) -> float:
    """Map a raw CLIP cosine similarity to a readable 0-1 relevance for display."""
    lo, hi = config.CLIP_REL_LOW, config.CLIP_REL_HIGH
    if hi <= lo:
        return max(0.0, min(1.0, sim))
    return max(0.0, min(1.0, (sim - lo) / (hi - lo)))


def infer_object_type(query: str) -> str | None:
    """'vehicle' | 'person' | None from the query wording (only when unambiguous)."""
    tokens = set(re.findall(r"[a-z]+", (query or "").lower()))
    veh = bool(tokens & _VEHICLE_WORDS)
    per = bool(tokens & _PERSON_WORDS)
    if veh and not per:
        return "vehicle"
    if per and not veh:
        return "person"
    return None


def media_url(path) -> str | None:
    """Turn an on-disk crop/frame path into a /media/... URL the API serves."""
    if not path:
        return None
    try:
        rel = Path(path).resolve().relative_to(config.DATA_DIR.resolve())
        return "/media/" + str(rel).replace("\\", "/")
    except (ValueError, OSError):
        return None


def _camera_names() -> dict:
    return {c["camera_id"]: c.get("name") for c in database.list_cameras()}


def _video_index() -> dict:
    return database.video_index()


def playback_fields(det: dict, vindex: dict) -> dict:
    """Resolve a detection to its recording clip + the exact seek offset so the
    UI can jump straight to that moment. offset = frame / native_fps (falls back
    to timestamp - clip start)."""
    vid = det.get("video_id")
    v = vindex.get(vid) if vid is not None else None
    if not v:
        return {}
    offset = None
    native_fps = v.get("native_fps")
    if det.get("frame_number") is not None and native_fps:
        offset = det["frame_number"] / native_fps
    elif det.get("timestamp") and v.get("start_time"):
        try:
            offset = (datetime.fromisoformat(det["timestamp"])
                      - datetime.fromisoformat(v["start_time"])).total_seconds()
        except ValueError:
            offset = None
    return {
        "video_id": vid,
        "video_url": f"/media/videos/{v['filename']}" if v.get("filename") else None,
        "offset_seconds": round(offset, 3) if offset is not None else None,
        "frame_width": v.get("width"),
        "frame_height": v.get("height"),
    }


def to_result_item(det: dict, score: float, cam_names: dict, vindex: dict | None = None) -> ResultItem:
    bbox = None
    if det.get("bbox_x") is not None:
        bbox = [det["bbox_x"], det["bbox_y"], det["bbox_w"], det["bbox_h"]]
    if vindex is None:
        vindex = _video_index()
    pb = playback_fields(det, vindex)
    return ResultItem(
        detection_id=det["detection_id"],
        camera_id=det.get("camera_id"),
        camera_name=cam_names.get(det.get("camera_id")),
        timestamp=det.get("timestamp"),
        class_label=det.get("class_label"),
        confidence=det.get("confidence") or 0.0,
        score=float(score),
        crop_url=media_url(det.get("crop_path")),
        bbox=bbox,
        attributes=det.get("attributes") or {},
        track_id=det.get("track_id"),
        visible_from=det.get("_visible_from"),
        visible_until=det.get("_visible_until"),
        track_appearances=det.get("_track_appearances"),
        video_id=pb.get("video_id"),
        video_url=pb.get("video_url"),
        offset_seconds=pb.get("offset_seconds"),
        frame_width=pb.get("frame_width"),
        frame_height=pb.get("frame_height"),
    )


def search_text(query: str, filters_obj: SearchFilters | None = None,
                top_k: int | None = None, include_scenes: bool = True,
                translated_query: str | None = None) -> SearchResponse:
    """Run a descriptive text search and return a ranked SearchResponse."""
    top_k = top_k or config.DEFAULT_TOP_K
    filters_obj = filters_obj or SearchFilters()

    clip_query = translated_query or query
    qvec = embedder.embed_text(clip_query)

    # If the query clearly names a vehicle or a person (and the caller hasn't set
    # object_type), bias results to that class so e.g. "a motorcycle" can't return
    # people. An explicit filter always wins.
    inferred_type = None
    if not filters_obj.object_type:
        inferred_type = infer_object_type(clip_query)
        if inferred_type:
            filters_obj = filters_obj.model_copy(update={"object_type": inferred_type})

    # Pull a larger candidate pool so filtering + de-dup still leave enough results.
    pool = min(max(top_k * 10, 200), 3000)
    ids, scores = vector_store.search("clip", qvec, top_k=pool)
    score_by_id = dict(zip(ids, scores))

    dets = database.get_detections(ids)                 # preserves FAISS rank order
    if not include_scenes:
        dets = [d for d in dets if d.get("class_label") != "scene"]
    dets = filters.apply_filters(dets, filters_obj)
    dets = result_utils.dedupe_detections(dets)         # one card per (camera, track)

    cam_names = _camera_names()
    vindex = _video_index()
    top_dets = dets[:top_k]
    results = []
    for d in top_dets:
        raw = float(score_by_id.get(d["detection_id"], 0.0))
        item = to_result_item(d, calibrate_relevance(raw), cam_names, vindex)
        item.raw_score = round(raw, 4)
        results.append(item)

    result_utils.apply_relative_scores(results)         # relative % within this set

    # No-strong-match guidance: honest signal when nothing really matched.
    top_raw = max((float(score_by_id.get(d["detection_id"], 0.0)) for d in top_dets), default=0.0)
    note = None
    if not results:
        note = "No matches. Try different wording or clear some filters."
    elif top_raw < config.CLIP_MIN_RELEVANT:
        note = "No strong match for this query - showing the closest results."

    database.log_audit("search", query_text=query, query_type="text",
                       result_count=len(results))
    return SearchResponse(query=query, translated_query=translated_query,
                          total=len(results), results=results, note=note,
                          object_type=inferred_type)


if __name__ == "__main__":
    tests = [
        ("a white truck", SearchFilters(), True),
        ("a car", SearchFilters(cameras=["CAM-01"]), False),
        ("a person", SearchFilters(object_type="person", colors=["red"]), False),
        ("a vehicle", SearchFilters(object_type="vehicle"), False),
    ]
    for q, f, scenes in tests:
        resp = search_text(q, f, top_k=3, include_scenes=scenes)
        active = {k: v for k, v in f.model_dump().items() if v not in (None, [], 0.0)}
        print(f"\nquery={q!r} filters={active} -> {resp.total} results")
        for r in resp.results:
            t = r.timestamp[11:19] if r.timestamp else "?"
            print(f"  {r.score:.3f} {r.class_label:<7} cam={r.camera_id} t={t} "
                  f"appears={r.track_appearances} attrs={r.attributes} url={r.crop_url}")
