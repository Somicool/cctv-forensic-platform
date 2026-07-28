"""Image / re-ID search.

Two modes:
  - visual (CLIP): upload any image -> visually similar detections
  - re-ID (OSNet): upload a PERSON photo -> the same person across cameras

Results are de-duplicated to one card per track and relatively ranked, same as
text search. Returns the SearchResponse shape.

    python -m app.search.image_search   # self-test on an ingested person crop
"""
from __future__ import annotations

from .. import config, database
from ..models.schemas import SearchFilters, SearchResponse
from ..ingestion import embedder, reid_embedder
from . import vector_store, filters, result_utils
from .text_search import to_result_item, _camera_names, _video_index


def search_by_image(image, filters_obj: SearchFilters | None = None,
                    top_k: int | None = None, use_reid: bool = False) -> SearchResponse:
    """Search by an uploaded image. use_reid=True routes to OSNet person re-ID,
    otherwise CLIP visual similarity."""
    top_k = top_k or config.DEFAULT_TOP_K
    filters_obj = filters_obj or SearchFilters()

    if use_reid:
        vec = reid_embedder.embed_person(image)
        index = "reid"
    else:
        vec = embedder.embed_image(image)
        index = "clip"

    pool = min(max(top_k * 10, 200), 3000)
    ids, scores = vector_store.search(index, vec, top_k=pool)
    score_by_id = dict(zip(ids, scores))
    dets = filters.apply_filters(database.get_detections(ids), filters_obj)
    dets = result_utils.dedupe_detections(dets)             # one card per (camera, track)

    cam_names = _camera_names()
    vindex = _video_index()
    results = []
    for d in dets[:top_k]:
        raw = float(score_by_id.get(d["detection_id"], 0.0))
        item = to_result_item(d, raw, cam_names, vindex)
        item.raw_score = round(raw, 4)
        results.append(item)

    result_utils.apply_relative_scores(results)             # relative % within this set

    database.log_audit("search", query_type="image_reid" if use_reid else "image_clip",
                       result_count=len(results))
    return SearchResponse(query="<uploaded image>", total=len(results), results=results)


if __name__ == "__main__":
    persons = database.query_detections(class_labels=["person"], limit=20)
    if not persons:
        print("No person detections; run scripts/ingest_all.py first.")
    else:
        ref = persons[0]
        print(f"reference person: det {ref['detection_id']} cam {ref['camera_id']}")
        resp = search_by_image(ref["crop_path"], top_k=5, use_reid=True)
        print(f"re-ID image search -> {resp.total} results (top should be the reference, ~1.0)")
        for r in resp.results:
            t = r.timestamp[11:19] if r.timestamp else "?"
            print(f"  {r.score:.3f} {r.class_label} cam={r.camera_id} det={r.detection_id} t={t}")
