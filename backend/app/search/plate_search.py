"""License plate search.

A full or partial plate string -> normalised SQL substring match on the plates
table -> map each hit back to its source detection -> ranked ResultItem list.
Plates are short strings, so this is a metadata (SQL) search, not a vector search.

    python -m app.search.plate_search   # self-test on an ingested plate (if any)
"""
from __future__ import annotations

from .. import config, database
from ..models.schemas import SearchFilters, SearchResponse
from .text_search import to_result_item, _camera_names, _video_index


def search_by_plate(plate: str, filters_obj: SearchFilters | None = None,
                    top_k: int | None = None) -> SearchResponse:
    top_k = top_k or config.DEFAULT_TOP_K
    filters_obj = filters_obj or SearchFilters()
    if not config.PLATE_RECOGNITION_ENABLED:
        return SearchResponse(query=plate, total=0, results=[])

    query = (plate or "").strip()
    if not query:
        return SearchResponse(query=plate, total=0, results=[])

    rows = database.search_plates(
        query, camera_ids=filters_obj.cameras,
        start_time=filters_obj.start_time, end_time=filters_obj.end_time,
        limit=top_k * 3,
    )
    dets = {d["detection_id"]: d
            for d in database.get_detections([r["detection_id"] for r in rows])}
    cam_names = _camera_names()
    vindex = _video_index()

    results = []
    for r in rows:
        d = dets.get(r["detection_id"])
        if not d:
            continue
        item = to_result_item(d, float(r.get("confidence") or 0.0), cam_names, vindex)
        item.attributes = {**(item.attributes or {}),
                           "plate_text": r.get("plate_text"),
                           "plate_confidence": r.get("confidence")}
        results.append(item)
        if len(results) >= top_k:
            break

    database.log_audit("search", query_text=plate, query_type="plate",
                       result_count=len(results))
    return SearchResponse(query=plate, total=len(results), results=results)


if __name__ == "__main__":
    print("PLATE_RECOGNITION_ENABLED:", config.PLATE_RECOGNITION_ENABLED)
    print("plates in DB:", database.count_plates())
    with database.get_conn() as conn:
        row = conn.execute("SELECT plate_text FROM plates LIMIT 1").fetchone()
    if not row:
        print("No plates in DB (sample clips likely have no readable plates). "
              "Run scripts/verify_plates.py for a dataset-independent check.")
    else:
        sample = row["plate_text"]
        print(f"searching for a partial of {sample!r}")
        resp = search_by_plate(sample[-4:], top_k=5)
        print(f"plate search -> {resp.total} results")
        for r in resp.results:
            print(f"  {r.attributes.get('plate_text')} cam={r.camera_id} det={r.detection_id}")
