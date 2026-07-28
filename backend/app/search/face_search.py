"""Face search (InsightFace). Bonus feature, ethics-gated.

Upload a face -> detect + embed -> search the 'face' FAISS index -> map each
matching face_id back to its source detection -> ResultItem (with age/gender).

    python -m app.search.face_search   # self-test on an ingested face's crop
"""
from __future__ import annotations

from .. import config, database
from ..models.schemas import SearchResponse
from ..ingestion import face_recognizer
from . import vector_store
from .text_search import to_result_item, _camera_names, _video_index


def search_by_face(image, top_k: int | None = None, threshold: float | None = None) -> SearchResponse:
    top_k = top_k or config.DEFAULT_TOP_K
    if not config.FACE_RECOGNITION_ENABLED:
        return SearchResponse(query="<face search disabled>", total=0, results=[])

    faces = face_recognizer.detect_faces(image)
    if not faces:
        return SearchResponse(query="<no face detected>", total=0, results=[])

    face = max(faces, key=lambda f: f["det_score"])         # most confident face
    thr = threshold if threshold is not None else config.FACE_SIM_THRESHOLD

    ids, scores = vector_store.search("face", face["embedding"], top_k=top_k * 3)
    score_by_face = dict(zip(ids, scores))
    face_rows = database.get_faces(ids)
    dets = {d["detection_id"]: d
            for d in database.get_detections([r["detection_id"] for r in face_rows])}
    cam_names = _camera_names()
    vindex = _video_index()

    results = []
    for r in face_rows:
        sc = score_by_face.get(r["face_id"], 0.0)
        if sc < thr:
            continue
        d = dets.get(r["detection_id"])
        if not d:
            continue
        item = to_result_item(d, sc, cam_names, vindex)
        item.attributes = {**(item.attributes or {}),
                           "age": r.get("age"), "gender": r.get("gender")}
        results.append(item)
        if len(results) >= top_k:
            break

    database.log_audit("search", query_type="face", result_count=len(results))
    return SearchResponse(query="<uploaded face>", total=len(results), results=results)


if __name__ == "__main__":
    print("FACE_RECOGNITION_ENABLED:", config.FACE_RECOGNITION_ENABLED)
    print("faces in DB:", database.count_faces())
    with database.get_conn() as conn:
        row = conn.execute(
            "SELECT f.face_id, d.crop_path FROM faces f "
            "JOIN detections d ON f.detection_id = d.detection_id LIMIT 1"
        ).fetchone()
    if not row:
        print("No faces in DB - re-ingest with faces enabled (CAM-04 has faces).")
    else:
        print(f"query using source crop of face {row['face_id']}: {row['crop_path']}")
        resp = search_by_face(row["crop_path"], top_k=5)
        print(f"face search -> {resp.total} results")
        for r in resp.results:
            print(f"  {r.score:.3f} cam={r.camera_id} det={r.detection_id} "
                  f"age={r.attributes.get('age')} gender={r.attributes.get('gender')}")
