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


def _norm(s: str) -> str:
    """Uppercase, keep only letters/digits (drops spaces / hyphens / IND, etc.)."""
    return "".join(ch for ch in (s or "").upper() if ch.isalnum())


def _match_score(q: str, p: str) -> float:
    """How well query `q` matches plate `p` (both normalised).

    1.0  -> q is an exact substring of p (covers full plate AND partials like the
            last 4 digits).
    <1.0 -> best character-for-character overlap of a q-length window inside p, so
            OCR-noisy reads (e.g. 'GJ21OC641S' when searching '6419') still surface
            as *probable* matches. Returns 0 when nothing lines up."""
    if not q or not p:
        return 0.0
    if q in p:
        return 1.0
    lq, lp = len(q), len(p)
    if lq <= lp:
        best = 0
        for i in range(lp - lq + 1):
            seg = p[i:i + lq]
            m = sum(1 for a, b in zip(q, seg) if a == b)
            if m > best:
                best = m
        return best / lq
    # query longer than the plate -> overlap against the whole plate
    m = sum(1 for a, b in zip(q, p) if a == b)
    return m / lq


def search_by_plate(plate: str, filters_obj: SearchFilters | None = None,
                    top_k: int | None = None) -> SearchResponse:
    top_k = top_k or config.DEFAULT_TOP_K
    filters_obj = filters_obj or SearchFilters()
    if not config.PLATE_RECOGNITION_ENABLED:
        return SearchResponse(query=plate, total=0, results=[])

    query = (plate or "").strip()
    qn = _norm(query)
    if not qn:
        return SearchResponse(query=plate, total=0, results=[])

    # Pull every stored plate (respecting camera/time filters) and rank them by
    # similarity to the query. Exact substring matches (score 1.0) come first,
    # then probable/fuzzy matches - so a partial or a slightly-misread plate still
    # returns the likely vehicles. The plates table is small, so scoring in Python
    # is cheap and lets us tolerate OCR confusions a plain SQL LIKE can't.
    rows = database.search_plates(
        "", camera_ids=filters_obj.cameras,
        start_time=filters_obj.start_time, end_time=filters_obj.end_time,
        limit=100000,
    )

    # Tiny queries (<=2 chars) would fuzzy-match almost anything, so require an
    # exact substring there; longer queries allow ~1 error per 3 characters.
    threshold = 1.0 if len(qn) <= 2 else config.PLATE_FUZZY_THRESHOLD

    scored = []
    for r in rows:
        s = _match_score(qn, _norm(r.get("plate_text")))
        if s >= threshold:
            scored.append((s, float(r.get("confidence") or 0.0), r))
    # best match first; ties broken by OCR confidence
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

    dets = {d["detection_id"]: d
            for d in database.get_detections([r["detection_id"] for _s, _c, r in scored])}
    cam_names = _camera_names()
    vindex = _video_index()

    results = []
    seen = set()                       # one card per vehicle detection (a track may
                                       # store several plate reads -> same detection)
    for s, _c, r in scored:
        did = r["detection_id"]
        if did in seen:
            continue
        d = dets.get(did)
        if not d:
            continue
        seen.add(did)
        # display score = match quality (exact=100%, probable=lower) so the UI
        # badge tells the investigator how confident the plate match is.
        item = to_result_item(d, s, cam_names, vindex)
        item.attributes = {**(item.attributes or {}),
                           "plate_text": r.get("plate_text"),
                           "plate_confidence": r.get("confidence"),
                           "plate_match": round(s, 3),
                           "plate_frames": r.get("votes"),      # supporting frames (ANPR)
                           "plate_source": r.get("source")}
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
