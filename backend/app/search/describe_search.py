"""Describe-and-filter search.

The officer types a plain description; we parse it into constraints, run the
normal CLIP visual search, de-duplicate to one card per track, keep only
detections that satisfy every HARD constraint, then rank relatively. Each result
is annotated with which parts it matched, so the result is provable.

If nothing satisfies every hard constraint we fall back to the closest visual
matches (still annotated) rather than showing an empty screen.
"""
from __future__ import annotations

from .. import config, database
from ..models.schemas import SearchFilters
from ..ingestion import embedder
from . import vector_store, filters as filt, query_parser, result_utils
from .text_search import to_result_item, _camera_names, _video_index, calibrate_relevance


def _has_hard(parsed: dict) -> bool:
    return any([parsed["object_type"], parsed["upper_color"], parsed["lower_color"],
                parsed["vehicle_color"], parsed["vehicle_type"],
                any(a in query_parser._EXTRACTED_ACCESSORIES for a in parsed["accessories"])])


def search(query: str, filters_obj: SearchFilters | None = None, top_k: int | None = None) -> dict:
    top_k = top_k or config.DEFAULT_TOP_K
    filters_obj = filters_obj or SearchFilters()
    parsed = query_parser.parse(query)

    qvec = embedder.embed_text(query)
    pool = min(max(top_k * 10, 300), 3000)
    ids, scores = vector_store.search("clip", qvec, top_k=pool)
    score_by_id = dict(zip(ids, scores))

    dets = database.get_detections(ids)                     # preserves CLIP rank
    dets = [d for d in dets if d.get("class_label") != "scene"]
    dets = filt.apply_filters(dets, filters_obj)            # camera / time / video base filters
    dets = result_utils.dedupe_detections(dets)            # one card per (camera, track)

    cam_names, vindex = _camera_names(), _video_index()
    strict, loose = [], []
    for d in dets:
        ev = query_parser.evaluate(d, parsed)
        raw = float(score_by_id.get(d["detection_id"], 0.0))
        item = to_result_item(d, calibrate_relevance(raw), cam_names, vindex).model_dump()
        item["raw_score"] = round(raw, 4)
        item["matched"] = ev["matched"]
        item["soft"] = ev["soft"]
        (strict if ev["passed"] else loose).append(item)

    note = None
    if strict:
        results = strict[:top_k]
    elif _has_hard(parsed):
        results = loose[:top_k]
        note = ("No detection matched every requirement - showing the closest visual "
                "matches. The ticks show what each one does match; drop a filter to broaden.")
    else:
        results = loose[:top_k]

    result_utils.apply_relative_scores(results)             # relative % within this set

    database.log_audit("describe_search", query_text=query, query_type="describe",
                       result_count=len(results))
    return {
        "query": query,
        "parsed": parsed,
        "chips": query_parser.to_chips(parsed),
        "total": len(results),
        "strict_total": len(strict),
        "results": results,
        "note": note,
    }
