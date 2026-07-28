"""Shared post-processing for search results.

Two steps every search endpoint (text / describe / image) runs:

  dedupe_detections   - a moving object is detected in dozens of consecutive
                        sampled frames, so collapse results to ONE per
                        (camera_id, track_id): the highest-ranked detection of
                        that track, annotated with how long it was visible and
                        how many detections it had.

  apply_relative_scores - CLIP cosine sims saturate (everything "looks 100%").
                        Instead show a RELATIVE score within the result set:
                        the best match ~97%, the rest scaled down by their raw
                        similarity and (for describe search) how many hard
                        constraints they matched.

Both helpers work on either ResultItem objects or plain dicts.
"""
from __future__ import annotations


def _get(it, key, default=None):
    return it.get(key, default) if isinstance(it, dict) else getattr(it, key, default)


def _set(it, key, value):
    if isinstance(it, dict):
        it[key] = value
    else:
        setattr(it, key, value)


def dedupe_detections(dets: list[dict]) -> list[dict]:
    """dets: detection dicts in ranked (best-first) order.

    Returns one dict per (camera_id, track_id) - the first (= highest-ranked,
    since the input is CLIP-ranked) - annotated with:
        _visible_from / _visible_until  (min / max timestamp of the track)
        _track_appearances              (how many detections the track had)
    Untracked rows (track_id is None, e.g. 'scene' frames) pass through as-is.
    """
    groups: dict = {}
    for d in dets:
        tid = d.get("track_id")
        if tid is None:
            continue
        groups.setdefault((d.get("camera_id"), tid), []).append(d)

    out, seen = [], set()
    for d in dets:
        tid = d.get("track_id")
        if tid is None:
            out.append(d)
            continue
        key = (d.get("camera_id"), tid)
        if key in seen:
            continue
        seen.add(key)
        grp = groups[key]
        times = sorted(x.get("timestamp") for x in grp if x.get("timestamp"))
        d = dict(d)                       # first occurrence = best rank for this track
        d["_visible_from"] = times[0] if times else d.get("timestamp")
        d["_visible_until"] = times[-1] if times else d.get("timestamp")
        d["_track_appearances"] = len(grp)
        out.append(d)
    return out


def apply_relative_scores(results):
    """Rewrite each result's `score` to a relative 0-1 within this set.

    best (highest raw CLIP) -> ~0.97, others scaled by raw ratio; for describe
    search the number of matched hard constraints is blended in so a result that
    ticks more boxes ranks higher than one that merely looks similar."""
    if not results:
        return results
    raws = [(_get(r, "raw_score") or 0.0) for r in results]
    top = max(raws) if raws else 0.0
    matched_counts = [len(_get(r, "matched") or []) for r in results]
    max_matched = max(matched_counts) if matched_counts else 0

    for r, mc in zip(results, matched_counts):
        raw = _get(r, "raw_score") or 0.0
        vis = (raw / top) if top > 0 else 0.0            # 0..1 vs the best in set
        score = 0.55 + 0.42 * vis                        # visual component (55%..97%)
        if max_matched > 0:                              # describe: weight by constraints met
            constraint = 0.55 + 0.42 * (mc / max_matched)
            score = 0.5 * score + 0.5 * constraint
        _set(r, "score", round(min(score, 0.98), 3))
    return results
