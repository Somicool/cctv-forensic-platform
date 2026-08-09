"""Single-camera track path for the interactive tracking viewer.

Given any detection, return the full per-frame bounding-box trajectory of its
ByteTrack track WITHIN its own recording - reusing the boxes, timestamps and
track ids stored during ingestion. No detection / tracking / ReID / OCR / CLIP
is re-run here; this only reads indexed metadata, so playback is instant and
cheap.

Kept as its own module (separate from cross_camera.py) so cross-camera replay
can be layered on later without touching this: a future viewer could request
one path per camera and stitch them into a single timeline, and this function's
response shape already carries the camera + clip identity needed for that.

    python -m app.search.track_path        # self-test on an ingested track
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .. import config, database
from ..models.schemas import TrackPathPoint, TrackPathResponse
from . import vector_store
from .text_search import media_url, _video_index, playback_fields, _camera_names

# --- identity-preserving playback tunables ---
# The target reference is a SET of high-quality views of the selected track, not the
# single clicked crop. One crop is a weak yardstick: an unlucky pose sets the bar for
# the whole track, and at the old 0.55 break threshold roughly half of
# DIFFERENT-person frames still cleared it (measured impostor p90 was ~0.67), so the
# box could follow a visually similar neighbour.
#
# Measured over 198 real tracks, comparing each frame against a reference SET:
#   refs   AUC     impostor p90   genuine frames kept at p90
#   1      0.916   0.643          78.4%     <- single clicked crop, the permissive case
#   5      0.942   0.674          80.3%     <- default
APPEARANCE_BREAK_SIM = 0.65      # below this the frame is a different person: stop
# At/above this it is confidently the same person: keep it even if the box jumped
# (the target reappeared from behind an obstacle). Above the measured same-person
# mean (0.743), so only strong evidence overrides motion continuity.
APPEARANCE_HOLD_SIM = 0.80
REFERENCE_VIEWS = 5              # max views in the target reference
# Quality gates for a view entering the reference. A blurred or tiny crop describes
# the scene more than the person, so it must not define the target.
REF_MIN_SHARPNESS = 25.0         # Laplacian variance floor
REF_MIN_AREA_FRAC = 0.45         # at least this fraction of the track's median area
REF_MAX_SIMILARITY = 0.97        # reject near-duplicates so the set spans poses
REF_MAX_CANDIDATES = 24          # bound the crop reads per request
# --- gap bridging limits -----------------------------------------------------
# Interpolated boxes exist so a ONE-frame miss (a passing obstruction, a dip in
# detector confidence) does not make the overlay blink out. They are guesses, so
# they are strictly bounded:
#
# Measured on a busy clip when this was 4.0 s with no distance limit: 18.3% of all
# boxes drawn were invented, some tracks were 50% invented, and the longest guess
# glided 692 px - 31% of the frame diagonal - in a straight line over 3.5 s. In a
# crowded scene a box travelling that far sits on OTHER PEOPLE for most of its
# journey, which is exactly the "it shows a different person" failure.
#
# A gap is now only filled when it is short AND the person barely moved across it.
# Where the path is genuinely unknown the box is hidden instead of guessed.
PREDICT_MAX_GAP_S = 1.5          # at 2 fps this is ~3 missed samples
PREDICT_MAX_DRIFT_FRAC = 0.06    # endpoints must be within 6% of the frame diagonal
PREDICT_MAX_RUN = 2              # never invent more than this many boxes in a row


def _center(bbox):
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def _unit(v):
    v = np.asarray(v, dtype="float32").ravel()
    n = float(np.linalg.norm(v))
    return None if n < 1e-6 else v / n


def _sharpness(path):
    """Laplacian variance of a stored crop. Cheap: no model, just the saved image."""
    try:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None or not im.size:
            return None
        return float(cv2.Laplacian(im, cv2.CV_64F).var())
    except Exception:
        return None


def reference_views(rows, ref_detection_id, max_views: int | None = None):
    """Target identity reference: up to `max_views` good, visually varied views.

    The clicked detection always anchors the set - it is what the investigator
    chose - and the rest are the best remaining views of the SAME stored track.

    Selection, in order:
      1. reject crops below the sharpness floor or well under the track's median
         size (a blurred or distant crop describes the scene, not the person)
      2. rank the survivors by sharpness x relative size x detector confidence
      3. take them greedily, skipping any whose embedding is a near-duplicate of one
         already chosen, so the set spans poses/distances instead of one moment

    Built ONCE per request from stored embeddings and stored crops. Nothing is
    learned from playback frames: continuously absorbing frames is how a tracker
    drifts onto the wrong person.
    """
    # resolved at call time, not import time, so the module setting can be overridden
    max_views = REFERENCE_VIEWS if max_views is None else max_views
    have = [r for r in rows if r.get("bbox_w") and r.get("bbox_h")
            and vector_store.get_vector("reid", r["detection_id"]) is not None]
    if not have or max_views <= 1:
        return [ref_detection_id], {}

    areas = sorted((r["bbox_w"] * r["bbox_h"]) for r in have)
    median_area = areas[len(areas) // 2] or 1.0
    # only read crops for the most promising candidates, biggest first
    have.sort(key=lambda r: r["bbox_w"] * r["bbox_h"], reverse=True)
    scored, diag = [], {}
    for r in have[:REF_MAX_CANDIDATES]:
        area = r["bbox_w"] * r["bbox_h"]
        rel = area / median_area
        sharp = _sharpness(r.get("crop_path")) if r.get("crop_path") else None
        diag[r["detection_id"]] = {"sharpness": sharp, "area_frac": round(rel, 3),
                                   "confidence": r.get("confidence")}
        if rel < REF_MIN_AREA_FRAC:
            continue
        if sharp is not None and sharp < REF_MIN_SHARPNESS:
            continue
        conf = float(r.get("confidence") or 0.5)
        quality = (sharp if sharp is not None else REF_MIN_SHARPNESS) * min(rel, 3.0) * conf
        scored.append((quality, r["detection_id"]))
    scored.sort(reverse=True)

    # Only vectors of the expected width are usable (see _appearance_check).
    want_dim = config.REID_DIM

    chosen, vecs = [], []
    anchor = _unit(vector_store.get_vector("reid", ref_detection_id))
    if anchor is not None and anchor.shape[0] == want_dim:
        chosen.append(ref_detection_id)
        vecs.append(anchor)
    for _q, did in scored:
        if len(chosen) >= max_views:
            break
        if did in chosen:
            continue
        v = _unit(vector_store.get_vector("reid", did))
        if v is None or v.shape[0] != want_dim:
            continue
        if vecs and max(float(np.dot(v, u)) for u in vecs) > REF_MAX_SIMILARITY:
            continue                                  # near-duplicate view
        chosen.append(did)
        vecs.append(v)
    if not chosen:
        chosen = [ref_detection_id]
    return chosen, diag


def _appearance_check(points, ref_detection_id, ref_ids=None):
    """Per-point ReID similarity to the TARGET REFERENCE SET.

    Motion continuity alone cannot tell "the same person walked on" from "a
    different person walked into the same place"; the stored embeddings can. Each
    frame is scored best-of-set against the reference, softened by the set mean so
    one lucky view cannot carry a bad match.

    Returns detection_id -> similarity, or {} when embeddings are unavailable (the
    caller then falls back to motion only).
    """
    raw = {}
    for p in points:
        v = _unit(vector_store.get_vector("reid", p.detection_id))
        if v is not None:
            raw[p.detection_id] = v
    if not raw:
        return {}
    # Keep only vectors of the EXPECTED width. A large number of stored ReID vectors
    # are malformed (width 1 instead of config.REID_DIM), and on some tracks they are
    # the MAJORITY - so neither "the first vector seen" nor "the most common width"
    # identifies a usable embedding. Getting this wrong made _appearance_check return
    # nothing, which silently dropped Track Person to motion-only tracking: no
    # identity verification (the box could follow a neighbour) and a clip truncated at
    # the first motion break.
    dim = config.REID_DIM
    vecs = {d: v for d, v in raw.items() if v.shape[0] == dim}
    if not vecs:
        return {}

    ids = [d for d in (ref_ids or [ref_detection_id]) if d in vecs]
    if not ids:
        # The chosen reference views have no usable vector here. Returning nothing
        # would drop Track Person to motion-only tracking - no identity check at all,
        # so the box may follow a neighbour and the clip stops at the first motion
        # break. Fall back to the track's own centroid instead, which is a weaker but
        # real identity reference.
        M = np.vstack(list(vecs.values())).reshape(-1, dim)
        centroid = M.mean(axis=0)
        n = float(np.linalg.norm(centroid))
        if n < 1e-6:
            return {}
        R = (centroid / n).reshape(1, dim)
    else:
        R = np.vstack([vecs[d] for d in ids]).reshape(-1, dim)

    out = {}
    for did, v in vecs.items():
        s = R @ v
        out[did] = float(0.7 * s.max() + 0.3 * s.mean()) if len(s) > 1 else float(s.max())
    return out


def _bridge_gaps(points, native_fps, stride_s, frame_w=None, frame_h=None):
    """Fill only the short, low-movement holes in a track; hide the rest.

    A person momentarily occluded or scored below threshold leaves a hole. Filling
    it keeps the overlay attached to the target. But a guess is only defensible
    while it stays near a real sighting, so a gap is bridged only when ALL of these
    hold (see the constants above for the measurements behind them):

      * it is no longer than PREDICT_MAX_GAP_S
      * the two real endpoints are within PREDICT_MAX_DRIFT_FRAC of each other
      * it needs no more than PREDICT_MAX_RUN invented boxes

    Otherwise the gap is left empty and the viewer simply shows no box - honest
    about not knowing, rather than drawing a straight line over other people.
    """
    if len(points) < 2 or stride_s <= 0:
        return points
    diag = math.hypot(frame_w or 1920, frame_h or 1080)
    max_drift = PREDICT_MAX_DRIFT_FRAC * diag
    max_steps = min(PREDICT_MAX_RUN, max(0, int(PREDICT_MAX_GAP_S / stride_s)))

    out = [points[0]]
    for a, b in zip(points, points[1:]):
        gap = b.offset_seconds - a.offset_seconds
        steps = int(round(gap / stride_s)) - 1
        if 0 < steps <= max_steps and gap <= PREDICT_MAX_GAP_S:
            ca, cb = _center(a.bbox), _center(b.bbox)
            drift = math.hypot(cb[0] - ca[0], cb[1] - ca[1])
            if drift <= max_drift:
                for k in range(1, steps + 1):
                    f = k / (steps + 1)
                    out.append(TrackPathPoint(
                        detection_id=a.detection_id,      # provenance: the real sighting
                        offset_seconds=round(a.offset_seconds + f * gap, 3),
                        frame_number=(int(a.frame_number + f * (b.frame_number - a.frame_number))
                                      if a.frame_number is not None and b.frame_number is not None
                                      else None),
                        timestamp=None,
                        bbox=[round(a.bbox[i] + f * (b.bbox[i] - a.bbox[i]), 2) for i in range(4)],
                        confidence=None, predicted=True))
        out.append(b)
    return out


def _target_frames(points, ref_detection_id, frame_w, frame_h, appearance):
    """Every frame of this track that is the TARGET - not just the run around the click.

    Why this replaced a contiguous-run trim
    ---------------------------------------
    The old logic walked outward from the clicked detection and stopped at the first
    break, then derived the clip's start/end from what survived. Measured on the four
    test cameras, 5 of 21 tracks were truncated inside the person's visible span - one
    returned 0.00-5.51 s of a 0.00-16.53 s appearance (33%). A single missed or
    ambiguous frame threw away everything after it.

    A break means "this frame is someone else", not "the target is gone for good". So
    every frame is judged on its own identity evidence and the whole target span is
    kept, with the impostor stretch simply excluded. The clip therefore runs from the
    target's FIRST valid appearance to its LAST, while no box is ever drawn on a frame
    that failed the identity check.

    Frames with no stored embedding fall back to motion continuity against the last
    accepted frame, so a missing vector does not silently drop a real sighting.
    """
    if not points:
        return points
    diag = math.hypot(frame_w or 1920, frame_h or 1080)
    gaps = [b.offset_seconds - a.offset_seconds
            for a, b in zip(points, points[1:]) if b.offset_seconds > a.offset_seconds]
    gaps.sort()
    stride = max(gaps[len(gaps) // 2] if gaps else 1.0, 1e-3)

    def motion_ok(prev, cur) -> bool:
        if prev is None:
            return True
        dt = cur.offset_seconds - prev.offset_seconds
        if dt > 12.0:
            return False
        ca, cb = _center(prev.bbox), _center(cur.bbox)
        dist = math.hypot(cb[0] - ca[0], cb[1] - ca[1])
        steps = min(max(1.0, dt / stride), 4.0)
        return dist <= 0.35 * diag * steps

    kept, prev = [], None
    for p in points:
        sim = appearance.get(p.detection_id)
        if sim is None:
            ok = motion_ok(prev, p)                   # no embedding: motion decides
        elif sim >= APPEARANCE_HOLD_SIM:
            ok = True                                 # confidently the target
        elif sim < APPEARANCE_BREAK_SIM:
            ok = False                                # confidently someone else
        else:
            ok = motion_ok(prev, p)                   # ambiguous band
        if p.detection_id == ref_detection_id:
            ok = True                                 # the investigator's own choice
        if ok:
            kept.append(p)
            prev = p
    return kept or [p for p in points if p.detection_id == ref_detection_id] or points


def _contiguous_segment(points, ref_detection_id, frame_w, frame_h, appearance=None):
    """Keep only the run of the track that is spatially/temporally continuous with
    the clicked detection.

    ByteTrack occasionally re-uses one track_id for more than one physical object
    (an "ID switch"): the box then teleports across the frame mid-playback. We
    detect those teleports (a large centroid jump, or a long time gap) and return
    only the segment that contains the detection the user actually clicked, so the
    box stays on that one object for its whole on-screen lifetime.
    """
    if len(points) <= 1:
        return points

    # typical sampling interval (median positive gap) for scaling the thresholds
    gaps = [b.offset_seconds - a.offset_seconds
            for a, b in zip(points, points[1:]) if b.offset_seconds > a.offset_seconds]
    gaps.sort()
    stride = gaps[len(gaps) // 2] if gaps else 1.0
    stride = max(stride, 1e-3)
    diag = math.hypot(frame_w or 1920, frame_h or 1080)
    HARD_GAP_S = 12.0                           # gone this long -> a separate appearance

    appearance = appearance or {}

    def continuous(a, b) -> bool:
        # The ID-switch signal is a fast spatial TELEPORT, not a mere time gap:
        # an object that is briefly occluded (or missed) should still reconnect as
        # long as it reappears near where it was heading. So we gate on SPEED, with
        # the allowance capped so a big jump after a long gap still splits.
        dt = b.offset_seconds - a.offset_seconds
        if dt > HARD_GAP_S:
            return False
        # Appearance overrides motion in BOTH directions. A box that no longer
        # looks like the clicked person ends the segment even if it moved
        # plausibly; a confident appearance match survives a jump that motion
        # alone would have called a teleport.
        sim = appearance.get(b.detection_id)
        if sim is not None:
            if sim < APPEARANCE_BREAK_SIM:
                return False
            if sim >= APPEARANCE_HOLD_SIM:
                return True
        ca, cb = _center(a.bbox), _center(b.bbox)
        dist = math.hypot(cb[0] - ca[0], cb[1] - ca[1])
        steps = min(max(1.0, dt / stride), 4.0)   # cap so long gaps aren't a free teleport
        return dist <= 0.35 * diag * steps        # ~35% of the diagonal per sampled step

    # anchor on the clicked detection (fallback: middle point)
    ref_i = next((i for i, p in enumerate(points) if p.detection_id == ref_detection_id), None)
    if ref_i is None:
        ref_i = len(points) // 2

    lo = ref_i
    while lo > 0 and continuous(points[lo - 1], points[lo]):
        lo -= 1
    hi = ref_i
    while hi < len(points) - 1 and continuous(points[hi], points[hi + 1]):
        hi += 1
    return points[lo:hi + 1]


def get_track_path(detection_id: int) -> TrackPathResponse:
    """Build the per-frame trajectory of the track that `detection_id` belongs to."""
    refs = database.get_detections([detection_id])
    if not refs:
        return TrackPathResponse(detection_id=detection_id, points=[])
    ref = refs[0]

    video_id = ref.get("video_id")
    track_id = ref.get("track_id")
    vindex = _video_index()
    cam_names = _camera_names()
    v = vindex.get(video_id) if video_id is not None else None

    # All rows of this (video, track); if the reference has no track id (rare),
    # fall back to just the reference detection so the viewer still works.
    rows = database.get_track_detections(video_id, track_id) or [ref]

    points = []
    for d in rows:
        if d.get("bbox_x") is None:                 # skip rows without a box
            continue
        pb = playback_fields(d, vindex)
        off = pb.get("offset_seconds")
        if off is None:                             # can't place it on the timeline
            continue
        conf = d.get("confidence")
        points.append(TrackPathPoint(
            detection_id=d["detection_id"],
            offset_seconds=off,
            frame_number=d.get("frame_number"),
            timestamp=d.get("timestamp"),
            bbox=[d["bbox_x"], d["bbox_y"], d["bbox_w"], d["bbox_h"]],
            confidence=float(conf) if conf is not None else None,
        ))
    points.sort(key=lambda p: p.offset_seconds)

    # Verify by APPEARANCE who each box actually is, so a box is never drawn on a
    # frame where the stored track was holding someone else.
    # Target identity reference: built once, from good views of the SELECTED track.
    stride_s = 0.0
    ref_ids, ref_diag = (reference_views(rows, detection_id)
                         if ref.get("class_label") == "person" else ([detection_id], {}))
    appearance = (_appearance_check(points, detection_id, ref_ids)
                  if ref.get("class_label") == "person" else {})
    fw = v.get("width") if v else None
    fh = v.get("height") if v else None
    if appearance:
        # identity evidence available: keep the target's WHOLE span
        points = _target_frames(points, detection_id, fw, fh, appearance)
    else:
        # no embeddings for this track: fall back to the motion-only contiguous run
        points = _contiguous_segment(points, detection_id, fw, fh)
    confs = [p.confidence for p in points if p.confidence is not None]
    real_points = len(points)

    # Fill short detection holes with interpolated boxes so a brief occlusion or a
    # dip in detector confidence does not look like losing the target.
    if len(points) > 1:
        gaps = sorted(b.offset_seconds - a.offset_seconds
                      for a, b in zip(points, points[1:])
                      if b.offset_seconds > a.offset_seconds)
        stride_s = gaps[len(gaps) // 2] if gaps else 0.0
        points = _bridge_gaps(points, v.get("native_fps") if v else None, stride_s,
                              v.get("width") if v else None,
                              v.get("height") if v else None)
    # Stretches inside the target's span with no box: the target is temporarily lost
    # (occluded, missed, or the track was holding someone else). Reported rather than
    # papered over with an interpolated box.
    lost_spans = []
    for a, b in zip(points, points[1:]):
        gap = b.offset_seconds - a.offset_seconds
        if gap > max(PREDICT_MAX_GAP_S, 2.5 * stride_s if stride_s else 0):
            lost_spans.append([round(a.offset_seconds, 2), round(b.offset_seconds, 2)])

    sims = [s for did, s in appearance.items()
            if did in {p.detection_id for p in points}]

    start_off = points[0].offset_seconds if points else None
    end_off = points[-1].offset_seconds if points else None
    duration = (round(end_off - start_off, 3)
                if start_off is not None and end_off is not None else None)

    return TrackPathResponse(
        detection_id=detection_id,
        video_id=video_id,
        video_url=f"/media/videos/{v['filename']}" if v and v.get("filename") else None,
        camera_id=ref.get("camera_id"),
        camera_name=cam_names.get(ref.get("camera_id")),
        track_id=track_id,
        class_label=ref.get("class_label"),
        frame_width=v.get("width") if v else None,
        frame_height=v.get("height") if v else None,
        native_fps=v.get("native_fps") if v else None,
        start_offset=start_off,
        end_offset=end_off,
        duration=duration,
        max_confidence=round(max(confs), 4) if confs else None,
        avg_confidence=round(sum(confs) / len(confs), 4) if confs else None,
        attributes=ref.get("attributes") or {},
        points=points,
        detected_points=real_points,
        predicted_points=sum(1 for p in points if p.predicted),
        identity_confidence=round(float(sum(sims) / len(sims)), 4) if sims else None,
        reference_views=list(ref_ids),
        identity_threshold=APPEARANCE_BREAK_SIM,
        lost_spans=lost_spans,
    )


if __name__ == "__main__":
    persons = database.query_detections(class_labels=["person"], limit=50)
    ref = next((p for p in persons if p.get("track_id") is not None), None)
    if not ref:
        print("No tracked person detections; run ingestion first.")
    else:
        resp = get_track_path(ref["detection_id"])
        print(f"track {resp.track_id} on {resp.camera_id}: {len(resp.points)} boxes, "
              f"span {resp.start_offset}->{resp.end_offset}s ({resp.duration}s), "
              f"conf max={resp.max_confidence}")
