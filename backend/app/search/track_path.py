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

import numpy as np

from .. import database
from ..models.schemas import TrackPathPoint, TrackPathResponse
from . import vector_store
from .text_search import media_url, _video_index, playback_fields, _camera_names

# --- identity-preserving playback tunables ---
# Below this the box is definitely a different person: end the segment.
APPEARANCE_BREAK_SIM = 0.55
# At/above this it is confidently the same person: keep it even if the box jumped
# (the target reappeared from behind an obstacle).
APPEARANCE_HOLD_SIM = 0.75
# Longest detection gap that may be filled with interpolated boxes, in seconds.
PREDICT_MAX_GAP_S = 4.0


def _center(bbox):
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def _appearance_check(points, ref_detection_id):
    """Per-point ReID similarity to the CLICKED person.

    Motion continuity alone cannot tell "the same person walked on" from "a
    different person walked into the same place". The stored person embeddings
    can, so the reference is compared against the rest of the track and any point
    that clearly is not the same person is treated as a break. Returns a dict of
    detection_id -> similarity, or {} when embeddings are unavailable (in which
    case the caller falls back to motion only)."""
    ids = [p.detection_id for p in points]
    ref_vec = vector_store.get_vector("reid", ref_detection_id)
    if ref_vec is None:
        return {}
    ref_vec = np.asarray(ref_vec, dtype="float32").ravel()
    n = float(np.linalg.norm(ref_vec))
    if n < 1e-6:
        return {}
    ref_vec = ref_vec / n
    out = {}
    for did in ids:
        v = vector_store.get_vector("reid", did)
        if v is None:
            continue
        v = np.asarray(v, dtype="float32").ravel()
        nv = float(np.linalg.norm(v))
        if nv < 1e-6:
            continue
        out[did] = float(np.dot(ref_vec, v / nv))
    return out


def _bridge_gaps(points, native_fps, stride_s):
    """Interpolate boxes across short detection gaps so the overlay stays locked.

    A person who is momentarily occluded, blurred or scored below threshold leaves
    a hole in the stored track. Rather than dropping the box (which looks like
    losing the target) the position is interpolated linearly between the two real
    sightings and flagged `predicted=True`. Only gaps up to PREDICT_MAX_GAP_S are
    filled: beyond that there is no honest basis for a box."""
    if len(points) < 2 or stride_s <= 0:
        return points
    out = [points[0]]
    for a, b in zip(points, points[1:]):
        gap = b.offset_seconds - a.offset_seconds
        steps = int(round(gap / stride_s)) - 1
        if 0 < steps <= int(PREDICT_MAX_GAP_S / stride_s):
            for k in range(1, steps + 1):
                f = k / (steps + 1)
                out.append(TrackPathPoint(
                    detection_id=a.detection_id,          # provenance: the real sighting
                    offset_seconds=round(a.offset_seconds + f * gap, 3),
                    frame_number=(int(a.frame_number + f * (b.frame_number - a.frame_number))
                                  if a.frame_number is not None and b.frame_number is not None
                                  else None),
                    timestamp=None,
                    bbox=[round(a.bbox[i] + f * (b.bbox[i] - a.bbox[i]), 2) for i in range(4)],
                    confidence=None, predicted=True))
        out.append(b)
    return out


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

    # Verify by APPEARANCE who each box actually is, then trim to the segment that
    # is continuous with the clicked detection, so the box never transfers to a
    # different person on a ByteTrack ID switch.
    appearance = (_appearance_check(points, detection_id)
                  if ref.get("class_label") == "person" else {})
    points = _contiguous_segment(points, detection_id,
                                 v.get("width") if v else None,
                                 v.get("height") if v else None,
                                 appearance=appearance)
    confs = [p.confidence for p in points if p.confidence is not None]
    real_points = len(points)

    # Fill short detection holes with interpolated boxes so a brief occlusion or a
    # dip in detector confidence does not look like losing the target.
    if len(points) > 1:
        gaps = sorted(b.offset_seconds - a.offset_seconds
                      for a, b in zip(points, points[1:])
                      if b.offset_seconds > a.offset_seconds)
        stride_s = gaps[len(gaps) // 2] if gaps else 0.0
        points = _bridge_gaps(points, v.get("native_fps") if v else None, stride_s)
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
