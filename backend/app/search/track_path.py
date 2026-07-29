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

from .. import database
from ..models.schemas import TrackPathPoint, TrackPathResponse
from .text_search import media_url, _video_index, playback_fields, _camera_names


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

    points, confs = [], []
    for d in rows:
        if d.get("bbox_x") is None:                 # skip rows without a box
            continue
        pb = playback_fields(d, vindex)
        off = pb.get("offset_seconds")
        if off is None:                             # can't place it on the timeline
            continue
        conf = d.get("confidence")
        if conf is not None:
            confs.append(float(conf))
        points.append(TrackPathPoint(
            detection_id=d["detection_id"],
            offset_seconds=off,
            frame_number=d.get("frame_number"),
            timestamp=d.get("timestamp"),
            bbox=[d["bbox_x"], d["bbox_y"], d["bbox_w"], d["bbox_h"]],
            confidence=float(conf) if conf is not None else None,
        ))
    points.sort(key=lambda p: p.offset_seconds)

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
