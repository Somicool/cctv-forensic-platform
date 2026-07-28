"""Metadata filtering for search results.

Applied AFTER the vector search narrows candidates: keeps only detections that
match the requested camera / time window / object type / colour / vehicle type
/ minimum confidence.
"""
from __future__ import annotations

from .. import config
from ..models.schemas import SearchFilters

_VEHICLE_LABELS = {config.DETECT_CLASSES[c] for c in config.VEHICLE_CLASSES}
_PERSON_LABELS = {config.DETECT_CLASSES[c] for c in config.PERSON_CLASSES}


def _object_type_labels(object_type):
    if object_type == "person":
        return _PERSON_LABELS
    if object_type == "vehicle":
        return _VEHICLE_LABELS
    return None


def match(det: dict, f: SearchFilters) -> bool:
    if f.cameras and det.get("camera_id") not in f.cameras:
        return False

    if f.video_id is not None and det.get("video_id") != f.video_id:
        return False

    ts = det.get("timestamp")
    if f.start_time and ts and ts < f.start_time:
        return False
    if f.end_time and ts and ts > f.end_time:
        return False

    if f.min_confidence and (det.get("confidence") or 0.0) < f.min_confidence:
        return False

    labels = _object_type_labels(f.object_type)
    if labels is not None and det.get("class_label") not in labels:
        return False

    attrs = det.get("attributes") or {}

    if f.colors:
        wanted = {c.lower() for c in f.colors}
        present = {attrs.get("color"), attrs.get("upper_color"), attrs.get("lower_color")}
        present = {c.lower() for c in present if c}
        if not (wanted & present):
            return False

    if f.vehicle_type and attrs.get("vehicle_type") != f.vehicle_type:
        return False

    return True


def apply_filters(detections: list[dict], f: SearchFilters | None) -> list[dict]:
    if f is None:
        return detections
    return [d for d in detections if match(d, f)]
