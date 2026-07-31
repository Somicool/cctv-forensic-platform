"""India-specific secondary detector plugins.

This package adds specialized detectors (auto-rickshaw now; tractor / Tata Ace /
etc. later) that run ALONGSIDE the primary YOLOv10 detector and merge into one
unified detection stream via class-aware NMS. The primary detector and the whole
downstream pipeline (ByteTrack, OCR, ReID, CLIP, FAISS, search, export) are left
unchanged - they never need to know which detector produced a box.
"""
from .plugins import (active, detect_frame, get_plugins, merge_detections,
                      reset, IoUTracker)

__all__ = ["active", "detect_frame", "get_plugins", "merge_detections", "reset",
           "IoUTracker"]
