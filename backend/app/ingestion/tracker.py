"""Multi-object tracking (YOLO + ByteTrack).

Runs detection + ByteTrack over a whole video in a single pass so each object
gets a persistent track_id ("car #1" stays "car #1" across frames). Saves the
sampled frames and a crop per detection, and returns TrackedDetection records,
a per-track summary, and the list of sampled frames (used for scene-level
embeddings). This is the main detection+tracking pass the pipeline uses.

CLI:
    python -m app.ingestion.tracker <video> --camera CAM-02 --fps 2
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

from .. import config, ingest_jobs, ingest_progress
from . import identity_guard, reid_embedder
from .detector import get_model
from .detectors import plugins as det_plugins


@dataclass
class TrackedDetection:
    camera_id: str
    video_name: str
    track_id: int
    frame_number: int         # index in the original video
    timestamp: str            # real-world ISO timestamp
    class_id: int
    class_label: str
    confidence: float
    bbox: tuple               # (x, y, w, h) pixels
    crop_path: str
    frame_path: str | None = None
    crop_img: object = None   # in-memory (downscaled) BGR crop for CLIP/attrs - avoids re-decode
    # Person ReID embedding computed during tracking for the appearance guard.
    # The pipeline REUSES this instead of embedding the crop again, so verifying
    # associations costs no extra OSNet forward passes.
    reid_vec: object = None
    # How the appearance guard resolved this detection's association:
    # accept / reacquire / reject-new / new. Diagnostic only.
    assoc: str | None = None


def _parse_start_time(start_time) -> datetime:
    if start_time is None:
        return datetime.now(timezone.utc)
    if isinstance(start_time, datetime):
        return start_time
    return datetime.fromisoformat(str(start_time))


def _padded_box(x1, y1, x2, y2, fw, fh):
    """Expand a detection box by CROP_PAD_FRAC (clamped to the frame) so the saved
    crop keeps some context - helps CLIP attributes and re-ID."""
    px = (x2 - x1) * config.CROP_PAD_FRAC
    py = (y2 - y1) * config.CROP_PAD_FRAC
    return (max(0, int(x1 - px)), max(0, int(y1 - py)),
            min(fw, int(x2 + px)), min(fh, int(y2 + py)))


def _downscale_copy(im, max_side: int = 320):
    """A detached, size-bounded copy of a crop for in-memory embedding/attributes
    (CLIP needs 224, re-ID 256; 320 is plenty). Always copies so it survives the
    next frame overwrite."""
    h, w = im.shape[:2]
    m = max(h, w)
    if m > max_side:
        f = max_side / m
        return cv2.resize(im, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    return im.copy()


def _crop_ok(crop, box, fw, fh) -> bool:
    """Reject extremely small, mostly-off-screen, or severely blurred crops
    before they enter the DB / embeddings."""
    if crop is None or not crop.size:
        return False
    h, w = crop.shape[:2]
    if min(w, h) < config.CROP_MIN_SIDE:
        return False
    x1, y1, x2, y2 = box
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    vis_w = max(0.0, min(x2, fw) - max(x1, 0.0))
    vis_h = max(0.0, min(y2, fh) - max(y1, 0.0))
    if (vis_w * vis_h) / (bw * bh) < config.CROP_MIN_VISIBLE:      # heavily truncated
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if cv2.Laplacian(gray, cv2.CV_64F).var() < config.CROP_BLUR_VAR_MIN:  # near-flat / blurred
        return False
    return True


def track_video(
    video_path,
    camera_id: str,
    start_time=None,
    fps: float | None = None,
    frame_root=None,
    crop_root=None,
    save_frames: bool = True,
    imgsz: int | None = None,
):
    """Detect + track objects through a video (whole clip in one call).

    Returns (detections, tracks, frames). Implemented on top of iter_track_chunks
    so the progressive and monolithic paths share ONE per-frame code path and
    produce byte-identical detections/tracks."""
    dets: list[TrackedDetection] = []
    tracks: dict[int, list] = defaultdict(list)
    frames_meta: list[dict] = []
    guard_stats = None
    for chunk in iter_track_chunks(video_path, camera_id, start_time=start_time, fps=fps,
                                   frame_root=frame_root, crop_root=crop_root,
                                   save_frames=save_frames, imgsz=imgsz,
                                   chunk_frames=None):
        dets.extend(chunk["dets"])
        frames_meta.extend(chunk["frames"])
        guard_stats = chunk.get("guard") or guard_stats
        for d in chunk["dets"]:
            tracks[d.track_id].append((d.frame_number, d.bbox))
    print(f"[track] {Path(video_path).name}: {len(frames_meta)} frames, {len(dets)} detections, "
          f"{len(tracks)} unique tracks (camera {camera_id})")
    if guard_stats:
        print(f"[track] appearance guard: {guard_stats['switches_blocked']} identity switches "
              f"blocked, {guard_stats['reacquired']} tracks re-acquired after occlusion, "
              f"reject rate {guard_stats['reject_rate']:.1%}")
    return dets, dict(tracks), frames_meta


def iter_track_chunks(video_path, camera_id: str, start_time=None, fps: float | None = None,
                      frame_root=None, crop_root=None, save_frames: bool = True,
                      imgsz: int | None = None, chunk_frames: int | None = None):
    """Detect + track over a video, YIELDING results in chunks of ~chunk_frames
    sampled frames while keeping ByteTrack state continuous across the whole clip
    (so track_ids - and therefore the final index - are identical to processing
    it in one pass). Each yielded chunk is a dict:
        {"dets": [...], "frames": [...], "sampled": int, "total_sampled": int}
    chunk_frames=None yields everything as one final chunk (monolithic behaviour).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    target_fps = fps or config.DEFAULT_FPS
    frame_root = Path(frame_root or config.FRAME_DIR)
    crop_root = Path(crop_root or config.CROP_DIR)
    base_time = _parse_start_time(start_time)

    cap = cv2.VideoCapture(str(video_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if native_fps <= 0:
        native_fps = 30.0
    stride = max(1, round(native_fps / target_fps))
    total_sampled = max(1, total_frames // stride) if total_frames else 0

    frame_dir = frame_root / camera_id / video_path.stem
    crop_dir = crop_root / camera_id / video_path.stem
    frame_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    # The India Vehicle Detector, when loaded, is the SOLE source of vehicle
    # detections; the COCO model then only detects people + general objects
    # (+bicycle). Without it, COCO keeps detecting vehicles as a fallback so
    # vehicle detection never disappears.
    use_plugins = det_plugins.active()
    primary_classes = (config.PRIMARY_NONVEHICLE_CLASSES if use_plugins
                       else config.PRIMARY_CLASSES)
    model = get_model()
    # Feed the tracker LOW-confidence boxes too: ByteTrack's second association
    # stage needs them to carry a track through occlusion. Detections are filtered
    # back to config.DETECT_CONF before they are stored, so the database keeps the
    # same quality floor it always had.
    track_conf = min(config.TRACK_INPUT_CONF, config.DETECT_CONF)
    results = model.track(
        source=str(video_path), stream=True, tracker=config.TRACKER_CFG,
        classes=list(primary_classes), conf=track_conf,
        imgsz=imgsz or config.YOLO_IMGSZ, vid_stride=stride, device=config.DEVICE, verbose=False,
    )
    sec_tracker = det_plugins.IoUTracker() if use_plugins else None
    # Appearance guard: motion may propose an association, appearance decides it.
    guard = identity_guard.IdentityGuard() if config.TRACK_APPEARANCE_GUARD else None

    buf_dets: list[TrackedDetection] = []
    buf_frames: list[dict] = []
    sampled = 0
    for r in results:
        if ingest_jobs.stop_requested():
            raise RuntimeError("ingest stopped by user")
        if sampled % 10 == 0:
            ingest_progress.set_progress(sampled, total_sampled)
        frame = r.orig_img
        original_frame = sampled * stride
        ts = base_time + timedelta(seconds=original_frame / native_fps)

        frame_path = None
        if save_frames:
            fp = frame_dir / f"frame_{sampled:06d}.jpg"
            cv2.imwrite(str(fp), frame)
            frame_path = str(fp)
            buf_frames.append({"frame_path": frame_path, "frame_number": original_frame,
                               "timestamp": ts.isoformat()})

        # ONE unified detection list per frame: primary YOLOv10 boxes (tracked by
        # ByteTrack) + any India-specific secondary boxes (class-aware NMS + a
        # lightweight per-class IoU tracker). Downstream can't tell them apart.
        fh, fw = frame.shape[:2]
        raw: list[dict] = []
        if r.boxes is not None and r.boxes.id is not None:
            for b in r.boxes:
                cls = int(b.cls[0])
                raw.append({"cls_id": cls,
                            "label": config.DETECT_CLASSES.get(cls, str(cls)),
                            "conf": float(b.conf[0]),
                            "xyxy": tuple(float(v) for v in b.xyxy[0].tolist()),
                            "track_id": int(b.id[0])})
        if use_plugins:
            sec = det_plugins.detect_frame(frame)
            # class-aware NMS: the more specific Indian class wins; generic
            # duplicates are dropped. primary keeps ByteTrack ids; surviving
            # secondary boxes get lightweight IoU-tracker ids.
            raw, kept_sec = det_plugins.merge_detections(raw, sec)
            for s in sec_tracker.update(kept_sec, sampled):
                raw.append({"cls_id": s["cls_id"], "label": s["label"], "conf": s["conf"],
                            "xyxy": s["xyxy"], "track_id": s["track_id"]})

        # Keep the low-confidence boxes only for ByteTrack's benefit - never store
        # them. This is what preserves the previous database quality floor.
        keep = []
        for d in raw:
            if d["conf"] < config.DETECT_CONF:
                continue
            x1, y1, x2, y2 = d["xyxy"]
            cx1, cy1, cx2, cy2 = _padded_box(x1, y1, x2, y2, fw, fh)
            crop = frame[cy1:cy2, cx1:cx2]
            if not _crop_ok(crop, (x1, y1, x2, y2), fw, fh):
                continue
            keep.append((d, crop))

        # ONE batched ReID pass over this frame's people, used both to verify the
        # associations here and (reused, not recomputed) by the pipeline later.
        small = [_downscale_copy(c) for _d, c in keep]
        person_i = [i for i, (d, _c) in enumerate(keep)
                    if d["cls_id"] in config.PERSON_CLASSES]
        embs = {}
        if guard is not None and person_i:
            try:
                vecs = reid_embedder.embed_persons([small[i] for i in person_i])
                embs = {person_i[k]: vecs[k] for k in range(len(person_i))}
            except Exception as exc:                      # never fail ingestion on this
                print(f"[track] appearance guard disabled for this frame: {exc}")
                embs = {}

        seen_ids = set()
        for i, (d, crop) in enumerate(keep):
            tid, label = d["track_id"], d["label"]
            x1, y1, x2, y2 = d["xyxy"]
            assoc = None
            vec = embs.get(i)
            if vec is not None:
                tid, assoc = guard.resolve(tid, vec, sampled)
                seen_ids.add(tid)
            cp = crop_dir / f"f{sampled:06d}_t{tid:04d}_{label}.jpg"
            cv2.imwrite(str(cp), crop)
            buf_dets.append(TrackedDetection(
                camera_id=camera_id, video_name=video_path.name, track_id=tid,
                frame_number=original_frame, timestamp=ts.isoformat(), class_id=d["cls_id"],
                class_label=label, confidence=d["conf"],
                bbox=(x1, y1, x2 - x1, y2 - y1),
                crop_path=str(cp), frame_path=frame_path, crop_img=small[i],
                reid_vec=vec, assoc=assoc))
        if guard is not None:
            guard.retire(sampled, seen_ids)
        sampled += 1

        if chunk_frames and sampled % chunk_frames == 0:
            yield {"dets": buf_dets, "frames": buf_frames, "sampled": sampled,
                   "total_sampled": total_sampled,
                   "guard": guard.summary() if guard else None}
            buf_dets, buf_frames = [], []

    yield {"dets": buf_dets, "frames": buf_frames, "sampled": sampled,
           "total_sampled": max(total_sampled, sampled),
           "guard": guard.summary() if guard else None}


def _cli():
    ap = argparse.ArgumentParser(description="Detect + track objects in a video")
    ap.add_argument("video")
    ap.add_argument("--camera", default="CAM-01")
    ap.add_argument("--fps", type=float, default=config.DEFAULT_FPS)
    ap.add_argument("--start-time", default=None, help="ISO datetime of first frame")
    args = ap.parse_args()
    dets, tracks, frames = track_video(args.video, args.camera,
                                       start_time=args.start_time, fps=args.fps)
    by_class = Counter(d.class_label for d in dets)
    print(f"Done. {len(dets)} detections across {len(tracks)} tracks, {len(frames)} frames.")
    print("By class:", dict(by_class))


if __name__ == "__main__":
    _cli()
