"""Re-run licence-plate OCR over already-stored vehicle crops and populate the
plates table - no full pipeline re-run needed.

Improves on the pipeline's single-crop pass with MULTI-FRAME VOTING: for each
vehicle track it OCRs the several largest crops and adds up confidence per
candidate plate string across frames, so a plate that reads consistently across
frames wins even if any single frame is weak. Inserts one plate per track.

Run via POST /api/recompute-plates (optionally scoped to a video_id).
"""
from __future__ import annotations

import os
from collections import defaultdict

from .. import config, database, ingest_jobs, ingest_progress
from . import plate_reader

_VEHICLE_LABELS = [config.DETECT_CLASSES[c] for c in config.VEHICLE_CLASSES]
_TOP_N = 6                      # largest crops per track to OCR


def recompute_plates(video_id=None, job_id=None) -> dict:
    ph = ",".join("?" * len(_VEHICLE_LABELS))
    q = ("SELECT detection_id, camera_id, track_id, video_id, frame_number, timestamp, "
         " crop_path, class_label, confidence, bbox_x, bbox_y, bbox_w, bbox_h "
         f"FROM detections WHERE class_label IN ({ph})")
    params = list(_VEHICLE_LABELS)
    if video_id is not None:
        q += " AND video_id=?"
        params.append(video_id)
    with database.get_conn() as conn:
        rows = [dict(r) for r in conn.execute(q, params).fetchall()]
        det_ids = [r["detection_id"] for r in rows]
        for i in range(0, len(det_ids), 400):     # clear old plates for these detections
            chunk = det_ids[i:i + 400]
            conn.execute(f"DELETE FROM plates WHERE detection_id IN ({','.join('?' * len(chunk))})", chunk)

    # video_id -> (source path, native_fps) so two-wheeler tracks can be adaptively
    # re-sampled from the original footage.
    vinfo = {v["video_id"]: v for v in database.list_videos()}

    tracks: dict = defaultdict(list)
    for r in rows:
        tracks[(r["camera_id"], r["track_id"])].append(r)

    total, done, found = len(tracks), 0, 0
    for _key, dets in tracks.items():
        done += 1
        dets.sort(key=lambda d: (d.get("bbox_w") or 0) * (d.get("bbox_h") or 0), reverse=True)
        top = [d for d in dets[:_TOP_N] if d.get("crop_path") and os.path.exists(d["crop_path"])]
        rep = top[0] if top else None
        if rep and config.ANPR_ENABLED:
            from . import anpr
            cls_label = rep.get("class_label")
            is_tw = any(config.DETECT_CLASSES.get(cid) == cls_label
                        for cid in config.ANPR_TWOWHEELER_CLASSES)
            v = vinfo.get(rep.get("video_id")) or {}
            vpath = (config.VIDEO_DIR / v["filename"]) if v.get("filename") else None
            tag = f"{rep.get('video_id')}_{rep['detection_id']}"
            cands = []
            # Two-wheelers / autos -> ADAPTIVE high-FPS re-sampling of the source video
            # around the track (recovers small/blurred bike & auto plates). Falls back
            # to crop-based ANPR if the video is unavailable or yields nothing.
            if is_tw and config.ANPR_ADAPTIVE_ENABLED and vpath and vpath.exists():
                adet = [{"frame_number": d.get("frame_number"),
                         "bbox": (d.get("bbox_x") or 0, d.get("bbox_y") or 0,
                                  d.get("bbox_w") or 0, d.get("bbox_h") or 0),
                         "confidence": d.get("confidence")} for d in dets]
                cands = anpr.read_plate_track_adaptive(
                    str(vpath), adet, v.get("native_fps"),
                    save_dir=str(config.PLATE_CROP_DIR), tag=tag)
            if not cands:
                cands = anpr.read_plate_track([d["crop_path"] for d in top],
                                              save_dir=str(config.PLATE_CROP_DIR), tag=tag)
            for text, c in [(c["text"], c) for c in cands[:config.PLATE_MAX_CANDIDATES]]:
                database.insert_plate({
                    "detection_id": rep["detection_id"], "camera_id": rep["camera_id"],
                    "timestamp": rep["timestamp"], "plate_text": text,
                    "confidence": round(c["conf"], 3), "crop_path": rep["crop_path"],
                    "votes": c.get("votes"), "source": c.get("source"),
                    "plate_crop": c.get("plate_crop"),
                })
                found += 1
        elif rep:
            votes: dict = {}                        # plate_text -> {score, conf, det}
            for d in top:
                for p in plate_reader.read_plates(d["crop_path"]):
                    v = votes.setdefault(p["text"], {"score": 0.0, "conf": 0.0, "det": d, "n": 0})
                    v["score"] += p["conf"]
                    v["n"] += 1
                    if p["conf"] > v["conf"]:
                        v["conf"], v["det"] = p["conf"], d
            # Store the top few plate reads per vehicle track (not just the single
            # best) so a confident misread never hides the correct plate.
            ranked = sorted(votes.items(), key=lambda kv: kv[1]["score"], reverse=True)
            for text, v in ranked[:config.PLATE_MAX_CANDIDATES]:
                d = v["det"]
                database.insert_plate({
                    "detection_id": d["detection_id"], "camera_id": d["camera_id"],
                    "timestamp": d["timestamp"], "plate_text": text,
                    "confidence": round(v["conf"], 3), "crop_path": d["crop_path"],
                    "votes": v["n"], "source": "paddle",
                })
                found += 1
        if done % 10 == 0 or done == total:
            ingest_progress.set_progress(done, total)
            if job_id:
                ingest_jobs.update(job_id, status="processing", done=done, total=total,
                                   current="reading plates")

    if job_id:
        ingest_jobs.update(job_id, status="done", done=total, total=total, current=None)
    ingest_progress.reset()
    return {"tracks": total, "plates_found": found, "video_id": video_id}


if __name__ == "__main__":
    print(recompute_plates())
