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
    q = ("SELECT detection_id, camera_id, track_id, timestamp, crop_path, bbox_w, bbox_h "
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

    tracks: dict = defaultdict(list)
    for r in rows:
        tracks[(r["camera_id"], r["track_id"])].append(r)

    total, done, found = len(tracks), 0, 0
    for _key, dets in tracks.items():
        done += 1
        dets.sort(key=lambda d: (d.get("bbox_w") or 0) * (d.get("bbox_h") or 0), reverse=True)
        votes: dict = {}                            # plate_text -> {score, conf, det}
        for d in dets[:_TOP_N]:
            cp = d.get("crop_path")
            if not cp or not os.path.exists(cp):
                continue
            for p in plate_reader.read_plates(cp):
                v = votes.setdefault(p["text"], {"score": 0.0, "conf": 0.0, "det": d})
                v["score"] += p["conf"]             # accumulate across frames (voting)
                if p["conf"] > v["conf"]:
                    v["conf"], v["det"] = p["conf"], d
        if votes:
            best = max(votes, key=lambda t: votes[t]["score"])
            v = votes[best]
            d = v["det"]
            database.insert_plate({
                "detection_id": d["detection_id"], "camera_id": d["camera_id"],
                "timestamp": d["timestamp"], "plate_text": best,
                "confidence": round(v["conf"], 3), "crop_path": d["crop_path"],
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
