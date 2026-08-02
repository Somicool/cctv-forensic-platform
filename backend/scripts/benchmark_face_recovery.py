"""Benchmark: face-extraction success rate BEFORE vs AFTER track-wide recovery.

OLD behaviour = a face is available only if ingestion already stored one for the
                person's track (that's what produced "No clear face available").
NEW behaviour = go back to the ORIGINAL frame + expanded bbox and scan the whole
                ByteTrack track (both directions), scoring every candidate.

    python -m scripts.benchmark_face_recovery --tracks 40
Run from the backend/ directory.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.getcwd())

from app import database, faces_gallery                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, default=40, help="person tracks to sample")
    ap.add_argument("--only-failing", action="store_true",
                    help="sample only tracks that had NO stored face (the old failures)")
    args = ap.parse_args()

    with database.get_conn() as c:
        rows = [dict(r) for r in c.execute(
            "SELECT d.video_id, d.track_id, MIN(d.detection_id) AS det, COUNT(1) AS n, "
            " MAX(CASE WHEN f.face_id IS NULL THEN 0 ELSE 1 END) AS had_face "
            "FROM detections d LEFT JOIN faces f ON f.detection_id = d.detection_id "
            "WHERE d.class_label='person' AND d.track_id IS NOT NULL "
            "GROUP BY d.video_id, d.track_id HAVING n >= 2 ORDER BY n DESC").fetchall()]
    if args.only_failing:
        rows = [r for r in rows if not r["had_face"]]
    rows = rows[:args.tracks]
    if not rows:
        print("No person tracks found."); return

    print(f"Evaluating {len(rows)} person tracks "
          f"({'previously-failing only' if args.only_failing else 'mixed'})\n")

    old_ok = new_ok = 0
    t_old = t_new = 0.0
    recovered = []
    for r in rows:
        # OLD: stored-face-only
        t = time.perf_counter()
        old = faces_gallery.best_face_for_detection(r["det"], deep=False)
        t_old += time.perf_counter() - t
        old_hit = bool(old and old.get("available"))

        # NEW: track-wide recovery from original frames
        t = time.perf_counter()
        new = faces_gallery.best_face_for_detection(r["det"], deep=True)
        t_new += time.perf_counter() - t
        new_hit = bool(new and new.get("available"))

        old_ok += old_hit; new_ok += new_hit
        if new_hit and not old_hit:
            recovered.append((r["track_id"], new.get("quality"), new.get("face_size"),
                              new.get("frontal"), new.get("source")))
        print(f"  track {r['track_id']:5d} n={r['n']:3d}  OLD={'HIT ' if old_hit else 'MISS'}"
              f"  NEW={'HIT ' if new_hit else 'MISS'}"
              f"{'  q=%.2f size=%s frontal=%s' % (new.get('quality'), new.get('face_size'), new.get('frontal')) if new_hit and new.get('quality') is not None else ''}")

    n = len(rows)
    print("\n================ FACE EXTRACTION SUCCESS RATE ================")
    print(f"  tracks evaluated       : {n}")
    print(f"  OLD (stored crop only) : {old_ok}/{n}  = {100*old_ok/n:.1f}%")
    print(f"  NEW (track-wide scan)  : {new_ok}/{n}  = {100*new_ok/n:.1f}%")
    print(f"  improvement            : +{100*(new_ok-old_ok)/n:.1f} points"
          f"  ({'%.1fx' % (new_ok/max(old_ok,1))} )")
    print(f"  newly recovered tracks : {len(recovered)}")
    print(f"  avg time/track OLD/NEW : {1000*t_old/n:.0f} ms / {1000*t_new/n:.0f} ms")
    print("==============================================================")


if __name__ == "__main__":
    main()
