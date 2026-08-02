"""Benchmark: FIRST detected face vs BEST representative face per person track.

OLD behaviour = take the first face found while walking the track in frame order.
NEW behaviour = score every detected face across the whole track on 9 quality
                factors and keep the highest-ranked one.

Both use the SAME candidate scan, so the comparison isolates the selection policy.

    python -m scripts.benchmark_best_face --tracks 20
Run from the backend/ directory.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.getcwd())

from app import config, database, faces_gallery            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, default=20)
    ap.add_argument("--min-dets", type=int, default=8)
    args = ap.parse_args()

    with database.get_conn() as c:
        rows = [dict(r) for r in c.execute(
            "SELECT video_id, track_id, COUNT(1) n FROM detections "
            "WHERE class_label='person' AND track_id IS NOT NULL "
            "GROUP BY video_id, track_id HAVING n >= ? ORDER BY n DESC", (args.min_dets,)).fetchall()]
    rows = rows[:args.tracks]
    if not rows:
        print("No person tracks found."); return

    print(f"Comparing FIRST-detected vs BEST-representative face on {len(rows)} tracks\n")
    fq, nq, fs, ns, fr, nr = [], [], [], [], [], []
    better = 0
    low_flagged = 0
    t0 = time.perf_counter()

    for r in rows:
        scan = faces_gallery.rank_faces_in_track(r["video_id"], r["track_id"])
        ranked, best = scan["ranked"], scan["best"]
        if not ranked or best is None:
            print(f"  track {r['track_id']:5d} n={r['n']:3d}  no face in track")
            continue
        # OLD: first face encountered in frame order
        first = min(ranked, key=lambda x: (x.get("frame_number") or 0))
        fq.append(first["quality"]);  nq.append(best["quality"])
        fs.append(first["sharpness"]); ns.append(best["sharpness"])
        fr.append(first["resolution"]); nr.append(best["resolution"])
        if best["quality"] > first["quality"] + 1e-6:
            better += 1
        if best.get("low_quality"):
            low_flagged += 1
        print(f"  track {r['track_id']:5d} n={r['n']:3d} faces={scan['faces_seen']:3d} "
              f"FIRST q={first['quality']:.3f} sharp={first['sharpness']:.3f} res={first['resolution']:6d} | "
              f"BEST q={best['quality']:.3f} sharp={best['sharpness']:.3f} res={best['resolution']:6d}"
              f"{'  [LOW]' if best.get('low_quality') else ''}")

    n = len(fq)
    if not n:
        print("\nNo comparable tracks."); return
    avg = lambda x: sum(x) / len(x)
    pct = lambda a, b: (100.0 * (b - a) / a) if a else float('inf')

    print("\n============ BEST-FACE SELECTION BENCHMARK ============")
    print(f"  tracks compared            : {n}")
    print(f"  avg QUALITY   first / best : {avg(fq):.3f}  ->  {avg(nq):.3f}   (+{pct(avg(fq),avg(nq)):.1f}%)")
    print(f"  avg SHARPNESS first / best : {avg(fs):.3f}  ->  {avg(ns):.3f}   (+{pct(max(avg(fs),1e-6),avg(ns)):.1f}%)")
    print(f"  avg RESOLUTION first/best  : {avg(fr):.0f} px -> {avg(nr):.0f} px (+{pct(avg(fr),avg(nr)):.1f}%)")
    print(f"  tracks with a BETTER face  : {better}/{n}  ({100*better/n:.1f}%)")
    print(f"  flagged low-quality        : {low_flagged}/{n}")
    print(f"  total scan time            : {time.perf_counter()-t0:.1f}s "
          f"({(time.perf_counter()-t0)/max(n,1):.2f}s/track, cap {config.FACE_TRACK_SCAN_FRAMES} frames)")
    print("======================================================")


if __name__ == "__main__":
    main()
