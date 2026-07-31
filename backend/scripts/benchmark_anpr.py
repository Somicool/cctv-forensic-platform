"""Benchmark: OLD OCR (plate_reader.read_plates_voted) vs NEW ANPR
(anpr.read_plate_track) on the existing footage's vehicle tracks.

Runs BOTH fully offline (Gemini disabled) for a fair OCR-quality comparison, on
the same sharpest crops per vehicle track. Reports plate yield, OCR confidence,
supporting frames, agreement and runtime. If a Gemini key is present and
--oracle is passed, Gemini reads each track once as a reference plate and we also
report exact / partial (last-4) accuracy of each method against it.

    python -m scripts.benchmark_anpr --tracks 15
    python -m scripts.benchmark_anpr --tracks 15 --oracle
Run from the backend/ directory.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.getcwd())

from app import config, database                         # noqa: E402
from app.ingestion import plate_reader, anpr, gemini_plate  # noqa: E402

VEH = ["car", "truck", "bus", "motorcycle", "bicycle", "auto-rickshaw",
       "tractor", "tempo", "mini-truck", "hcv", "lcv"]


def _norm(s):
    return "".join(c for c in (s or "").upper() if c.isalnum())


def _partial_match(a, b, n=4):
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return False
    return a[-n:] == b[-n:] or a in b or b in a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, default=15, help="vehicle tracks to sample")
    ap.add_argument("--min-crops", type=int, default=3)
    ap.add_argument("--oracle", action="store_true", help="use Gemini as reference (needs key)")
    args = ap.parse_args()

    ph = ",".join("?" * len(VEH))
    with database.get_conn() as c:
        rows = [dict(r) for r in c.execute(
            f"SELECT detection_id, video_id, track_id, crop_path, bbox_w, bbox_h "
            f"FROM detections WHERE class_label IN ({ph}) AND crop_path IS NOT NULL", VEH).fetchall()]
    tracks = defaultdict(list)
    for r in rows:
        tracks[(r["video_id"], r["track_id"])].append(r)
    groups = [g for g in tracks.values() if len(g) >= args.min_crops]
    groups.sort(key=lambda g: len(g), reverse=True)
    groups = groups[:args.tracks]
    if not groups:
        print("No vehicle tracks with enough crops - run ingestion first.")
        return
    print(f"Benchmarking OLD OCR vs NEW ANPR on {len(groups)} vehicle tracks "
          f"(>= {args.min_crops} crops each)\n")

    global _HAS_GEMINI_KEY
    _HAS_GEMINI_KEY = bool(os.environ.get(getattr(config, "GEMINI_API_KEY_ENV", "GEMINI_API_KEY")))
    gem_on = config.GEMINI_ENABLED
    config.GEMINI_ENABLED = False                        # fair, fully-offline comparison

    old_t = new_t = 0.0
    old_yield = new_yield = 0
    old_confs, new_confs, new_frames = [], [], []
    agree = 0
    old_exact = new_exact = old_partial = new_partial = 0
    n_oracle = 0

    for g in groups:
        g.sort(key=lambda d: (d.get("bbox_w") or 0) * (d.get("bbox_h") or 0), reverse=True)
        paths = [d["crop_path"] for d in g[:config.PLATE_VOTE_FRAMES]
                 if d.get("crop_path") and os.path.exists(d["crop_path"])]
        if not paths:
            continue

        t = time.perf_counter()
        old = plate_reader.read_plates_voted(paths)
        old_t += time.perf_counter() - t
        old_p = old[0]["text"] if old else None
        old_c = old[0]["conf"] if old else 0.0

        t = time.perf_counter()
        new = anpr.read_plate_track(paths)
        new_t += time.perf_counter() - t
        new_p = new[0]["text"] if new else None
        new_c = new[0]["conf"] if new else 0.0
        new_f = new[0].get("frames") if new else 0

        if old_p:
            old_yield += 1; old_confs.append(old_c)
        if new_p:
            new_yield += 1; new_confs.append(new_c); new_frames.append(new_f or 0)
        if old_p and new_p and _norm(old_p) == _norm(new_p):
            agree += 1

        ref = None
        if args.oracle and _HAS_GEMINI_KEY:              # oracle bypasses the fairness flag
            g_read = gemini_plate.read_plate(paths[0])
            ref = plate_reader._candidate(g_read[0]) if g_read else None
        if ref:
            n_oracle += 1
            if old_p and _norm(old_p) == _norm(ref): old_exact += 1
            if new_p and _norm(new_p) == _norm(ref): new_exact += 1
            if old_p and _partial_match(old_p, ref): old_partial += 1
            if new_p and _partial_match(new_p, ref): new_partial += 1

        print(f"  track: OLD={str(old_p):14s}({old_c:.2f})  NEW={str(new_p):14s}"
              f"({new_c:.2f}, {new_f}f){'  ref='+ref if ref else ''}")

    config.GEMINI_ENABLED = gem_on
    n = len(groups)

    def avg(x):
        return sum(x) / len(x) if x else 0.0

    print("\n==================== ANPR BENCHMARK ====================")
    print(f"  tracks evaluated        : {n}")
    print(f"  plate yield   OLD / NEW : {old_yield} / {new_yield}")
    print(f"  mean OCR conf OLD / NEW : {avg(old_confs):.3f} / {avg(new_confs):.3f}")
    print(f"  mean supporting frames  : {avg(new_frames):.1f} (NEW)")
    print(f"  OLD/NEW agree on plate  : {agree}/{min(old_yield, new_yield)}")
    print(f"  runtime/track OLD / NEW : {1000*old_t/n:.0f} ms / {1000*new_t/n:.0f} ms "
          f"(+{100*(new_t-old_t)/max(old_t,1e-9):.0f}%)")
    if n_oracle:
        print("  --- vs Gemini reference (oracle) ---")
        print(f"  exact  OLD / NEW : {old_exact}/{n_oracle}  /  {new_exact}/{n_oracle}")
        print(f"  partial OLD / NEW: {old_partial}/{n_oracle}  /  {new_partial}/{n_oracle}")
    else:
        print("  (exact/partial accuracy vs ground truth: pass --oracle with a Gemini key,")
        print("   or supply a labelled set - not available offline without labels)")
    print("========================================================")


if __name__ == "__main__":
    main()
