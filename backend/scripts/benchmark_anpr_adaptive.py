"""Benchmark: OLD crop-based OCR vs NEW adaptive ANPR, specifically on
two-wheelers / auto-rickshaws.

OLD = plate_reader.read_plates_voted on the sparse 2-FPS saved crops.
NEW = anpr.read_plate_track_adaptive (re-open the video, dense re-sampling +
      frame scoring + plate-region OCR + voting).
Both run fully offline (Gemini disabled) for a fair comparison. If a Gemini key
is present and --oracle is passed, Gemini reads each track's best frame as a
reference and we report exact / partial (last-4) accuracy against it.

    python -m scripts.benchmark_anpr_adaptive --video 104 --tracks 20 --oracle
Run from the backend/ directory.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.getcwd())

from app import config, database                              # noqa: E402
from app.ingestion import plate_reader, anpr, gemini_plate    # noqa: E402

TW_LABELS = {config.DETECT_CLASSES.get(c) for c in config.ANPR_TWOWHEELER_CLASSES}
TW_LABELS |= {"motorcycle", "scooter", "auto-rickshaw"}       # ensure the core three


def norm(s):
    return "".join(ch for ch in (s or "").upper() if ch.isalnum())


def partial(a, b, n=4):
    a, b = norm(a), norm(b)
    return bool(a) and bool(b) and (a[-n:] == b[-n:] or a in b or b in a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=int, default=None, help="restrict to one video_id")
    ap.add_argument("--tracks", type=int, default=20)
    ap.add_argument("--min-crops", type=int, default=2)
    ap.add_argument("--oracle", action="store_true")
    args = ap.parse_args()

    labels = [l for l in TW_LABELS if l]
    ph = ",".join("?" * len(labels))
    q = ("SELECT detection_id, video_id, track_id, camera_id, frame_number, timestamp, "
         " crop_path, class_label, confidence, bbox_x, bbox_y, bbox_w, bbox_h "
         f"FROM detections WHERE class_label IN ({ph})")
    params = list(labels)
    if args.video is not None:
        q += " AND video_id=?"; params.append(args.video)
    with database.get_conn() as c:
        rows = [dict(r) for r in c.execute(q, params).fetchall()]
    tracks = defaultdict(list)
    for r in rows:
        tracks[(r["video_id"], r["track_id"])].append(r)
    groups = [g for g in tracks.values() if len(g) >= args.min_crops]
    groups.sort(key=lambda g: max((d["bbox_w"] or 0) * (d["bbox_h"] or 0) for d in g), reverse=True)
    groups = groups[:args.tracks]
    if not groups:
        print("No two-wheeler/auto tracks found."); return

    vinfo = {v["video_id"]: v for v in database.list_videos()}
    print(f"Benchmarking OLD vs NEW-adaptive on {len(groups)} two-wheeler/auto tracks "
          f"(classes: {sorted(labels)})\n")

    _HAS_KEY = bool(os.environ.get(getattr(config, "GEMINI_API_KEY_ENV", "GEMINI_API_KEY")))
    gem = config.GEMINI_ENABLED
    config.GEMINI_ENABLED = False                             # fair offline comparison

    old_t = new_t = 0.0
    old_yield = new_yield = 0
    old_confs, new_confs = [], []
    old_exact = new_exact = old_partial = new_partial = n_ref = 0

    for g in groups:
        g.sort(key=lambda d: (d["bbox_w"] or 0) * (d["bbox_h"] or 0), reverse=True)
        top = [d for d in g[:config.PLATE_VOTE_FRAMES] if d["crop_path"] and os.path.exists(d["crop_path"])]
        if not top:
            continue
        cls = top[0]["class_label"]
        v = vinfo.get(top[0]["video_id"]) or {}
        vpath = (config.VIDEO_DIR / v["filename"]) if v.get("filename") else None

        t = time.perf_counter()
        old = plate_reader.read_plates_voted([d["crop_path"] for d in top])
        old_t += time.perf_counter() - t
        old_p = old[0]["text"] if old else None
        old_c = old[0]["conf"] if old else 0.0

        adet = [{"frame_number": d["frame_number"],
                 "bbox": (d["bbox_x"] or 0, d["bbox_y"] or 0, d["bbox_w"] or 0, d["bbox_h"] or 0),
                 "confidence": d["confidence"]} for d in g]
        t = time.perf_counter()
        new = (anpr.read_plate_track_adaptive(str(vpath), adet, v.get("native_fps"))
               if vpath and vpath.exists() else [])
        if not new:
            new = anpr.read_plate_track([d["crop_path"] for d in top])
        new_t += time.perf_counter() - t
        new_p = new[0]["text"] if new else None
        new_c = new[0]["conf"] if new else 0.0
        new_f = new[0].get("frames") if new else 0

        if old_p: old_yield += 1; old_confs.append(old_c)
        if new_p: new_yield += 1; new_confs.append(new_c)

        ref = None
        if args.oracle and _HAS_KEY and vpath and vpath.exists():
            # Gemini oracle: read the sharpest adaptively-sampled frame's plate crop
            g_read = gemini_plate.read_plate(top[0]["crop_path"])
            ref = plate_reader._candidate(g_read[0]) if g_read else None
        if ref:
            n_ref += 1
            if old_p and norm(old_p) == norm(ref): old_exact += 1
            if new_p and norm(new_p) == norm(ref): new_exact += 1
            if old_p and partial(old_p, ref): old_partial += 1
            if new_p and partial(new_p, ref): new_partial += 1

        print(f"  {cls:12s} OLD={str(old_p):12s}({old_c:.2f})  NEW={str(new_p):12s}"
              f"({new_c:.2f},{new_f}f){'  ref='+ref if ref else ''}")

    config.GEMINI_ENABLED = gem
    n = len(groups)
    avg = lambda x: (sum(x) / len(x)) if x else 0.0

    print("\n============ ADAPTIVE ANPR BENCHMARK (two-wheelers/autos) ============")
    print(f"  tracks                  : {n}")
    print(f"  plate yield  OLD / NEW  : {old_yield} / {new_yield}")
    print(f"  mean conf    OLD / NEW  : {avg(old_confs):.3f} / {avg(new_confs):.3f}")
    print(f"  runtime/track OLD / NEW : {1000*old_t/n:.0f} ms / {1000*new_t/n:.0f} ms "
          f"(+{100*(new_t-old_t)/max(old_t,1e-9):.0f}%)")
    if n_ref:
        print("  --- vs Gemini reference ---")
        print(f"  exact   OLD / NEW : {old_exact}/{n_ref}  /  {new_exact}/{n_ref}")
        print(f"  partial OLD / NEW : {old_partial}/{n_ref}  /  {new_partial}/{n_ref}")
    print("======================================================================")


if __name__ == "__main__":
    main()
