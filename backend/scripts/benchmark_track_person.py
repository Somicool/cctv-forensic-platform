"""Benchmark the Track Person target reference: single clicked crop vs multi-view.

What is measured, and how (read before quoting)
----------------------------------------------
There are no human per-frame labels for this footage, so a ReID referee decides
whether a kept frame is the target. To keep that honest the referee is NOT the
reference set being tested - it is the full stored track's own median embedding,
which both configurations see identically. A frame is judged:

  correct    similarity to the referee >= the impostor-safe floor
  incorrect  below it, i.e. the box is on someone the referee says is not the target

  target switch        two CONSECUTIVE kept frames that disagree in appearance -
                       the box moved from one person to another mid-segment
  false target switch  a frame trimmed away despite strongly matching the target
                       (>= APPEARANCE_HOLD_SIM) - the target was dropped wrongly
  reacquisition        a bridged detection gap: the box was lost then re-attached

    python -m scripts.benchmark_track_person --cameras test1 test2 test3 test4
Run from the backend/ directory with the API not required (reads stored data).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

from app import database                                   # noqa: E402
from app.search import track_path, vector_store            # noqa: E402

BREAK = 0.55          # the referee's floor: impostor-safe, measured on this footage


def unit(v):
    v = np.asarray(v, dtype="float32").ravel()
    n = float(np.linalg.norm(v))
    return None if n < 1e-6 else v / n


MALFORMED = {"n": 0}


def referee(rows):
    """Median-direction embedding of the whole stored track - config independent."""
    vs = [unit(vector_store.get_vector("reid", r["detection_id"])) for r in rows]
    vs = [v for v in vs if v is not None]
    if not vs:
        return None
    # Some stored vectors are malformed (size 1 instead of the embedding width).
    # Keep the majority width and count the rest rather than crashing.
    widths = {}
    for v in vs:
        widths[v.shape[0]] = widths.get(v.shape[0], 0) + 1
    keep = max(widths, key=widths.get)
    MALFORMED["n"] += sum(n for w, n in widths.items() if w != keep)
    vs = [v for v in vs if v.shape[0] == keep]
    if len(vs) < 2:
        return None
    M = np.vstack(vs)
    med = M.mean(axis=0)
    n = float(np.linalg.norm(med))
    return None if n < 1e-6 else med / n


def evaluate(label, tracks, single_view: bool):
    """single_view=True reproduces the old behaviour: reference = clicked crop only."""
    saved_refs = track_path.REFERENCE_VIEWS
    saved_break = track_path.APPEARANCE_BREAK_SIM
    if single_view:
        track_path.REFERENCE_VIEWS = 1
        track_path.APPEARANCE_BREAK_SIM = 0.55
    stats = {"tracks": 0, "kept": 0, "correct": 0, "incorrect": 0,
             "switches": 0, "false_switches": 0, "reacq": 0, "predicted": 0,
             "conf": [], "refs": []}
    try:
        for vid, tid, rows in tracks:
            ref_row = max(rows, key=lambda r: (r.get("bbox_w") or 0) * (r.get("bbox_h") or 0))
            judge = referee(rows)
            if judge is None:
                continue
            resp = track_path.get_track_path(ref_row["detection_id"])
            pts = resp.points
            if not pts:
                continue
            stats["tracks"] += 1
            stats["refs"].append(len(resp.reference_views or []))
            if resp.identity_confidence is not None:
                stats["conf"].append(resp.identity_confidence)
            stats["predicted"] += resp.predicted_points or 0

            kept_ids, seq = set(), []
            for p in pts:
                if p.predicted:
                    continue
                v = unit(vector_store.get_vector("reid", p.detection_id))
                if v is None or v.shape != judge.shape:
                    continue
                kept_ids.add(p.detection_id)
                s = float(np.dot(judge, v))
                seq.append(s)
                stats["kept"] += 1
                if s >= BREAK:
                    stats["correct"] += 1
                else:
                    stats["incorrect"] += 1
            # a switch = adjacent kept frames that disagree with each other
            for a, b in zip(seq, seq[1:]):
                if (a >= BREAK) != (b >= BREAK):
                    stats["switches"] += 1
            # trimmed despite strongly matching the target
            for r in rows:
                if r["detection_id"] in kept_ids:
                    continue
                v = unit(vector_store.get_vector("reid", r["detection_id"]))
                if (v is not None and v.shape == judge.shape
                        and float(np.dot(judge, v)) >= track_path.APPEARANCE_HOLD_SIM):
                    stats["false_switches"] += 1
            # bridged gaps = re-attachments after a miss
            run = False
            for p in pts:
                if p.predicted and not run:
                    stats["reacq"] += 1
                    run = True
                elif not p.predicted:
                    run = False
    finally:
        track_path.REFERENCE_VIEWS = saved_refs
        track_path.APPEARANCE_BREAK_SIM = saved_break

    k = max(stats["kept"], 1)
    print(f"\n  {label}")
    print(f"    reference views per target : mean "
          f"{np.mean(stats['refs']) if stats['refs'] else 0:.2f}")
    print(f"    identity threshold        : {0.55 if single_view else saved_break}")
    print(f"    tracks tested             : {stats['tracks']}")
    print(f"    kept (real) target frames : {stats['kept']}")
    print(f"    CORRECT target frames     : {stats['correct']} ({100*stats['correct']/k:.1f}%)")
    print(f"    INCORRECT target frames   : {stats['incorrect']} ({100*stats['incorrect']/k:.1f}%)")
    print(f"    target switches           : {stats['switches']}")
    print(f"    false target switches     : {stats['false_switches']}")
    print(f"    reacquisitions (bridged)  : {stats['reacq']}")
    print(f"    predicted boxes           : {stats['predicted']}")
    print(f"    mean identity confidence  : "
          f"{np.mean(stats['conf']) if stats['conf'] else 0:.3f}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", nargs="*", default=["test1", "test2", "test3", "test4"])
    ap.add_argument("--min-dets", type=int, default=5)
    ap.add_argument("--crowded", action="store_true",
                    help="also report the busiest camera, where neighbours are close")
    args = ap.parse_args()

    cams = list(args.cameras)
    if args.crowded:
        cams.append("somi")

    ph = ",".join("?" * len(cams))
    with database.get_conn() as c:
        keys = [dict(r) for r in c.execute(
            f"SELECT video_id, track_id, camera_id, COUNT(1) n FROM detections "
            f"WHERE camera_id IN ({ph}) AND class_label='person' AND track_id IS NOT NULL "
            f"GROUP BY video_id, track_id HAVING n >= ? ORDER BY n DESC",
            (*cams, args.min_dets)).fetchall()]
    tracks = []
    for k in keys[:150]:
        rows = database.get_track_detections(k["video_id"], k["track_id"]) or []
        if len(rows) >= args.min_dets:
            tracks.append((k["video_id"], k["track_id"], rows))

    print("======== TRACK PERSON TARGET REFERENCE BENCHMARK ========")
    print(f"cameras: {cams}")
    print(f"targets: {len(tracks)} person tracks with >= {args.min_dets} detections")
    print(f"gap bridging (unchanged): max {track_path.PREDICT_MAX_GAP_S}s, "
          f"drift <= {100*track_path.PREDICT_MAX_DRIFT_FRAC:.0f}%, "
          f"max run {track_path.PREDICT_MAX_RUN}")

    before = evaluate("BEFORE - single clicked crop, threshold 0.55", tracks, True)
    after = evaluate(f"AFTER  - multi-view reference, threshold "
                     f"{track_path.APPEARANCE_BREAK_SIM}", tracks, False)

    kb, ka = max(before["kept"], 1), max(after["kept"], 1)
    print("\n  --- change ---")
    print(f"    incorrect target frames : {100*before['incorrect']/kb:.1f}% -> "
          f"{100*after['incorrect']/ka:.1f}%")
    print(f"    target switches         : {before['switches']} -> {after['switches']}")
    print(f"    false target switches   : {before['false_switches']} -> "
          f"{after['false_switches']}")
    print(f"    mean identity confidence: "
          f"{np.mean(before['conf']) if before['conf'] else 0:.3f} -> "
          f"{np.mean(after['conf']) if after['conf'] else 0:.3f}")
    if MALFORMED["n"]:
        print(f"\n  NOTE: {MALFORMED['n']} stored ReID vectors had the wrong width and were")
        print("  ignored. track_path already skips these rather than failing playback.")
    print("\n  Referee is the stored track's own mean embedding, identical for both")
    print("  configurations; these are ReID-refereed proxies, not human labels.")


if __name__ == "__main__":
    main()
