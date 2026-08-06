"""Benchmark Journey Reconstruction.

Identity accuracy is measured WITHOUT hand labels by using ByteTrack tracks as
ground truth: two detections of the SAME (video, track) are the same person;
detections from different tracks in the SAME camera at overlapping times are
different people. That gives an honest same/different set to score against.

Reports: ReID/identity accuracy (TPR, precision), false-match rate, journey
reconstruction stats, runtime, and GPU memory delta.

    python -m scripts.benchmark_journey --pairs 300 --journeys 12
Run from the backend/ directory.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.getcwd())

from app import config, database, journey            # noqa: E402
from app.search import vector_store                  # noqa: E402


def _sig(ref_det, cand_det):
    """Fused identity score between two detections using the same signals the
    journey module uses (face / reid / body / attrs)."""
    parts = {}
    for name, key in (("reid", "reid"), ("body", "clip")):
        a = vector_store.get_vector(key, ref_det["detection_id"])
        b = vector_store.get_vector(key, cand_det["detection_id"])
        s = journey._cos(a, b)
        if s is not None:
            parts[name] = s
    fa = journey._face_emb_for_track(ref_det.get("video_id"), ref_det.get("track_id"))
    fb = journey._face_emb_for_track(cand_det.get("video_id"), cand_det.get("track_id"))
    s = journey._cos(fa, fb)
    if s is not None:
        parts["face"] = s
    a = journey._attr_score(ref_det.get("attributes"), cand_det.get("attributes"))
    if a is not None:
        parts["attrs"] = a
    if not parts:
        return None, False
    w = sum(journey.SIGNAL_WEIGHTS.get(k, 0) for k in parts)
    fused = sum(journey.SIGNAL_WEIGHTS.get(k, 0) * v for k, v in parts.items()) / w
    if parts.get("face", 0) >= journey.FACE_STRONG:
        fused = max(fused, parts["face"])
    strong = ("face" in parts) or ("reid" in parts)
    return fused, strong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--journeys", type=int, default=12)
    args = ap.parse_args()
    random.seed(42)

    with database.get_conn() as c:
        rows = [dict(r) for r in c.execute(
            "SELECT detection_id, video_id, camera_id, track_id, timestamp, attributes "
            "FROM detections WHERE class_label='person' AND track_id IS NOT NULL").fetchall()]
    for r in rows:
        r["attributes"] = database._row_to_detection(r)["attributes"] if isinstance(r.get("attributes"), str) else (r.get("attributes") or {})
    by_track = {}
    for r in rows:
        by_track.setdefault((r["video_id"], r["track_id"]), []).append(r)
    tracks = [v for v in by_track.values() if len(v) >= 2]
    print(f"person detections={len(rows)}  usable tracks={len(tracks)}")

    # ---- identity accuracy -------------------------------------------------
    n = args.pairs // 2
    same, diff = [], []
    for _ in range(n):
        t = random.choice(tracks)
        a, b = random.sample(t, 2)
        same.append((a, b))
    keys = list(by_track.keys())
    tries = 0
    while len(diff) < n and tries < n * 40:
        tries += 1
        k1, k2 = random.sample(keys, 2)
        if k1[0] != k2[0]:                                  # different video -> ambiguous
            continue
        a, b = random.choice(by_track[k1]), random.choice(by_track[k2])
        diff.append((a, b))

    t0 = time.perf_counter()
    tp = fn = 0
    for a, b in same:
        f, strong = _sig(a, b)
        thr = journey.IDENTITY_MIN if strong else journey.IDENTITY_MIN_WEAK
        if f is not None and f >= thr: tp += 1
        else: fn += 1
    fp = tn = 0
    for a, b in diff:
        f, strong = _sig(a, b)
        thr = journey.IDENTITY_MIN if strong else journey.IDENTITY_MIN_WEAK
        if f is not None and f >= thr: fp += 1
        else: tn += 1
    id_time = time.perf_counter() - t0

    tpr = tp / max(tp + fn, 1)
    fmr = fp / max(fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    acc = (tp + tn) / max(tp + fn + fp + tn, 1)

    # ---- journey reconstruction --------------------------------------------
    gpu0 = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(); gpu0 = torch.cuda.memory_allocated() / 1048576
    except Exception:
        pass

    cands = [t[0] for t in sorted(tracks, key=len, reverse=True)[:args.journeys]]
    multi = single = 0
    confs, cams, rej, times = [], [], 0, []
    for d in cands:
        s = time.perf_counter()
        res = journey.reconstruct(d["detection_id"], persist=False)
        times.append(time.perf_counter() - s)
        if res.get("error"):
            continue
        p = res["primary"]
        confs.append(p["confidence"]); cams.append(p["stats"]["cameras"])
        rej += len(p["rejected_transitions"])
        if p["stats"]["cameras"] > 1: multi += 1
        else: single += 1

    gpu_delta = None
    try:
        import torch
        if torch.cuda.is_available() and gpu0 is not None:
            gpu_delta = round(torch.cuda.max_memory_allocated() / 1048576 - gpu0, 1)
    except Exception:
        pass

    avg = lambda x: (sum(x) / len(x)) if x else 0.0
    print("\n================= JOURNEY RECONSTRUCTION BENCHMARK =================")
    print(f"  identity pairs tested      : {len(same)} same / {len(diff)} different")
    print(f"  ReID/identity accuracy     : {100*acc:.1f}%")
    print(f"  true-positive rate (recall): {100*tpr:.1f}%   ({tp}/{tp+fn})")
    print(f"  precision                  : {100*prec:.1f}%")
    print(f"  FALSE MATCH rate           : {100*fmr:.1f}%   ({fp}/{fp+tn})")
    print(f"  identity scoring runtime   : {1000*id_time/max(len(same)+len(diff),1):.1f} ms/pair")
    print(f"  --- journeys ---")
    print(f"  journeys reconstructed     : {len(confs)}/{len(cands)}")
    print(f"  multi-camera journeys      : {multi}   single-camera: {single}")
    print(f"  avg cameras per journey    : {avg(cams):.2f}")
    print(f"  avg confidence             : {avg(confs):.3f}")
    print(f"  impossible transitions cut : {rej}")
    print(f"  runtime per journey        : {1000*avg(times):.0f} ms")
    print(f"  extra GPU memory           : {gpu_delta if gpu_delta is not None else 'n/a'} MB")
    print("===================================================================")


if __name__ == "__main__":
    main()
