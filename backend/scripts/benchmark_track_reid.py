"""Benchmark: single-crop (previous) vs track-level (new) person re-identification.

Ground truth without manual labels
----------------------------------
SAME identity  : one ByteTrack track is split into two DISJOINT halves of its
                 views. Both halves are the same person seen at different times,
                 distances and poses - the closest label-free proxy for the
                 cross-camera situation.
DIFFERENT ident: two different tracks in the SAME camera.

Previous method = cosine between ONE representative crop embedding per side,
                  accepted at the old journey threshold (0.78).
New method      = set-to-set descriptor comparison over multiple views, fused
                  across face / ReID / clothing / colour / accessories / body,
                  accepted at the swept track_identity.IDENTITY_ACCEPT.

LIMITATIONS - read before quoting these numbers
-----------------------------------------------
1. Same-identity pairs come from view-halves of ONE track in ONE camera. They
   contain real pose/distance/lighting variation but NO camera change, so
   cross-camera recall in the field will be lower than the recall printed here.
2. Different-identity pairs assume two ByteTrack tracks in one camera are two
   different people. A ByteTrack ID switch breaks that assumption, so the printed
   false-match rate is a PESSIMISTIC upper bound.
3. This cannot measure "motorcycle -> walking accuracy" per camera. That needs
   human identity labels across the four cameras, which do not exist in this
   dataset. The posture breakdown below only shows recall on tracks where a
   ridden vehicle was observed - not verified cross-camera identity.

    python -m scripts.benchmark_track_reid --cameras test1 test2 test3 test4
Run from the backend/ directory.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())

from app import track_identity                                # noqa: E402

OLD_THRESHOLD = 0.78          # journey.IDENTITY_MIN, single-crop cosine
NEW_THRESHOLD = track_identity.IDENTITY_ACCEPT   # swept track-level accept


def _split(desc):
    """Two disjoint half-descriptors of the same track (different views)."""
    out = []
    for half in (0, 1):
        d = {k: v for k, v in desc.items() if not k.startswith("_")}
        for key in ("_reid", "_clip", "_face"):
            m = desc.get(key)
            if m is None or len(m) < 2:
                d[key] = m
            else:
                idx = [i for i in range(len(m)) if i % 2 == half]
                d[key] = m[idx] if idx else m
        out.append(d)
    return out


def _old_score(a, b):
    """Previous behaviour: ONE crop vs ONE crop (single ReID embedding)."""
    ra, rb = a.get("_reid"), b.get("_reid")
    if ra is None or rb is None or not len(ra) or not len(rb):
        return None
    x, y = np.asarray(ra[0], "float32"), np.asarray(rb[0], "float32")
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx < 1e-6 or ny < 1e-6:
        return None
    return float(np.dot(x, y) / (nx * ny))


def _new_score(a, b):
    return track_identity.compare(a, b)["identity"]


def _metrics(same_scores, diff_scores, thr):
    tp = sum(1 for s in same_scores if s >= thr)
    fn = len(same_scores) - tp
    fp = sum(1 for s in diff_scores if s >= thr)
    tn = len(diff_scores) - fp
    n = len(same_scores) + len(diff_scores)
    pc = lambda a, b: (100.0 * a / b) if b else 0.0            # noqa: E731
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": pc(tp, len(same_scores)),
        "frr": pc(fn, len(same_scores)),
        "fmr": pc(fp, len(diff_scores)),
        "precision": pc(tp, tp + fp),
        "accuracy": pc(tp + tn, n),
        "f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "avg_same": float(np.mean(same_scores)) if same_scores else 0.0,
        "avg_diff": float(np.mean(diff_scores)) if diff_scores else 0.0,
    }


def _auc(same_scores, diff_scores):
    s, d = np.asarray(same_scores), np.asarray(diff_scores)
    if not len(s) or not len(d):
        return 0.0
    gt = (s[:, None] > d[None, :]).sum() + 0.5 * (s[:, None] == d[None, :]).sum()
    return float(gt / (len(s) * len(d)))


def _score_all(pairs, scorer):
    out = []
    for a, b in pairs:
        s = scorer(a, b)
        if s is not None:
            out.append(s)
    return out


def _block(title, m, auc=None):
    print(f"  --- {title} ---")
    print(f"  recall (true match)   : {m['recall']:5.1f}%   ({m['tp']}/{m['tp'] + m['fn']})")
    print(f"  FALSE REJECT rate     : {m['frr']:5.1f}%")
    print(f"  FALSE MATCH rate      : {m['fmr']:5.1f}%   ({m['fp']}/{m['fp'] + m['tn']})")
    print(f"  precision             : {m['precision']:5.1f}%")
    print(f"  accuracy              : {m['accuracy']:5.1f}%")
    print(f"  F1                    : {m['f1']:.3f}" + (f"   AUC {auc:.4f}" if auc else ""))
    print(f"  avg score  same/diff  : {m['avg_same']:.3f} / {m['avg_diff']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", nargs="*", default=["test1", "test2", "test3", "test4"])
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--sweep", action="store_true", help="print the full threshold sweep")
    args = ap.parse_args()
    random.seed(42)

    t0 = time.perf_counter()
    stats = track_identity.build_all(camera_ids=args.cameras, min_dets=3)
    t_build = time.perf_counter() - t0
    ds = [d for d in track_identity.list_descriptors(args.cameras)
          if d.get("_reid") is not None and len(d["_reid"]) >= 2]
    print(f"cameras: {args.cameras}")
    print(f"descriptors built: {stats['built']} new / {stats['tracks']} tracks "
          f"in {t_build:.2f}s")
    print(f"tracks with >=2 views: {len(ds)}")
    if len(ds) < 4:
        print("Not enough tracks to benchmark.")
        return

    n_face = sum(1 for d in ds if d.get("has_face"))
    riders = [d for d in ds if d.get("vehicle_context")]
    walkers = [d for d in ds if not d.get("vehicle_context")]
    print(f"tracks with a face embedding: {n_face}/{len(ds)}"
          + ("   (face signal UNAVAILABLE - identity rests on ReID/clothing/body)"
             if not n_face else ""))
    print(f"tracks observed riding a vehicle: {len(riders)} | walking: {len(walkers)}")

    same = [_split(d) for d in ds]
    by_cam: dict = {}
    for d in ds:
        by_cam.setdefault(d["camera_id"], []).append(d)
    diff, tries = [], 0
    while len(diff) < args.pairs and tries < args.pairs * 20:
        tries += 1
        cam = random.choice(list(by_cam))
        if len(by_cam[cam]) < 2:
            continue
        diff.append(tuple(random.sample(by_cam[cam], 2)))

    t0 = time.perf_counter()
    o_same, o_diff = _score_all(same, _old_score), _score_all(diff, _old_score)
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    n_same, n_diff = _score_all(same, _new_score), _score_all(diff, _new_score)
    t_new = time.perf_counter() - t0

    old_m = _metrics(o_same, o_diff, OLD_THRESHOLD)
    new_m = _metrics(n_same, n_diff, NEW_THRESHOLD)

    print("\n============ TRACK-LEVEL RE-ID BENCHMARK ============")
    print(f"  same-identity pairs: {len(n_same)}   different-identity pairs: {len(n_diff)}")
    _block(f"PREVIOUS  single crop, threshold {OLD_THRESHOLD:.2f}", old_m,
           _auc(o_same, o_diff))
    _block(f"NEW  track-level multi-view, threshold {NEW_THRESHOLD:.2f}", new_m,
           _auc(n_same, n_diff))

    print("  --- headline change ---")
    print(f"  accuracy      {old_m['accuracy']:5.1f}% -> {new_m['accuracy']:5.1f}%")
    print(f"  recall        {old_m['recall']:5.1f}% -> {new_m['recall']:5.1f}%")
    print(f"  false reject  {old_m['frr']:5.1f}% -> {new_m['frr']:5.1f}%")
    print(f"  false match   {old_m['fmr']:5.1f}% -> {new_m['fmr']:5.1f}%")
    print(f"  precision     {old_m['precision']:5.1f}% -> {new_m['precision']:5.1f}%")
    print(f"  avg confidence on true pairs  {old_m['avg_same']:.3f} -> {new_m['avg_same']:.3f}")

    print("  --- recall by observed posture (NOT verified cross-camera identity) ---")
    for label, subset in (("riding ", riders), ("walking", walkers)):
        p = [_split(d) for d in subset]
        om = _metrics(_score_all(p, _old_score), [], OLD_THRESHOLD)
        nm = _metrics(_score_all(p, _new_score), [], NEW_THRESHOLD)
        print(f"  {label} tracks  old {om['recall']:5.1f}% -> new {nm['recall']:5.1f}%"
              f"  (n={len(p)})")

    print(f"  runtime: old {1000 * t_old / max(len(o_same) + len(o_diff), 1):.2f} ms/pair"
          f" | new {1000 * t_new / max(len(n_same) + len(n_diff), 1):.2f} ms/pair")

    if args.sweep:
        print("\n  --- threshold sweep (new method) ---")
        print(f"  {'thr':>5} {'recall%':>8} {'FMR%':>7} {'prec%':>7} {'acc%':>7} {'F1':>6}")
        for thr in np.arange(0.60, 0.961, 0.02):
            m = _metrics(n_same, n_diff, float(thr))
            mark = "  <- default" if abs(thr - NEW_THRESHOLD) < 0.01 else ""
            print(f"  {thr:5.2f} {m['recall']:8.1f} {m['fmr']:7.1f} {m['precision']:7.1f} "
                  f"{m['accuracy']:7.1f} {m['f1']:6.3f}{mark}")

    print("=====================================================")
    print("  Caveats: same-identity pairs never cross a camera, so field recall is")
    print("  lower than shown; different-identity pairs assume no ByteTrack ID switch,")
    print("  so the false-match rate is a pessimistic upper bound.")


if __name__ == "__main__":
    main()
