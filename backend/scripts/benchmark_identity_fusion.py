"""Benchmark: ReID-dominated scoring vs the Identity Fusion Engine.

The confirmation threshold is HELD FIXED at track_identity.IDENTITY_ACCEPT. The
question this answers is: with the threshold unchanged, does fusing many
partly-independent evidence sources recover matches that a ReID-dominated score
rejected, and at what cost in false matches?

Three scorers are compared on identical pairs:
  REID-ONLY   raw track-level ReID set-to-set similarity (the signal that used to
              dominate the decision)
  LEGACY      the previous weighted mean (reid carried ~56% of available weight
              once face is unavailable), no corroboration, no contradiction
  FUSION      the Identity Fusion Engine: 8 appearance sources in 5 independent
              groups, corroboration uplift, contradiction penalty, plus the
              spatio-temporal context sources when available

Ground truth is label-free, exactly as in benchmark_track_reid:
  SAME identity  = two disjoint view-halves of one ByteTrack track
  DIFFERENT      = two different tracks in the same camera
See that file for the limitations; they apply here unchanged.

    python -m scripts.benchmark_identity_fusion --sweep
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

from app import identity_fusion, track_identity                # noqa: E402

THR = track_identity.IDENTITY_ACCEPT      # held fixed, never lowered
LEGACY_WEIGHTS = {"face": 0.40, "reid": 0.28, "clothing": 0.10,
                  "colour": 0.10, "accessories": 0.05, "body": 0.07}


def _split(desc):
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


def _reid_only(a, b):
    return track_identity.set_similarity(a.get("_reid"), b.get("_reid"))


def _legacy(a, b):
    sig = {}
    for name, key in (("face", "_face"), ("reid", "_reid"), ("clothing", "_clip")):
        s = track_identity.set_similarity(a.get(key), b.get(key))
        if s is not None:
            sig[name] = s
    for name, fn in (("colour", track_identity._colour_sim),
                     ("accessories", track_identity._accessory_sim),
                     ("body", track_identity._body_sim)):
        s = fn(a, b)
        if s is not None:
            sig[name] = s
    if not sig:
        return None
    w = sum(LEGACY_WEIGHTS.get(k, 0.0) for k in sig)
    return sum(LEGACY_WEIGHTS.get(k, 0.0) * v for k, v in sig.items()) / max(w, 1e-6)


def _fusion(a, b):
    return track_identity.compare(a, b)["identity"]


def _metrics(same, diff, thr):
    tp = sum(1 for s in same if s >= thr)
    fn = len(same) - tp
    fp = sum(1 for s in diff if s >= thr)
    tn = len(diff) - fp
    pc = lambda x, y: (100.0 * x / y) if y else 0.0            # noqa: E731
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": pc(tp, len(same)), "frr": pc(fn, len(same)),
            "fmr": pc(fp, len(diff)), "precision": pc(tp, tp + fp),
            "accuracy": pc(tp + tn, len(same) + len(diff)),
            "f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
            "avg_same": float(np.mean(same)) if same else 0.0,
            "avg_diff": float(np.mean(diff)) if diff else 0.0}


def _auc(same, diff):
    s, d = np.asarray(same), np.asarray(diff)
    if not len(s) or not len(d):
        return 0.0
    return float(((s[:, None] > d[None, :]).sum()
                  + 0.5 * (s[:, None] == d[None, :]).sum()) / (len(s) * len(d)))


def _scores(pairs, fn):
    out = []
    for a, b in pairs:
        s = fn(a, b)
        if s is not None:
            out.append(s)
    return out


def _row(name, m, auc):
    print(f"  {name:<12} {m['recall']:7.1f} {m['frr']:7.1f} {m['fmr']:7.1f} "
          f"{m['precision']:8.1f} {m['accuracy']:7.1f} {m['f1']:6.3f} {auc:6.3f} "
          f"{m['avg_same']:8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", nargs="*", default=["test1", "test2", "test3", "test4"])
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    random.seed(42)

    track_identity.build_all(camera_ids=args.cameras, min_dets=3)
    ds = [d for d in track_identity.list_descriptors(args.cameras)
          if d.get("_reid") is not None and len(d["_reid"]) >= 2]
    if len(ds) < 4:
        print("Not enough tracks to benchmark.")
        return
    n_face = sum(1 for d in ds if d.get("has_face"))
    print(f"cameras: {args.cameras}   tracks: {len(ds)}   with face: {n_face}/{len(ds)}")
    print(f"confirmation threshold HELD at {THR:.2f} for every scorer")

    same = [_split(d) for d in ds]
    by_cam: dict = {}
    for d in ds:
        by_cam.setdefault(d["camera_id"], []).append(d)
    diff, tries = [], 0
    while len(diff) < args.pairs and tries < args.pairs * 20:
        tries += 1
        cam = random.choice(list(by_cam))
        if len(by_cam[cam]) >= 2:
            diff.append(tuple(random.sample(by_cam[cam], 2)))
    print(f"same-identity pairs: {len(same)}   different-identity pairs: {len(diff)}")

    scorers = (("REID-ONLY", _reid_only), ("LEGACY", _legacy), ("FUSION", _fusion))
    print("\n=========== IDENTITY FUSION ENGINE BENCHMARK ===========")
    print(f"  {'scorer':<12} {'recall%':>7} {'FRR%':>7} {'FMR%':>7} {'prec%':>8} "
          f"{'acc%':>7} {'F1':>6} {'AUC':>6} {'avgConf':>8}")
    table = {}
    for name, fn in scorers:
        t0 = time.perf_counter()
        s, d = _scores(same, fn), _scores(diff, fn)
        dt = time.perf_counter() - t0
        m = _metrics(s, d, THR)
        table[name] = (m, _auc(s, d), 1000 * dt / max(len(s) + len(d), 1))
        _row(name, m, table[name][1])

    ro, lg, fu = table["REID-ONLY"][0], table["LEGACY"][0], table["FUSION"][0]
    print("\n  --- what fusion changed (threshold unchanged at "
          f"{THR:.2f}) ---")
    print(f"  recall     ReID-only {ro['recall']:5.1f}%  legacy {lg['recall']:5.1f}%"
          f"  -> fusion {fu['recall']:5.1f}%")
    print(f"  false rej  ReID-only {ro['frr']:5.1f}%  legacy {lg['frr']:5.1f}%"
          f"  -> fusion {fu['frr']:5.1f}%")
    print(f"  false mat  ReID-only {ro['fmr']:5.1f}%  legacy {lg['fmr']:5.1f}%"
          f"  -> fusion {fu['fmr']:5.1f}%")
    print(f"  precision  ReID-only {ro['precision']:5.1f}%  legacy {lg['precision']:5.1f}%"
          f"  -> fusion {fu['precision']:5.1f}%")
    print(f"  accuracy   ReID-only {ro['accuracy']:5.1f}%  legacy {lg['accuracy']:5.1f}%"
          f"  -> fusion {fu['accuracy']:5.1f}%")
    print(f"  recovered  {fu['tp'] - lg['tp']:+d} true matches, "
          f"{fu['fp'] - lg['fp']:+d} false matches vs legacy")
    print(f"  runtime    fusion {table['FUSION'][2]:.3f} ms/pair")

    # how much of the decision ReID actually carries now
    ev = track_identity.appearance_evidence(*same[0])
    avail = [k for k in identity_fusion.APPEARANCE_SIGNALS if k in ev]
    wsum = sum(identity_fusion.APPEARANCE_SIGNALS[k]["weight"] for k in avail)
    print("\n  --- evidence weighting on this dataset ---")
    print(f"  available appearance sources: {', '.join(avail)}")
    for k in avail:
        w = identity_fusion.APPEARANCE_SIGNALS[k]["weight"]
        print(f"    {identity_fusion.APPEARANCE_SIGNALS[k]['label']:<24} "
              f"{100 * w / wsum:5.1f}% of the decision")

    if args.sweep:
        print("\n  --- corroboration-uplift sweep (threshold fixed) ---")
        print(f"  {'uplift':>7} {'recall%':>8} {'FMR%':>7} {'prec%':>7} {'acc%':>7} {'F1':>6}")
        original = identity_fusion.CORROBORATION_MAX
        for u in (0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85):
            identity_fusion.CORROBORATION_MAX = u
            s, d = _scores(same, _fusion), _scores(diff, _fusion)
            m = _metrics(s, d, THR)
            mark = "  <- default" if abs(u - original) < 1e-9 else ""
            print(f"  {u:7.2f} {m['recall']:8.1f} {m['fmr']:7.1f} {m['precision']:7.1f} "
                  f"{m['accuracy']:7.1f} {m['f1']:6.3f}{mark}")
        identity_fusion.CORROBORATION_MAX = original

    # worked example of the explanation attached to every confirmed match
    print("\n  --- example explanation (one confirmed pair) ---")
    for a, b in same:
        r = track_identity.compare(a, b)
        if r["identity"] >= THR and r.get("fusion"):
            f = r["fusion"]
            print(f"  identity {r['identity']:.3f} ({r['tier']})   "
                  f"appearance {f['appearance_score']:.3f}  uplift +{f['uplift']:.3f}  "
                  f"penalty -{f['penalty']:.3f}")
            for c in f["contributions"]:
                if c["kind"] == "appearance" and c["value"] is not None:
                    print(f"    {c['label']:<24} {c['pct']:3d}%  weight {c['weight']:.2f}  "
                          f"share {100 * c['share']:4.1f}%  {c['verdict']}")
            for line in f["explanation"]:
                print(f"    - {line}")
            break
    print("========================================================")


if __name__ == "__main__":
    main()
