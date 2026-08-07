"""Benchmark identity-preserving tracking: baseline ByteTrack vs the tuned stack.

Two full tracking passes are run over the same videos and compared:

  BASELINE  ultralytics bytetrack.yaml, YOLO conf 0.4, no appearance guard
            (exactly what the project did before this change)
  TUNED     bytetrack_cctv.yaml, YOLO conf TRACK_INPUT_CONF, appearance guard on

Nothing is written to the database: crops and frames go to a temp directory that
is deleted afterwards, so the ingested dataset is untouched.

How the metrics are defined (read this before quoting them)
----------------------------------------------------------
There are no human identity labels for this footage, so APPEARANCE is used as the
referee: person ReID embeddings decide whether two boxes are the same human. Every
metric below is therefore a ReID-based proxy, not a hand-annotated MOT score. They
are still directly comparable between the two runs because both are judged by the
identical referee.

  ID switches        adjacent detections inside ONE track whose appearance
                     similarity falls below BREAK_SIM - the track changed person
  fragmentation      tracks that look like the same person in the same camera and
                     are close in time, but were given different ids. Reported as
                     mean track ids per distinct person
  precision (purity) share of a track's detections that match the track's own
                     medoid appearance - how clean the identity inside a track is
  recall (coverage)  share of the sampled frames inside a track's lifetime that
                     actually carry a box - how little the track drops out
  track duration     seconds from a track's first to last sighting
  cross-camera       same split-half protocol as benchmark_identity_fusion, run on
                     the tracks this pass produced, scored by the fusion engine

    python -m scripts.benchmark_tracking --cameras test1 test2 test3 test4
Run from the backend/ directory. Stop the servers first - this uses the GPU.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.getcwd())

from app import config, database, identity_fusion, track_identity   # noqa: E402
from app.ingestion import reid_embedder                             # noqa: E402

BREAK_SIM = 0.55        # below this, two boxes are different people
SAME_SIM = 0.75         # at/above this, two tracks are the same person
LINK_GAP_S = 45.0       # max time gap when linking fragments of one person


# ---------------------------------------------------------------- helpers
def _unit(v):
    v = np.asarray(v, dtype="float32").ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else v


def _videos(cameras):
    with database.get_conn() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT video_id, camera_id, filename, native_fps FROM videos").fetchall()]
    out = []
    for r in rows:
        if cameras and r["camera_id"] not in cameras:
            continue
        p = config.VIDEO_DIR / r["filename"]
        if p.exists():
            out.append({**r, "path": p})
    return out


def _run_pass(label, videos, tuned: bool, fps: float, quiet: bool = False,
              min_sim=None, recent_min=None):
    """One full tracking pass. Returns per-camera track dicts + timing."""
    from app.ingestion import tracker

    saved = (config.TRACKER_CFG, config.TRACK_APPEARANCE_GUARD,
             config.TRACK_REID_MIN_SIM, config.TRACK_REID_RECENT_MIN)
    if tuned:
        config.TRACK_APPEARANCE_GUARD = True
        if min_sim is not None:
            config.TRACK_REID_MIN_SIM = min_sim
        if recent_min is not None:
            config.TRACK_REID_RECENT_MIN = recent_min
    else:
        config.TRACKER_CFG = "bytetrack.yaml"
        config.TRACK_APPEARANCE_GUARD = False

    tmp = tempfile.mkdtemp(prefix=f"bench_{label}_")
    per_cam, guard_stats, t_total = {}, [], 0.0
    try:
        for v in videos:
            dets = []
            t0 = time.perf_counter()
            for chunk in tracker.iter_track_chunks(
                    v["path"], v["camera_id"], fps=fps,
                    frame_root=os.path.join(tmp, "frames"),
                    crop_root=os.path.join(tmp, "crops"),
                    save_frames=False, chunk_frames=None):
                dets.extend(chunk["dets"])
                if chunk.get("guard"):
                    guard_stats.append(chunk["guard"])
            t_total += time.perf_counter() - t0

            people = [d for d in dets if d.class_id in config.PERSON_CLASSES]
            # make sure every person has an embedding, whichever pass this is
            need = [d for d in people if getattr(d, "reid_vec", None) is None]
            if need:
                vecs = reid_embedder.embed_persons([d.crop_img for d in need])
                for d, vec in zip(need, vecs):
                    d.reid_vec = vec
            tracks = defaultdict(list)
            for d in people:
                tracks[d.track_id].append(d)
            for t in tracks.values():
                t.sort(key=lambda d: d.frame_number)
            per_cam[v["camera_id"]] = {"tracks": dict(tracks),
                                       "native_fps": v.get("native_fps") or 30.0,
                                       "fps": fps}
            if not quiet:
                print(f"    {v['camera_id']:<8} {len(people):5d} person boxes  "
                      f"{len(tracks):4d} tracks")
    finally:
        (config.TRACKER_CFG, config.TRACK_APPEARANCE_GUARD,
         config.TRACK_REID_MIN_SIM, config.TRACK_REID_RECENT_MIN) = saved
        shutil.rmtree(tmp, ignore_errors=True)
    return {"per_cam": per_cam, "seconds": t_total, "guard": guard_stats}


# ---------------------------------------------------------------- metrics
def _track_metrics(per_cam):
    switches = adjacent = 0
    purities, coverages, durations, sizes = [], [], [], []
    n_tracks = 0
    for cam, blob in per_cam.items():
        step_s = 1.0 / max(blob["fps"], 1e-6)
        for tid, dets in blob["tracks"].items():
            if len(dets) < 2:
                n_tracks += 1
                sizes.append(len(dets))
                durations.append(0.0)
                purities.append(1.0)
                coverages.append(1.0)
                continue
            n_tracks += 1
            embs = [_unit(d.reid_vec) for d in dets]
            # --- ID switches: appearance breaks between adjacent detections ---
            for a, b in zip(embs, embs[1:]):
                adjacent += 1
                if float(np.dot(a, b)) < BREAK_SIM:
                    switches += 1
            # --- purity against the track's own medoid ---
            M = np.stack(embs)
            sim = M @ M.T
            medoid = int(np.argmax(sim.sum(axis=1)))
            purities.append(float((sim[medoid] >= BREAK_SIM).mean()))
            # --- coverage of the track's own lifetime ---
            span_frames = (dets[-1].frame_number - dets[0].frame_number)
            fps_native = blob["native_fps"]
            span_s = span_frames / max(fps_native, 1e-6)
            expected = max(1, int(round(span_s / step_s)) + 1)
            coverages.append(min(1.0, len(dets) / expected))
            durations.append(span_s)
            sizes.append(len(dets))

    # --- fragmentation: how many ids one person was split across ---
    frags = []
    for cam, blob in per_cam.items():
        items = []
        for tid, dets in blob["tracks"].items():
            if len(dets) < 2:
                continue
            items.append((tid, np.stack([_unit(d.reid_vec) for d in dets]),
                          dets[0].frame_number, dets[-1].frame_number))
        if not items:
            continue
        fps_native = blob["native_fps"]
        parent = {t[0]: t[0] for t in items}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(len(items)):
            for jj in range(i + 1, len(items)):
                a, b = items[i], items[jj]
                gap = (max(a[2], b[2]) - min(a[3], b[3])) / max(fps_native, 1e-6)
                if gap > LINK_GAP_S:
                    continue
                if float((a[1] @ b[1].T).max()) >= SAME_SIM:
                    ra, rb = find(a[0]), find(b[0])
                    if ra != rb:
                        parent[ra] = rb
        clusters = defaultdict(int)
        for tid in parent:
            clusters[find(tid)] += 1
        frags.extend(clusters.values())

    mean = lambda xs: (statistics.mean(xs) if xs else 0.0)              # noqa: E731
    return {
        "tracks": n_tracks,
        "detections": sum(sizes),
        "id_switches": switches,
        "adjacent_pairs": adjacent,
        "switch_rate": (100.0 * switches / adjacent) if adjacent else 0.0,
        "fragmentation": mean(frags),
        "identities": len(frags),
        "precision": 100.0 * mean(purities),
        "recall": 100.0 * mean(coverages),
        "avg_duration_s": mean(durations),
        "avg_detections": mean(sizes),
    }


def _cross_camera(per_cam, pairs=300, seed=42):
    """Cross-camera recall / false-match / confidence via the fusion engine.

    Same label-free protocol as benchmark_identity_fusion: same-identity pairs are
    disjoint view-halves of one track, different-identity pairs are two tracks in
    one camera."""
    random.seed(seed)
    descs = []
    for cam, blob in per_cam.items():
        for tid, dets in blob["tracks"].items():
            if len(dets) < 4:
                continue
            embs = np.stack([_unit(d.reid_vec) for d in dets])
            idx = np.linspace(0, len(embs) - 1, min(10, len(embs))).astype(int)
            descs.append({"camera_id": cam, "track_id": tid, "_reid": embs[idx],
                          "_clip": None, "_face": None,
                          "upper_color": None, "lower_color": None,
                          "accessories": [], "body_ratio": float(np.median(
                              [d.bbox[3] / max(d.bbox[2], 1e-6) for d in dets]))})
    if len(descs) < 4:
        return None

    def half(d, which):
        m = d["_reid"]
        sel = [i for i in range(len(m)) if i % 2 == which] or list(range(len(m)))
        return {**d, "_reid": m[sel]}

    same = [(half(d, 0), half(d, 1)) for d in descs]
    by_cam = defaultdict(list)
    for d in descs:
        by_cam[d["camera_id"]].append(d)
    diff, tries = [], 0
    while len(diff) < pairs and tries < pairs * 20:
        tries += 1
        cam = random.choice(list(by_cam))
        if len(by_cam[cam]) >= 2:
            diff.append(tuple(random.sample(by_cam[cam], 2)))

    thr = track_identity.IDENTITY_ACCEPT
    s = [track_identity.compare(a, b)["identity"] for a, b in same]
    d = [track_identity.compare(a, b)["identity"] for a, b in diff]
    tp = sum(1 for x in s if x >= thr)
    fp = sum(1 for x in d if x >= thr)
    return {"tracks_compared": len(descs), "same_pairs": len(s), "diff_pairs": len(d),
            "recall": 100.0 * tp / max(len(s), 1),
            "false_match": 100.0 * fp / max(len(d), 1),
            "precision": 100.0 * tp / max(tp + fp, 1),
            "avg_identity_confidence": float(np.mean(s)) if s else 0.0,
            "threshold": thr}


# ---------------------------------------------------------------- report
def _delta(a, b, higher_better=True, pct=True):
    d = b - a
    good = (d > 0) if higher_better else (d < 0)
    arrow = "better" if (good and abs(d) > 1e-9) else ("worse" if abs(d) > 1e-9 else "same")
    unit = "pp" if pct else ""
    return f"{a:8.2f} -> {b:8.2f}  ({d:+.2f}{unit}, {arrow})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", nargs="*", default=["test1", "test2", "test3", "test4"])
    ap.add_argument("--fps", type=float, default=config.DEFAULT_FPS)
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the guard thresholds instead of a single comparison")
    args = ap.parse_args()

    if args.sweep:
        videos = _videos(args.cameras)
        print("=== GUARD THRESHOLD SWEEP (tracking pass re-run per setting) ===")
        print(f"  {'set':>5} {'recent':>7} {'switch%':>8} {'frag':>6} {'purity%':>8} "
              f"{'cover%':>7} {'dur s':>7} {'tracks':>7} {'refused':>8}")
        for min_sim, recent in ((0.62, 0.00), (0.62, 0.45), (0.62, 0.55), (0.62, 0.65),
                                (0.70, 0.55), (0.70, 0.65), (0.78, 0.65)):
            r = _run_pass("sw", videos, tuned=True, fps=args.fps, quiet=True,
                          min_sim=min_sim, recent_min=recent)
            m = _track_metrics(r["per_cam"])
            ref = sum(x.get("switches_blocked", 0) for x in r["guard"])
            print(f"  {min_sim:5.2f} {recent:7.2f} {m['switch_rate']:8.2f} "
                  f"{m['fragmentation']:6.2f} {m['precision']:8.2f} {m['recall']:7.2f} "
                  f"{m['avg_duration_s']:7.2f} {m['tracks']:7d} {ref:8d}")
        return

    videos = _videos(args.cameras)
    if not videos:
        print(f"No source videos found for {args.cameras} in {config.VIDEO_DIR}")
        return
    print("======== IDENTITY-PRESERVING TRACKING BENCHMARK ========")
    print(f"videos: {[v['camera_id'] for v in videos]}   sampling {args.fps} fps")
    print(f"guard thresholds: continue >= {config.TRACK_REID_MIN_SIM}, "
          f"re-acquire >= {config.TRACK_REACQUIRE_MIN_SIM}, "
          f"lost window {config.TRACK_LOST_WINDOW} frames")

    print("\n[1/2] BASELINE pass (bytetrack.yaml, conf 0.4, no guard)")
    base = _run_pass("base", videos, tuned=False, fps=args.fps)
    print("\n[2/2] TUNED pass (bytetrack_cctv.yaml, low-conf input, appearance guard)")
    tune = _run_pass("tuned", videos, tuned=True, fps=args.fps)

    bm, tm = _track_metrics(base["per_cam"]), _track_metrics(tune["per_cam"])

    print("\n--- SINGLE-CAMERA TRACKING ---")
    print(f"  {'metric':<28} {'baseline':>8}    {'tuned':>8}")
    print(f"  {'ID switches (count)':<28} {_delta(bm['id_switches'], tm['id_switches'], False, False)}")
    print(f"  {'ID switch rate %':<28} {_delta(bm['switch_rate'], tm['switch_rate'], False)}")
    print(f"  {'fragmentation (ids/person)':<28} {_delta(bm['fragmentation'], tm['fragmentation'], False, False)}")
    print(f"  {'precision (purity) %':<28} {_delta(bm['precision'], tm['precision'], True)}")
    print(f"  {'recall (coverage) %':<28} {_delta(bm['recall'], tm['recall'], True)}")
    print(f"  {'avg track duration s':<28} {_delta(bm['avg_duration_s'], tm['avg_duration_s'], True, False)}")
    print(f"  {'avg detections / track':<28} {_delta(bm['avg_detections'], tm['avg_detections'], True, False)}")
    print(f"  {'tracks created':<28} {bm['tracks']:8d} -> {tm['tracks']:8d}")
    print(f"  {'person detections stored':<28} {bm['detections']:8d} -> {tm['detections']:8d}")

    bc, tc = _cross_camera(base["per_cam"]), _cross_camera(tune["per_cam"])
    if bc and tc:
        print(f"\n--- CROSS-CAMERA IDENTITY (fusion engine, threshold {tc['threshold']}) ---")
        print(f"  {'cross-camera recall %':<28} {_delta(bc['recall'], tc['recall'], True)}")
        print(f"  {'false match rate %':<28} {_delta(bc['false_match'], tc['false_match'], False)}")
        print(f"  {'precision %':<28} {_delta(bc['precision'], tc['precision'], True)}")
        print(f"  {'avg identity confidence':<28} "
              f"{_delta(bc['avg_identity_confidence'], tc['avg_identity_confidence'], True, False)}")
        print(f"  pairs: baseline {bc['same_pairs']}/{bc['diff_pairs']}  "
              f"tuned {tc['same_pairs']}/{tc['diff_pairs']}")

    if tune["guard"]:
        # one guard per video, so sum across them rather than reading the last
        keys = ("checked", "accepted", "rejected", "reacquired",
                "new_identities", "switches_blocked")
        g = {k: sum(x.get(k, 0) for x in tune["guard"]) for k in keys}
        g["reject_rate"] = g["rejected"] / max(g["checked"], 1)
        print("\n--- APPEARANCE GUARD ACTIVITY (tuned pass) ---")
        print(f"  associations checked      : {g['checked']}")
        print(f"  accepted                  : {g['accepted']}")
        print(f"  REFUSED (switch prevented): {g['switches_blocked']}  "
              f"({g['reject_rate']:.1%} of checks)")
        print(f"  re-acquired after occlusion: {g['reacquired']}")
        print(f"  new identities created    : {g['new_identities']}")

    print("\n--- RUNTIME ---")
    print(f"  baseline tracking pass : {base['seconds']:.1f} s")
    print(f"  tuned tracking pass    : {tune['seconds']:.1f} s "
          f"({tune['seconds'] / max(base['seconds'], 1e-6):.2f}x)")
    print("  note: the tuned pass embeds people during tracking, but the pipeline")
    print("  reuses those vectors instead of re-embedding, so end-to-end ingestion")
    print("  does not pay for them twice.")
    print("\n========================================================")
    print("  Metrics are ReID-refereed proxies, not hand-annotated MOT scores;")
    print("  both runs are judged by the identical referee so the comparison holds.")


if __name__ == "__main__":
    main()
