"""Project cleanup: keep ONLY the current .mp4 videos + their data, drop the rest.

Keeps every video whose .mp4 file is present in the video dir (currently ids
86-90) and removes everything tied to any other video:
  * DB rows: detections / tracks / faces / plates / videos  (cameras kept)
  * their crop + frame folders on disk
  * old / orphaned FAISS vectors (indexes are rebuilt keeping only kept ids -
    this also clears ~240k stale vectors and is the biggest size win)
  * leftover non-mp4 source files (the .avi originals)
  * stale export zips (regenerable)

KEPT intact: app code, .venv, models, cctv.db (pruned + backed up first),
camera_config.json, the cameras table (drives the map), and the 5 .mp4 clips.

Dry-run by default (prints the plan, changes nothing). Pass --apply to execute.
STOP THE BACKEND before --apply so the DB / FAISS aren't in use.

    python scripts/cleanup_project.py            # preview
    python scripts/cleanup_project.py --apply     # execute
"""
import os
import shutil
import sqlite3
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app import config, database                      # noqa: E402


def dir_size(p: Path) -> int:
    total = 0
    if p.exists():
        for root, _d, files in os.walk(p):
            for f in files:
                try:
                    total += (Path(root) / f).stat().st_size
                except OSError:
                    pass
    return total


def mb(n) -> str:
    return f"{n / 1048576:.1f} MB"


def main(apply: bool):
    VIDEO_DIR = config.VIDEO_DIR

    # ---- keep set = videos whose .mp4 file is present on disk ----
    keep_ids, keep_cams, keep_files = set(), set(), set()
    for v in database.list_videos():
        fn = v.get("filename") or ""
        if fn.lower().endswith(".mp4") and (VIDEO_DIR / fn).exists():
            keep_ids.add(v["video_id"]); keep_cams.add(v["camera_id"]); keep_files.add(fn)
    if not keep_ids:
        print("ABORT: no kept videos (no present .mp4). Refusing to wipe everything.")
        return
    print("KEEP video_ids:", sorted(keep_ids))
    print("KEEP cameras  :", sorted(keep_cams))
    print("KEEP files    :", len(keep_files), "mp4(s)")

    kp = ",".join("?" * len(keep_ids))
    kc = ",".join("?" * len(keep_cams))

    # ---- capture ids for FAISS rebuild BEFORE any delete ----
    with database.get_conn() as conn:
        kept_det_ids = [r[0] for r in conn.execute(
            f"SELECT detection_id FROM detections WHERE video_id IN ({kp})", tuple(keep_ids))]
        kept_face_ids = [r[0] for r in conn.execute(
            f"SELECT face_id FROM faces WHERE camera_id IN ({kc})", tuple(keep_cams))]
        rows = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("detections", "tracks", "faces", "plates", "videos", "cameras")}
    print(f"kept detection_ids={len(kept_det_ids)}  kept face_ids={len(kept_face_ids)}")
    print("DB rows now:", rows)

    # ---- crop/frame/face camera dirs to delete ----
    del_dirs = []
    keep_sz = drop_sz = 0
    for base in (config.CROP_DIR, config.FRAME_DIR, config.FACE_DIR):
        if not base.exists():
            continue
        for sub in base.iterdir():
            if not sub.is_dir():
                continue
            if sub.name in keep_cams:
                keep_sz += dir_size(sub)
            else:
                sz = dir_size(sub)
                drop_sz += sz
                del_dirs.append((sub, sz))

    # ---- non-kept source files (the .avi) ----
    del_videos, del_vid_sz = [], 0
    for f in VIDEO_DIR.iterdir():
        if f.is_file() and f.name not in keep_files:
            del_videos.append(f); del_vid_sz += f.stat().st_size

    exp_sz = dir_size(config.EXPORT_DIR)
    faiss_files = list(config.FAISS_DIR.glob("*.index"))
    faiss_sz = sum(f.stat().st_size for f in faiss_files)

    print("\n--- PLAN ---")
    print(f"crop/frame data KEPT : {mb(keep_sz)}")
    print(f"crop/frame dirs DROP : {len(del_dirs)} dirs, {mb(drop_sz)}")
    print(f"source files DROP    : {len(del_videos)} files, {mb(del_vid_sz)}  ({[f.name[:30] for f in del_videos]})")
    print(f"exports cleared      : {mb(exp_sz)}")
    print(f"FAISS now            : {mb(faiss_sz)} -> rebuilt to kept-only")
    print(f"~space freed (files) : {mb(drop_sz + del_vid_sz + exp_sz)}  (+ FAISS shrink + DB VACUUM)")

    if not apply:
        print("\nDRY RUN - nothing changed. Re-run with --apply to execute.")
        return

    # ===================== APPLY =====================
    print("\n=== APPLYING ===")
    bak = config.DB_PATH.with_name("cctv.db.bak")
    shutil.copy2(config.DB_PATH, bak)
    print(f"[backup] DB -> {bak.name} ({mb(bak.stat().st_size)})")

    # 1) rebuild FAISS in memory first (abort if clip result looks wrong) ----
    import faiss                                        # noqa: E402
    import numpy as np                                  # noqa: E402
    from app.search import vector_store                 # noqa: E402
    vector_store._indexes.clear()

    def build(name, ids):
        old = vector_store.get_index(name)
        before = old.ntotal
        dim = {"clip": config.CLIP_DIM, "reid": config.REID_DIM, "face": config.FACE_DIM}[name]
        new = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        vecs, keep = [], []
        for i in ids:
            try:
                vecs.append(old.reconstruct(int(i))); keep.append(int(i))
            except (RuntimeError, TypeError, ValueError):
                continue
        if vecs:
            new.add_with_ids(np.ascontiguousarray(np.stack(vecs), "float32"),
                             np.ascontiguousarray(keep, "int64"))
        return new, before

    clip_idx, clip_before = build("clip", kept_det_ids)
    reid_idx, reid_before = build("reid", kept_det_ids)   # non-persons skipped automatically
    face_idx, face_before = build("face", kept_face_ids)

    if clip_idx.ntotal < 0.5 * len(kept_det_ids):
        print(f"ABORT: clip rebuild kept only {clip_idx.ntotal}/{len(kept_det_ids)} - "
              f"unexpected, refusing to delete. (DB/FAISS untouched.)")
        return

    for name, idx in (("clip", clip_idx), ("reid", reid_idx), ("face", face_idx)):
        tmp = config.FAISS_DIR / f"{name}.index.tmp"
        faiss.write_index(idx, str(tmp))
        os.replace(tmp, config.FAISS_DIR / f"{name}.index")
    print(f"[faiss] clip {clip_before}->{clip_idx.ntotal}  reid {reid_before}->{reid_idx.ntotal}  "
          f"face {face_before}->{face_idx.ntotal}")

    # 2) prune DB ----
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.execute(f"DELETE FROM detections WHERE video_id NOT IN ({kp})", tuple(keep_ids))
    conn.execute(f"DELETE FROM tracks     WHERE video_id NOT IN ({kp})", tuple(keep_ids))
    conn.execute(f"DELETE FROM faces      WHERE camera_id NOT IN ({kc})", tuple(keep_cams))
    conn.execute(f"DELETE FROM plates     WHERE camera_id NOT IN ({kc})", tuple(keep_cams))
    conn.execute(f"DELETE FROM videos     WHERE video_id NOT IN ({kp})", tuple(keep_ids))
    conn.execute("DELETE FROM exports")
    conn.commit()
    conn.close()
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.isolation_level = None
    conn.execute("VACUUM")
    conn.close()
    print("[db] pruned + VACUUM")

    # 3) delete crop/frame/face dirs ----
    for d, _sz in del_dirs:
        shutil.rmtree(d, ignore_errors=True)
    print(f"[files] deleted {len(del_dirs)} crop/frame dirs")

    # 4) delete non-kept source files ----
    for f in del_videos:
        try:
            f.unlink()
        except OSError:
            pass
    print(f"[files] deleted {len(del_videos)} source files")

    # 5) clear exports ----
    if config.EXPORT_DIR.exists():
        for f in config.EXPORT_DIR.iterdir():
            if f.is_file():
                f.unlink()
            else:
                shutil.rmtree(f, ignore_errors=True)
    print("[files] cleared exports")

    print(f"\nDONE. data/ now: {mb(dir_size(config.DATA_DIR))}. Restart the backend.")


if __name__ == "__main__":
    main("--apply" in sys.argv)
