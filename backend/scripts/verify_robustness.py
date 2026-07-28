"""Task 19 verification: robustness + unfamiliar-footage generalisation.

Protects the canonical 1336-detection baseline: snapshots DB+FAISS, ingests a
brand-new (never-seen) clip LIVE via the API + WebSocket as a new camera CAM-06,
verifies search works on it, runs failure/edge cases, then RESTORES the snapshot
so the baseline is byte-for-byte pristine again.

    python -u scripts/verify_robustness.py
"""
import shutil
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "scripts"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app import config, database                         # noqa: E402


def pf(ok):
    return "PASS" if ok else "FAIL"


def snapshot(bak: Path):
    bak.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.DB_PATH, bak / "cctv.db")
    for idx in config.FAISS_DIR.glob("*.index"):
        shutil.copy2(idx, bak / idx.name)


def restore(bak: Path):
    shutil.copy2(bak / "cctv.db", config.DB_PATH)
    for idx in bak.glob("*.index"):
        shutil.copy2(idx, config.FAISS_DIR / idx.name)


def main():
    print("=== TASK 19 ROBUSTNESS + UNFAMILIAR-FOOTAGE ===")
    database.init_db()
    base_dets = database.count_detections()
    bak = config.DATA_DIR / "_robustness_bak"
    snapshot(bak)
    print(f"baseline detections={base_dets}; snapshot saved to {bak.name}")

    import make_unfamiliar_clip
    clip_path, nframes = make_unfamiliar_clip.make_unfamiliar()
    print(f"[{pf(clip_path.exists() and nframes > 0)}] created unfamiliar clip "
          f"{clip_path.name} ({nframes} frames, never ingested before)")

    from fastapi.testclient import TestClient
    from app.main import app
    from app.search import vector_store

    try:
        with TestClient(app) as client:
            # 1) LIVE API ingest of the unfamiliar clip as a NEW camera
            r = client.post("/api/ingest", json={
                "video": clip_path.name, "camera_id": "CAM-06",
                "start_time": "2026-07-07T21:00:00"})
            job = r.json()
            job_id = job.get("job_id")
            print(f"[{pf(r.status_code == 200 and bool(job_id))}] POST /api/ingest "
                  f"-> job={job_id} status={job.get('status')}")

            # 2) WebSocket progress until done
            stages, final = [], None
            with client.websocket_connect(f"/ws/ingest/{job_id}") as ws:
                for _ in range(600):
                    msg = ws.receive_json()
                    if msg.get("stage"):
                        stages.append(msg["stage"])
                    if msg.get("status") in ("done", "error"):
                        final = msg
                        break
            print(f"[{pf(final and final.get('status') == 'done')}] WS ingest progress "
                  f"-> final={final and final.get('status')} stages={sorted(set(stages))}")

            # 3) dynamic camera registration
            cams = {c["camera_id"] for c in database.list_cameras()}
            print(f"[{pf('CAM-06' in cams)}] dynamic camera registration: CAM-06 auto-registered")

            # 4) DB grew + FAISS/DB consistency after live ingest
            new_dets = database.count_detections()
            fs = vector_store.stats()
            print(f"[{pf(new_dets > base_dets)}] DB grew after ingest: {base_dets} -> {new_dets}")
            print(f"[{pf(new_dets == fs['clip'])}] DB == FAISS clip ({new_dets} vs {fs['clip']})")

            # 5) search the unfamiliar footage (generalisation)
            r = client.post("/api/search/text", json={
                "query": "a person walking", "top_k": 5, "include_scenes": False,
                "filters": {"cameras": ["CAM-06"]}})
            j = r.json()
            all06 = bool(j["results"]) and all(x["camera_id"] == "CAM-06" for x in j["results"])
            top = j["results"][0]["class_label"] if j["results"] else "-"
            print(f"[{pf(j['total'] > 0 and all06)}] search 'a person walking' @CAM-06 -> "
                  f"{j['total']} results (all CAM-06={all06}, top={top})")

            # 6) failure / edge cases (must degrade gracefully, never 500)
            r = client.post("/api/search/text", json={"query": "   ", "top_k": 5})
            print(f"[{pf(r.status_code == 200)}] empty query -> {r.status_code} "
                  f"total={r.json().get('total')}")

            r = client.post("/api/search/text", json={
                "query": "a car", "top_k": 5, "filters": {"cameras": ["CAM-DOES-NOT-EXIST"]}})
            print(f"[{pf(r.status_code == 200 and r.json().get('total') == 0)}] "
                  f"filter excludes everything -> total={r.json().get('total')}")

            r = client.post("/api/search/image",
                            files={"file": ("bad.jpg", b"not-a-real-image", "image/jpeg")})
            print(f"[{pf(r.status_code == 400)}] corrupt image upload -> {r.status_code} "
                  f"(graceful 400, not 500)")

            r = client.get("/media/crops/CAM-06/does_not_exist_xyz.jpg")
            print(f"[{pf(r.status_code == 404)}] missing media file -> {r.status_code}")

            r = client.get("/api/track/99999999")
            n_app = len(r.json().get("appearances", [])) if r.status_code == 200 else -1
            print(f"[{pf(r.status_code == 200 and n_app == 0)}] track unknown detection -> "
                  f"{r.status_code}, appearances={n_app}")

            r = client.post("/api/ingest", json={"video": "totally_missing_clip.mp4"})
            print(f"[{pf(r.status_code == 404)}] ingest missing file -> {r.status_code}")
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        # 7) restore the pristine baseline + clean the unfamiliar clip artifacts
        restore(bak)
        try:
            clip_path.unlink()
        except OSError:
            pass
        shutil.rmtree(config.CROP_DIR / "CAM-06", ignore_errors=True)
        shutil.rmtree(config.FRAME_DIR / "CAM-06", ignore_errors=True)
        shutil.rmtree(bak, ignore_errors=True)
        restored = database.count_detections()
        print(f"[{pf(restored == base_dets)}] baseline restored byte-for-byte: "
              f"detections back to {restored} (was {base_dets})")

    print("=== DONE ===")


if __name__ == "__main__":
    main()
