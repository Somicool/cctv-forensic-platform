"""Task 15 verification: exercise the FastAPI routes with the in-process test
client (no live server needed).

Does NOT mutate the ingested DB: the ingest route is checked via its validation
(404) path, and the WebSocket via a seeded job - real ingest correctness is
already covered by reingest_and_verify.py.

    python -u scripts/verify_api.py
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from fastapi.testclient import TestClient          # noqa: E402
from app import database, ingest_jobs              # noqa: E402
from app.main import app                           # noqa: E402
from app.search.text_search import media_url       # noqa: E402


def pf(ok):
    return "PASS" if ok else "FAIL"


def main():
    # Real ids/paths from the ingested DB for the read-only route tests.
    with database.get_conn() as conn:
        person = conn.execute(
            "SELECT detection_id, crop_path FROM detections "
            "WHERE class_label='person' AND crop_path IS NOT NULL LIMIT 1").fetchone()
        anycrop = conn.execute(
            "SELECT crop_path FROM detections WHERE crop_path IS NOT NULL "
            "AND class_label!='scene' LIMIT 1").fetchone()
        facerow = conn.execute(
            "SELECT d.crop_path FROM faces f JOIN detections d "
            "ON f.detection_id=d.detection_id LIMIT 1").fetchone()

    with TestClient(app) as client:
        print("=== TASK 15 API VERIFICATION ===")

        r = client.get("/api/health")
        j = r.json()
        print(f"[{pf(r.status_code == 200 and j['status'] == 'ok')}] GET /api/health -> "
              f"{r.status_code} device={j.get('device')} cameras={j.get('cameras')}")

        r = client.get("/api/cameras")
        cams = r.json()
        print(f"[{pf(r.status_code == 200 and len(cams) > 0)}] GET /api/cameras -> "
              f"{len(cams)} cameras")

        # text search (English), scenes excluded -> object result on top
        r = client.post("/api/search/text",
                        json={"query": "a white truck", "top_k": 3, "include_scenes": False})
        j = r.json()
        top = j["results"][0]["class_label"] if j["results"] else "-"
        print(f"[{pf(r.status_code == 200 and j['total'] > 0 and top == 'truck')}] "
              f"POST /api/search/text 'a white truck' -> {j['total']} (top={top})")

        # text search (Hindi -> translated to English internally)
        r = client.post("/api/search/text",
                        json={"query": "सफ़ेद ट्रक", "language": "hi", "top_k": 3,
                              "include_scenes": False})
        j = r.json()
        print(f"[{pf(r.status_code == 200 and j['total'] > 0 and bool(j.get('translated_query')))}] "
              f"POST /api/search/text hi -> {j['total']} translated={j.get('translated_query')!r}")

        # image upload (CLIP visual similarity)
        if anycrop:
            data = Path(anycrop["crop_path"]).read_bytes()
            r = client.post("/api/search/image",
                            files={"file": ("crop.jpg", data, "image/jpeg")},
                            data={"top_k": "5"})
            j = r.json()
            print(f"[{pf(r.status_code == 200 and j['total'] > 0)}] "
                  f"POST /api/search/image (upload) -> {j['total']} results")

        # face upload (self-match against an ingested face)
        if facerow:
            data = Path(facerow["crop_path"]).read_bytes()
            r = client.post("/api/search/face",
                            files={"file": ("face.jpg", data, "image/jpeg")},
                            data={"top_k": "5"})
            j = r.json()
            print(f"[{pf(r.status_code == 200 and j['total'] > 0)}] "
                  f"POST /api/search/face (upload) -> {j['total']} results")

        # plate search (SQL only; 0 on our data, but route + shape must work)
        r = client.post("/api/search/plate", json={"plate": "GJ05"})
        j = r.json()
        print(f"[{pf(r.status_code == 200 and 'results' in j)}] "
              f"POST /api/search/plate 'GJ05' -> {j['total']} results")

        # cross-camera track
        if person:
            r = client.get(f"/api/track/{person['detection_id']}")
            j = r.json()
            n = len(j.get("appearances", []))
            print(f"[{pf(r.status_code == 200 and n > 0)}] "
                  f"GET /api/track/{person['detection_id']} -> {n} appearances")

        # /media static file serving
        if anycrop:
            url = media_url(anycrop["crop_path"])
            r = client.get(url)
            ok = r.status_code == 200 and r.headers.get("content-type", "").startswith("image")
            print(f"[{pf(ok)}] GET {url} -> {r.status_code} ({r.headers.get('content-type')})")

        # audit log
        r = client.get("/api/audit?limit=5")
        print(f"[{pf(r.status_code == 200 and isinstance(r.json(), list))}] "
              f"GET /api/audit -> {len(r.json())} entries")

        # ingest validation path (missing file -> 404, and NO DB mutation)
        before = database.count_detections()
        r = client.post("/api/ingest", json={"video": "does_not_exist_xyz.mp4"})
        after = database.count_detections()
        print(f"[{pf(r.status_code == 404 and after == before)}] "
              f"POST /api/ingest (missing file) -> {r.status_code}, dets {before}=={after}")

        # websocket progress via a seeded job (proves streaming, no real ingest)
        ingest_jobs.set_job("verifyjob", {"job_id": "verifyjob", "video": "x",
                                          "status": "done", "stage": "done", "pct": 100,
                                          "message": "done", "stats": {"object_detections": 5}})
        with client.websocket_connect("/ws/ingest/verifyjob") as ws:
            msg = ws.receive_json()
        print(f"[{pf(msg.get('status') == 'done' and msg.get('pct') == 100)}] "
              f"WS /ws/ingest (seeded job) -> status={msg.get('status')} pct={msg.get('pct')}")

        with client.websocket_connect("/ws/ingest/unknownjob") as ws:
            msg = ws.receive_json()
        print(f"[{pf(msg.get('status') == 'unknown')}] "
              f"WS /ws/ingest (unknown job) -> status={msg.get('status')}")

        print("=== DONE ===")


if __name__ == "__main__":
    main()
