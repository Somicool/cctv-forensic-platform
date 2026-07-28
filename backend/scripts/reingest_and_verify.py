"""Clean re-ingest of all clips + integrity/verification checks in ONE process
(no concurrency, so the DB and FAISS stay consistent). Prints PASS/FAIL lines.

    python -u scripts/reingest_and_verify.py
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import config, database                       # noqa: E402
from app.ingestion import pipeline                     # noqa: E402
from app.search import vector_store, text_search, face_search, plate_search  # noqa: E402


def main():
    database.init_db()
    pipeline.reset_all()
    pipeline.ingest_directory(start_time="2026-07-07T20:00:00")

    dets = database.count_detections()
    faces = database.count_faces()
    fs = vector_store.stats()
    print("\n=== POST-INGEST VERIFICATION ===")
    print(f"DB detections={dets}  faces={faces}  faiss={fs}")

    ok_consistency = dets == fs["clip"]
    print(f"[{'PASS' if ok_consistency else 'FAIL'}] DB detections == faiss clip "
          f"({dets} vs {fs['clip']})")

    r = text_search.search_text("a white truck", top_k=3, include_scenes=False)
    top = r.results[0].class_label if r.results else "-"
    print(f"[{'PASS' if r.total > 0 else 'FAIL'}] text 'a white truck' -> "
          f"{r.total} results (top={top})")

    with database.get_conn() as conn:
        row = conn.execute(
            "SELECT f.face_id, d.crop_path FROM faces f "
            "JOIN detections d ON f.detection_id = d.detection_id LIMIT 1"
        ).fetchone()
    if row:
        fr = face_search.search_by_face(row["crop_path"], top_k=5)
        ok_face = fr.total > 0
        print(f"[{'PASS' if ok_face else 'FAIL'}] face search -> {fr.total} results")
        for x in fr.results[:3]:
            print(f"    cam={x.camera_id} det={x.detection_id} score={x.score:.3f} "
                  f"age={x.attributes.get('age')} gender={x.attributes.get('gender')}")
    else:
        print("[WARN] no faces in DB to test face search")

    plates = database.count_plates()
    print(f"plates read from footage: {plates}")
    if plates:
        with database.get_conn() as conn:
            prow = conn.execute("SELECT plate_text FROM plates LIMIT 1").fetchone()
        pr = plate_search.search_by_plate(prow["plate_text"], top_k=5)
        print(f"[{'PASS' if pr.total > 0 else 'FAIL'}] plate search "
              f"'{prow['plate_text']}' -> {pr.total} results")
    else:
        print("[WARN] no readable plates in sample footage (expected for these clips) "
              "- plate code path is proven separately by scripts/verify_plates.py")

    print("=== DONE ===")


if __name__ == "__main__":
    main()
