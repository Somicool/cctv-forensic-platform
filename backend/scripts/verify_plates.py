"""Deterministic Task 13 verification.

Proves the licence-plate code path works end to end - EasyOCR read + regex
filter + DB insert + partial-string plate search - WITHOUT depending on whether
the sample footage happens to contain a readable plate. Leaves the DB pristine
(the one test plate it inserts is deleted afterwards).

    python -u scripts/verify_plates.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import database                                 # noqa: E402
from app.ingestion import plate_reader                   # noqa: E402
from app.search import plate_search                      # noqa: E402


def _synthetic_plate(text="GJ05AB1234"):
    img = np.full((130, 420, 3), 255, np.uint8)
    cv2.rectangle(img, (8, 8), (412, 122), (0, 0, 0), 3)
    cv2.putText(img, text, (22, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0, 0, 0), 6)
    return img


def main():
    database.init_db()
    print("=== TASK 13 PLATE VERIFICATION ===")

    # 1) OCR + regex on a clean synthetic plate --------------------------
    plates = plate_reader.read_plates(_synthetic_plate("GJ05AB1234"))
    ok_read = any("AB1234" in p["text"] for p in plates)
    print(f"[{'PASS' if ok_read else 'FAIL'}] read_plates(synthetic) -> "
          f"{[(p['text'], round(p['conf'], 2)) for p in plates]}")

    # 2) DB insert + partial plate search, attached to a real detection --
    with database.get_conn() as conn:
        det = conn.execute(
            "SELECT detection_id, camera_id, timestamp FROM detections "
            "WHERE class_label IN ('car','truck','bus','motorcycle','bicycle') LIMIT 1"
        ).fetchone()
    if not det:
        print("[WARN] no vehicle detection to attach a test plate to - skipping search round-trip")
        print("=== DONE ===")
        return

    before = database.count_plates()
    pid = database.insert_plate({
        "detection_id": det["detection_id"], "camera_id": det["camera_id"],
        "timestamp": det["timestamp"], "plate_text": "GJ05AB1234",
        "confidence": 0.99, "crop_path": None,
    })
    try:
        resp = plate_search.search_by_plate("ab1234", top_k=5)      # partial + lowercase
        hit = any(r.detection_id == det["detection_id"] for r in resp.results)
        print(f"[{'PASS' if hit else 'FAIL'}] plate search 'ab1234' (partial) -> "
              f"{resp.total} results, matched the test plate={hit}")
    finally:
        with database.get_conn() as conn:
            conn.execute("DELETE FROM plates WHERE plate_id=?", (pid,))
    after = database.count_plates()
    print(f"[{'PASS' if after == before else 'FAIL'}] cleanup restored plate count "
          f"({before} -> {after})")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
