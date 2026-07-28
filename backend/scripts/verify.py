"""Sanity-check the backend building blocks built so far.

    python scripts/verify.py

Checks: every module imports, the SQLite helpers round-trip, and the FAISS
vector store adds/searches/persists. Does not load the heavy ML models.
"""
import sys
import traceback
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

_results = []


def check(name, fn):
    try:
        fn()
        _results.append((name, True, ""))
        print(f"[PASS] {name}")
    except Exception as e:  # noqa: BLE001
        _results.append((name, False, str(e)))
        print(f"[FAIL] {name}: {e}")
        traceback.print_exc()


def t_imports():
    import app.config          # noqa: F401
    import app.database        # noqa: F401
    from app.models import schemas          # noqa: F401
    from app.ingestion import (             # noqa: F401
        video_processor, detector, tracker, embedder, attribute_extractor,
    )
    from app.search import vector_store     # noqa: F401


def t_config():
    from app import config
    assert config.CLIP_DIM == 512
    assert 0 in config.PERSON_CLASSES
    assert config.VEHICLE_CLASSES
    print(f"   device={config.DEVICE} classes={list(config.DETECT_CLASSES.values())}")


def t_db():
    from app import database as db
    db.init_db()
    vid = db.add_video("CAM-01", "verify.mp4", fps=2,
                       start_time="2026-07-07T20:00:00", duration=6, status="done")
    det_id = db.insert_detection({
        "video_id": vid, "camera_id": "CAM-01", "track_id": 1, "frame_number": 0,
        "timestamp": "2026-07-07T20:00:01", "class_label": "car", "confidence": 0.9,
        "bbox_x": 1, "bbox_y": 2, "bbox_w": 3, "bbox_h": 4, "crop_path": "x.jpg",
        "attributes": {"color": "red", "vehicle_type": "hatchback"},
    })
    got = db.get_detections([det_id])
    assert got and got[0]["class_label"] == "car", got
    assert got[0]["attributes"]["color"] == "red", got
    q = db.query_detections(camera_ids=["CAM-01"], class_labels=["car"])
    assert any(r["detection_id"] == det_id for r in q), "query_detections missed the row"
    print(f"   db: video_id={vid} det_id={det_id} total={db.count_detections()}")
    # clean up the test rows
    with db.get_conn() as conn:
        conn.execute("DELETE FROM detections WHERE detection_id=?", (det_id,))
        conn.execute("DELETE FROM videos WHERE video_id=?", (vid,))


def t_vector_store():
    import numpy as np
    from app import config
    from app.search import vector_store as vs
    vs.reset("clip")
    v = np.random.randn(4, config.CLIP_DIM).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    vs.add("clip", v, [101, 102, 103, 104])
    ids, scores = vs.search("clip", v[2], top_k=2)
    assert ids[0] == 103, f"expected 103 first, got {ids}"
    vs.save("clip")
    vs._indexes.clear()                       # force reload from disk
    ids2, _ = vs.search("clip", v[2], top_k=1)
    assert ids2[0] == 103, ids2
    print(f"   vector_store: top={ids[0]} score={scores[0]:.3f} stats={vs.stats()}")
    vs.reset("clip")


if __name__ == "__main__":
    check("imports", t_imports)
    check("config", t_config)
    check("database round-trip", t_db)
    check("vector_store add/search/persist", t_vector_store)
    passed = sum(1 for _, ok, _ in _results if ok)
    print(f"\n{passed}/{len(_results)} checks passed")
    sys.exit(0 if passed == len(_results) else 1)
