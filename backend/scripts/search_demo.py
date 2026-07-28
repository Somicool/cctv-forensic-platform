"""Quick descriptive-search demo over whatever has been ingested.

    python scripts/search_demo.py "a person walking" "a car" "a truck"

Embeds each text query with CLIP, searches the FAISS 'clip' index, and prints
the top matching detections (camera, timestamp, class, attributes, score).
This is the end-to-end proof: text query -> ranked footage matches.
"""
import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import database                    # noqa: E402
from app.ingestion import embedder          # noqa: E402
from app.search import vector_store         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="+")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    for q in args.queries:
        qvec = embedder.embed_text(q)
        ids, scores = vector_store.search("clip", qvec, top_k=args.k)
        dets = database.get_detections(ids)
        score_by_id = dict(zip(ids, scores))
        print(f"\nquery: {q!r} -> {len(dets)} results")
        for d in dets:
            attrs = d.get("attributes") or {}
            print(f"  {score_by_id[d['detection_id']]:.3f}  {d['class_label']:<7} "
                  f"cam={d['camera_id']} t={d['timestamp'][11:19]} "
                  f"attrs={attrs}")


if __name__ == "__main__":
    main()
