"""Batch-ingest videos in data/videos/ through the full pipeline.

    python scripts/ingest_all.py [--reset] [--fps 2] [--video NAME] [--start-time ISO]

  --reset       clear existing detections/tracks/videos + FAISS first
  --video NAME  ingest a single file in data/videos/ (else all videos)
  --start-time  ISO datetime of the first frame (real-world clock for the clip)
"""
import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import config, database          # noqa: E402
from app.ingestion import pipeline        # noqa: E402
from app.search import vector_store       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="clear existing data first")
    ap.add_argument("--fps", type=float, default=config.DEFAULT_FPS)
    ap.add_argument("--video", default=None, help="single filename in data/videos/")
    ap.add_argument("--start-time", default=None, help="ISO datetime of first frame")
    args = ap.parse_args()

    database.init_db()
    if args.reset:
        pipeline.reset_all()

    if args.video:
        pipeline.ingest_video(config.VIDEO_DIR / args.video, fps=args.fps, start_time=args.start_time)
    else:
        pipeline.ingest_directory(fps=args.fps, start_time=args.start_time)

    print(f"\nDONE. detections in DB: {database.count_detections()} | faiss: {vector_store.stats()}")


if __name__ == "__main__":
    main()
