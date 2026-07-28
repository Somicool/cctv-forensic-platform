"""One-time migration: make already-ingested non-mp4 clips browser-playable.

Older clips were ingested as .avi (before auto-transcode was wired into the
pipeline). Browsers can't play .avi in a <video>, so the player shows a black
frame + a "format isn't playable" notice even though search works.

For every video record whose file is a non-mp4 still sitting in the video dir,
this creates an H.264 .mp4 proxy (keeping the original .avi on disk as evidence)
and repoints the DB record's filename to the .mp4, so search results + library
playback use the playable file. Seek offsets are unchanged - the transcode keeps
the same fps / duration / frame size, and offsets come from frame_number/fps.

Idempotent: once a record points at a .mp4 it's skipped, and ensure_mp4 reuses an
existing .mp4. Non-destructive: originals are left in place (the library hides a
source whose .mp4 twin exists, so no duplicate/unprocessed card appears).

    python -u scripts/make_existing_playable.py
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app import config, database                    # noqa: E402
from app.ingestion import transcode                 # noqa: E402


def main():
    database.init_db()
    todo = []
    for v in database.list_videos():
        fn = v.get("filename")
        if not fn:
            continue
        src = config.VIDEO_DIR / fn
        if src.suffix.lower() == ".mp4":
            continue                                  # already playable
        if not src.exists():
            print(f"[skip] video {v['video_id']} {fn!r}: file not on disk")
            continue
        todo.append((v["video_id"], src))

    print(f"{len(todo)} clip(s) to make playable.\n", flush=True)
    changed = 0
    for i, (vid, src) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] transcoding {src.name} ...", flush=True)
        out = transcode.ensure_mp4(src)
        if out.suffix.lower() == ".mp4" and out.exists() and out.stat().st_size > 10_000:
            with database.get_conn() as conn:
                conn.execute("UPDATE videos SET filename=? WHERE video_id=?",
                             (out.name, vid))
            changed += 1
            print(f"      -> repointed video {vid} to {out.name}", flush=True)
        else:
            print(f"      -> FAILED (kept {src.name}); browser still can't play it",
                  flush=True)

    print(f"\ndone. {changed}/{len(todo)} clip(s) now playable. Refresh the app.",
          flush=True)


if __name__ == "__main__":
    main()
