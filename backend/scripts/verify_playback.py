"""Verify the recording index + seek-to-moment data.

Confirms each ingested clip is indexed like a real recording (camera, start time,
duration, native fps, frame size) and that search results carry a video_url +
offset_seconds that land inside the clip, so the UI can jump to the exact moment.

    python -u scripts/verify_playback.py
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app import config, database                         # noqa: E402
from app.search import text_search                       # noqa: E402


def pf(ok):
    return "PASS" if ok else "FAIL"


def main():
    database.init_db()
    print("=== PLAYBACK / SEEK VERIFICATION ===")

    vindex = database.video_index()
    print(f"recordings indexed: {len(vindex)}")
    meta_ok = True
    for v in vindex.values():
        line = (f"  {v['camera_id']}  {v['filename']}  start={v.get('start_time')}  "
                f"dur={v.get('duration')}s  native_fps={v.get('native_fps')}  "
                f"{v.get('width')}x{v.get('height')}")
        print(line)
        if not (v.get("duration") and v.get("native_fps") and v.get("width")):
            meta_ok = False
    print(f"[{pf(meta_ok)}] every recording has duration + native_fps + frame size")

    r = text_search.search_text("a person", top_k=5, include_scenes=False)
    print(f"search 'a person' -> {r.total} results")
    ok_any = False
    for it in r.results[:5]:
        v = vindex.get(it.video_id) or {}
        dur = v.get("duration")
        within = (it.offset_seconds is not None and it.offset_seconds >= 0
                  and (dur is None or it.offset_seconds <= dur + 1))
        has_file = bool(it.video_url) and (config.VIDEO_DIR / Path(it.video_url).name).exists()
        print(f"  det={it.detection_id} cam={it.camera_id} off={it.offset_seconds}s "
              f"dur={dur} url={it.video_url} {it.frame_width}x{it.frame_height} "
              f"within_clip={within} file_exists={has_file}")
        if within and has_file:
            ok_any = True
    print(f"[{pf(ok_any)}] a result seeks to a valid moment in an existing recording")

    it0 = r.results[0] if r.results else None
    fields_ok = bool(it0 and it0.video_url and it0.offset_seconds is not None
                     and it0.frame_width and it0.frame_height and it0.bbox)
    print(f"[{pf(fields_ok)}] result carries video_url + offset_seconds + frame size + bbox")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
