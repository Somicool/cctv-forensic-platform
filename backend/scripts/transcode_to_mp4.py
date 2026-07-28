"""Transcode recordings to browser-playable H.264 MP4 (faststart) - the format
real VMS/NVR software stores for playback and scrubbing. Converts any non-mp4
clip in the video dir (or one given file) and removes the original. Idempotent.

    python scripts/transcode_to_mp4.py           # convert every non-mp4 in data/videos
    python scripts/transcode_to_mp4.py <file>    # convert one file
"""
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import config                                   # noqa: E402
import imageio_ffmpeg                                     # noqa: E402

_CONVERTIBLE = {".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv"}


def transcode(src, remove_original: bool = True) -> Path:
    src = Path(src)
    if src.suffix.lower() == ".mp4":
        return src
    dst = src.with_suffix(".mp4")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-i", str(src),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)]
    print(f"transcoding {src.name} -> {dst.name} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not dst.exists():
        print(result.stderr[-1000:])
        raise RuntimeError(f"ffmpeg failed for {src}")
    print(f"  wrote {dst.name} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")
    if remove_original and dst.exists() and dst != src:
        src.unlink()
        print(f"  removed original {src.name}")
    return dst


def main(argv):
    if argv:
        transcode(argv[0])
        return
    vids = [p for p in config.VIDEO_DIR.iterdir() if p.suffix.lower() in _CONVERTIBLE]
    if not vids:
        print("no non-mp4 recordings to convert")
        return
    for v in vids:
        transcode(v)


if __name__ == "__main__":
    main(sys.argv[1:])
