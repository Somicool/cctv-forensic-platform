"""Download small, freely-licensed REAL sample clips (a few MB each) so we can
test detection / tracking / search without giant research datasets.

Sources:
  - OpenCV 'vtest.avi'  (pedestrian plaza)               -> CAM-02_plaza.avi
  - Intel IoT DevKit sample-videos (Apache-2.0):
      car-detection.mp4              (traffic / vehicles) -> CAM-01_traffic.mp4
      person-bicycle-car-detection.mp4 (mixed street)    -> CAM-03_street.mp4
      face-demographics-walking.mp4  (walking, faces)     -> CAM-04_entrance.mp4

Each file is named CAM-XX_* so the pipeline auto-assigns the camera id.

    python scripts/fetch_sample_video.py          # fetch the whole recommended set
    python scripts/fetch_sample_video.py --plaza  # only the OpenCV plaza clip
"""
import argparse
import sys
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import config  # noqa: E402

# GitHub /raw/ redirects to the real bytes (handles Git-LFF); media.* is a
# direct LFS fallback; raw.githubusercontent covers non-LFS files.
CLIPS = {
    "CAM-02_plaza.avi": [
        "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi",
        "https://github.com/opencv/opencv/raw/4.x/samples/data/vtest.avi",
    ],
    "CAM-01_traffic.mp4": [
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4",
        "https://media.githubusercontent.com/media/intel-iot-devkit/sample-videos/master/car-detection.mp4",
    ],
    "CAM-03_street.mp4": [
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4",
        "https://media.githubusercontent.com/media/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4",
    ],
    "CAM-04_entrance.mp4": [
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4",
        "https://media.githubusercontent.com/media/intel-iot-devkit/sample-videos/master/face-demographics-walking.mp4",
    ],
}

MIN_BYTES = 100_000   # anything smaller is almost certainly an LFS pointer, not a video


def _download(dest_name: str, urls, force: bool = False):
    dest = config.VIDEO_DIR / dest_name
    if dest.exists() and dest.stat().st_size > MIN_BYTES and not force:
        print(f"  {dest_name} already present ({dest.stat().st_size/1024/1024:.1f} MB), skipping")
        return dest
    for url in urls:
        try:
            print(f"  {dest_name} <- {url}")
            urllib.request.urlretrieve(url, dest)
            size = dest.stat().st_size
            if size < MIN_BYTES:
                print(f"    too small ({size} B, likely an LFS pointer) - trying next url")
                continue
            print(f"    OK ({size/1024/1024:.1f} MB)")
            return dest
        except Exception as e:  # noqa: BLE001
            print(f"    failed: {e}")
    print(f"    !! could not fetch {dest_name}")
    return None


def fetch(names=None, force: bool = False):
    names = names or list(CLIPS)
    config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    got = [d for d in (_download(n, CLIPS[n], force) for n in names) if d]
    print(f"\ndownloaded/present: {len(got)}/{len(names)} clips in {config.VIDEO_DIR}")
    return got


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plaza", action="store_true", help="only the OpenCV plaza clip")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()
    fetch(["CAM-02_plaza.avi"] if args.plaza else None, force=args.force)
