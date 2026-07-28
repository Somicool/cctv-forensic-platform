"""Warm-up script: load every model once to force all weight files to
download and cache locally.

Run this ONCE after installing dependencies:

    d:\\hackathon\\.venv\\Scripts\\python.exe backend\\scripts\\warmup_models.py

After it finishes, every model's weights live on disk, so the app runs
instantly and fully offline (no surprise downloads during a live demo).

Each model is wrapped in its own try/except so one failure (e.g. torchreid
not installed) doesn't stop the others. A summary is printed at the end.
"""
import sys
import time
from pathlib import Path

# Make the `app` package importable when run from anywhere.
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import config  # noqa: E402

results = {}


def step(name):
    """Small helper for consistent, timed logging per model."""
    def deco(fn):
        print(f"\n[ {name} ] loading ...")
        t0 = time.time()
        try:
            fn()
            dt = time.time() - t0
            results[name] = f"OK ({dt:.1f}s)"
            print(f"[ {name} ] done in {dt:.1f}s")
        except Exception as e:  # noqa: BLE001
            results[name] = f"FAILED: {type(e).__name__}: {e}"
            print(f"[ {name} ] FAILED: {e}")
    return deco


def main():
    print("=" * 60)
    print("Model warm-up - downloading & caching all weights")
    print("=" * 60)

    # --- Environment / GPU report ---
    try:
        import torch
        print(f"torch {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"VRAM: {vram:.1f} GB")
    except Exception as e:  # noqa: BLE001
        print(f"torch not available: {e}")

    gpu = config.DEVICE == "cuda"

    # --- 1. YOLO (detection) ---
    @step("YOLOv10")
    def _yolo():
        from ultralytics import YOLO
        YOLO(config.YOLO_MODEL)

    # --- 2. OpenCLIP (text-image search + attributes) ---
    @step("OpenCLIP")
    def _clip():
        import open_clip
        open_clip.create_model_and_transforms(
            config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED
        )

    # --- 3. OSNet (person re-ID) ---
    @step("OSNet")
    def _osnet():
        # torchreid's package layout varies by build: newer PyPI nests under
        # torchreid.reid.utils, older exposes torchreid.utils.
        try:
            from torchreid.reid.utils import FeatureExtractor
        except ImportError:
            from torchreid.utils import FeatureExtractor
        FeatureExtractor(
            model_name=config.REID_MODEL,
            model_path="",
            device=config.DEVICE,
        )

    # --- 4. InsightFace (face recognition) ---
    @step("InsightFace")
    def _face():
        from insightface.app import FaceAnalysis
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        app = FaceAnalysis(name=config.FACE_MODEL, providers=providers)
        app.prepare(ctx_id=0 if gpu else -1, det_size=(640, 640))

    # --- 5. EasyOCR (license plates) ---
    @step("EasyOCR")
    def _ocr():
        import easyocr
        easyocr.Reader(config.OCR_LANGS, gpu=gpu)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("WARM-UP SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name:<14} {status}")
    failed = [n for n, s in results.items() if s.startswith("FAILED")]
    if failed:
        print(f"\n{len(failed)} model(s) need attention: {', '.join(failed)}")
        sys.exit(1)
    print("\nAll models cached. You're ready to run offline.")


if __name__ == "__main__":
    main()
