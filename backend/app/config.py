"""Central configuration: paths, model settings, thresholds, vocabularies.

Everything tunable lives here so the rest of the code stays clean.
"""
from pathlib import Path

try:
    import torch
    _CUDA = torch.cuda.is_available()
except Exception:  # torch not installed yet (e.g. before env setup)
    _CUDA = False

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # .../backend

# Load backend/.env (if present) so secrets like GEMINI_API_KEY are available as
# environment variables without hardcoding them. Safe no-op if python-dotenv or
# the file is missing.
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except Exception:
    pass
DATA_DIR = BASE_DIR / "data"
VIDEO_DIR = DATA_DIR / "videos"                            # source footage
FRAME_DIR = DATA_DIR / "frames"                            # extracted frames
CROP_DIR = DATA_DIR / "crops"                              # detected object crops
FACE_DIR = DATA_DIR / "faces"                              # face crops
FAISS_DIR = DATA_DIR / "faiss_indexes"                     # saved vector indexes
EXPORT_DIR = DATA_DIR / "exports"                          # forensic exports
DB_PATH = DATA_DIR / "cctv.db"
CAMERA_CONFIG_PATH = DATA_DIR / "camera_config.json"

for _d in (VIDEO_DIR, FRAME_DIR, CROP_DIR, FACE_DIR, FAISS_DIR, EXPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Device / VRAM
# ------------------------------------------------------------------
DEVICE = "cuda" if _CUDA else "cpu"
# On a 4GB GPU we load models one stage at a time and clear the cache
# between stages during ingestion. Set False on bigger cards to keep
# models resident for speed.
LOW_VRAM = True

# ------------------------------------------------------------------
# Frame extraction
# ------------------------------------------------------------------
DEFAULT_FPS = 2            # frames sampled per second of source video (denser
                          # sampling = better attribute coverage for describe-
                          # search; applies to newly processed videos)
UPLOAD_FAST_FPS = 1        # sparser sampling for "analyse this upload now" so the
                          # clip is searchable quickly; skips the CPU-heavy face
                          # + plate bonus stages too (see routes_library upload)

# ------------------------------------------------------------------
# Detection (YOLO) + tracking (ByteTrack)
# ------------------------------------------------------------------
YOLO_MODEL = "yolov10b.pt"                 # auto-downloaded by ultralytics
# COCO class ids we index. Person + vehicles are the core (they match the
# problem statement); a few carried-object classes broaden search coverage
# (e.g. "person with a backpack", "abandoned bag").
DETECT_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
}
# Groupings the attribute extractor uses to decide which attributes apply.
PERSON_CLASSES = {0}
VEHICLE_CLASSES = {1, 2, 3, 5, 7}
DETECT_CONF = 0.4
TRACKER_CFG = "bytetrack.yaml"             # ultralytics built-in ByteTrack


def _pick_imgsz() -> int:
    """Adaptive YOLO inference resolution. Wide HD/4K CCTV loses small/distant
    objects at the default 640, so we run larger where the GPU can afford it.
    Chosen once from the GPU's total VRAM (RTX 3050 6GB -> 960)."""
    try:
        if _CUDA:
            import torch
            gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if gb >= 10:
                return 1280
            if gb >= 5:
                return 960          # RTX 3050 6GB
            return 768
    except Exception:
        pass
    return 640                      # CPU / unknown -> keep the default


YOLO_IMGSZ = _pick_imgsz()          # detection inference resolution (see _pick_imgsz)

# ------------------------------------------------------------------
# Object crops (padding + quality gating before embedding)
# ------------------------------------------------------------------
CROP_PAD_FRAC = 0.12               # padding added around each detection bbox for
                                   # the saved crop (context helps CLIP / re-ID);
                                   # the stored bbox stays tight for overlay accuracy
CROP_MIN_SIDE = 24                 # reject crops whose shorter side is under this (px)
CROP_MIN_VISIBLE = 0.5             # reject if <50% of the detection box is on-screen
CROP_BLUR_VAR_MIN = 8.0            # reject near-flat / severely blurred crops
                                   # (variance-of-Laplacian floor)
CLIP_PAD_SQUARE = True             # letterbox crops to square before CLIP preprocess
                                   # so the whole object survives the 224 centre-crop

# ------------------------------------------------------------------
# CLIP (descriptive text-image search + zero-shot attributes)
# ------------------------------------------------------------------
CLIP_MODEL = "ViT-B-16"
CLIP_PRETRAINED = "laion2b_s34b_b88k"
CLIP_DIM = 512

# ------------------------------------------------------------------
# Person re-ID (OSNet)
# ------------------------------------------------------------------
REID_MODEL = "osnet_x1_0"
REID_DIM = 512
REID_SIM_THRESHOLD = 0.75

# ------------------------------------------------------------------
# Face recognition (InsightFace) - bonus, ethics-gated
# ------------------------------------------------------------------
FACE_MODEL = "buffalo_l"
FACE_DIM = 512
FACE_SIM_THRESHOLD = 0.5
FACE_RECOGNITION_ENABLED = True            # master on/off switch
FACE_DET_MIN = 0.5                         # min InsightFace det_score to trust a face
FACE_VOTE_FRAMES = 5                        # frames per track aggregated for gender/age

# ------------------------------------------------------------------
# License plate OCR (EasyOCR) - bonus
# ------------------------------------------------------------------
OCR_LANGS = ["en"]                         # add "hi" for Hindi script if needed
# Loose Indian plate pattern, e.g. GJ 05 AB 1234
PLATE_REGEX = r"[A-Z]{2}[ -]?\d{1,2}[ -]?[A-Z]{1,3}[ -]?\d{3,4}"
OCR_USE_GPU = True                         # EasyOCR (fallback engine) on GPU
PLATE_RECOGNITION_ENABLED = True           # master on/off switch
PLATE_VOTE_FRAMES = 6                       # frames per vehicle track aggregated for OCR
PLATE_MIN_VOTES = 2                         # a plate seen in >=2 frames is trusted;
                                            # a single-frame read needs high confidence
PLATE_SINGLE_CONF = 0.50                    # confidence floor for a 1-frame plate

# --- Hybrid OCR engine selection (swappable) ---
OCR_ENGINE = "paddle"                       # "paddle" (PP-OCRv4) | "easyocr"; auto-falls
                                            # back to easyocr if PaddleOCR can't load
PADDLE_USE_GPU = False                      # CPU keeps the 6GB VRAM for torch/CLIP/re-ID
PADDLE_LANG = "en"

# --- Gemini Vision fallback (optional, difficult cases only) ---
GEMINI_ENABLED = True                       # master switch; still a no-op without a key
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"       # env var holding the API key
GEMINI_MODEL = "gemini-flash-lite-latest"
PLATE_GEMINI_CONF = 0.55                     # PaddleOCR best-conf below this -> try Gemini
                                            # (only once per vehicle track, key permitting)

# ------------------------------------------------------------------
# Processing modes: Fast (default, quick indexing/demos) vs Accurate
# (full forensic pipeline). Every knob that differs between the two lives here,
# so the single ingest_video() reads a preset instead of duplicating code.
# ------------------------------------------------------------------
PROCESSING_MODE = "fast"                     # default mode for all ingestion
FAST_ACTIVITY_THRESHOLD = 9.0                # mean inter-frame diff -> 2 FPS if busier, else 1
# Progressive processing: index the video in chunks of this many SAMPLED frames
# so early portions become searchable within seconds instead of waiting for the
# whole clip. Final index is identical - only the processing ORDER changes.
PROGRESSIVE_CHUNK_FRAMES = 80

MODE_PRESETS = {
    "fast": {
        "adaptive_fps": True, "fps": None, "fps_min": 1, "fps_max": 2,
        "imgsz": 736,                        # lower detection resolution (faster)
        "clip_batch": 48,                    # bigger CLIP batches
        "region_split": False,               # cheaper CLIP-only attributes
        "voting": False,                     # single best frame for OCR / face
        # Face recognition (GPU) + licence-plate OCR (CPU) are the slow "bonus"
        # stages and do NOT affect describe / image / text search. Fast mode skips
        # them so uploads index quickly; use Accurate mode for face/plate search.
        "do_faces": False, "do_plates": False,
        "face_min_w": 56, "face_min_h": 112, "face_min_det": 0.60,  # skip tiny/poor faces
        "plate_min_w": 120, "plate_min_h": 70, "plate_blur_min": 40.0,  # only big, sharp plates
        "incremental": True, "index_chunk": 256,   # searchable before the clip finishes
    },
    "accurate": {
        "adaptive_fps": False, "fps": 3,     # higher, fixed sampling
        "imgsz": YOLO_IMGSZ,                 # full adaptive resolution (960 on RTX 3050)
        "clip_batch": 32,
        "region_split": True,                # full attribute extraction (upper/lower + HSV)
        "voting": True,                      # temporal OCR + gender voting
        "do_faces": True, "do_plates": True,
        "face_min_w": 40, "face_min_h": 80, "face_min_det": FACE_DET_MIN,
        "plate_min_w": 60, "plate_min_h": 40, "plate_blur_min": 0.0,
        "incremental": False, "index_chunk": 0,
    },
}
# If False, accept general (non-Indian) plate-like strings too, not only the
# strict LL DD L DDDD Indian format - needed for foreign/stock footage.
PLATE_STRICT_INDIAN = False
PLATE_MIN_CONF = 0.10                       # OCR confidence floor for a plate candidate

# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------
DEFAULT_TOP_K = 60
# CLIP text-image cosine sims sit ~0.18-0.32 even for correct matches, so raw
# scores read as misleadingly low. Map them to a readable 0-1 "relevance" for
# display, and treat anything below the floor as a weak / no-strong match.
CLIP_REL_LOW = 0.18            # maps to 0% relevance
CLIP_REL_HIGH = 0.30           # maps to 100% relevance
CLIP_MIN_RELEVANT = 0.225      # top raw sim below this -> "no strong match"

# ------------------------------------------------------------------
# Attribute vocabularies (used for CLIP zero-shot classification)
# ------------------------------------------------------------------
COLORS = ["red", "blue", "white", "black", "silver", "grey",
          "yellow", "green", "brown", "orange", "maroon", "purple", "pink"]
# Confidence weight applied to the HSV colour reading when it disagrees with the
# CLIP zero-shot colour. 1.0 = balanced (HSV wins only when its pixel support
# beats CLIP's probability) instead of the old unconditional HSV priority.
HSV_COLOR_WEIGHT = 1.0
VEHICLE_TYPES = ["sedan", "hatchback", "SUV", "pickup truck", "van",
                 "auto-rickshaw", "bus", "truck", "motorcycle", "bicycle"]
ACCESSORIES = ["cap", "helmet", "backpack", "handbag", "sunglasses", "face mask"]

# ------------------------------------------------------------------
# API
# ------------------------------------------------------------------
CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
