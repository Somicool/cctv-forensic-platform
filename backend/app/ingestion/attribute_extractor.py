"""CLIP zero-shot attribute extraction (+ region split + HSV colour backup).

Given a detection crop and its class, reads off human-friendly attributes:
  - vehicles: colour + vehicle type (from the whole-crop CLIP embedding)
  - persons : upper/lower clothing colour + accessories

Colour bug fix: CLIP on the WHOLE person crop lets the dominant colour win both
"top" and "bottom" (blue jacket + red pants -> "top: red"). So for persons with
the crop image available we split it into the upper 40% and lower 60% and
classify each half independently, and cross-check each half with an HSV
dominant-colour reading. HSV is more reliable for a pure colour, so if CLIP and
HSV disagree we prefer HSV.

    python -m app.ingestion.attribute_extractor
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import config
from . import embedder

_LOGIT_SCALE = 100.0        # CLIP-style temperature so softmax is decisive
_text_cache: dict[str, np.ndarray] = {}

# Accessories grouped into buckets, each with a competing "none" option.
_ACCESSORY_GROUPS = [
    ("headwear", [("cap", "a person wearing a cap or hat"),
                  ("helmet", "a person wearing a helmet")],
     "a person with bare head and no headwear"),
    ("carry", [("backpack", "a person carrying a backpack"),
               ("handbag", "a person carrying a handbag or purse")],
     "a person not carrying any bag"),
    ("eyewear", [("sunglasses", "a person wearing sunglasses")],
     "a person not wearing sunglasses"),
    ("mask", [("face mask", "a person wearing a face mask")],
     "a person not wearing a face mask"),
]


def _text_embs(key: str, prompts: list[str]) -> np.ndarray:
    if key not in _text_cache:
        _text_cache[key] = embedder.embed_texts(prompts)
    return _text_cache[key]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp((x - x.max()) * _LOGIT_SCALE)
    return e / e.sum()


def _best(img_emb: np.ndarray, key: str, prompts: list[str], labels: list[str]):
    sims = _text_embs(key, prompts) @ img_emb          # [K]
    probs = _softmax(sims)
    i = int(probs.argmax())
    return labels[i], round(float(probs[i]), 3)


# ------------------------------------------------------------------ HSV colour
def _hue_to_color(h: int, v: float) -> str:
    """OpenCV hue (0-179) + brightness -> one of config.COLORS (approximate).
    Now distinguishes purple / pink instead of folding violet-magenta into red."""
    if h < 8 or h >= 172:
        return "maroon" if v < 120 else "red"
    if h < 20:
        return "brown" if v < 120 else "orange"
    if h < 33:
        return "yellow"
    if h < 95:
        return "green"                          # includes teal (no teal in palette)
    if h < 130:
        return "blue"
    if h < 150:
        return "purple"                         # violet
    return "pink" if v >= 120 else "purple"     # magenta -> pink (bright) / purple (dark)


def _center(bgr, fw: float = 0.7, fh: float = 0.9):
    """Central sub-region of a crop - drops background pixels near the edges
    (the main cause of wrong colours), especially after crop padding."""
    if bgr is None or bgr.size == 0:
        return bgr
    h, w = bgr.shape[:2]
    cw, ch = max(1, int(w * fw)), max(1, int(h * fh))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return bgr[y0:y0 + ch, x0:x0 + cw]


def _normalize_illumination(bgr):
    """Lift very dark / pull down blown-out regions (V channel only, so hue is
    preserved). Applied only where beneficial - well-exposed crops pass through."""
    if bgr is None or bgr.size == 0:
        return bgr
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    mv = float(v.mean())
    if mv < 70:
        scale = min(2.2, 110.0 / max(mv, 1.0))
    elif mv > 205:
        scale = 195.0 / mv
    else:
        return bgr
    hsv[..., 2] = np.clip(v.astype("float32") * scale, 0, 255).astype("uint8")
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def prep_region(bgr, fw: float = 0.7, fh: float = 0.9):
    """Centre-sample (background suppression) + illumination-normalise a region
    before colour reading. Used by both live ingest and colour recompute."""
    return _normalize_illumination(_center(bgr, fw, fh))


def _hsv_dominant(bgr):
    """(dominant colour, confidence 0-1) of a region via HSV, or (None, 0.0)."""
    if bgr is None or bgr.size == 0:
        return None, 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()
    if V.size == 0:
        return None, 0.0
    # Achromatic checks first (black / white / grey), confidence = pixel share.
    dark = float((V < 50).mean())
    if dark > 0.5:
        return "black", dark
    gray_mask = S < 45
    gfrac = float(gray_mask.mean())
    if gfrac > 0.55:
        meanv = float(V[gray_mask].mean()) if gray_mask.any() else float(V.mean())
        return ("white" if meanv > 170 else "grey"), gfrac
    # Chromatic: dominant hue among sufficiently saturated, bright pixels.
    m = (S > 55) & (V > 55)
    mfrac = float(m.mean())
    if mfrac < 0.12:
        return None, 0.0
    hue = H[m]
    dom = int(np.bincount(hue, minlength=180).argmax())
    band = float(((hue >= dom - 12) & (hue <= dom + 12)).mean())   # hue concentration
    return _hue_to_color(dom, float(V[m].mean())), round(mfrac * band, 3)


def fuse_clip_hsv(clip_color, clip_p, region_bgr):
    """Confidence-weighted fusion of the CLIP zero-shot colour and the HSV
    dominant colour. Replaces the old unconditional 'trust HSV': HSV now only
    overrides CLIP when its pixel support (times HSV_COLOR_WEIGHT) beats CLIP's
    probability."""
    hsv_color, hsv_p = _hsv_dominant(region_bgr)
    if not hsv_color:
        return clip_color, round(float(clip_p), 3)
    if hsv_color == clip_color:
        return clip_color, round(min(0.99, float(clip_p) + 0.15), 3)   # agreement boost
    if config.HSV_COLOR_WEIGHT * hsv_p >= clip_p:
        return hsv_color, round(min(0.99, 0.5 + 0.5 * hsv_p), 3)
    return clip_color, round(float(clip_p), 3)


def _region_color(region, which: str):
    """Colour of one body region: centre-sample + illumination-normalise, then
    CLIP zero-shot fused (confidence-weighted) with the HSV reading."""
    if region is None or region.size == 0:
        return None, 0.0
    garment = "top" if which == "upper" else "trousers"
    reg = prep_region(region)
    emb = embedder.embed_image(reg)
    clip_color, score = _best(
        emb, f"person_{which}",
        [f"a person wearing a {c} colored {garment}" for c in config.COLORS],
        config.COLORS,
    )
    return fuse_clip_hsv(clip_color, score, reg)


def _person_colors_split(img):
    """Split a person crop upper 40% / lower 60% and colour each half."""
    h = img.shape[0]
    cut = max(1, int(round(0.40 * h)))
    uc, ucs = _region_color(img[:cut, :], "upper")
    lc, lcs = _region_color(img[cut:, :], "lower")
    return uc, ucs, lc, lcs


# ------------------------------------------------------------------ public API
def _as_bgr(crop_image):
    if isinstance(crop_image, np.ndarray):
        return crop_image
    if isinstance(crop_image, (str,)):
        return cv2.imread(crop_image)
    return None


def extract(crop_image, class_id: int) -> dict:
    """Extract attributes from a crop IMAGE (path or ndarray). Uses the region
    split for person colours."""
    img = _as_bgr(crop_image)
    emb = embedder.embed_image(img if img is not None else crop_image)
    return _extract(emb, class_id, img)


def extract_from_embedding(img_emb, class_id: int, crop_path: str | None = None,
                           region_split: bool = True) -> dict:
    """Same as extract() but reuses a precomputed whole-crop CLIP embedding (the
    pipeline already embeds every crop). When region_split is True and crop_path
    is given, the crop is loaded so person upper/lower colours use the region
    split and vehicles get an HSV cross-check (Accurate mode). When False (Fast
    mode), colours come from the whole-crop embedding only - cheaper, no extra
    image load or region embeddings, while search quality stays good."""
    img = None
    if region_split and crop_path and class_id in (config.PERSON_CLASSES | config.VEHICLE_CLASSES):
        img = cv2.imread(str(crop_path))
    return _extract(img_emb, class_id, img)


def extract_batch(crops, class_ids, whole_embs, region_split: bool = True) -> list[dict]:
    """Vectorised attribute extraction for a whole video's detections.

    Reuses the precomputed whole-crop CLIP embeddings (`whole_embs`) for vehicle
    colour/type and person accessories, and - crucially - collects EVERY person's
    upper/lower region crop and embeds them in a couple of large CLIP batches
    instead of two batch-1 calls per person. Same results as calling
    extract_from_embedding per detection, but far fewer GPU round-trips."""
    n = len(crops)
    results: list[dict] = [{} for _ in range(n)]
    veh_color_prompts = [f"a surveillance photo of a {c} colored vehicle" for c in config.COLORS]
    veh_type_prompts = [f"a surveillance photo of a {t}" for t in config.VEHICLE_TYPES]
    up_prompts = [f"a person wearing a {c} colored top" for c in config.COLORS]
    lo_prompts = [f"a person wearing {c} colored trousers" for c in config.COLORS]

    for i in range(n):
        cid = class_ids[i]
        emb = whole_embs[i]
        img = crops[i]
        if cid in config.VEHICLE_CLASSES:
            color, cs = _best(emb, "veh_color", veh_color_prompts, config.COLORS)
            if region_split and img is not None and getattr(img, "size", 0):
                color, cs = fuse_clip_hsv(color, cs, prep_region(img, 0.7, 0.7))
            vtype, ts = _best(emb, "veh_type", veh_type_prompts, config.VEHICLE_TYPES)
            results[i] = {"color": color, "color_score": cs,
                          "vehicle_type": vtype, "vehicle_type_score": ts}
        elif cid in config.PERSON_CLASSES:
            accessories = []
            for key, items, none_prompt in _ACCESSORY_GROUPS:
                labels = [lbl for lbl, _ in items] + ["none"]
                prompts = [p for _, p in items] + [none_prompt]
                probs = _softmax(_text_embs(f"acc_{key}", prompts) @ emb)
                winner = labels[int(probs.argmax())]
                if winner != "none":
                    accessories.append(winner)
            results[i]["accessories"] = accessories
            if not (region_split and img is not None and getattr(img, "size", 0)):
                uc, ucs = _best(emb, "upper", up_prompts, config.COLORS)   # whole-crop fallback
                lc, lcs = _best(emb, "lower", lo_prompts, config.COLORS)
                results[i].update(upper_color=uc, upper_color_score=ucs,
                                  lower_color=lc, lower_color_score=lcs)

    if region_split:
        up_imgs, lo_imgs, idx = [], [], []
        for i in range(n):
            if class_ids[i] not in config.PERSON_CLASSES:
                continue
            img = crops[i]
            if img is None or not getattr(img, "size", 0):
                continue
            cut = max(1, int(round(0.40 * img.shape[0])))
            up, lo = prep_region(img[:cut, :]), prep_region(img[cut:, :])
            if up is None or lo is None or not up.size or not lo.size:
                continue
            up_imgs.append(up); lo_imgs.append(lo); idx.append(i)
        if idx:
            up_embs = embedder.embed_crops(up_imgs)        # ONE batched pass each
            lo_embs = embedder.embed_crops(lo_imgs)
            for k, i in enumerate(idx):
                uc, ucs = _best(up_embs[k], "upper", up_prompts, config.COLORS)
                uc, ucs = fuse_clip_hsv(uc, ucs, up_imgs[k])
                lc, lcs = _best(lo_embs[k], "lower", lo_prompts, config.COLORS)
                lc, lcs = fuse_clip_hsv(lc, lcs, lo_imgs[k])
                results[i].update(upper_color=uc, upper_color_score=ucs,
                                  lower_color=lc, lower_color_score=lcs)
    return results


def _extract(img_emb, class_id: int, img=None) -> dict:
    attrs: dict = {}

    if class_id in config.VEHICLE_CLASSES:
        color, cs = _best(
            img_emb, "veh_color",
            [f"a surveillance photo of a {c} colored vehicle" for c in config.COLORS],
            config.COLORS,
        )
        if img is not None and getattr(img, "size", 0):     # background-suppressed HSV fuse
            color, cs = fuse_clip_hsv(color, cs, prep_region(img, 0.7, 0.7))
        vtype, ts = _best(
            img_emb, "veh_type",
            [f"a surveillance photo of a {t}" for t in config.VEHICLE_TYPES],
            config.VEHICLE_TYPES,
        )
        attrs.update(color=color, color_score=cs, vehicle_type=vtype, vehicle_type_score=ts)

    if class_id in config.PERSON_CLASSES:
        if img is not None and img.size:
            uc, ucs, lc, lcs = _person_colors_split(img)        # split + HSV (accurate)
        else:
            uc, ucs = _best(                                    # whole-crop fallback
                img_emb, "upper",
                [f"a person wearing a {c} colored top" for c in config.COLORS], config.COLORS)
            lc, lcs = _best(
                img_emb, "lower",
                [f"a person wearing {c} colored trousers" for c in config.COLORS], config.COLORS)
        attrs.update(upper_color=uc, upper_color_score=ucs,
                     lower_color=lc, lower_color_score=lcs)

        accessories = []
        for key, items, none_prompt in _ACCESSORY_GROUPS:
            labels = [lbl for lbl, _ in items] + ["none"]
            prompts = [p for _, p in items] + [none_prompt]
            probs = _softmax(_text_embs(f"acc_{key}", prompts) @ img_emb)
            winner = labels[int(probs.argmax())]
            if winner != "none":
                accessories.append(winner)
        attrs["accessories"] = accessories

    return attrs


if __name__ == "__main__":
    test_dir = config.CROP_DIR / "test"
    crops = sorted(test_dir.glob("*.jpg")) if test_dir.exists() else []
    if not crops:
        print(f"No test crops in {test_dir}. Run: python -m app.ingestion.detector")
    for p in crops:
        label = p.stem.split("_")[-1]
        cid = next((k for k, v in config.DETECT_CLASSES.items() if v == label), 0)
        print(f"{p.name} ({label}) -> {extract(str(p), cid)}")
