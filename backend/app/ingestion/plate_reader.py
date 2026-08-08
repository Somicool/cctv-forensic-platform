"""License plate OCR - hybrid pipeline.

Primary engine : PaddleOCR (PP-OCRv4) via ocr_engines (swappable; EasyOCR fallback).
Preprocessing  : crop refinement (morphological text-region proposals),
                 perspective correction (deskew of detected quads), CLAHE contrast,
                 unsharp sharpening, adaptive upscaling for small plates.
Validation     : Indian registration formats (with OCR-confusion repair) + a
                 tightened general fallback that rejects obvious non-plate text;
                 partial matching for search is preserved (short substrings still
                 match in plate_search / DB).
Temporal vote  : read_plates_voted() aggregates several high-quality frames of one
                 tracked vehicle into a single final plate + confidence.
Fallback       : Gemini Vision, difficult cases only (low PaddleOCR confidence),
                 optional and key-gated - absent -> PaddleOCR only, no errors.

    python -m app.ingestion.plate_reader   # deterministic self-test on synthetic plates
"""
from __future__ import annotations

import re

import cv2
import numpy as np

from .. import config
from . import ocr_engines, gemini_plate

# Anchored, separator-free Indian plate shape used to validate a cleaned candidate.
_STRICT = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}$")
_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# Conservative, genuinely-ambiguous confusions only (note: no 'L'/'A'/'T').
_TO_DIGIT = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1",
                           "Z": "2", "S": "5", "B": "8", "G": "6"})
_TO_ALPHA = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"})
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def get_reader():
    """Backward-compatible accessor - now returns the active OCR engine."""
    return ocr_engines.get_engine()


def is_indian_plate(text: str) -> bool:
    """True if `text` matches the strict Indian registration format."""
    return bool(_STRICT.match(text or ""))


# ------------------------------------------------------------------ validation
def _clean(text: str) -> str:
    """Uppercase and drop everything that isn't a letter or digit."""
    return "".join(ch for ch in text.upper() if ch.isalnum())


def _coerce_plate(cleaned: str) -> str | None:
    """Fit `cleaned` to LL DD L(1-3) DDDD, fixing <=2 OCR confusions. None if it can't."""
    n = len(cleaned)
    best = None
    for a in (2, 1):                       # RTO number: 1 or 2 digits
        for c in (4, 3):                   # serial: 4 or 3 digits
            b = n - 2 - a - c              # middle series letters
            if not 1 <= b <= 3:
                continue
            cand = (cleaned[:2].translate(_TO_ALPHA)
                    + cleaned[2:2 + a].translate(_TO_DIGIT)
                    + cleaned[2 + a:2 + a + b].translate(_TO_ALPHA)
                    + cleaned[2 + a + b:].translate(_TO_DIGIT))
            if _STRICT.match(cand):
                edits = sum(1 for x, y in zip(cleaned, cand) if x != y)
                if edits <= 2 and (best is None or edits < best[1]):
                    best = (cand, edits)
    return best[0] if best else None


def _candidate(text: str) -> str | None:
    """Normalised plate string for `text`, or None if it isn't plate-like.

    Order: exact Indian -> Indian confusion-repair -> (optional) general plate.
    The general fallback is tightened (>=6 chars, >=2 letters AND >=2 digits) to
    reject obvious false positives like bus route text or ad strings, while real
    plates (which always carry several digits) still pass. Partial matching for
    SEARCH is unaffected - that happens later against the DB substring index.
    """
    cleaned = _clean(text)
    if not cleaned:
        return None
    if _STRICT.match(cleaned):                     # already a clean Indian plate
        return cleaned
    if 7 <= len(cleaned) <= 11:                    # try Indian confusion recovery
        ind = _coerce_plate(cleaned)
        if ind:
            return ind
    if not config.PLATE_STRICT_INDIAN:             # general (foreign) plate fallback
        n_alpha = sum(ch.isalpha() for ch in cleaned)
        n_digit = sum(ch.isdigit() for ch in cleaned)
        if 6 <= len(cleaned) <= 11 and n_alpha >= 2 and n_digit >= 2:
            return cleaned
    return None


# ------------------------------------------------------------------ preprocessing
def _upscale(img, min_side: int = 480):
    """Adaptive resize: enlarge small crops/regions so OCR has enough pixels."""
    if img is None or not getattr(img, "size", 0):
        return img
    h, w = img.shape[:2]
    m = min(h, w)
    if m and m < min_side:
        f = min_side / m
        img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_CUBIC)
    return img


def _enhance(img):
    """Grayscale + CLAHE contrast + unsharp mask - a cleaner variant for OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = _clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    return cv2.addWeighted(gray, 1.6, blur, -0.6, 0)


def _order_quad(pts):
    """Order 4 points as tl, tr, br, bl."""
    pts = np.array(pts, dtype="float32").reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def _deskew(img, quad):
    """Perspective-correct a detected text quad to an axis-aligned rectangle."""
    try:
        src = _order_quad(quad)
    except Exception:
        return None
    (tl, tr, br, bl) = src
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if w < 8 or h < 8:
        return None
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h))


def _refine_regions(img, max_regions: int = 3):
    """Lightweight plate-region proposals: morphological black-hat highlights dark
    text on lighter plates; connect into blobs; keep plate-shaped rectangles.
    Returns upscaled candidate regions (a cheap stand-in for a plate detector)."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        rect = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect)
        grad = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)
        grad = np.absolute(grad)
        mn, mx = grad.min(), grad.max()
        if mx - mn < 1e-3:
            return []
        grad = (255 * (grad - mn) / (mx - mn)).astype("uint8")
        grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rect)
        thr = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        H, W = gray.shape[:2]
        cand = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if h < 8 or w < 20:
                continue
            ar = w / float(h)
            if 1.8 <= ar <= 8.0 and w * h >= 0.01 * W * H:
                pad_x, pad_y = int(w * 0.08), int(h * 0.30)
                x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
                x1, y1 = min(W, x + w + pad_x), min(H, y + h + pad_y)
                cand.append((w * h, img[y0:y1, x0:x1]))
        cand.sort(key=lambda t: t[0], reverse=True)
        return [_upscale(r, 320) for _a, r in cand[:max_regions] if r.size]
    except Exception:
        return []


# ------------------------------------------------------------------ read
def read_plates(image, min_conf: float | None = None) -> list[dict]:
    """Detect plate-like text in one image (path or BGR ndarray) using the active
    OCR engine over several enhanced/refined/deskewed variants."""
    min_conf = config.PLATE_MIN_CONF if min_conf is None else min_conf
    img = cv2.imread(image) if isinstance(image, str) else image
    if img is None:
        return []

    engine = ocr_engines.get_engine()
    base = _upscale(img, 480)
    reads = []

    def _confident_plate(rs) -> bool:
        """A valid Indian-format plate read confidently enough to stop refining."""
        for _b, t, c in rs:
            if c >= config.PLATE_EARLY_EXIT_CONF and _candidate(t):
                return True
        return False

    base_reads = list(engine.readtext(base, allowlist=_ALLOWLIST))     # upscaled original
    reads += base_reads
    # The variants below exist to rescue hard plates. When the plain read already
    # gives a confident, well-formed plate they cannot improve the answer, so each
    # extra OCR call is skipped - OCR dominates ingestion time.
    if not _confident_plate(reads):
        reads += engine.readtext(_upscale(_enhance(img), 480), allowlist=_ALLOWLIST)
    if not _confident_plate(reads):
        for region in _refine_regions(img):                           # refined proposals
            reads += engine.readtext(region, allowlist=_ALLOWLIST)
            if _confident_plate(reads):
                break
    # Perspective-correct + re-read the most confident detected quads (deskew) -
    # the biggest win for angled / small plates.
    if not _confident_plate(reads):
        for box, _t, _c in sorted(base_reads, key=lambda r: r[2], reverse=True)[:4]:
            deskewed = _deskew(base, box)
            if deskewed is not None and deskewed.size:
                reads += engine.readtext(_upscale(deskewed, 240), allowlist=_ALLOWLIST)
                if _confident_plate(reads):
                    break

    found: dict[str, dict] = {}
    for _bbox, text, conf in reads:
        if conf < min_conf:
            continue
        plate = _candidate(text)
        if plate and (plate not in found or conf > found[plate]["conf"]):
            found[plate] = {"text": plate, "conf": float(conf), "bbox": None}

    # Plates split across boxes (e.g. "GJ05" + "AB1234"): try the joined line.
    joined = "".join(t for _b, t, c in reads if c >= min_conf)
    plate = _candidate(joined)
    if plate and plate not in found:
        confs = [c for _b, _t, c in reads if c >= min_conf]
        found[plate] = {"text": plate, "conf": float(sum(confs) / len(confs)) if confs else 0.0,
                        "bbox": None}

    return sorted(found.values(), key=lambda x: x["conf"], reverse=True)


def read_plates_voted(crop_paths, max_frames: int | None = None,
                      min_conf: float | None = None) -> list[dict]:
    """Read plates across several crops of the SAME vehicle track and vote.

    Each normalised plate string accumulates a vote count + confidence across
    frames. If the best PaddleOCR candidate is still weak (< PLATE_GEMINI_CONF)
    and Gemini is available (enabled + API key + SDK), Gemini is queried ONCE on
    the best crop and its answer is folded into the vote. Returns candidates
    sorted by (votes, summed confidence): text, conf, votes, score, source.
    """
    if max_frames is None:
        max_frames = config.PLATE_VOTE_FRAMES
    tally: dict[str, dict] = {}

    def _add(text, conf, source):
        t = tally.setdefault(text, {"text": text, "conf": 0.0, "votes": 0,
                                    "score": 0.0, "bbox": None, "source": source})
        t["votes"] += 1
        t["score"] += conf
        t["conf"] = max(t["conf"], conf)
        if source == "gemini":
            t["source"] = "gemini"

    paths = list(crop_paths)[:max_frames]
    for cp in paths:
        for p in read_plates(cp, min_conf=min_conf):
            _add(p["text"], p["conf"], "paddle")

    results = sorted(tally.values(), key=lambda x: (x["votes"], x["score"]), reverse=True)
    best_conf = results[0]["conf"] if results else 0.0

    # Gemini fallback: difficult cases only, one call per track, key permitting.
    if (config.GEMINI_ENABLED and best_conf < config.PLATE_GEMINI_CONF
            and paths and gemini_plate.available()):
        g = gemini_plate.read_plate(paths[0])
        if g:
            gtext, gconf = g
            plate = _candidate(gtext)
            if plate:
                _add(plate, gconf, "gemini")
                results = sorted(tally.values(), key=lambda x: (x["votes"], x["score"]),
                                 reverse=True)

    return results


if __name__ == "__main__":
    def synth(text, scale=1.7, th=5, h=130, w=420):
        img = np.full((h, w, 3), 255, np.uint8)
        cv2.rectangle(img, (8, 8), (w - 8, h - 8), (0, 0, 0), 3)
        cv2.putText(img, text, (22, int(h * 0.69)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th)
        return img

    print(f"engine: {ocr_engines.get_engine().name}")
    for label in ("GJ05AB1234", "MH12DE1433", "ABC1234"):
        plates = read_plates(synth(label))
        print(f"{label}: {[(p['text'], round(p['conf'], 2)) for p in plates]}")
