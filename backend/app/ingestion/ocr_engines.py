"""Swappable OCR engine layer.

A thin abstraction so the plate reader doesn't care which OCR backend is used.
Every engine exposes the SAME method:

    readtext(image, allowlist=None) -> [(box, text, conf), ...]

where `box` is a 4-point polygon (list of [x, y]) and `conf` is 0-1. This matches
EasyOCR's tuple shape, so downstream code is engine-agnostic.

Primary engine is PaddleOCR (PP-OCRv4). If it can't be imported/loaded (e.g. not
installed in this environment), get_engine() transparently falls back to EasyOCR
so plate recognition keeps working. Select via config.OCR_ENGINE.
"""
from __future__ import annotations

from .. import config


class OcrEngine:
    name = "base"

    def readtext(self, image, allowlist=None):  # pragma: no cover - interface
        raise NotImplementedError


class EasyOcrEngine(OcrEngine):
    """Original engine, kept as a reliable fallback."""
    name = "easyocr"

    def __init__(self):
        import easyocr
        self._reader = easyocr.Reader(config.OCR_LANGS, gpu=config.OCR_USE_GPU)

    def readtext(self, image, allowlist=None):
        if allowlist:
            return list(self._reader.readtext(image, allowlist=allowlist))
        return list(self._reader.readtext(image))


class PaddleOcrEngine(OcrEngine):
    """PaddleOCR PP-OCRv4 (primary). allowlist is ignored here - the plate reader
    post-filters to A-Z/0-9 anyway, so results stay identical in shape."""
    name = "paddle"

    def __init__(self):
        from paddleocr import PaddleOCR
        self._ocr = PaddleOCR(use_angle_cls=True, lang=config.PADDLE_LANG,
                              use_gpu=config.PADDLE_USE_GPU, show_log=False)

    def readtext(self, image, allowlist=None):
        res = self._ocr.ocr(image, cls=True)
        out = []
        if not res:
            return out
        page = res[0]
        if not page:
            return out
        for line in page:
            try:
                box, (text, conf) = line[0], line[1]
            except (ValueError, TypeError, IndexError):
                continue
            out.append((box, text, float(conf)))
        return out


_engine: OcrEngine | None = None


def get_engine() -> OcrEngine:
    """Lazily build and cache the configured OCR engine, with graceful fallback."""
    global _engine
    if _engine is None:
        want = str(getattr(config, "OCR_ENGINE", "paddle")).lower()
        if want == "paddle":
            try:
                _engine = PaddleOcrEngine()
            except Exception as exc:  # noqa: BLE001 - keep OCR working no matter what
                print(f"[ocr] PaddleOCR unavailable ({exc}); falling back to EasyOCR")
                _engine = EasyOcrEngine()
        else:
            _engine = EasyOcrEngine()
        print(f"[ocr] active engine: {_engine.name}")
    return _engine
