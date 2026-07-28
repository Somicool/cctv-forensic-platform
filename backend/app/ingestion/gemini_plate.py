"""Optional Gemini Vision fallback for difficult license plates.

Uses the Gemini REST API via `requests` (NOT the google-generativeai SDK) so it
does NOT pull a newer protobuf that would conflict with paddlepaddle's pinned
protobuf 3.20.x. Completely optional and OFF unless BOTH are true:
  * config.GEMINI_ENABLED, and
  * an API key is present in the env var config.GEMINI_API_KEY_ENV.

If the key is missing (or a request fails), available()/read_plate() degrade to
False/None and the pipeline simply continues on PaddleOCR - no errors. read_plate
is only ever called by the plate reader for hard cases (low PaddleOCR confidence),
never for every crop.
"""
from __future__ import annotations

import base64
import os

import cv2

from .. import config

_PROMPT = (
    "You are reading a single vehicle number plate from a CCTV crop. "
    "Return ONLY the plate characters using A-Z and 0-9 with no spaces, "
    "punctuation, or extra words. If it is unreadable, return exactly NONE."
)
_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _key() -> str | None:
    return os.environ.get(getattr(config, "GEMINI_API_KEY_ENV", "GEMINI_API_KEY"))


def available() -> bool:
    """True only if enabled, a key is set, and `requests` is importable."""
    if not getattr(config, "GEMINI_ENABLED", False):
        return False
    if not _key():
        return False
    try:
        import requests  # noqa: F401
        return True
    except Exception:
        return False


def read_plate(image):
    """Ask Gemini for the plate string in `image` (path or BGR ndarray) via the
    REST API. Returns (text, confidence) or None. Never raises - any failure -> None."""
    try:
        import requests
        key = _key()
        if not key:
            return None
        img = cv2.imread(image) if isinstance(image, str) else image
        if img is None:
            return None
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        model = getattr(config, "GEMINI_MODEL", "gemini-1.5-flash")
        url = _ENDPOINT.format(model=model)
        payload = {
            "contents": [{
                "parts": [
                    {"text": _PROMPT},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ]
            }],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 16},
        }
        resp = requests.post(url, params={"key": key}, json=payload, timeout=20)
        if resp.status_code != 200:
            return None
        data = resp.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip().upper()
        text = "".join(ch for ch in text if ch.isalnum())
        if not text or text == "NONE":
            return None
        return text, 0.9
    except Exception:
        return None
