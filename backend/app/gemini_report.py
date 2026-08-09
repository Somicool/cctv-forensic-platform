"""Gemini Vision narration for the Case File evidence report.

Given the annotated frame (the subject boxed in its original scene) plus the
structured facts already held in the database, Gemini writes the situational
description an officer needs next to each exhibit: what the location looks like,
what the subject visibly appears to be doing, and what else in frame matters.

Uses the Gemini REST API via `requests`, matching ingestion/gemini_plate.py, so it
does NOT pull the google-generativeai SDK (whose protobuf conflicts with
paddlepaddle's pinned 3.20.x).

Fails soft, always. No key, disabled, network error or malformed reply -> None, and
case_report falls back to a description built purely from stored attributes. The
report is never blocked by this module.

Accuracy guardrails baked into the prompts:
  * describe only what is visible; never identify a person or guess a name;
  * never assert that a crime occurred or attribute intent;
  * hedge ("appears to") and state when image quality limits what can be said.
Everything it returns is labelled as machine-generated in the PDF and needs
officer verification.
"""
from __future__ import annotations

import base64
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import cv2

from . import config

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_GUARDRAILS = (
    "Rules you must follow:\n"
    "- Describe ONLY what is visibly present. Never invent details.\n"
    "- Never state or guess the identity, name, nationality, religion or caste of "
    "any person. Never claim a match to a known individual.\n"
    "- Never state or imply that a crime occurred, and never attribute intent or "
    "guilt. You are describing a scene, not accusing anyone.\n"
    "- Use cautious wording such as 'appears to' for anything uncertain.\n"
    "- If the image is too low quality, dark or distant to judge something, say so "
    "instead of guessing.\n"
)

_ITEM_PROMPT = (
    "You are assisting a police investigator by describing one CCTV exhibit.\n"
    "The FIRST image is the full camera frame with the subject of interest outlined "
    "in a red rectangle. The SECOND image is a close-up of that same subject.\n\n"
    "{guardrails}\n"
    "Recorded facts for this exhibit (already verified by the system - stay "
    "consistent with them):\n{facts}\n\n"
    "Reply with ONLY a JSON object, no markdown fences, using exactly these keys:\n"
    '{{"scene": "2-3 sentences on the location and setting: type of place, road or '
    'premises, lighting, time-of-day cues, weather, how busy it is.",\n'
    ' "subject": "2-3 sentences describing ONLY the outlined subject\'s visible '
    'appearance: clothing and colours, build, headwear, footwear, anything carried; '
    'for a vehicle its type, colour, condition and markings.",\n'
    ' "actions": "1-2 sentences on what the subject appears to be doing and its '
    'apparent direction of movement within the frame.",\n'
    ' "context": "1-2 sentences on other people, vehicles or objects nearby that an '
    'investigator should notice, including approximate positions.",\n'
    ' "quality": "1 sentence on how clearly the subject is captured and what that '
    'limits.",\n'
    ' "observations": ["2-4 short factual bullet points an investigator would '
    'want flagged"]}}'
)

_SUMMARY_PROMPT = (
    "You are assisting a police investigator by summarising a CCTV case file.\n\n"
    "{guardrails}\n"
    "Case details:\n{case}\n\n"
    "Exhibits in chronological order (system-recorded facts and the per-exhibit "
    "descriptions already produced):\n{items}\n\n"
    "Reply with ONLY a JSON object, no markdown fences, using exactly these keys:\n"
    '{{"overview": "3-5 sentences summarising what this collection of evidence '
    'shows overall.",\n'
    ' "movement": "3-5 sentences describing the sequence across cameras and times: '
    'where subjects were seen, in what order, and any gaps. Say plainly if the '
    'exhibits are from a single camera or are too few to establish movement.",\n'
    ' "corroboration": "2-3 sentences on which details are consistent across '
    'exhibits and which are not.",\n'
    ' "followups": ["3-6 concrete next investigative steps, each one line"],\n'
    ' "limitations": ["2-4 honest limitations of this evidence"]}}'
)


def _key() -> str | None:
    return os.environ.get(getattr(config, "GEMINI_API_KEY_ENV", "GEMINI_API_KEY"))


def _model() -> str:
    return (getattr(config, "GEMINI_REPORT_MODEL", "")
            or getattr(config, "GEMINI_MODEL", "gemini-flash-latest"))


def _model_chain() -> list[str]:
    """Models to try in order.

    The preferred report model and the lighter plate-reading model draw on separate
    free-tier quotas, so when the first is rate-limited (HTTP 429) the second can
    still narrate the report instead of the officer getting a report with no
    descriptions at all."""
    chain = [_model()]
    alt = getattr(config, "GEMINI_MODEL", "")
    if alt and alt not in chain:
        chain.append(alt)
    return chain


def available() -> bool:
    """True only if report narration is enabled, a key is set and requests imports."""
    if not getattr(config, "GEMINI_REPORT_ENABLED", True):
        return False
    if not getattr(config, "GEMINI_ENABLED", False):
        return False
    if not _key():
        return False
    try:
        import requests  # noqa: F401
        return True
    except Exception:
        return False


def _b64_jpeg(image, max_side: int = 1280) -> str | None:
    """JPEG-encode a path or BGR array, downscaled so uploads stay small."""
    try:
        img = cv2.imread(str(image)) if isinstance(image, (str, os.PathLike)) else image
        if img is None or not getattr(img, "size", 0):
            return None
        h, w = img.shape[:2]
        scale = max_side / float(max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
    except Exception:
        return None


def _parse_json(text: str) -> dict | None:
    """Gemini occasionally wraps JSON in ```json fences despite instructions."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()
    try:
        out = json.loads(t)
        return out if isinstance(out, dict) else None
    except ValueError:
        m = re.search(r"\{.*\}", t, re.S)          # salvage the first JSON object
        if not m:
            return None
        try:
            out = json.loads(m.group(0))
            return out if isinstance(out, dict) else None
        except ValueError:
            return None


def _call_model(model: str, parts, max_tokens: int) -> tuple[dict | None, str | None]:
    """One request to one model. Returns (parsed_json, error_message)."""
    try:
        import requests
        key = _key()
        if not key:
            return None, "no API key set"
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens,
                                 "responseMimeType": "application/json"},
        }
        resp = requests.post(_ENDPOINT.format(model=model), params={"key": key},
                             json=payload,
                             timeout=getattr(config, "GEMINI_REPORT_TIMEOUT", 60))
        if resp.status_code != 200:
            detail = ""
            try:
                detail = (resp.json().get("error") or {}).get("message", "")[:160]
            except Exception:
                detail = resp.text[:160]
            if resp.status_code == 429:
                err = f"{model}: API quota exceeded (HTTP 429)"
            else:
                err = f"{model}: HTTP {resp.status_code} {detail}"
            print(f"[gemini-report] {err}")
            return None, err
        body = resp.json()
        cands = body.get("candidates") or []
        if not cands:
            err = f"{model}: no candidates (promptFeedback={body.get('promptFeedback')})"
            print(f"[gemini-report] {err}")
            return None, err
        text = "".join(p.get("text", "")
                       for p in cands[0].get("content", {}).get("parts", []))
        if not text.strip():
            # The 2.5-series models spend part of the output budget on internal
            # reasoning (usageMetadata.thoughtsTokenCount). If maxOutputTokens is
            # too tight the reply comes back EMPTY with finishReason MAX_TOKENS
            # rather than as an error, which is why this is worth naming.
            err = (f"{model}: empty reply "
                   f"(finishReason={cands[0].get('finishReason')})")
            print(f"[gemini-report] {err} usage={body.get('usageMetadata')}")
            return None, err
        out = _parse_json(text)
        if out is None:
            err = f"{model}: reply was not JSON"
            print(f"[gemini-report] {err}: {text[:160]!r}")
            return None, err
        out["model"] = model
        return out, None
    except Exception as exc:
        err = f"{model}: request failed ({exc})"
        print(f"[gemini-report] {err}")
        return None, err


def _generate(parts, max_tokens: int) -> tuple[dict | None, str | None]:
    """Try each model in turn; return the first success, else the last error."""
    err = None
    for model in _model_chain():
        out, err = _call_model(model, parts, max_tokens)
        if out is not None:
            return out, None
    return None, err


def describe_exhibit(frame_image, subject_image, facts: dict) -> tuple[dict | None, str | None]:
    """Situational description for one exhibit. Returns (description, error)."""
    if not available():
        return None, "narration disabled or no API key"
    prompt = _ITEM_PROMPT.format(
        guardrails=_GUARDRAILS,
        facts=json.dumps(facts, indent=2, ensure_ascii=False, default=str))
    parts = [{"text": prompt}]
    for img in (frame_image, subject_image):
        b64 = _b64_jpeg(img) if img is not None else None
        if b64:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})
    if len(parts) == 1:                      # no usable imagery -> nothing to describe
        return None, "no usable image for this exhibit"
    # Budget must cover the model's internal reasoning tokens as well as the JSON
    # itself; too small and the reply arrives empty (see _call_model).
    return _generate(parts, max_tokens=4096)


def describe_exhibits(jobs: list[dict]) -> list[tuple[dict | None, str | None]]:
    """Describe several exhibits concurrently, preserving input order.

    Each job: {"frame": path|array, "subject": path|array, "facts": dict}."""
    if not jobs:
        return []
    if not available():
        return [(None, "narration disabled or no API key")] * len(jobs)
    workers = max(1, min(int(getattr(config, "GEMINI_REPORT_WORKERS", 4)), len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(
            lambda j: describe_exhibit(j.get("frame"), j.get("subject"), j.get("facts")),
            jobs))


def summarise_case(case: dict, items: list[dict]) -> tuple[dict | None, str | None]:
    """Overall case narrative + suggested follow-ups. Text only, no images."""
    if not available():
        return None, "narration disabled or no API key"
    prompt = _SUMMARY_PROMPT.format(
        guardrails=_GUARDRAILS,
        case=json.dumps(case, indent=2, ensure_ascii=False, default=str),
        items=json.dumps(items, indent=2, ensure_ascii=False, default=str)[:24000])
    return _generate([{"text": prompt}], max_tokens=6144)
