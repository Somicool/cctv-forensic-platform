"""Natural-language query understanding for descriptive search.

Turns a plain description like:
    "man with a black backpack, blue shirt and white trousers, holding an umbrella"
into structured constraints the search can verify against the attributes the
pipeline already extracts:

    {
      "object_type": "person",
      "gender": "male",
      "upper_color": "blue",
      "lower_color": "white",
      "vehicle_color": None,
      "vehicle_type": None,
      "accessories": ["backpack", "umbrella"],
      "colors_any": ["black"],
    }

Constraints split into two kinds:
  * HARD  - we store this attribute per detection, so we can filter exactly
            (object type, upper/lower clothing colour, vehicle colour/type, and
            the accessories the extractor knows: cap/helmet/backpack/handbag/
            sunglasses/face mask).
  * SOFT  - mentioned but not something we extract as a fact (e.g. "umbrella",
            gender, or a colour describing a bag). These don't exclude results;
            they only label + lean on CLIP visual ranking.

The raw query is always still used for CLIP similarity, so SOFT terms and any
wording we don't recognise still influence the ranking.
"""
from __future__ import annotations

import re

from .. import config

_PERSON_WORDS = {"man", "men", "woman", "women", "person", "people", "boy", "girl",
                 "guy", "lady", "pedestrian", "kid", "child", "someone", "male",
                 "female", "he", "she", "worker", "officer"}
_VEHICLE_WORDS = {"car", "truck", "van", "suv", "sedan", "hatchback", "bus",
                  "motorcycle", "motorbike", "bike", "scooter", "bicycle", "cycle",
                  "auto", "rickshaw", "vehicle", "jeep", "pickup", "lorry", "taxi"}
_MALE = {"man", "men", "boy", "guy", "male", "gentleman", "he"}
_FEMALE = {"woman", "women", "girl", "lady", "female", "she"}

# clothing nouns -> which colour slot a nearby colour word sets
_UPPER = {"shirt", "tshirt", "top", "jacket", "hoodie", "kurta", "blouse",
          "sweater", "coat", "jersey", "vest", "shirts"}
_LOWER = {"pant", "pants", "trouser", "trousers", "jeans", "shorts", "skirt",
          "lower", "lowers", "pyjama", "pajama"}

# accessory synonyms -> canonical label. Only cap/helmet/backpack/handbag/
# sunglasses/"face mask" are actually extracted; "umbrella" is recognised but
# stays SOFT (see module docstring).
_ACCESSORY_SYN = {
    "backpack": "backpack", "bagpack": "backpack", "rucksack": "backpack", "bag": "backpack",
    "handbag": "handbag", "purse": "handbag",
    "cap": "cap", "hat": "cap",
    "helmet": "helmet",
    "sunglasses": "sunglasses", "shades": "sunglasses", "goggles": "sunglasses",
    "mask": "face mask", "facemask": "face mask",
    "umbrella": "umbrella",
}

_ACCESSORY_NOUNS = set(_ACCESSORY_SYN)          # bag/backpack/cap/... boundary words
_COLORS = {c.lower() for c in config.COLORS}
_VTYPES = {t.lower() for t in config.VEHICLE_TYPES}
_EXTRACTED_ACCESSORIES = {a.lower() for a in config.ACCESSORIES}   # what we can verify
_PERSON_LABELS = {config.DETECT_CLASSES[c] for c in config.PERSON_CLASSES}
_VEHICLE_LABELS = {config.DETECT_CLASSES[c] for c in config.VEHICLE_CLASSES}


def _tokens(q: str) -> list[str]:
    return re.findall(r"[a-z]+", (q or "").lower())


def parse(query: str) -> dict:
    """Free text -> structured constraints (see module docstring)."""
    toks = _tokens(query)
    ts = set(toks)
    out = {"object_type": None, "gender": None, "upper_color": None,
           "lower_color": None, "vehicle_color": None, "vehicle_type": None,
           "accessories": [], "colors_any": []}

    is_person, is_vehicle = bool(ts & _PERSON_WORDS), bool(ts & _VEHICLE_WORDS)
    if is_person and not is_vehicle:
        out["object_type"] = "person"
    elif is_vehicle and not is_person:
        out["object_type"] = "vehicle"

    if (ts & _MALE) and not (ts & _FEMALE):
        out["gender"] = "male"
    elif (ts & _FEMALE) and not (ts & _MALE):
        out["gender"] = "female"

    joined = " ".join(toks)
    for t in _VTYPES:
        if re.search(rf"\b{re.escape(t)}\b", joined):
            out["vehicle_type"] = t
            break

    # bind each colour to the FIRST meaningful noun that follows it, so
    # "black backpack, blue shirt" gives blue->top (not black->top). A colour
    # that lands on a bag/accessory noun isn't storable, so it stays SOFT.
    leftover = []
    for i, tk in enumerate(toks):
        if tk not in _COLORS:
            continue
        target = None
        for w in toks[i + 1:i + 5]:
            if w in _COLORS:                 # next colour starts a new phrase
                break
            if w in _UPPER:
                target = "upper"; break
            if w in _LOWER:
                target = "lower"; break
            if w in _VEHICLE_WORDS:
                target = "vehicle"; break
            if w in _ACCESSORY_NOUNS:        # e.g. "black backpack" -> not storable
                target = "accessory"; break
        if target == "upper" and not out["upper_color"]:
            out["upper_color"] = tk
        elif target == "lower" and not out["lower_color"]:
            out["lower_color"] = tk
        elif target == "vehicle" and not out["vehicle_color"]:
            out["vehicle_color"] = tk
        else:
            leftover.append(tk)
    out["colors_any"] = list(dict.fromkeys(leftover))

    accs = []
    for tk in toks:
        a = _ACCESSORY_SYN.get(tk)
        if a and a not in accs:
            accs.append(a)
    out["accessories"] = accs
    return out


def evaluate(det: dict, parsed: dict) -> dict:
    """Check a detection against parsed constraints.

    Returns {"passed": bool, "matched": [labels satisfied], "soft": [visual-only
    labels]}. HARD constraints must all pass for `passed` to be True; SOFT ones
    never exclude - they just annotate / lean on CLIP ranking.
    """
    attrs = det.get("attributes") or {}
    label = det.get("class_label")
    det_accessories = {a.lower() for a in (attrs.get("accessories") or [])}
    matched, soft = [], []
    passed = True

    if parsed.get("object_type"):
        want = parsed["object_type"]
        ok = (want == "person" and label in _PERSON_LABELS) or \
             (want == "vehicle" and label in _VEHICLE_LABELS)
        if ok:
            matched.append(want)
        else:
            passed = False

    for parsed_key, attr_key, human in (("upper_color", "upper_color", "top"),
                                        ("lower_color", "lower_color", "bottom")):
        w = parsed.get(parsed_key)
        if w:
            if (attrs.get(attr_key) or "").lower() == w:
                matched.append(f"{w} {human}")
            else:
                passed = False

    if parsed.get("vehicle_color"):
        w = parsed["vehicle_color"]
        if (attrs.get("color") or "").lower() == w:
            matched.append(f"{w} vehicle")
        else:
            passed = False

    if parsed.get("vehicle_type"):
        w = parsed["vehicle_type"]
        if (attrs.get("vehicle_type") or "").lower() == w:
            matched.append(w)
        else:
            passed = False

    # accessories: extractable ones are HARD, the rest (umbrella) are SOFT
    for a in parsed.get("accessories", []):
        if a in _EXTRACTED_ACCESSORIES:
            if a in det_accessories:
                matched.append(a)
            else:
                passed = False
        else:
            soft.append(a)

    # colours with no clear target: match any slot (SOFT - never excludes)
    slot_colors = {(attrs.get(k) or "").lower()
                   for k in ("upper_color", "lower_color", "color")}
    for c in parsed.get("colors_any", []):
        (matched if c in slot_colors else soft).append(c)

    # gender is only known from a clear face, not stored per detection -> SOFT
    if parsed.get("gender"):
        soft.append(parsed["gender"])

    return {"passed": passed, "matched": matched, "soft": soft}


def to_chips(parsed: dict) -> list[dict]:
    """Human-readable constraint chips for the UI. kind = 'hard' | 'soft'."""
    chips = []
    if parsed.get("object_type"):
        chips.append({"label": parsed["object_type"], "kind": "hard"})
    if parsed.get("gender"):
        chips.append({"label": parsed["gender"], "kind": "soft"})
    if parsed.get("upper_color"):
        chips.append({"label": f"{parsed['upper_color']} top", "kind": "hard"})
    if parsed.get("lower_color"):
        chips.append({"label": f"{parsed['lower_color']} bottom", "kind": "hard"})
    if parsed.get("vehicle_color"):
        chips.append({"label": f"{parsed['vehicle_color']} vehicle", "kind": "hard"})
    if parsed.get("vehicle_type"):
        chips.append({"label": parsed["vehicle_type"], "kind": "hard"})
    for a in parsed.get("accessories", []):
        chips.append({"label": a, "kind": "hard" if a in _EXTRACTED_ACCESSORIES else "soft"})
    for c in parsed.get("colors_any", []):
        chips.append({"label": c, "kind": "soft"})
    return chips


if __name__ == "__main__":
    tests = [
        "man with a black backpack, blue shirt and white trousers, holding an umbrella",
        "woman in a red top with sunglasses",
        "white SUV",
        "person wearing a helmet on a motorcycle",
    ]
    for q in tests:
        p = parse(q)
        print(f"\nQ: {q}\n   parsed: {p}\n   chips : {[c['label']+'('+c['kind']+')' for c in to_chips(p)]}")
