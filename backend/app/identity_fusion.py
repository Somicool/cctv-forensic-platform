"""Identity Fusion Engine - weighted multi-evidence identity decisions.

Why this exists
---------------
Person ReID used to dominate the identity decision: it carried ~56% of the
available weight once face was unavailable, so a moderate OSNet score sank an
otherwise well-corroborated match. A single learned embedding is exactly the
signal most damaged by the things forensic footage does constantly - posture
change, mounting a motorcycle, resolution loss, backlight.

This engine reduces ReID to ONE component (20% of appearance weight) and decides
identity from the agreement of many partly-independent evidence sources:

    BIOMETRIC / APPEARANCE          CONTEXT (corroborating only)
      face similarity                 timeline consistency
      person ReID similarity          camera GPS proximity
      clothing embedding              estimated travel time
      upper clothing colour           camera direction / facing
      lower clothing colour           journey continuity
      bag / backpack
      helmet / cap
      body proportions

Three-stage decision
--------------------
1. WEIGHTED EVIDENCE MEAN over whatever appearance signals exist, renormalised so
   a missing signal is neither a bonus nor a penalty.
2. CORROBORATION UPLIFT. Signals are grouped into INDEPENDENT evidence groups
   (face / reid / clothing / accessories / body). When several independent groups
   agree strongly, the score is lifted toward certainty - this is what lets a
   moderate ReID score still yield a confident identification. The lift is a
   bounded fraction of the remaining headroom, so corroboration can never
   manufacture identity out of nothing.
3. CONTRADICTION PENALTY. Independent groups that clearly disagree pull the score
   down multiplicatively, which is what pays for the uplift in false-match terms.

Context signals (timeline, GPS, travel time, direction, continuity) never add
positive identity mass on their own: timing consistency is not evidence of WHO
someone is, only of whether a sighting is possible. They scale the corroboration
uplift and can apply an impossibility penalty. That keeps the engine forensically
honest while still folding them into the final number.

The fused score stays on the same 0..1 scale as before, so the swept operating
point in track_identity.IDENTITY_ACCEPT keeps its calibrated meaning.
"""
from __future__ import annotations

# Rejected experiment, recorded so it is not retried: each source was also passed
# through a logistic centred between its measured same/different distributions
# (reid 0.746, clothing 0.869, body 0.730) to express it as evidence strength
# rather than raw cosine. It made discrimination WORSE on this dataset
# (AUC 0.984 -> 0.965, recall at 2% false-match 87.2% -> 71.8%), because the
# logistic saturates and discards the ordering information in the tails. Raw
# similarities are fused directly.

# ---------------------------------------------------------------- signal model
# weight  : share of the appearance decision (renormalised over available signals)
# group   : independence group - only ONE vote per group counts as corroboration
# strong  : at/above this the signal is positive corroborating evidence
# weak    : at/below this the signal actively contradicts the match
# label   : investigator-facing name used in the explanation
APPEARANCE_SIGNALS = {
    "face":        {"weight": 0.30, "group": "face",        "strong": 0.55, "weak": 0.30,
                    "label": "Face"},
    "reid":        {"weight": 0.20, "group": "reid",        "strong": 0.82, "weak": 0.62,
                    "label": "Person ReID"},
    "clothing":    {"weight": 0.08, "group": "clothing",    "strong": 0.93, "weak": 0.78,
                    "label": "Clothing appearance"},
    "upper_color": {"weight": 0.09, "group": "clothing",    "strong": 0.99, "weak": 0.20,
                    "label": "Upper clothing colour"},
    "lower_color": {"weight": 0.09, "group": "clothing",    "strong": 0.99, "weak": 0.20,
                    "label": "Lower clothing colour"},
    "bag":         {"weight": 0.06, "group": "accessories", "strong": 0.99, "weak": 0.10,
                    "label": "Bag / backpack"},
    "headwear":    {"weight": 0.06, "group": "accessories", "strong": 0.99, "weak": 0.10,
                    "label": "Helmet / cap"},
    "body":        {"weight": 0.12, "group": "body",        "strong": 0.90, "weak": 0.55,
                    "label": "Body proportions"},
}

CONTEXT_SIGNALS = {
    "timeline":      {"weight": 0.30, "label": "Timeline consistency"},
    "gps_proximity": {"weight": 0.22, "label": "Camera GPS proximity"},
    "travel_time":   {"weight": 0.26, "label": "Estimated travel time"},
    "direction":     {"weight": 0.12, "label": "Camera direction"},
    "continuity":    {"weight": 0.10, "label": "Journey continuity"},
}

# Corroboration / contradiction limits, chosen by sweep (see
# scripts/benchmark_identity_fusion.py) on recall at a FIXED 2% false-match rate,
# so extra recall is never bought with false matches:
#   uplift 0.00 (off)  -> AUC 0.984  recall 84.6%
#   uplift 0.20        -> AUC 0.984  recall 84.6%
#   uplift 0.40        -> AUC 0.984  recall 87.2%   <- default
#   uplift 0.75        -> AUC 0.984  recall 87.2%   (no further gain)
# Reference: the previous ReID-dominated weighted mean scored AUC 0.971 / 79.5%.
CORROBORATION_MAX = 0.40      # max fraction of the remaining headroom a full
                              # independent agreement may close
CONTRADICTION_MAX = 0.30      # max multiplicative penalty from disagreeing groups
# Score alignment (see _align): the fused score reaches the 2% false-match
# operating point at 0.894; the configured threshold stays 0.86.
ALIGN_FROM = 0.894
ALIGN_TO = 0.86
MIN_CORROBORATING_GROUPS = 2  # one signal agreeing with itself is not corroboration
# Context quality below this cannot support any uplift (implausible transition).
CONTEXT_FLOOR = 0.25


def _align(score: float) -> float:
    """Monotonic score alignment so an existing threshold keeps its meaning.

    Fusing more sources changes the SCALE of the output, not just its quality: the
    fused score that sits at the 2% false-match operating point is 0.894, whereas
    the previous ReID-dominated score reached that point at 0.860. Re-using 0.86 on
    the new scale would silently make the bar stricter and throw away the recall
    the fusion just gained.

    This maps 0.894 -> IDENTITY_ACCEPT (0.86) with two straight segments through
    (0,0) and (1,1). Being strictly monotonic it cannot change the ranking of
    candidates or the AUC - it only restates the same decision on the scale the
    configured threshold was calibrated for, so the threshold does not move."""
    if ALIGN_FROM <= 0.0 or ALIGN_FROM >= 1.0:
        return score
    if score <= ALIGN_FROM:
        return score * (ALIGN_TO / ALIGN_FROM)
    return ALIGN_TO + (score - ALIGN_FROM) * (1.0 - ALIGN_TO) / (1.0 - ALIGN_FROM)


def _weighted_mean(values: dict, spec: dict):
    """Mean of available signals weighted by `spec`, plus the weight actually used."""
    used = {k: v for k, v in values.items() if k in spec and v is not None}
    if not used:
        return None, 0.0, {}
    w_total = sum(spec[k]["weight"] for k in used)
    if w_total <= 0:
        return None, 0.0, {}
    score = sum(spec[k]["weight"] * used[k] for k in used) / w_total
    return score, w_total, dict(used)


def _group_votes(values: dict) -> dict:
    """Per independence group: its strongest agreement and worst contradiction.

    Only one vote per group, so three colour-derived signals cannot pretend to be
    three independent witnesses."""
    groups: dict = {}
    for key, spec in APPEARANCE_SIGNALS.items():
        v = values.get(key)
        if v is None:
            continue
        g = groups.setdefault(spec["group"], {"strong": False, "weak": False, "signals": []})
        g["signals"].append(key)
        if v >= spec["strong"]:
            g["strong"] = True
        if v <= spec["weak"]:
            g["weak"] = True
    # a group that has any strong vote is not counted as contradicting
    for g in groups.values():
        if g["strong"]:
            g["weak"] = False
    return groups


def fuse(appearance: dict, context: dict | None = None) -> dict:
    """Fuse all evidence into one identity score with a full contribution breakdown.

    `appearance` and `context` map signal name -> similarity in 0..1 (or None when
    the signal is unavailable). Returns the fused score plus, for every signal, its
    value, weight and share of the final decision - the explanation shown for each
    confirmed match."""
    context = context or {}
    base, weight_used, strength = _weighted_mean(appearance, APPEARANCE_SIGNALS)
    if base is None:
        return {"score": 0.0, "appearance_score": None, "context_score": None,
                "contributions": [], "corroboration": None, "explanation": []}

    ctx_score, ctx_weight, _ = _weighted_mean(context, CONTEXT_SIGNALS)
    # unknown context is neutral, not disqualifying (many deployments have no GPS)
    ctx_quality = 0.5 if ctx_score is None else ctx_score

    groups = _group_votes(appearance)
    n_groups = len(groups)
    n_strong = sum(1 for g in groups.values() if g["strong"])
    n_weak = sum(1 for g in groups.values() if g["weak"])

    # ---- stage 2: corroboration uplift ------------------------------------
    # Agreement is NET of disagreement: a witness that contradicts cancels one that
    # agrees, so a mixed evidence picture earns no uplift at all. Without this the
    # uplift also rescues look-alikes that happen to share two attributes.
    net_strong = n_strong - n_weak
    uplift = 0.0
    if net_strong >= MIN_CORROBORATING_GROUPS and n_groups > 1:
        agreement = (net_strong - 1) / (n_groups - 1)        # 0 at one group, 1 at all
        support = 0.0 if ctx_quality < CONTEXT_FLOOR else ctx_quality
        uplift = CORROBORATION_MAX * max(0.0, min(1.0, agreement)) * support
    lifted = base + uplift * (1.0 - base)

    # ---- stage 3: contradiction penalty -----------------------------------
    penalty = 0.0
    if n_weak and n_groups:
        penalty = CONTRADICTION_MAX * (n_weak / n_groups)
    # an implausible transition (impossible speed, sighting before the reference)
    # is itself a contradiction, applied through the context score
    if ctx_score is not None and ctx_score < CONTEXT_FLOOR:
        penalty = max(penalty, CONTRADICTION_MAX * (1.0 - ctx_score / CONTEXT_FLOOR))
    raw = max(0.0, min(1.0, lifted * (1.0 - penalty)))
    fused = _align(raw)                    # monotonic, keeps the threshold's meaning

    # ---- explanation: what each source contributed -------------------------
    contributions = []
    for key, spec in APPEARANCE_SIGNALS.items():
        v = appearance.get(key)
        if v is None:
            contributions.append({"signal": key, "label": spec["label"], "value": None,
                                  "weight": spec["weight"], "group": spec["group"],
                                  "share": 0.0, "verdict": "unavailable",
                                  "kind": "appearance"})
            continue
        st = strength.get(key, v)
        share = (spec["weight"] * st) / (weight_used * base) if base > 0 else 0.0
        contributions.append({
            "signal": key, "label": spec["label"], "value": round(float(v), 4),
            "pct": round(float(v) * 100), "strength": round(float(st), 4),
            "strength_pct": round(float(st) * 100),
            "weight": spec["weight"], "group": spec["group"],
            "share": round(share, 4), "kind": "appearance",
            "verdict": ("corroborates" if v >= spec["strong"] else
                        "contradicts" if v <= spec["weak"] else "neutral"),
        })
    for key, spec in CONTEXT_SIGNALS.items():
        v = context.get(key)
        contributions.append({
            "signal": key, "label": spec["label"],
            "value": None if v is None else round(float(v), 4),
            "pct": None if v is None else round(float(v) * 100),
            "weight": spec["weight"], "group": "context", "share": 0.0,
            "kind": "context",
            "verdict": ("unavailable" if v is None else
                        "corroborates" if v >= 0.66 else
                        "contradicts" if v < CONTEXT_FLOOR else "neutral"),
        })
    contributions.sort(key=lambda c: (c["kind"] != "appearance", -c["share"], -c["weight"]))

    strong_labels = [APPEARANCE_SIGNALS[s]["label"]
                     for g in groups.values() if g["strong"] for s in g["signals"]
                     if appearance.get(s) is not None
                     and appearance[s] >= APPEARANCE_SIGNALS[s]["strong"]]
    explanation = [f"Weighted appearance evidence: {round(base * 100)}%"]
    if uplift > 0:
        explanation.append(
            f"{n_strong} of {n_groups} independent evidence groups agree strongly "
            f"({', '.join(strong_labels[:4])}) -> confidence raised to {round(lifted * 100)}%")
    elif n_groups:
        explanation.append(f"Only {n_strong} of {n_groups} independent groups agree strongly "
                           "- no corroboration uplift applied")
    if penalty > 0:
        explanation.append(f"{n_weak} evidence group(s) contradict the match "
                           f"-> confidence reduced by {round(penalty * 100)}%")
    if ctx_score is not None:
        explanation.append(f"Context (timeline / distance / travel time): {round(ctx_score * 100)}%")
    else:
        explanation.append("Context unavailable (no camera GPS) - treated as neutral")

    return {
        "score": round(float(fused), 4),
        "raw_score": round(float(raw), 4),
        "appearance_score": round(float(base), 4),
        "context_score": None if ctx_score is None else round(float(ctx_score), 4),
        "uplift": round(float(uplift), 4),
        "penalty": round(float(penalty), 4),
        "corroboration": {"groups": n_groups, "strong": n_strong, "contradicting": n_weak,
                          "detail": {k: {"strong": v["strong"], "contradicts": v["weak"],
                                         "signals": v["signals"]} for k, v in groups.items()}},
        "contributions": contributions,
        "explanation": explanation,
    }
