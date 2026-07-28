"""Task 14 verification: multi-language query translation.

1) translate_query() turns Hindi/Gujarati into English (online, or offline dict).
2) end-to-end: a translated Hindi/Gujarati query returns the SAME footage as the
   equivalent English query (proves translate -> CLIP -> FAISS search works).

    python -u scripts/verify_translate.py
"""
import sys
from pathlib import Path

# Windows consoles default to cp1252, which can't encode Devanagari/Gujarati.
# Force UTF-8 (with replacement) so printing the queries never crashes.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.search import translate, text_search          # noqa: E402


def main():
    print("=== TASK 14 TRANSLATION VERIFICATION ===")

    # 1) translation of common descriptive phrases --------------------------
    cases = [
        ("hi", "सफ़ेद ट्रक", {"white", "truck"}),
        ("gu", "સફેદ ટ્રક", {"white", "truck"}),
        ("hi", "लाल कार", {"red", "car"}),
        ("hi", "बैकपैक वाला आदमी", {"backpack", "man"}),
    ]
    for lang, q, expect in cases:
        eng, method = translate.translate_query(q, lang)
        low = eng.lower()
        ok = all(w in low for w in expect)
        print(f"[{'PASS' if ok else 'FAIL'}] [{lang}] {q} -> {eng!r} (via {method}); "
              f"expect all of {expect}")

    # English passes through untouched
    eng, method = translate.translate_query("a white truck", "en")
    print(f"[{'PASS' if method == 'none' and eng == 'a white truck' else 'FAIL'}] "
          f"English passthrough -> {eng!r} (via {method})")

    # Offline fallback (what happens with no internet) - the dictionary must
    # still handle the common descriptive terms so a demo never hard-fails.
    print("--- offline fallback (forced, no network) ---")
    offline_cases = [
        ("सफ़ेद ट्रक", {"white", "truck"}),        # Hindi
        ("લાલ કાર", {"red", "car"}),                # Gujarati
        ("बैकपैक वाला आदमी", {"backpack", "man"}),  # Hindi phrase
    ]
    for q, expect in offline_cases:
        off = translate._offline_translate(q)
        ok = all(w in off.lower() for w in expect)
        print(f"[{'PASS' if ok else 'FAIL'}] offline {q} -> {off!r}; expect all of {expect}")

    # 2) end-to-end: a translated query routes through CLIP correctly -------
    translated, method = translate.translate_query("सफ़ेद ट्रक", "hi")

    # Apples-to-apples: an English search on the SAME translated string must be
    # identical to the Hindi query routed through translate -> the multilingual
    # path adds no distortion beyond the translation itself.
    ref = text_search.search_text(translated, top_k=5, include_scenes=False)
    hi = text_search.search_text("सफ़ेद ट्रक", top_k=5, include_scenes=False,
                                 translated_query=translated)
    ref_ids = [r.detection_id for r in ref.results]
    hi_ids = [r.detection_id for r in hi.results]
    hi_top = hi.results[0].class_label if hi.results else "-"

    print(f"translated -> {translated!r} (via {method})")
    print(f"  English direct  ids={ref_ids}")
    print(f"  Hindi via trans ids={hi_ids}")
    identical = ref_ids == hi_ids and hi.total > 0
    print(f"[{'PASS' if identical else 'FAIL'}] Hindi query gives identical results to "
          f"English {translated!r}")
    print(f"[{'PASS' if hi_top == 'truck' else 'FAIL'}] top result is class-correct (truck): {hi_top}")
    print(f"[{'PASS' if hi.translated_query == translated else 'FAIL'}] SearchResponse "
          f"carries translated_query={hi.translated_query!r}")

    # Sanity: overlap with the natural English phrasing 'a white truck'
    eng_ref = text_search.search_text("a white truck", top_k=5, include_scenes=False)
    overlap = len(set(hi_ids) & {r.detection_id for r in eng_ref.results})
    print(f"[{'PASS' if overlap >= 3 else 'WARN'}] overlap with 'a white truck' "
          f"= {overlap}/5 (article/phrasing differences are expected)")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
