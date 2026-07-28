"""Run a spread of natural-language queries against the running API and print
clear results - a quick way to eyeball search quality on the ingested footage.

Backend must be running (uvicorn app.main:app --port 8000).
    python scripts/demo_search.py
"""
import json
import sys
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

API = "http://localhost:8000/api/search/text"
QUERIES = [
    "a white truck",
    "a red car",
    "a person carrying a backpack",
    "a man in a white shirt",
    "a motorcycle",
    "a woman walking",
    "a person with an umbrella",
]


def search(q):
    body = json.dumps({"query": q, "top_k": 3, "include_scenes": False}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def main():
    for q in QUERIES:
        try:
            r = search(q)
        except Exception as exc:  # noqa: BLE001
            print(f"\nQUERY '{q}' -> ERROR: {exc}")
            continue
        bias = f" [bias={r.get('object_type')}]" if r.get("object_type") else ""
        print(f"\nQUERY: '{q}'  -> {r['total']} results{bias}")
        if r.get("note"):
            print(f"   NOTE: {r['note']}")
        for x in r["results"]:
            t = (x.get("timestamp") or "")[11:19] or "?"
            attrs = {k: v for k, v in (x.get("attributes") or {}).items()
                     if k != "kind" and not k.endswith("_score")}
            print(f"   rel={round((x.get('score') or 0) * 100):3d}% (raw {x.get('raw_score')})  "
                  f"{(x.get('class_label') or ''):<8} {x.get('camera_id') or ''}  "
                  f"t={t}  jump={x.get('offset_seconds')}s  {attrs}")
    print("\n(in the dashboard, clicking a result plays that clip from the 'jump' second)")


if __name__ == "__main__":
    main()
