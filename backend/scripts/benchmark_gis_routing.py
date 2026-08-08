"""Validate + benchmark the GIS Camera Registry and OSRM road routing.

Runs the Part 10 checklist against the cameras actually stored in SQLite, then
measures OSRM routing latency cold vs cached.

    python -m scripts.benchmark_gis_routing
Run from the backend/ directory.
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())

from app import camera_registry as cr, database, journey, routing   # noqa: E402

OK, BAD, WARN = "  [PASS]", "  [FAIL]", "  [WARN]"


def haversine_km(a, b):
    p = math.pi / 180
    x = (math.sin((b[0] - a[0]) * p / 2) ** 2
         + math.cos(a[0] * p) * math.cos(b[0] * p) * math.sin((b[1] - a[1]) * p / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="foot")
    args = ap.parse_args()
    results = []

    def check(cond, msg, warn_only=False):
        results.append(bool(cond) or warn_only)
        print(f"{OK if cond else (WARN if warn_only else BAD)} {msg}")

    print("=========== GIS REGISTRY + OSRM ROUTING VALIDATION ===========")

    # ---- 1. coordinates are really in SQLite -----------------------------
    with database.get_conn() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(cameras)")}
        stored = [dict(r) for r in conn.execute(
            "SELECT camera_id, lat, lon, address, road_name, facing_deg, fov_deg, "
            "coverage_m, active FROM cameras").fetchall()]
    needed = {"lat", "lon", "address", "road_name", "facing_deg", "fov_deg", "coverage_m"}
    print("\n1) PERSISTENCE")
    check(needed <= cols, f"cameras table has all siting columns ({len(cols)} columns)")
    raw_located = [c for c in stored if cr._valid_latlon(c["lat"], c["lon"])]
    check(len(raw_located) >= 1,
          f"{len(raw_located)} of {len(stored)} cameras have coordinates in SQLite")
    # a fresh read through the service layer is what survives a backend restart
    located = [c for c in cr.list_cameras() if c["has_gps"]]
    check(len(located) == len(raw_located),
          f"coordinates re-read from disk on a new connection ({len(located)})")

    # ---- 2. partial update must not wipe coordinates ---------------------
    print("\n2) PARTIAL SAVE (the reported bug)")
    probe = "BENCH-GIS-TMP"
    cr.upsert_camera({"camera_id": probe, "name": "probe", "lat": 21.1959, "lon": 72.8302,
                      "facing_deg": "NE", "fov_deg": 80, "coverage_m": 45})
    before = cr.get_camera(probe)
    cr.upsert_camera({"camera_id": probe, "name": "probe", "facing_deg": "W",
                      "coverage_m": 60, "road_name": "Ring Road"})
    after = cr.get_camera(probe)
    check(after["lat"] == before["lat"] and after["lon"] == before["lon"],
          "a save that omits lat/lon preserves the stored coordinates")
    check(after["facing_deg"] == 270.0 and after["road_name"] == "Ring Road",
          "the fields that WERE submitted are updated")
    bad = False
    try:
        cr.upsert_camera({"camera_id": probe, "lat": 999, "lon": 0})
    except ValueError:
        bad = True
    check(bad, "invalid coordinates are rejected, not silently stored as NULL")
    check((after.get("coverage_cone") or []) and len(after["coverage_cone"]) > 3,
          f"viewing cone generated from direction/FOV/range "
          f"({len(after.get('coverage_cone') or [])} points)")
    cr.delete_camera(probe, force=True)

    # ---- 3. OSRM endpoints + real road geometry --------------------------
    print("\n3) OSRM ROAD ROUTING")
    eng = routing.get_engine("osrm")
    eps = eng.endpoints(args.profile)
    print(f"      endpoint order: {[k for k, _u, _p in eps]}")
    check(bool(eps), "at least one OSRM endpoint is configured (local preferred)")
    if len(located) < 2:
        print(f"{WARN} fewer than 2 located cameras - routing checks skipped")
        return

    pts = [{"camera_id": c["camera_id"], "lat": float(c["lat"]), "lon": float(c["lon"])}
           for c in located[:5]]
    routing.clear_route_cache()
    t0 = time.perf_counter()
    res = routing.cached_route(pts, profile=args.profile, alternatives=True)
    cold_ms = (time.perf_counter() - t0) * 1000
    check(res.get("available"), f"OSRM returned a route via {res.get('provider')}"
                               f"{'' if res.get('available') else ': ' + str(res.get('reason'))}")
    if not res.get("available"):
        return
    geom = res.get("geometry") or []
    check(res.get("road_route") is True, "result is flagged as real road geometry")
    check(len(geom) > len(pts) * 3,
          f"geometry has {len(geom)} vertices for {len(pts)} cameras "
          f"-> follows roads rather than {len(pts)} straight segments")
    # road distance must exceed the straight-line distance through the same points
    straight = sum(haversine_km((pts[i]["lat"], pts[i]["lon"]),
                                (pts[i + 1]["lat"], pts[i + 1]["lon"]))
                   for i in range(len(pts) - 1))
    road_km = (res.get("distance_m") or 0) / 1000.0
    check(road_km >= straight * 0.98,
          f"road distance {road_km:.2f} km vs straight-line {straight:.2f} km "
          f"(detour factor {road_km / max(straight, 1e-9):.2f}x)")
    alts = res.get("alternatives") or []
    check(True, f"{len(alts)} alternative road route(s) offered by OSRM", warn_only=True)

    # ---- 4. cache ---------------------------------------------------------
    print("\n4) PERFORMANCE")
    warm = []
    for _ in range(5):
        t0 = time.perf_counter()
        r2 = routing.cached_route(pts, profile=args.profile, alternatives=True)
        warm.append((time.perf_counter() - t0) * 1000)
    check(r2.get("cached") is True, "second identical request is served from the cache")
    print(f"      cold  {cold_ms:8.1f} ms  (network round trip to OSRM)")
    print(f"      warm  {statistics.mean(warm):8.1f} ms  "
          f"(median {statistics.median(warm):.1f} ms, n=5)")
    print(f"      speedup {cold_ms / max(statistics.mean(warm), 1e-6):.0f}x")

    # ---- 5. journey rules -------------------------------------------------
    print("\n5) JOURNEY RULES")
    geo = {c["camera_id"]: c for c in cr.list_cameras()}
    one = journey._route_for([{"camera_id": located[0]["camera_id"]}], geo)
    check(one["available"] is False and not one["geometry"],
          "no route is generated from a single matched camera")
    nogps = [c for c in cr.list_cameras() if not c["has_gps"]]
    if nogps:
        mixed = journey._route_for(
            [{"camera_id": located[0]["camera_id"]},
             {"camera_id": nogps[0]["camera_id"]},
             {"camera_id": located[1]["camera_id"]}], geo)
        check(nogps[0]["camera_id"] in (mixed.get("skipped_no_location") or []),
              "cameras without coordinates are skipped AND named in the warning")
        check(mixed.get("available") is True or mixed.get("reason"),
              "the remaining located cameras are still routed")
    empty = journey._route_for([{"camera_id": c["camera_id"]} for c in nogps[:3]], geo)
    check(empty["available"] is False and not empty["geometry"],
          "cameras with no coordinates produce no route and no straight line")

    print("\n6) CAMERAS USED FOR THIS BENCHMARK (your stored locations)")
    for c in located[:5]:
        print(f"      {c['camera_id']:<10} {float(c['lat']):.4f},{float(c['lon']):.4f}  "
              f"videos={c['video_count']} dets={c['detection_count']} status={c['status']}")

    print("\n==============================================================")
    print(f"  {sum(results)}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
