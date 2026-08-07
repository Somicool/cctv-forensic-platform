"""Benchmark the Journey Engine: reconstruction accuracy, runtime, GPU, routing time.

What is measured
----------------
1. TRANSITION ACCURACY (synthetic, exact ground truth). Camera layouts and travel
   modes are generated with known coordinates, timings and speeds, then fed
   through the engine. Because the truth is constructed we can check exactly:
     - are impossible transitions rejected, and possible ones kept?
     - is the travel mode recovered (walking / scooter / motorcycle / car)?
     - is distance/speed arithmetic correct against a reference haversine?
   This is real accuracy, not an estimate.

2. END-TO-END RUNTIME on the ingested dataset: how long `journey.reconstruct`
   takes and how that time splits between identity matching and the engine.

3. GPU: the engine is pure arithmetic. Allocated VRAM is sampled before and after
   to demonstrate the claim rather than assert it.

4. ROUTE GENERATION TIME for the configured provider (the null provider when no
   routing backend is set, which is the honest default).

    python -m scripts.benchmark_journey_engine
Run from the backend/ directory.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.getcwd())

from app import database, journey, journey_engine, routing        # noqa: E402

BASE = datetime(2026, 8, 7, 9, 0, 0)
# metres per degree of latitude, used to place synthetic cameras a known distance apart
M_PER_DEG = 111_320.0


def _cam(cid, lat, lon, facing=None, fov=None, coverage=60.0):
    return {"camera_id": cid, "lat": lat, "lon": lon, "facing_deg": facing,
            "fov_deg": fov, "coverage_m": coverage, "name": cid}


def _node(cid, t_offset_s, dwell_s=0.0, vehicles=None, ident=0.9):
    t0 = BASE + timedelta(seconds=t_offset_s)
    t1 = t0 + timedelta(seconds=dwell_s)
    return {"camera_id": cid, "first_seen": t0.isoformat(), "last_seen": t1.isoformat(),
            "identity_score": ident, "dwell_seconds": dwell_s,
            "vehicle_context": vehicles or [], "detection_id": None}


def _case(name, dist_m, travel_s, expect_mode, expect_plausible,
          vehicles=None, coverage=60.0):
    """One synthetic transition with exactly known distance, time and mode."""
    dlat = dist_m / M_PER_DEG
    geo = {"A": _cam("A", 21.170000, 72.830000, facing=0.0, fov=90.0, coverage=coverage),
           "B": _cam("B", 21.170000 + dlat, 72.830000, coverage=coverage)}
    nodes = [_node("A", 0, vehicles=vehicles), _node("B", travel_s, vehicles=vehicles)]
    return {"name": name, "geo": geo, "nodes": nodes,
            "expect_mode": expect_mode, "expect_plausible": expect_plausible,
            "expect_km": round(dist_m / 1000.0, 4),
            "expect_kmh": (round((dist_m / 1000.0) / (travel_s / 3600.0), 2)
                           if travel_s > 0 else None)}


def transition_cases():
    """Ground-truth transitions covering every mode plus the impossible ones."""
    return [
        # ---- mode recovery from measured speed (no vehicle observed) ----
        _case("walking 200 m in 3 min (4 km/h)", 200, 180, "walking", True),
        _case("walking 400 m in 6 min (4 km/h)", 400, 360, "walking", True),
        _case("scooter 1 km in 4 min (15 km/h)", 1000, 240, "scooter", True),
        _case("scooter 800 m in 2.5 min (19 km/h)", 800, 150, "scooter", True),
        _case("motorcycle 2 km in 3 min (40 km/h)", 2000, 180, "motorcycle", True),
        _case("motorcycle 1.5 km in 2 min (45 km/h)", 1500, 120, "motorcycle", True),
        _case("car 3 km in 3.2 min (56 km/h)", 3000, 193, "car", True),
        # ---- observation beats the speed band ----
        _case("motorcycle observed, slow traffic", 300, 180, "motorcycle", True,
              vehicles=["motorcycle"]),
        _case("scooter observed, slow traffic", 250, 200, "scooter", True,
              vehicles=["scooter"]),
        _case("car observed in congestion", 400, 300, "car", True, vehicles=["car"]),
        # ---- impossible: too fast for any road transport ----
        _case("10 km in 2 min (300 km/h)", 10000, 120, "unknown", False),
        _case("5 km in 60 s (300 km/h)", 5000, 60, "unknown", False),
        # ---- simultaneous sightings: coverage decides ----
        _case("simultaneous, 2 km apart, small coverage", 2000, -5, "overlap", False),
        _case("simultaneous, 30 m apart, overlapping coverage", 30, -5, "overlap", True,
              coverage=80.0),
    ]


def bench_transitions():
    cases = transition_cases()
    mode_ok = plaus_ok = dist_ok = speed_ok = 0
    failures = []
    t0 = time.perf_counter()
    for c in cases:
        legs, _rej = journey_engine.build_legs(c["nodes"], c["geo"])
        leg = legs[0]
        m_ok = leg["mode"] == c["expect_mode"]
        p_ok = leg["plausible"] == c["expect_plausible"]
        d_ok = (leg["distance_km"] is not None
                and abs(leg["distance_km"] - c["expect_km"]) <= max(0.002, 0.01 * c["expect_km"]))
        s_ok = (c["expect_kmh"] is None or leg["avg_speed_kmh"] is None
                or abs(leg["avg_speed_kmh"] - c["expect_kmh"]) <= 0.5)
        mode_ok += m_ok
        plaus_ok += p_ok
        dist_ok += d_ok
        speed_ok += s_ok
        if not (m_ok and p_ok and d_ok and s_ok):
            failures.append(f"{c['name']}: mode={leg['mode']} (want {c['expect_mode']}) "
                            f"plausible={leg['plausible']} (want {c['expect_plausible']}) "
                            f"km={leg['distance_km']} (want {c['expect_km']}) "
                            f"kmh={leg['avg_speed_kmh']} (want {c['expect_kmh']})")
    dt = time.perf_counter() - t0
    n = len(cases)
    pc = lambda x: 100.0 * x / n                                   # noqa: E731
    print("--- 1. TRANSITION ACCURACY (synthetic, exact ground truth) ---")
    print(f"  cases                       : {n}")
    print(f"  travel mode correct         : {pc(mode_ok):5.1f}%  ({mode_ok}/{n})")
    print(f"  plausible/impossible correct: {pc(plaus_ok):5.1f}%  ({plaus_ok}/{n})")
    print(f"  distance within 1%          : {pc(dist_ok):5.1f}%  ({dist_ok}/{n})")
    print(f"  speed within 0.5 km/h       : {pc(speed_ok):5.1f}%  ({speed_ok}/{n})")
    print(f"  engine time                 : {1000 * dt / n:.3f} ms per transition")
    for f in failures:
        print(f"  FAIL {f}")
    return not failures


def bench_direction():
    """Camera-direction reasoning: did the person leave through the field of view?"""
    a_in = _cam("A", 21.17, 72.83, facing=0.0, fov=90.0)     # looking north
    b_north = _cam("B", 21.180, 72.830)
    b_south = _cam("B", 21.160, 72.830)
    d1 = journey_engine.direction_for(a_in, b_north)
    d2 = journey_engine.direction_for(a_in, b_south)
    no_geo = journey_engine.direction_for({"lat": None, "lon": None}, b_north)
    ok = (d1["left_through_view"] is True and d1["compass"] == "N"
          and d2["left_through_view"] is False and d2["compass"] == "S"
          and no_geo["bearing_deg"] is None)
    print("\n--- 2. CAMERA DIRECTION REASONING ---")
    print(f"  travel north, camera faces north : {d1['compass']} "
          f"left_through_view={d1['left_through_view']}")
    print(f"  travel south, camera faces north : {d2['compass']} "
          f"left_through_view={d2['left_through_view']}")
    print(f"  no coordinates                   : {no_geo['note']}")
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    return ok


def bench_routing(runs=20):
    engine = routing.get_engine()
    pts = [{"camera_id": "A", "lat": 21.17, "lon": 72.83},
           {"camera_id": "B", "lat": 21.18, "lon": 72.84},
           {"camera_id": "C", "lat": 21.19, "lon": 72.85}]
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        res = engine.route(pts, profile="foot")
        times.append((time.perf_counter() - t0) * 1000.0)
    print("\n--- 4. ROUTE GENERATION ---")
    print(f"  active provider : {engine.name}")
    print(f"  available       : {res.available}")
    print(f"  mean time       : {statistics.mean(times):.3f} ms  "
          f"(median {statistics.median(times):.3f} ms, n={runs})")
    print(f"  geometry points : {len(res.geometry)}")
    if not res.available:
        print(f"  reason          : {res.reason}")
        print("  no straight-line fallback is emitted - the map stays empty by design")
    print("  configured providers:")
    for p in routing.providers():
        print(f"    {p['name']:<12} available={str(p['available']):<5} "
              f"active={str(p['active']):<5} {p.get('configure') or ''}")
    return statistics.mean(times)


def _vram_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e6
    except Exception:
        pass
    return None


def bench_end_to_end(limit=3):
    with database.get_conn() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT detection_id FROM detections WHERE class_label='person' "
            "AND track_id IS NOT NULL GROUP BY video_id, track_id "
            "ORDER BY COUNT(1) DESC LIMIT ?", (limit,)).fetchall()]
    print("\n--- 3. END-TO-END RUNTIME + GPU (ingested dataset) ---")
    if not rows:
        print("  no person tracks in the database - skipped")
        return
    before = _vram_mb()
    totals, engine_times, route_times, cams = [], [], [], []
    for r in rows:
        t0 = time.perf_counter()
        res = journey.reconstruct(r["detection_id"], persist=False)
        total = (time.perf_counter() - t0) * 1000.0
        if res.get("error"):
            print(f"  detection {r['detection_id']}: {res['error']}")
            continue
        primary = res["primary"]
        nodes, legs = primary["nodes"], primary["legs"]
        t1 = time.perf_counter()
        for _ in range(50):
            journey_engine.build_legs(nodes, res.get("camera_geo") or {})
            journey_engine.build_timeline(nodes, legs)
            journey_engine.stats(nodes, legs)
        engine_ms = (time.perf_counter() - t1) * 1000.0 / 50
        totals.append(total)
        engine_times.append(engine_ms)
        route_times.append(primary.get("route_ms") or 0.0)
        cams.append(primary["stats"]["cameras_visited"])
    after = _vram_mb()
    if totals:
        print(f"  journeys reconstructed : {len(totals)}")
        print(f"  full reconstruct       : mean {statistics.mean(totals):.1f} ms "
              f"(median {statistics.median(totals):.1f} ms)")
        print(f"  Journey Engine portion : mean {statistics.mean(engine_times):.3f} ms "
              f"({100 * statistics.mean(engine_times) / statistics.mean(totals):.2f}% of total)")
        print(f"  route step             : mean {statistics.mean(route_times):.3f} ms")
        print(f"  cameras per journey    : {cams}")
    if before is None:
        print("  GPU: CUDA not available in this process - engine is CPU-only regardless")
    else:
        print(f"  GPU VRAM allocated     : {before:.1f} MB before -> {after:.1f} MB after "
              f"(delta {after - before:+.1f} MB)")
        print("  the Journey Engine runs no model: any delta comes from the identity "
              "matching step, not from route reconstruction")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journeys", type=int, default=3)
    args = ap.parse_args()
    print("============ JOURNEY ENGINE BENCHMARK ============")
    ok1 = bench_transitions()
    ok2 = bench_direction()
    bench_end_to_end(args.journeys)
    bench_routing()
    print("\n==================================================")
    print(f"  correctness: {'PASS' if (ok1 and ok2) else 'FAIL'}")
    print("  Note: transition accuracy is measured on synthetic layouts with exact")
    print("  ground truth. Cross-camera IDENTITY accuracy is a separate concern,")
    print("  measured by benchmark_identity_fusion.py.")


if __name__ == "__main__":
    main()
