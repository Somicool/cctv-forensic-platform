"""Route Engine abstraction for the Journey Engine.

A route engine turns an ordered list of camera coordinates into a real travelled
path. Only the INTERFACE and the provider registry exist right now - no road
routing is implemented yet, on purpose.

Adding OSRM / GraphHopper / Valhalla / Google / OpenStreetMap later means writing
one class with `route(points, profile)` and registering it here. Nothing else in
the project changes: the Journey Engine and the frontend consume the same
`RouteResult` shape regardless of provider.

    class OsrmEngine(RouteEngine):
        name, available() -> True when the host is reachable
        def route(self, points, profile="foot"): ... -> RouteResult

    register(OsrmEngine())
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict


@dataclass
class RouteLeg:
    """One camera-to-camera hop of a route."""
    from_camera: str | None = None
    to_camera: str | None = None
    distance_m: float | None = None          # road distance (None until a real engine runs)
    duration_s: float | None = None          # engine-estimated travel time
    geometry: list = field(default_factory=list)   # [[lat, lon], ...] road polyline


@dataclass
class RouteResult:
    """Uniform result contract every provider must return."""
    provider: str
    available: bool
    profile: str | None = None
    legs: list = field(default_factory=list)
    distance_m: float | None = None
    duration_s: float | None = None
    geometry: list = field(default_factory=list)   # full [[lat, lon], ...] path
    reason: str | None = None                     # why unavailable, when applicable

    def to_dict(self) -> dict:
        d = asdict(self)
        d["legs"] = [asdict(l) if not isinstance(l, dict) else l for l in self.legs]
        return d


class RouteEngine:
    """Interface every routing provider implements."""
    name = "base"
    profiles = ("foot", "bike", "car")

    def available(self) -> bool:                      # pragma: no cover - interface
        return False

    def route(self, points: list[dict], profile: str = "foot") -> RouteResult:
        raise NotImplementedError


class NullRouteEngine(RouteEngine):
    """Default engine: NO routing backend configured.

    Deliberately returns an unavailable result with an empty geometry so the UI
    shows an honest message instead of drawing a misleading straight line between
    cameras."""
    name = "none"

    def available(self) -> bool:
        return False

    def route(self, points, profile="foot") -> RouteResult:
        return RouteResult(
            provider=self.name, available=False, profile=profile, geometry=[],
            reason=("No routing engine configured - road-accurate routes are not "
                    "available yet. Straight lines are intentionally not drawn."))


# ------------------------------------------------------------------ providers
def _http_json(url: str, timeout: float = 6.0, headers: dict | None = None):
    """Minimal GET returning parsed JSON. stdlib only - routing must not add deps."""
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "sentinel-journey"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class _HttpRouteEngine(RouteEngine):
    """Shared plumbing for HTTP routing backends.

    A provider is `available()` only when it has been explicitly configured, so an
    unreachable or unconfigured backend degrades to the honest "no routing"
    message rather than a broken map."""
    base_url: str | None = None
    api_key: str | None = None
    requires_key = False

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or self.base_url or "").rstrip("/") or None
        self.api_key = api_key or self.api_key

    def available(self) -> bool:
        if self.requires_key and not self.api_key:
            return False
        return bool(self.base_url)

    def _unavailable(self, profile, reason) -> RouteResult:
        return RouteResult(provider=self.name, available=False, profile=profile,
                           geometry=[], reason=reason)

    def _legs_from_geometry(self, points, geometry, distance_m, duration_s) -> list:
        """Split a full polyline back into camera-to-camera legs by nearest vertex."""
        if not geometry or len(points) < 2:
            return []
        idx = []
        for p in points:
            best, bi = None, 0
            for i, (lat, lon) in enumerate(geometry):
                d = (lat - p["lat"]) ** 2 + (lon - p["lon"]) ** 2
                if best is None or d < best:
                    best, bi = d, i
            idx.append(bi)
        legs = []
        for (a, b), (ia, ib) in zip(zip(points, points[1:]), zip(idx, idx[1:])):
            lo, hi = sorted((ia, ib))
            legs.append(RouteLeg(from_camera=a["camera_id"], to_camera=b["camera_id"],
                                 geometry=geometry[lo:hi + 1]))
        # spread engine totals across legs proportionally to their vertex count
        total_pts = sum(max(1, len(l.geometry)) for l in legs) or 1
        for l in legs:
            frac = max(1, len(l.geometry)) / total_pts
            if distance_m is not None:
                l.distance_m = round(distance_m * frac, 1)
            if duration_s is not None:
                l.duration_s = round(duration_s * frac, 1)
        return legs


class OsrmEngine(_HttpRouteEngine):
    """OSRM (also the backend behind the public OpenStreetMap routing demo).

    Configure with ROUTE_OSRM_URL, e.g. http://localhost:5000 for a self-hosted
    instance. Nothing is contacted unless that variable is set."""
    name = "osrm"
    profiles = ("foot", "bike", "car")
    _OSRM_PROFILE = {"foot": "foot", "bike": "bike", "car": "driving"}

    def route(self, points, profile="foot") -> RouteResult:
        if not self.available():
            return self._unavailable(profile, "OSRM not configured (set ROUTE_OSRM_URL)")
        coords = ";".join(f"{p['lon']},{p['lat']}" for p in points)
        url = (f"{self.base_url}/route/v1/{self._OSRM_PROFILE.get(profile, 'foot')}/"
               f"{coords}?overview=full&geometries=geojson&steps=false")
        try:
            data = _http_json(url)
        except Exception as exc:
            return self._unavailable(profile, f"OSRM request failed: {exc}")
        routes = data.get("routes") or []
        if not routes:
            return self._unavailable(profile, "OSRM returned no route between these cameras")
        r = routes[0]
        geom = [[lat, lon] for lon, lat in (r.get("geometry") or {}).get("coordinates", [])]
        return RouteResult(provider=self.name, available=True, profile=profile,
                           geometry=geom, distance_m=r.get("distance"),
                           duration_s=r.get("duration"),
                           legs=self._legs_from_geometry(points, geom, r.get("distance"),
                                                         r.get("duration")))


class GraphHopperEngine(_HttpRouteEngine):
    """GraphHopper. Configure ROUTE_GRAPHHOPPER_KEY (and optionally a self-hosted
    ROUTE_GRAPHHOPPER_URL)."""
    name = "graphhopper"
    base_url = "https://graphhopper.com/api/1"
    profiles = ("foot", "bike", "car")
    requires_key = True
    _GH_PROFILE = {"foot": "foot", "bike": "bike", "car": "car"}

    def route(self, points, profile="foot") -> RouteResult:
        if not self.available():
            return self._unavailable(profile, "GraphHopper not configured "
                                              "(set ROUTE_GRAPHHOPPER_KEY)")
        pts = "&".join(f"point={p['lat']},{p['lon']}" for p in points)
        url = (f"{self.base_url}/route?{pts}&vehicle={self._GH_PROFILE.get(profile, 'foot')}"
               f"&points_encoded=false&key={self.api_key}")
        try:
            data = _http_json(url)
        except Exception as exc:
            return self._unavailable(profile, f"GraphHopper request failed: {exc}")
        paths = data.get("paths") or []
        if not paths:
            return self._unavailable(profile, "GraphHopper returned no route")
        p0 = paths[0]
        geom = [[lat, lon] for lon, lat in (p0.get("points") or {}).get("coordinates", [])]
        dur = (p0.get("time") or 0) / 1000.0 or None
        return RouteResult(provider=self.name, available=True, profile=profile,
                           geometry=geom, distance_m=p0.get("distance"), duration_s=dur,
                           legs=self._legs_from_geometry(points, geom,
                                                         p0.get("distance"), dur))


class ValhallaEngine(_HttpRouteEngine):
    """Valhalla. Configure ROUTE_VALHALLA_URL (self-hosted or a hosted endpoint)."""
    name = "valhalla"
    profiles = ("foot", "bike", "car")
    _COSTING = {"foot": "pedestrian", "bike": "bicycle", "car": "auto"}

    def route(self, points, profile="foot") -> RouteResult:
        if not self.available():
            return self._unavailable(profile, "Valhalla not configured (set ROUTE_VALHALLA_URL)")
        body = {"locations": [{"lat": p["lat"], "lon": p["lon"]} for p in points],
                "costing": self._COSTING.get(profile, "pedestrian"),
                "directions_options": {"units": "kilometers"}}
        url = f"{self.base_url}/route?json={urllib.parse.quote(json.dumps(body))}"
        try:
            data = _http_json(url)
        except Exception as exc:
            return self._unavailable(profile, f"Valhalla request failed: {exc}")
        trip = data.get("trip") or {}
        shapes = [l.get("shape") for l in (trip.get("legs") or []) if l.get("shape")]
        geom = [pt for s in shapes for pt in _decode_polyline(s, precision=6)]
        if not geom:
            return self._unavailable(profile, "Valhalla returned no route geometry")
        summary = trip.get("summary") or {}
        dist_m = (summary.get("length") or 0) * 1000.0 or None
        return RouteResult(provider=self.name, available=True, profile=profile,
                           geometry=geom, distance_m=dist_m, duration_s=summary.get("time"),
                           legs=self._legs_from_geometry(points, geom, dist_m,
                                                         summary.get("time")))


class GoogleRoutesEngine(_HttpRouteEngine):
    """Google Directions. Configure ROUTE_GOOGLE_KEY.

    Note for deployment: sending camera coordinates to Google means sending case
    location data to a third party. Prefer a self-hosted OSRM or Valhalla for
    sensitive investigations."""
    name = "google"
    base_url = "https://maps.googleapis.com/maps/api/directions"
    profiles = ("foot", "bike", "car")
    requires_key = True
    _G_MODE = {"foot": "walking", "bike": "bicycling", "car": "driving"}

    def route(self, points, profile="foot") -> RouteResult:
        if not self.available():
            return self._unavailable(profile, "Google routing not configured "
                                              "(set ROUTE_GOOGLE_KEY)")
        origin = f"{points[0]['lat']},{points[0]['lon']}"
        dest = f"{points[-1]['lat']},{points[-1]['lon']}"
        way = "|".join(f"{p['lat']},{p['lon']}" for p in points[1:-1])
        url = (f"{self.base_url}/json?origin={origin}&destination={dest}"
               f"&mode={self._G_MODE.get(profile, 'walking')}&key={self.api_key}"
               + (f"&waypoints={urllib.parse.quote(way)}" if way else ""))
        try:
            data = _http_json(url)
        except Exception as exc:
            return self._unavailable(profile, f"Google request failed: {exc}")
        routes = data.get("routes") or []
        if not routes:
            return self._unavailable(
                profile, f"Google returned no route ({data.get('status', 'unknown status')})")
        r = routes[0]
        geom = _decode_polyline((r.get("overview_polyline") or {}).get("points") or "")
        dist = sum((l.get("distance") or {}).get("value", 0) for l in r.get("legs", [])) or None
        dur = sum((l.get("duration") or {}).get("value", 0) for l in r.get("legs", [])) or None
        return RouteResult(provider=self.name, available=True, profile=profile,
                           geometry=geom, distance_m=dist, duration_s=dur,
                           legs=self._legs_from_geometry(points, geom, dist, dur))


def _decode_polyline(encoded: str, precision: int = 5) -> list:
    """Decode an encoded polyline into [[lat, lon], ...] (Google / Valhalla format)."""
    if not encoded:
        return []
    factor = float(10 ** precision)
    out, index, lat, lon = [], 0, 0, 0
    while index < len(encoded):
        for axis in (0, 1):
            shift, result = 0, 0
            while index < len(encoded):
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            if axis == 0:
                lat += delta
            else:
                lon += delta
        out.append([lat / factor, lon / factor])
    return out


# ------------------------------------------------------------------ registry
_ENGINES: dict[str, RouteEngine] = {}
_ACTIVE: str = "none"


def register(engine: RouteEngine, activate: bool = False) -> None:
    """Register a provider (and optionally make it active)."""
    _ENGINES[engine.name] = engine
    if activate:
        set_active(engine.name)


def set_active(name: str) -> bool:
    global _ACTIVE
    if name in _ENGINES:
        _ACTIVE = name
        return True
    return False


def get_engine(name: str | None = None) -> RouteEngine:
    """Active engine (or a named one). Falls back to the null engine."""
    return _ENGINES.get(name or _ACTIVE, _ENGINES["none"])


def providers() -> list[dict]:
    """Every known provider, whether it is configured, and how to configure it."""
    return [{"name": n, "registered": True, "available": e.available(),
             "active": n == _ACTIVE, "profiles": list(getattr(e, "profiles", [])),
             "configure": CONFIG_HINT.get(n)}
            for n, e in _ENGINES.items()]


CONFIG_HINT = {
    "none": None,
    "osrm": "ROUTE_OSRM_URL (e.g. http://localhost:5000 for a self-hosted OSRM/OSM instance)",
    "graphhopper": "ROUTE_GRAPHHOPPER_KEY (+ optional ROUTE_GRAPHHOPPER_URL)",
    "valhalla": "ROUTE_VALHALLA_URL",
    "google": "ROUTE_GOOGLE_KEY (sends camera coordinates to a third party)",
}


def configure_from_env() -> str:
    """Register every provider and activate whichever one is configured.

    Order of preference favours self-hosted engines, which keep case location data
    inside the deployment. ROUTE_ENGINE forces a specific provider."""
    register(NullRouteEngine())
    register(OsrmEngine(base_url=os.getenv("ROUTE_OSRM_URL")))
    register(GraphHopperEngine(base_url=os.getenv("ROUTE_GRAPHHOPPER_URL"),
                               api_key=os.getenv("ROUTE_GRAPHHOPPER_KEY")))
    register(ValhallaEngine(base_url=os.getenv("ROUTE_VALHALLA_URL")))
    register(GoogleRoutesEngine(api_key=os.getenv("ROUTE_GOOGLE_KEY")))

    forced = (os.getenv("ROUTE_ENGINE") or "").strip().lower()
    if forced and forced in _ENGINES:
        set_active(forced)
        return forced
    for name in ("osrm", "valhalla", "graphhopper", "google"):
        if _ENGINES[name].available():
            set_active(name)
            return name
    set_active("none")
    return "none"


configure_from_env()
