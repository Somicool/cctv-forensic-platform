// Interactive journey map - self-contained SVG (no map library, works fully
// offline). Projects camera GPS to screen space, draws the travel polyline with
// direction arrows, and shows a marker per camera with arrival time + confidence.
// Cameras without GPS are laid out on a time axis instead, so the journey is still
// readable before GPS is assigned to those cameras.
import { useMemo, useState } from 'react'

const W = 760, H = 420, PAD = 54

export default function JourneyMap({ journey, geo, onSelect, activeIdx }) {
  const [hover, setHover] = useState(null)
  const nodes = journey?.nodes || []
  const legs = journey?.legs || []
  // A real route is only drawn when the route engine returns one. Straight lines
  // between cameras are intentionally NOT drawn - they would misrepresent the
  // actual path taken. See app/routing.py.
  const route = journey?.route || {}
  const routeReady = !!(route.available && (route.geometry || []).length > 1)

  // One projector shared by the cameras, the road route and the coverage cones, so
  // everything lands in the same coordinate space and the road geometry actually
  // fits on screen (the bounds include the route, not just the cameras).
  const { pts, hasGps, project, routePts, altPts, cones } = useMemo(() => {
    const coords = nodes.map((n) => {
      const g = geo?.[n.camera_id] || {}
      return (g.lat != null && g.lon != null) ? { lat: Number(g.lat), lon: Number(g.lon) } : null
    })
    const has = coords.every(Boolean) && coords.length > 1
    if (!has) {
      // no GPS: spread along a time axis (still shows order + gaps)
      const n = Math.max(nodes.length - 1, 1)
      return {
        hasGps: false, project: null, routePts: [], altPts: [], cones: [],
        pts: nodes.map((_n, i) => ({
          x: PAD + (i / n) * (W - PAD * 2),
          y: H / 2 + (i % 2 === 0 ? -34 : 34),
        })),
      }
    }
    const geomAll = (route.geometry || [])
    const coneRings = nodes.map((n) => (geo?.[n.camera_id] || {}).coverage_cone || null)
    const all = [
      ...coords.map((c) => [c.lat, c.lon]),
      ...geomAll,
      ...coneRings.filter(Boolean).flat(),
    ]
    const lats = all.map((p) => p[0]), lons = all.map((p) => p[1])
    const minLat = Math.min(...lats), maxLat = Math.max(...lats)
    const minLon = Math.min(...lons), maxLon = Math.max(...lons)
    const spanLat = Math.max(maxLat - minLat, 1e-4), spanLon = Math.max(maxLon - minLon, 1e-4)
    const proj = (lat, lon) => ({
      x: PAD + ((lon - minLon) / spanLon) * (W - PAD * 2),
      y: H - PAD - ((lat - minLat) / spanLat) * (H - PAD * 2),        // north = up
    })
    return {
      hasGps: true,
      project: proj,
      pts: coords.map((c) => proj(c.lat, c.lon)),
      routePts: geomAll.map(([la, lo]) => proj(la, lo)),
      altPts: (journey?.route?.alternatives || [])
        .filter((a) => !a.primary && (a.geometry || []).length > 1)
        .map((a) => ({ label: a.label, confidence: a.confidence,
                       pts: a.geometry.map(([la, lo]) => proj(la, lo)) })),
      cones: coneRings.map((ring) => ring ? ring.map(([la, lo]) => proj(la, lo)) : null),
    }
  }, [nodes, geo, route, journey])

  if (!nodes.length) return <div className="jn-map-empty">No journey to display.</div>

  return (
    <div className="jn-map">
      <svg viewBox={`0 0 ${W} ${H}`} className="jn-svg" role="img" aria-label="Journey map">
        <defs>
          <marker id="jn-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--fp-accent)" />
          </marker>
          <pattern id="jn-grid" width="38" height="38" patternUnits="userSpaceOnUse">
            <path d="M38 0H0V38" fill="none" stroke="rgba(255,255,255,.045)" strokeWidth="1" />
          </pattern>
        </defs>
        <rect width={W} height={H} fill="url(#jn-grid)" />

        {/* Camera coverage cones: what each camera can actually observe, from its
            stored direction / field of view / coverage distance. */}
        {hasGps && cones.map((ring, i) => ring && (
          <polygon key={'cone' + i} points={ring.map((p) => `${p.x},${p.y}`).join(' ')}
                   fill="var(--fp-accent)" fillOpacity="0.10"
                   stroke="var(--fp-accent)" strokeOpacity="0.30" strokeWidth="1" />
        ))}

        {/* ROAD ROUTE. Drawn from the routing engine's own geometry (OSRM over
            OpenStreetMap), never from the camera points - joining the cameras
            directly would assert a path straight through buildings. When routing
            is unavailable nothing is drawn and the reason is shown instead. */}
        {altPts.map((a, i) => (
          <polyline key={'alt' + i} points={a.pts.map((p) => `${p.x},${p.y}`).join(' ')}
                    fill="none" stroke="var(--fp-warn)" strokeOpacity="0.45"
                    strokeWidth="2" strokeDasharray="5 4" />
        ))}
        {routeReady && routePts.length > 1 && (
          <polyline points={routePts.map((p) => `${p.x},${p.y}`).join(' ')} fill="none"
                    stroke="var(--fp-accent)" strokeWidth="3" strokeLinejoin="round"
                    strokeLinecap="round" markerEnd="url(#jn-arrow)" />
        )}
        {/* travelling pulse along the real road path (Part 6 animation) */}
        {routeReady && routePts.length > 1 && (
          <circle r="5" fill="var(--fp-accent)">
            <animateMotion dur={`${Math.max(6, Math.min(24, routePts.length / 8))}s`}
                           repeatCount="indefinite"
                           path={'M' + routePts.map((p) => `${p.x},${p.y}`).join(' L')} />
          </circle>
        )}
        {!routeReady && pts.length > 1 && (
          <text x={W / 2} y={26} textAnchor="middle" className="jn-leg-t">
            {route.reason || 'Road route unavailable.'}
          </text>
        )}

        {/* leg labels (time / distance / mode) */}
        {pts.slice(0, -1).map((p, i) => {
          const q = pts[i + 1], leg = legs[i] || {}
          const mx = (p.x + q.x) / 2, my = (p.y + q.y) / 2 - 8
          const bits = []
          if (leg.travel_seconds != null && leg.travel_seconds > 0) bits.push(fmtDur(leg.travel_seconds))
          if (leg.distance_km != null) bits.push(`${leg.distance_km} km`)
          if (leg.mode && leg.mode !== 'unknown') bits.push(leg.mode)
          if (!bits.length) return null
          return <text key={'l' + i} x={mx} y={my} className={'jn-leg-t' + (leg.plausible === false ? ' bad' : '')}
                       textAnchor="middle">{bits.join(' · ')}</text>
        })}

        {/* camera markers */}
        {pts.map((p, i) => {
          const n = nodes[i]
          const on = activeIdx === i || hover === i
          return (
            <g key={n.camera_id + i} className="jn-node" onMouseEnter={() => setHover(i)}
               onMouseLeave={() => setHover(null)} onClick={() => onSelect?.(i)}>
              <circle cx={p.x} cy={p.y} r={on ? 17 : 13} fill="var(--fp-surface)"
                      stroke={on ? 'var(--fp-accent)' : 'var(--fp-border-2)'} strokeWidth={on ? 3 : 2} />
              <text x={p.x} y={p.y + 4} textAnchor="middle" className="jn-node-n">{i + 1}</text>
              <text x={p.x} y={p.y - (on ? 24 : 20)} textAnchor="middle" className="jn-node-t">
                {String(n.first_seen || '').slice(11, 16)}
              </text>
              <text x={p.x} y={p.y + (on ? 32 : 28)} textAnchor="middle" className="jn-node-c">
                {shortCam(n.camera_id)}
              </text>
            </g>
          )
        })}
      </svg>
      {altPts.length > 0 && (
        <div className="jn-map-alts">
          <b>Possible road routes</b>
          {(route.alternatives || []).map((a) => (
            <span key={a.label} className={'jn-alt' + (a.primary ? ' on' : '')}>
              {a.label} · {Math.round((a.confidence || 0) * 100)}%
              {a.distance_m != null ? ` · ${(a.distance_m / 1000).toFixed(2)} km` : ''}
            </span>
          ))}
          <span className="jn-map-warn">
            The sightings fix the cameras, not the roads between them — routes are ranked, not certain.
          </span>
        </div>
      )}
      <div className="jn-map-foot">
        {!hasGps
          ? 'Sequence view — add camera locations in the Camera Registry to enable the geographic map, distance and speed.'
          : routeReady
            ? `Road route via ${route.provider}${route.cached ? ' (cached)' : ''} · ${routePts.length} road points`
            : <span className="jn-map-warn">⚠ {route.reason || journey?.map_notice || 'Road route unavailable.'}</span>}
        {(route.skipped_no_location || []).length > 0 && (
          <span className="jn-map-warn"> · {route.skipped_no_location.length} matched camera(s) ignored:
            no stored coordinates ({route.skipped_no_location.join(', ')})</span>
        )}
      </div>
    </div>
  )
}

function shortCam(id) {
  const s = String(id || '')
  return s.length > 16 ? s.slice(0, 15) + '…' : s
}
function fmtDur(s) {
  s = Math.round(s || 0)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  return m < 60 ? `${m}m` : `${Math.floor(m / 60)}h ${m % 60}m`
}
