// Camera Registry map - real OpenStreetMap basemap via Leaflet (already a
// dependency), showing every stored camera with its viewing cone.
//
// Cameras WITHOUT stored coordinates cannot be placed and are reported in a
// footer instead of being dropped silently or guessed at a default location.
import { useEffect, useMemo, useRef } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

const OSM_TILES = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
const OSM_ATTR = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

// Cameras closer together than this are treated as one place on the map, because
// separate markers would be drawn on top of each other and only the last one
// would be clickable. ~1e-5 degrees is about 1 metre.
const SAME_PLACE = 1e-5

export default function CameraMap({
  cameras, selected, onSelect, height = 420,
  route = null,            // {geometry, alternatives, available, reason, provider}
  sequence = null,         // ordered [{camera_id, label, timestamp}] for a journey
}) {
  const divRef = useRef(null)
  const mapRef = useRef(null)
  const layerRef = useRef(null)

  const located = useMemo(
    () => (cameras || []).filter((c) => c.has_gps && c.lat != null && c.lon != null),
    [cameras])
  const missing = useMemo(
    () => (cameras || []).filter((c) => !c.has_gps).map((c) => c.camera_id),
    [cameras])

  // Group co-located cameras so every one of them stays reachable. Their real
  // coordinates are kept - markers are never nudged apart, because moving a
  // camera's plotted position would misrepresent where it actually is.
  const places = useMemo(() => {
    const out = []
    located.forEach((c) => {
      const lat = Number(c.lat), lon = Number(c.lon)
      const hit = out.find((p) => Math.abs(p.lat - lat) < SAME_PLACE
                                && Math.abs(p.lon - lon) < SAME_PLACE)
      if (hit) hit.cameras.push(c)
      else out.push({ lat, lon, cameras: [c] })
    })
    return out
  }, [located])
  const shared = useMemo(() => places.filter((p) => p.cameras.length > 1), [places])
  const order = useMemo(() => {
    const m = new Map()
    ;(sequence || []).forEach((s, i) => m.set(s.camera_id, { n: i + 1, ...s }))
    return m
  }, [sequence])

  // Create the map once. Everything is guarded: this panel is decorative relative
  // to the registry form beside it, and a Leaflet failure (offline tiles, a
  // container Leaflet thinks is already initialised under React's double-invoked
  // effects) must never take the page down with it.
  useEffect(() => {
    if (mapRef.current || !divRef.current) return undefined
    let map
    try {
      if (divRef.current._leaflet_id) divRef.current._leaflet_id = undefined
      map = L.map(divRef.current, { zoomControl: true, attributionControl: true })
      L.tileLayer(OSM_TILES, { maxZoom: 19, attribution: OSM_ATTR }).addTo(map)
      map.setView([20.5937, 78.9629], 5)               // sane default before fitting
      mapRef.current = map
      layerRef.current = L.layerGroup().addTo(map)
    } catch (err) {
      console.error('[CameraMap] could not initialise the map', err)
      mapRef.current = null
      return undefined
    }
    return () => {
      try { map.remove() } catch { /* already gone */ }
      mapRef.current = null
      layerRef.current = null
    }
  }, [])

  // redraw markers + cones whenever the registry changes
  useEffect(() => {
    const map = mapRef.current, layer = layerRef.current
    if (!map || !layer) return
    try {
      layer.clearLayers()
    } catch { return }
    if (!located.length) return

    // --- road route, from the routing engine's own geometry only -------------
    // Cameras are never joined with a straight line: that would assert a path
    // through whatever lies between them.
    ;(route?.alternatives || []).forEach((alt) => {
      if (!alt.primary && (alt.geometry || []).length > 1) {
        L.polyline(alt.geometry, { color: '#e0b341', weight: 3, opacity: 0.45,
                                   dashArray: '6 5' }).addTo(layer)
      }
    })
    if (route?.available && (route.geometry || []).length > 1) {
      L.polyline(route.geometry, { color: '#6ea8ff', weight: 5, opacity: 0.95,
                                   lineJoin: 'round' }).addTo(layer)
    }

    // --- one marker per PLACE, listing every camera sited there --------------
    places.forEach((p) => {
      const ids = p.cameras.map((c) => c.camera_id)
      const on = ids.includes(selected)
      const seq = ids.map((id) => order.get(id)).filter(Boolean)
      const inJourney = seq.length > 0

      p.cameras.forEach((c) => {
        if (c.coverage_cone?.length > 2) {
          L.polygon(c.coverage_cone, {
            color: '#6ea8ff', weight: 1, opacity: 0.5,
            fillColor: '#6ea8ff', fillOpacity: on ? 0.26 : 0.12,
          }).addTo(layer)
        }
      })

      const label = inJourney ? seq.map((s) => s.n).join('/')
        : (p.cameras.length > 1 ? String(p.cameras.length) : '')
      const colour = inJourney ? '#6ea8ff' : (sequence ? '#8b93a7' : '#6ea8ff')
      const marker = L.marker([p.lat, p.lon], {
        icon: L.divIcon({
          className: 'cm-pin' + (on ? ' on' : '') + (inJourney ? ' seq' : '')
                     + (sequence && !inJourney ? ' dim' : ''),
          html: `<span>${esc(label || '\u25cf')}</span>`,
          iconSize: [26, 26], iconAnchor: [13, 13],
        }),
        title: ids.join(', '),
      }).addTo(layer)

      const blocks = p.cameras.map((c) => {
        const s = order.get(c.camera_id)
        const rows = [
          ['Camera ID', c.camera_id],
          ['Name', c.name],
          ['Latitude', Number(c.lat).toFixed(6)],
          ['Longitude', Number(c.lon).toFixed(6)],
          ['Address', c.address],
          ['Road', c.road_name],
          ['Direction', c.facing ? `${c.facing} (${Math.round(c.facing_deg)}°)` : null],
          ['Field of view', c.fov_deg ? `${Math.round(c.fov_deg)}°` : null],
          ['Coverage', c.coverage_m ? `${Math.round(c.coverage_m)} m` : null],
          ['Videos', c.video_count ?? 0],
          ['Investigations', c.investigation_count ?? 0],
          ['Status', c.status],
          ...(s ? [['Journey stop', `#${s.n}`], ['Seen at', s.label || s.timestamp]] : []),
        ].filter(([, v]) => v !== null && v !== undefined && v !== '')
        return `<b>${esc(c.name || c.camera_id)}</b>` +
          rows.map(([k, v]) => `<div><span>${esc(k)}</span> ${esc(String(v))}</div>`).join('') +
          (c.cone_estimated ? '<i>cone uses default field of view / range</i>' : '')
      })
      marker.bindPopup(
        `<div class="cm-pop">${blocks.join('<hr/>')}` +
        (p.cameras.length > 1
          ? `<i>${p.cameras.length} cameras are recorded at this exact location</i>`
          : '') + '</div>',
        { maxHeight: 300 })
      marker.on('click', () => onSelect?.(ids[0]))
    })

    try {
      const bounds = L.latLngBounds(places.map((p) => [p.lat, p.lon]))
      located.forEach((c) => (c.coverage_cone || []).forEach((q) => bounds.extend(q)))
      ;(route?.geometry || []).forEach((q) => bounds.extend(q))
      if (bounds.isValid()) map.fitBounds(bounds.pad(0.25), { maxZoom: 17 })
    } catch (err) {
      console.error('[CameraMap] could not fit bounds', err)
    }
  }, [located, places, order, selected, onSelect, route, sequence])

  // keep Leaflet's internal size in sync with the panel
  useEffect(() => {
    const t = setTimeout(() => mapRef.current?.invalidateSize(), 120)
    return () => clearTimeout(t)
  }, [height, located.length])

  return (
    <div className="cm-wrap">
      <div ref={divRef} className="cm-map" style={{ height }} />
      <div className="cm-foot">
        {located.length
          ? `${located.length} camera${located.length > 1 ? 's' : ''} at ${places.length} location${places.length > 1 ? 's' : ''}`
          : 'No camera has stored coordinates yet — add latitude and longitude to place it.'}
        {route && (route.available
          ? ` · road route via ${route.provider}${route.cached ? ' (cached)' : ''}`
          : <span className="cm-warn"> · {route.reason || 'Road route unavailable.'}</span>)}
        {shared.length > 0 && (
          <span className="cm-warn"> · {shared.map((p) => p.cameras.length).reduce((a, b) => a + b, 0)}
            {' '}cameras share {shared.length} location{shared.length > 1 ? 's' : ''} —
            they overlap as one pin ({shared.map((p) => p.cameras.map((c) => c.camera_id).join('=')).join('; ')}).
            Check their coordinates if they are meant to be apart.</span>
        )}
        {missing.length > 0 && (
          <span className="cm-warn"> · {missing.length} without coordinates, not shown:
            {' '}{missing.slice(0, 6).join(', ')}{missing.length > 6 ? '…' : ''}</span>
        )}
      </div>
    </div>
  )
}

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (m) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]))
}
