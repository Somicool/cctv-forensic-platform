// Cross-camera tracking view: trace a detection across cameras and see its
// journey on a map + timeline, with the source clip for the selected sighting.
import { useEffect, useMemo, useState } from 'react'
import { MapContainer, TileLayer, CircleMarker, Popup, Polyline, Tooltip } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import { getVideos, trackDetection } from '../api'
import VideoPlayer from './VideoPlayer'

const SURAT = [21.195, 72.83]

export default function TrackingView({ cameras, initialDetectionId, onBack }) {
  const [detId, setDetId] = useState(initialDetectionId ? String(initialDetectionId) : '')
  const [track, setTrack] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState(0)
  const [videos, setVideos] = useState([])
  const [sortBy, setSortBy] = useState('time')       // 'time' | 'sim' (list only)

  useEffect(() => { getVideos().then(setVideos).catch(() => setVideos([])) }, [])

  useEffect(() => {
    if (initialDetectionId) {
      setDetId(String(initialDetectionId))
      doTrace(initialDetectionId)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialDetectionId])

  async function doTrace(id) {
    const target = id ?? parseInt(detId, 10)
    if (Number.isNaN(target)) { setError('Enter a detection ID.'); return }
    setLoading(true); setError(null); setSelected(0)
    try {
      const data = await trackDetection(target)
      setTrack(data)
      if (!data.appearances?.length) setError('No appearances found for this detection.')
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || 'Trace failed')
      setTrack(null)
    } finally {
      setLoading(false)
    }
  }

  const camGeo = useMemo(
    () => Object.fromEntries((cameras || []).map((c) => [c.camera_id, c])), [cameras])
  const camsWithGeo = (cameras || []).filter((c) => c.lat != null && c.lon != null)
  const center = useMemo(() => {
    if (camsWithGeo.length) {
      const la = camsWithGeo.reduce((s, c) => s + c.lat, 0) / camsWithGeo.length
      const lo = camsWithGeo.reduce((s, c) => s + c.lon, 0) / camsWithGeo.length
      return [la, lo]
    }
    return SURAT
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameras])

  const appearances = track?.appearances || []
  // Map polyline stays chronological; the list can be re-sorted independently.
  const listAppearances = [...appearances].sort((a, b) =>
    sortBy === 'sim'
      ? (b.similarity || 0) - (a.similarity || 0)
      : (a.timestamp || '').localeCompare(b.timestamp || ''))
  const pathPts = appearances
    .map((a) => camGeo[a.camera_id])
    .filter((c) => c && c.lat != null)
    .map((c) => [c.lat, c.lon])

  const sel = appearances[selected]
  const selVideo = sel ? videos.find((v) => v.camera_id === sel.camera_id) : null

  return (
    <div className="track-view">
      <div className="track-controls">
        {onBack && (
          <button className="btn ghost" onClick={onBack}>← Back to search</button>
        )}
        <input className="search-input" placeholder="Detection ID — e.g. 9020"
               value={detId}
               onChange={(e) => setDetId(e.target.value.replace(/[^0-9]/g, ''))}
               onKeyDown={(e) => { if (e.key === 'Enter') doTrace() }} />
        <button className="btn primary" onClick={() => doTrace()} disabled={loading}>
          {loading ? 'Tracing…' : 'Trace journey'}
        </button>
        {track && (
          <span className="muted small">
            ref #{track.reference_detection_id}
            {track.reference_class ? ` · ${track.reference_class}` : ''}
          </span>
        )}
      </div>

      {track?.summary && (
        <div className="track-summary">
          <span><b>{track.summary.total_appearances}</b> appearances</span>
          <span><b>{track.summary.unique_cameras}</b> camera{track.summary.unique_cameras === 1 ? '' : 's'}</span>
          {track.summary.first_seen && <span>first <b>{fmtClock(track.summary.first_seen)}</b></span>}
          {track.summary.last_seen && <span>last <b>{fmtClock(track.summary.last_seen)}</b></span>}
          {track.summary.span_seconds != null && <span>over <b>{fmtSpan(track.summary.span_seconds)}</b></span>}
        </div>
      )}

      {error && <div className="banner error">{error}</div>}

      <div className="track-layout">
        <div className="track-map-wrap">
          <MapContainer center={center} zoom={12} className="track-map" scrollWheelZoom>
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                       attribution="&copy; OpenStreetMap" />

            {camsWithGeo.map((c) => (
              <CircleMarker key={c.camera_id} center={[c.lat, c.lon]} radius={7}
                pathOptions={{ color: '#4a5878', fillColor: '#4a5878', fillOpacity: 0.55 }}>
                <Tooltip>{c.camera_id} · {c.name}</Tooltip>
              </CircleMarker>
            ))}

            {pathPts.length > 1 && (
              <Polyline positions={pathPts}
                pathOptions={{ color: '#00d4ff', weight: 3, opacity: 0.7, dashArray: '6 6' }} />
            )}

            {appearances.map((a, i) => {
              const c = camGeo[a.camera_id]
              if (!c || c.lat == null) return null
              const isSel = i === selected
              return (
                <CircleMarker key={i} center={[c.lat, c.lon]} radius={isSel ? 11 : 8}
                  pathOptions={{
                    color: isSel ? '#00d4ff' : '#2ed573',
                    fillColor: isSel ? '#00d4ff' : '#2ed573', fillOpacity: 0.85,
                  }}
                  eventHandlers={{ click: () => setSelected(i) }}>
                  <Popup>
                    <b>{a.camera_id}</b><br />
                    {a.timestamp ? a.timestamp.replace('T', ' ').slice(0, 19) : ''}<br />
                    similarity {Math.round((a.similarity || 0) * 100)}%
                    {a.crop_url && (
                      <><br /><img src={a.crop_url} alt="" style={{ width: 90, marginTop: 4, borderRadius: 4 }} /></>
                    )}
                  </Popup>
                </CircleMarker>
              )
            })}
          </MapContainer>

          {sel && (
            <div className="track-video">
              <div className="track-video-head">
                {sel.camera_id}{sel.camera_name ? ` · ${sel.camera_name}` : ''}
                {sel.timestamp ? ` · ${sel.timestamp.replace('T', ' ').slice(0, 19)}` : ''}
              </div>
              <VideoPlayer key={sel.detection_id} src={sel.video_url || selVideo?.url}
                           offset={sel.offset_seconds || 0} />
              {sel.crop_url && <img className="track-ref-crop" src={sel.crop_url} alt="appearance" />}
            </div>
          )}
        </div>

        <div className="track-timeline">
          <div className="track-head track-head-row">
            <span>Appearances</span>
            {appearances.length > 1 && (
              <div className="lang-toggle track-sort">
                <button className={sortBy === 'time' ? 'active' : ''} onClick={() => setSortBy('time')}>Time</button>
                <button className={sortBy === 'sim' ? 'active' : ''} onClick={() => setSortBy('sim')}>Match</button>
              </div>
            )}
          </div>
          {appearances.length === 0 && !loading && (
            <div className="muted small">Trace a detection to see its journey across cameras.</div>
          )}
          {listAppearances.map((a) => {
            const i = appearances.indexOf(a)
            return (
              <button key={a.detection_id} className={`timeline-item ${i === selected ? 'active' : ''}`}
                      onClick={() => setSelected(i)}>
                <div className="ti-thumb">
                  {a.crop_url ? <img src={a.crop_url} alt="" /> : <div className="thumb-empty">—</div>}
                </div>
                <div className="ti-body">
                  <div className="ti-top">
                    <span className="cam">{a.camera_id}{a.camera_name ? ` · ${a.camera_name}` : ''}</span>
                    <span className="sim">{Math.round((a.similarity || 0) * 100)}%</span>
                  </div>
                  <div className="ti-time mono">
                    {a.timestamp ? a.timestamp.replace('T', ' ').slice(0, 19) : ''}
                  </div>
                  {a.detection_id === track?.reference_detection_id && <div className="ti-ref">reference</div>}
                </div>
              </button>
            )
          })}
        </div>
      </div>
    </div>
  )
}

function fmtClock(ts) {
  if (!ts) return ''
  const t = ts.replace('T', ' ')
  return t.length >= 19 ? t.slice(0, 19) : t
}

function fmtSpan(sec) {
  if (sec == null) return ''
  sec = Math.round(sec)
  if (sec < 60) return sec + 's'
  const m = Math.floor(sec / 60), s = sec % 60
  if (m < 60) return m + 'm' + (s ? ' ' + s + 's' : '')
  const h = Math.floor(m / 60), mm = m % 60
  if (h < 24) return h + 'h' + (mm ? ' ' + mm + 'm' : '')
  const d = Math.floor(h / 24), hh = h % 24
  return d + 'd' + (hh ? ' ' + hh + 'h' : '')
}
