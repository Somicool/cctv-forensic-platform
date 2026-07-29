// Evidence Gallery - shows ONLY the current investigation's evidence:
// search matches + bookmarked/selected items (never every extracted frame).
// Filter by category / camera / time / confidence; click a card for a detailed
// evidence viewer. Uses existing backend fields only.
import { useMemo, useState } from 'react'
import { useInvestigation } from '../context/investigation'
import VideoPlayer from '../components/VideoPlayer'
import TrackingViewer from '../components/TrackingViewer'
import { trackDetection } from '../api'
import { IcEvidence, IcSearch } from '../components/icons'

const VEHICLES = new Set(['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'van', 'auto'])
const CATS = [
  { id: 'person', label: 'Person' },
  { id: 'vehicle', label: 'Vehicle' },
  { id: 'face', label: 'Face' },
  { id: 'plate', label: 'License Plate' },
]

function kindsOf(r) {
  const a = r.attributes || {}
  const k = new Set()
  if (r.class_label === 'person') k.add('person')
  if (VEHICLES.has((r.class_label || '').toLowerCase())) k.add('vehicle')
  if (a.gender != null || a.age != null) k.add('face')
  if (a.plate_text) k.add('plate')
  return k
}

export default function Evidence() {
  const { evidence, inEvidence, toggleEvidence } = useInvestigation()

  const [cats, setCats] = useState(new Set())
  const [camera, setCamera] = useState('')
  const [minConf, setMinConf] = useState(0)
  const [from, setFrom] = useState('')
  const [to, setTo] = useState('')
  const [detail, setDetail] = useState(null)

  // Only items the investigator explicitly added to evidence (bookmarked with
  // the ＋ button) - never raw search matches or every extracted frame.
  const items = useMemo(
    () => evidence.map((r) => ({ ...r, _bookmarked: true })), [evidence])

  const cameraOptions = useMemo(
    () => [...new Set(items.map((r) => r.camera_id).filter(Boolean))].sort(), [items])

  const filtered = useMemo(() => items.filter((r) => {
    if (cats.size) { const k = kindsOf(r); if (![...cats].some((c) => k.has(c))) return false }
    if (camera && r.camera_id !== camera) return false
    if ((r.confidence || 0) * 100 < minConf) return false
    if (from && r.timestamp && r.timestamp < from) return false
    if (to && r.timestamp && r.timestamp > to + ':59') return false
    return true
  }), [items, cats, camera, minConf, from, to])

  const toggleCat = (id) => setCats((p) => { const n = new Set(p); n.has(id) ? n.delete(id) : n.add(id); return n })

  return (
    <div className="fp-page">
      <div className="fp-page-head">
        <div>
          <h1 className="fp-page-title">Evidence Gallery</h1>
          <p className="fp-page-desc">Evidence you've added to the current investigation.</p>
        </div>
      </div>

      {/* filters */}
      <div className="fp-card eg-filters">
        <div className="eg-fgroup">
          <span className="lbl">Type</span>
          {CATS.map((c) => (
            <button key={c.id} className={'eg-chip ' + (cats.has(c.id) ? 'on' : '')} onClick={() => toggleCat(c.id)}>{c.label}</button>
          ))}
        </div>
        <span className="eg-sep" />
        <div className="eg-fgroup">
          <span className="lbl">Camera</span>
          <select className="eg-select" value={camera} onChange={(e) => setCamera(e.target.value)}>
            <option value="">All</option>
            {cameraOptions.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div className="eg-fgroup">
          <span className="lbl">Time</span>
          <input className="eg-time" type="datetime-local" value={from} onChange={(e) => setFrom(e.target.value)} />
          <span className="lbl">to</span>
          <input className="eg-time" type="datetime-local" value={to} onChange={(e) => setTo(e.target.value)} />
        </div>
        <div className="eg-fgroup">
          <span className="lbl">Min conf {minConf}%</span>
          <input className="eg-slider" type="range" min="0" max="100" value={minConf} onChange={(e) => setMinConf(Number(e.target.value))} />
        </div>
        <span className="eg-count">{filtered.length} of {items.length} item{items.length === 1 ? '' : 's'}</span>
      </div>

      {items.length === 0 ? (
        <div className="fp-panel">
          <div className="fp-empty">
            <div className="ic"><IcEvidence size={26} /></div>
            <h4>No evidence yet</h4>
            <p>Run a search in the Investigation Workspace, then bookmark matches with the ＋ button. They'll appear here as evidence.</p>
          </div>
        </div>
      ) : filtered.length === 0 ? (
        <div className="fp-panel"><div className="fp-empty"><div className="ic"><IcSearch size={24} /></div><h4>No items match these filters</h4><p>Try clearing a filter or lowering the minimum confidence.</p></div></div>
      ) : (
        <div className="ws-cards">
          {filtered.map((r) => (
            <div key={r.detection_id} className="ws-rc" onClick={() => setDetail(r)} style={{ cursor: 'pointer' }}>
              <div className="ws-rc-thumb">
                {r.crop_url ? <img src={r.crop_url} alt={r.class_label} loading="lazy" /> : <div className="empty">{r.class_label}</div>}
                {r.score != null && <span className={'ws-score ' + scoreTier(r.score)}>{Math.round((r.score || 0) * 100)}%</span>}
              </div>
              <div className="ws-rc-body">
                <div className="ws-rc-top"><span className="lb">{r.class_label}</span><span className="cam">{r.camera_id}</span></div>
                <div className="ws-rc-attr">{attrText(r.attributes)}</div>
                {r.attributes?.plate_text && <div className="eg-plate">{r.attributes.plate_text}</div>}
                <div className="eg-conf">conf {Math.round((r.confidence || 0) * 100)}%</div>
                <div className="ws-rc-ts">{fmtTs(r.timestamp)}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {detail && <EvidenceViewer item={detail} onClose={() => setDetail(null)}
        inEvidence={inEvidence} toggleEvidence={toggleEvidence} />}
    </div>
  )
}

/* ---------------------------- detailed evidence viewer ---------------------------- */
function EvidenceViewer({ item, onClose, inEvidence, toggleEvidence }) {
  const [track, setTrack] = useState(null)
  const [tracing, setTracing] = useState(false)
  const [showTrack, setShowTrack] = useState(false)
  const a = item.attributes || {}
  const saved = inEvidence(item.detection_id)

  async function trace() {
    setTracing(true)
    try { setTrack(await trackDetection(item.detection_id)) } catch { setTrack({ appearances: [] }) } finally { setTracing(false) }
  }

  return (
    <div className="ws-overlay" onClick={onClose}>
      <div className="ws-modal eg-viewer" onClick={(e) => e.stopPropagation()}>
        <button className="ws-modal-x" onClick={onClose}>×</button>
        <h3 style={{ textTransform: 'capitalize' }}>{item.class_label} · <span className="muted" style={{ fontSize: 13 }}>{item.camera_id}</span></h3>
        <div className="eg-view-grid">
          <div>
            <VideoPlayer key={item.detection_id} src={item.video_url} offset={item.offset_seconds}
              bbox={item.bbox} frameW={item.frame_width} frameH={item.frame_height} />
            {track && (
              <div className="ws-track">
                <div className="ws-track-h">{track.appearances.length} appearance(s) across cameras</div>
                {track.appearances.slice(0, 15).map((ap, i) => (
                  <div className="ws-track-row" key={i}><span className="cam">{ap.camera_id}</span><span className="mono">{fmtTs(ap.timestamp).slice(11)}</span><span className="sim">{Math.round((ap.similarity || 0) * 100)}%</span></div>
                ))}
              </div>
            )}
          </div>
          <div>
            {item.crop_url && <img src={item.crop_url} alt="" style={{ width: '100%', borderRadius: 10, border: '1px solid var(--fp-border)', marginBottom: 12 }} />}
            <div className="eg-meta-row"><span className="k">Object type</span><span className="v">{item.class_label}</span></div>
            <div className="eg-meta-row"><span className="k">Camera</span><span className="v">{item.camera_name || item.camera_id}</span></div>
            <div className="eg-meta-row"><span className="k">Timestamp</span><span className="v">{fmtTs(item.timestamp)}</span></div>
            {item.score != null && <div className="eg-meta-row"><span className="k">Match score</span><span className="v">{Math.round(item.score * 100)}%</span></div>}
            <div className="eg-meta-row"><span className="k">Confidence</span><span className="v">{Math.round((item.confidence || 0) * 100)}%</span></div>
            {attrText(a) && <div className="eg-meta-row"><span className="k">Attributes</span><span className="v">{attrText(a)}</span></div>}
            {a.plate_text && <div className="eg-meta-row"><span className="k">Plate</span><span className="v"><span className="eg-plate">{a.plate_text}</span></span></div>}
            {item.track_appearances > 1 && <div className="eg-meta-row"><span className="k">Track sightings</span><span className="v">{item.track_appearances}×</span></div>}
            <div className="eg-view-actions">
              <button className={'ws-btn-sm ' + (saved ? '' : 'primary')} onClick={() => toggleEvidence(item)}>{saved ? '✓ In evidence' : '＋ Add to evidence'}</button>
              {item.track_id != null && <button className="ws-btn-sm" onClick={() => setShowTrack(true)} title="Follow this object in the video">⤳ Track object</button>}
              <button className="ws-btn-sm" onClick={trace} disabled={tracing}>{tracing ? 'Tracing…' : '⤳ Track across cameras'}</button>
              {item.crop_url && <a className="ws-btn-sm" href={item.crop_url} download>Download crop</a>}
            </div>
          </div>
        </div>
      </div>
      {showTrack && <TrackingViewer detection={item} onClose={() => setShowTrack(false)}
        onAddEvidence={toggleEvidence} inEvidence={inEvidence} />}
    </div>
  )
}

/* helpers */
function fmtTs(ts) { return ts ? ts.replace('T', ' ').slice(0, 19) : '—' }
function scoreTier(s) { if ((s || 0) >= 0.7) return 'high'; if ((s || 0) >= 0.4) return 'mid'; return 'low' }
function attrText(a) {
  if (!a) return ''
  const p = []
  if (a.color) p.push(a.color)
  if (a.upper_color) p.push('top: ' + a.upper_color)
  if (a.lower_color) p.push('btm: ' + a.lower_color)
  if (a.vehicle_type) p.push(a.vehicle_type)
  if (Array.isArray(a.accessories) && a.accessories.length) p.push(a.accessories.join(', '))
  if (a.age) p.push('age ' + a.age)
  if (a.gender) p.push(a.gender)
  return p.join(' · ')
}
