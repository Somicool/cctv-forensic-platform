// Journey - reconstructed cross-camera movement of one person.
// Reads journeys stored by the backend (permanent, per investigation) and shows
// an interactive map, a timeline, statistics and per-camera actions.
import { useEffect, useMemo, useState } from 'react'
import { listJourneys, getJourney, deleteJourney, exportJourneyApi, journeyToCaseFile } from '../api'
import { useInvestigation } from '../context/investigation'
import JourneyMap from '../components/JourneyMap'
import VideoPlayer from '../components/VideoPlayer'
import TrackingViewer from '../components/TrackingViewer'
import { IcClock } from '../components/icons'

const hhmm = (t) => (t ? String(t).slice(11, 19) : '—')
const fmtDate = (t) => { try { return new Date(t).toLocaleString() } catch { return t || '—' } }
function fmtDur(s) {
  if (s == null) return '—'
  s = Math.round(s)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  return m < 60 ? `${m}m ${s % 60}s` : `${Math.floor(m / 60)}h ${m % 60}m`
}
const MODE_ICON = {
  walking: '🚶', scooter: '🛵', motorcycle: '🏍', car: '🚗',
  'two-wheeler': '🛵', vehicle: '🚗', overlap: '⇄', unknown: '·',
}

export default function Journey() {
  const { toggleEvidence, inEvidence } = useInvestigation()
  const [list, setList] = useState(null)
  const [jid, setJid] = useState(null)
  const [j, setJ] = useState(null)
  const [which, setWhich] = useState('primary')      // 'primary' | alt index
  const [active, setActive] = useState(0)
  const [jump, setJump] = useState(null)
  const [track, setTrack] = useState(null)
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState(null)

  async function load() {
    const rows = await listJourneys().catch(() => [])
    setList(rows || [])
    if ((rows || []).length && jid == null) open(rows[0].journey_id)
  }
  useEffect(() => { load() }, [])          // eslint-disable-line

  async function open(id) {
    setBusy(true)
    try { const d = await getJourney(id); setJ(d); setJid(id); setWhich('primary'); setActive(0) }
    finally { setBusy(false) }
  }
  async function remove(id) {
    if (!window.confirm('Delete this reconstructed journey?')) return
    await deleteJourney(id)
    if (jid === id) { setJ(null); setJid(null) }
    load()
  }

  const journey = useMemo(() => {
    if (!j) return null
    return which === 'primary' ? j.primary : (j.alternatives || [])[Number(which)] || j.primary
  }, [j, which])

  const nodes = journey?.nodes || []
  const legs = journey?.legs || []
  const st = journey?.stats || {}
  const tl = journey?.timeline || []

  // Server-side export so the file matches what the backend actually stored
  // (per-leg evidence, rejected transitions, and the road route when available).
  async function doExport(fmt) {
    setBusy(true)
    try {
      const d = await exportJourneyApi(jid, fmt)
      const blob = new Blob([JSON.stringify(d, null, 2)], { type: 'application/json' })
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob)
      a.download = `journey_${jid}_${fmt}.${fmt === 'geojson' ? 'geojson' : 'json'}`
      a.click()
      URL.revokeObjectURL(a.href)
      setNote(`Exported ${fmt}`)
    } catch (e) {
      setNote(`Export failed: ${e.message || e}`)
    } finally { setBusy(false) }
  }

  async function saveToCase() {
    setBusy(true)
    try {
      const r = await journeyToCaseFile(jid, j?.investigation || null)
      setNote(`Sealed ${r.sightings_sealed} sightings into the case file · ${r.export_id}`)
    } catch (e) {
      setNote(`Save failed: ${e.message || e}`)
    } finally { setBusy(false) }
  }

  function exportJourney() {
    const blob = new Blob([JSON.stringify(j, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob); a.download = `journey_${jid}.json`; a.click()
  }
  function addAllToCase() {
    nodes.forEach((n) => {
      if (!inEvidence(n.detection_id)) {
        toggleEvidence({
          detection_id: n.detection_id, camera_id: n.camera_id, camera_name: n.camera_name,
          timestamp: n.timestamp, class_label: 'person', confidence: n.identity_score,
          score: n.identity_score, crop_url: n.crop_url, attributes: n.attributes || {},
          video_url: n.video_url, offset_seconds: n.offset_seconds, track_id: n.track_id,
        })
      }
    })
  }

  return (
    <div className="fp-page">
      <div className="fp-page-head">
        <div>
          <h1 className="fp-page-title">Journey Reconstruction</h1>
          <p className="fp-page-desc">Probable movement of the same person across cameras — stored permanently.</p>
        </div>
        {j && (
          <div className="jn-head-actions">
            <button className="fp-btn" onClick={addAllToCase}>＋ Add Journey to Case File</button>
            <button className="fp-btn" onClick={exportJourney}>Export Journey</button>
          </div>
        )}
      </div>

      {list === null ? <div className="cf-empty">Loading journeys…</div>
        : list.length === 0 ? (
          <div className="cf-empty">
            No journeys yet. Search for a <b>person</b> in the Investigation Workspace and
            click <b>⇢ Journey</b> on a result to reconstruct their movement across cameras.
          </div>) : (
          <>
            {/* saved journeys */}
            <div className="jn-strip">
              {list.map((r) => (
                <button key={r.journey_id} className={'jn-chip ' + (jid === r.journey_id ? 'on' : '')}
                        onClick={() => open(r.journey_id)}>
                  <b>#{r.journey_id}</b> {r.camera_count} cams · {Math.round((r.confidence || 0) * 100)}%
                  <span className="d">{fmtDate(r.created_at).split(',')[0]}</span>
                </button>
              ))}
            </div>

            {busy && !j ? <div className="cf-empty">Loading…</div> : j && (
              <>
                {/* journey variant selector */}
                <div className="jn-vars">
                  <button className={'dash-chip ' + (which === 'primary' ? 'on' : '')}
                          onClick={() => { setWhich('primary'); setActive(0) }}>
                    Primary · {Math.round((j.primary?.confidence || 0) * 100)}%
                  </button>
                  {(j.alternatives || []).map((a, i) => (
                    <button key={i} className={'dash-chip ' + (which === String(i) ? 'on' : '')}
                            onClick={() => { setWhich(String(i)); setActive(0) }}>
                      {a.label} · {Math.round(a.confidence * 100)}%
                    </button>
                  ))}
                  <button className="fp-btn sm" style={{ marginLeft: 'auto', borderColor: 'var(--fp-danger)', color: '#ffb3bb' }}
                          onClick={() => remove(jid)}>Delete</button>
                </div>

                {/* statistics */}
                <div className="fp-stats">
                  <Stat v={st.cameras_visited ?? st.cameras} l="Cameras Visited" />
                  <Stat v={st.distance_km != null ? `${st.distance_km} km` : '—'} l="Total Distance" />
                  <Stat v={fmtDur(st.travel_seconds ?? st.span_seconds)} l="Total Time" />
                  <Stat v={st.avg_speed_kmh != null ? `${st.avg_speed_kmh} km/h` : '—'} l="Average Speed" />
                  <Stat v={st.estimated_transport || '—'} l="Estimated Transport" />
                  <Stat v={`${Math.round((journey.confidence || 0) * 100)}%`} l="Confidence" />
                </div>

                {/* journey-level actions */}
                <div className="jn-acts">
                  <button className="fp-btn sm" disabled={busy}
                          onClick={() => doExport('summary')}>⭳ Export Summary</button>
                  <button className="fp-btn sm" disabled={busy}
                          onClick={() => doExport('json')}>⭳ Export Full JSON</button>
                  <button className="fp-btn sm" disabled={busy}
                          onClick={() => doExport('geojson')}>⭳ Export GeoJSON</button>
                  <button className="fp-btn sm" disabled={busy}
                          onClick={saveToCase}>⛨ Save to Case File</button>
                  {note && <span className="jn-note">{note}</span>}
                </div>

                {journey.map_notice && (
                  <div className="cf-alert err">
                    <b>{journey.map_notice}</b>{' '}
                    Add latitude/longitude for the involved cameras in the <b>Camera Registry</b>.
                    {j.route_engine && ` Route engine: ${j.route_engine.active}.`}
                  </div>
                )}

                {journey.rejected_transitions?.length > 0 && (
                  <div className="cf-alert err">
                    <b>Rejected transitions:</b> {journey.rejected_transitions.join(' · ')}
                  </div>
                )}

                {/* Track-level candidates: always shown, never "no match found" */}
                {(j.matching?.best_per_camera || []).length > 0 && (
                  <section className="fp-panel" style={{ marginBottom: 18 }}>
                    <div className="fp-panel-title">
                      <span>Probable matches per camera</span>
                      <span className="muted">
                        {j.matching.mode} · compared {j.matching.compared_tracks} of {j.matching.searched_tracks} tracks
                        {j.matching.accept_threshold != null &&
                          ` · confirmed at ≥${Math.round(j.matching.accept_threshold * 100)}%`}
                      </span>
                    </div>
                    <div className="tm-grid">
                      {j.matching.best_per_camera.map((c) => (
                        <div key={`${c.video_id}:${c.track_id}`} className="tm-card">
                          <div className="tm-top">
                            <span className="cam">{c.camera_id}</span>
                            <span className={'tm-conf ' + (c.confidence >= 0.7 ? 'hi' : c.confidence >= 0.5 ? 'mid' : 'lo')}>
                              {Math.round(c.confidence * 100)}%
                            </span>
                          </div>
                          <div className="tm-ts mono">
                            {hhmm(c.first_seen)} · track {c.track_id}
                            {c.tier && <span className={'tm-tier ' + c.tier}>{c.tier}</span>}
                          </div>
                          <div className="tm-bars">
                            <Bar l="Face" v={c.face_pct} />
                            <Bar l="ReID" v={c.reid_pct} />
                            <Bar l="Clothing" v={c.clothing_pct} />
                            <Bar l="Accessories" v={c.accessories_pct} />
                            <Bar l="Body" v={c.body_pct} />
                          </div>
                          <div className="tm-travel">
                            {MODE_ICON[(c.travel_method || '').split(' ')[0]] || '·'} {c.travel_method}
                            <span className="tm-trans">{c.transition}</span>
                          </div>
                          {(c.reasons || []).length > 0 && (
                            <div className="tm-reasons">{c.reasons.slice(0, 4).join(' · ')}</div>
                          )}
                          {c.fusion && <Evidence f={c.fusion} />}
                          <div className="jn-tl-acts">
                            <button className="ws-btn-sm" onClick={() => setJump({
                              detection_id: c.detection_id, camera_id: c.camera_id,
                              timestamp: c.first_seen, video_url: null, offset_seconds: null,
                            })}>⤿ Jump to Video</button>
                            <button className="ws-btn-sm" onClick={() => setTrack({
                              detection_id: c.detection_id, camera_id: c.camera_id, attributes: {},
                            })}>⤳ Track Person</button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </section>
                )}

                <div className="fp-split">
                  {/* map */}
                  <section className="fp-panel">
                    <div className="fp-panel-title"><span>Movement map</span>
                      <span className="muted">{st.gps_available ? 'GPS' : 'sequence view'}</span></div>
                    <JourneyMap journey={journey} geo={j.camera_geo} activeIdx={active} onSelect={setActive} />
                  </section>

                  {/* timeline */}
                  <aside className="fp-panel">
                    <div className="fp-panel-title"><span><IcClock size={16} /> Timeline</span>
                      <span className="muted">{nodes.length} stops</span></div>
                    <div className="jn-timeline">
                      {nodes.map((n, i) => (
                        <div key={n.detection_id} className={'jn-tl-item ' + (active === i ? 'on' : '')}
                             onClick={() => setActive(i)}>
                          <div className="jn-tl-time">{hhmm(n.first_seen)}</div>
                          <div className="jn-tl-body">
                            <div className="jn-tl-top">
                              <span className="cam">{n.camera_name || n.camera_id}</span>
                              <span className={'jn-ev ' + n.evidence_strength}>{n.evidence_strength}</span>
                            </div>
                            <div className="jn-tl-meta">
                              match {Math.round(n.identity_score * 100)}%
                              {n.sightings > 1 ? ` · ${n.sightings} sightings` : ''}
                              {n.dwell_seconds ? ` · stayed ${fmtDur(n.dwell_seconds)}` : ''}
                            </div>
                            <div className="jn-tl-acts">
                              {n.video_url && <button className="ws-btn-sm" onClick={(e) => { e.stopPropagation(); setJump(n) }}>⤿ Jump to Video</button>}
                              <button className="ws-btn-sm" onClick={(e) => { e.stopPropagation(); setTrack(n) }}>⤳ Track Person</button>
                            </div>
                          </div>
                          {n.crop_url && <img className="jn-tl-thumb" src={n.crop_url} alt="" loading="lazy" />}
                          {i < nodes.length - 1 && (
                            <div className={'jn-tl-leg' + (legs[i]?.plausible === false ? ' bad' : '')}>
                              ↓ {MODE_ICON[legs[i]?.mode] || '·'} {legs[i]?.mode_label || legs[i]?.mode}
                              {legs[i]?.travel_seconds > 0 ? ` · ${fmtDur(legs[i].travel_seconds)}` : ''}
                              {legs[i]?.distance_km != null ? ` · ${legs[i].distance_km} km` : ''}
                              {legs[i]?.avg_speed_kmh != null ? ` · ${legs[i].avg_speed_kmh} km/h` : ''}
                              {legs[i]?.direction?.compass ? ` · heading ${legs[i].direction.compass}` : ''}
                              {legs[i]?.travel?.basis && (
                                <div className="jn-tl-basis">{legs[i].travel.basis}</div>
                              )}
                              {legs[i]?.direction?.left_through_view === false && (
                                <div className="jn-tl-basis">{legs[i].direction.note}</div>
                              )}
                            </div>
                          )}
                        </div>
                      ))}
                      <div className="jn-tl-end">
                        ◼ {tl[tl.length - 1]?.end_state || 'Exited Area'} · last seen {hhmm(st.last_seen)}
                        {tl[tl.length - 1]?.end_note && (
                          <div className="jn-tl-basis">{tl[tl.length - 1].end_note}</div>
                        )}
                      </div>
                    </div>
                  </aside>
                </div>
              </>
            )}
          </>
        )}

      {jump && (
        <div className="vi-overlay" onMouseDown={() => setJump(null)}>
          <div className="vi-modal" style={{ width: 'min(880px,96vw)' }} onMouseDown={(e) => e.stopPropagation()}>
            <div className="vi-head"><div className="vi-title">{jump.camera_name || jump.camera_id} · {hhmm(jump.timestamp)}</div>
              <button className="vi-x" onClick={() => setJump(null)}>×</button></div>
            <div className="vi-body">
              <VideoPlayer key={jump.detection_id} src={jump.video_url} offset={jump.offset_seconds} autoPlay />
            </div>
          </div>
        </div>
      )}
      {track && <TrackingViewer detection={{ detection_id: track.detection_id, class_label: 'person',
        camera_id: track.camera_id, attributes: track.attributes || {} }} onClose={() => setTrack(null)} />}
    </div>
  )
}

function Stat({ v, l }) {
  return <div className="fp-card fp-stat"><div className="v">{v ?? '—'}</div><div className="l">{l}</div></div>
}

function Bar({ l, v }) {
  const pct = Math.max(0, Math.min(100, v || 0))
  const col = pct >= 70 ? 'var(--fp-success)' : pct >= 45 ? 'var(--fp-warn)' : '#ff8a94'
  return (
    <div className="tm-bar">
      <span className="k">{l}</span>
      <span className="t"><i style={{ width: pct + '%', background: col }} /></span>
      <b>{pct}%</b>
    </div>
  )
}

// Why the engine reached this confidence: contribution of every evidence source.
function Evidence({ f }) {
  const [open, setOpen] = useState(false)
  const used = (f.contributions || []).filter((c) => c.value !== null)
  const appearance = used.filter((c) => c.kind === 'appearance')
  const context = used.filter((c) => c.kind === 'context')
  return (
    <div className="tm-ev">
      <button className="tm-ev-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? '▾' : '▸'} Why this match
        <span className="muted">
          appearance {Math.round((f.appearance_score || 0) * 100)}%
          {f.uplift > 0 && ` · corroboration +${Math.round(f.uplift * 100)}%`}
          {f.penalty > 0 && ` · contradiction −${Math.round(f.penalty * 100)}%`}
        </span>
      </button>
      {open && (
        <div className="tm-ev-body">
          {appearance.map((c) => (
            <div key={c.signal} className={'tm-ev-row ' + c.verdict}>
              <span className="k">{c.label}</span>
              <span className="v mono">{c.pct}%</span>
              <span className="w mono">w {c.weight.toFixed(2)}</span>
              <span className="s mono">{Math.round(c.share * 100)}% of score</span>
            </div>
          ))}
          {context.length > 0 && <div className="tm-ev-sep">Context (corroborating only)</div>}
          {context.map((c) => (
            <div key={c.signal} className={'tm-ev-row ' + c.verdict}>
              <span className="k">{c.label}</span>
              <span className="v mono">{c.pct}%</span>
              <span className="w mono">w {c.weight.toFixed(2)}</span>
              <span className="s">{c.verdict}</span>
            </div>
          ))}
          {(f.explanation || []).map((line, i) => (
            <div key={i} className="tm-ev-note">· {line}</div>
          ))}
        </div>
      )}
    </div>
  )
}
