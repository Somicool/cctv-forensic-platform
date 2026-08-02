// Face Gallery - saved best faces + find the same individual across footage.
// Reuses the existing InsightFace face index (no re-detection). Saved faces are
// permanent (server-side) and only removed on explicit delete.
import { useEffect, useMemo, useState } from 'react'
import { listSavedFaces, deleteSavedFace, findSimilarFaces } from '../api'
import VideoPlayer from '../components/VideoPlayer'
import TrackingViewer from '../components/TrackingViewer'
import { IcFace, IcSearch } from '../components/icons'

const fmtTs = (t) => (t ? t.replace('T', ' ').slice(0, 19) : '—')
const fmtDate = (t) => { try { return new Date(t).toLocaleString() } catch { return t } }

export default function FaceGallery() {
  const [faces, setFaces] = useState(null)
  const [q, setQ] = useState('')
  const [detail, setDetail] = useState(null)     // saved face open in the viewer

  async function load() { setFaces(await listSavedFaces().catch(() => [])) }
  useEffect(() => { load() }, [])

  const filtered = useMemo(() => {
    const list = faces || []
    const s = q.trim().toLowerCase()
    if (!s) return list
    return list.filter((f) => [f.investigation, f.camera_id, f.gender].some((x) => (x || '').toLowerCase().includes(s)))
  }, [faces, q])

  async function onDelete(id) {
    if (!window.confirm('Delete this saved face? This cannot be undone.')) return
    await deleteSavedFace(id)
    if (detail?.saved_id === id) setDetail(null)
    load()
  }

  return (
    <div className="fp-page">
      <div className="fp-page-head">
        <div>
          <h1 className="fp-page-title">Face Gallery</h1>
          <p className="fp-page-desc">Saved faces from your investigations. Find the same individual across all indexed footage.</p>
        </div>
      </div>

      <div className="fp-quicksearch">
        <IcSearch size={20} />
        <input placeholder="Search saved faces — investigation, camera…" value={q} onChange={(e) => setQ(e.target.value)} />
      </div>

      {faces === null ? <div className="dash-empty">Loading saved faces…</div>
        : faces.length === 0 ? (
          <div className="dash-empty">
            No saved faces yet. In the <b>Investigation Workspace</b>, search for people and click
            <b> Save Face</b> on a person result — the clearest face is stored here permanently.
          </div>)
          : filtered.length === 0 ? <div className="dash-empty">No saved faces match “{q}”.</div>
            : (
              <div className="fg-grid">
                {filtered.map((f) => (
                  <div key={f.saved_id} className="fg-card" onClick={() => setDetail(f)} title="Open face">
                    <div className="fg-thumb">
                      {(f.preview_crop_url || f.face_crop_url)
                        ? <img src={f.preview_crop_url || f.face_crop_url} alt="face" loading="lazy" />
                        : <div className="ph"><IcFace size={34} /></div>}
                      {f.confidence != null && <span className="fg-q">Q {Math.round(f.confidence * 100)}</span>}
                      {f.low_quality && <span className="fg-lowq">LOW QUALITY</span>}
                    </div>
                    <div className="fg-body">
                      <div className="fg-inv">{f.investigation || 'Unassigned case'}</div>
                      <div className="fg-meta">{f.camera_id || '—'}</div>
                      <div className="fg-meta mono">{fmtTs(f.timestamp)}</div>
                      <div className="fg-saved">Saved {fmtDate(f.created_at)}</div>
                    </div>
                  </div>
                ))}
              </div>)}

      {detail && <FaceViewer face={detail} onClose={() => setDetail(null)} onDelete={onDelete} />}
    </div>
  )
}

/* ---------------------------- saved-face viewer ---------------------------- */
function FaceViewer({ face, onClose, onDelete }) {
  const [tab, setTab] = useState('view')          // view | similar
  const [sim, setSim] = useState(null)
  const [loading, setLoading] = useState(false)
  const [jump, setJump] = useState(null)          // similar result to play
  const [track, setTrack] = useState(null)        // detection to track

  useEffect(() => { const esc = (e) => { if (e.key === 'Escape') onClose() }; window.addEventListener('keydown', esc); return () => window.removeEventListener('keydown', esc) }, [onClose])

  async function findSimilar() {
    setTab('similar'); setLoading(true)
    try { setSim(await findSimilarFaces(face.saved_id, 60)) } catch { setSim({ results: [] }) } finally { setLoading(false) }
  }

  function exportFace() {
    // client-side export of the saved-face evidence (image link + JSON record)
    const rec = { ...face }; delete rec.embedding
    const blob = new Blob([JSON.stringify(rec, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob); a.download = `face_${face.saved_id}.json`; a.click()
    if (face.face_crop_url) window.open(face.face_crop_url, '_blank')
  }

  return (
    <div className="vi-overlay" onMouseDown={onClose}>
      <div className="vi-modal fg-modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="vi-head">
          <div className="vi-title"><span className="vi-badge">FACE</span> Saved Face
            <span className="vi-plate" style={{ background: 'var(--fp-accent)', color: '#04121a' }}>#{face.saved_id}</span>
          </div>
          <div className="vi-head-actions">
            <button className={'fp-btn sm ' + (tab === 'view' ? 'primary' : '')} onClick={() => setTab('view')}>View Face</button>
            <button className={'fp-btn sm ' + (tab === 'similar' ? 'primary' : '')} onClick={findSimilar}>Find Similar Person</button>
            <button className="fp-btn sm" onClick={exportFace}>Export</button>
            <button className="fp-btn sm" onClick={() => onDelete(face.saved_id)} style={{ borderColor: 'var(--fp-danger)', color: '#ffb3bb' }}>Delete</button>
            <button className="vi-x" onClick={onClose}>×</button>
          </div>
        </div>

        <div className="vi-body">
          {tab === 'view' ? (
            <div className="fg-view">
              <div>
                <img className="fg-view-face" src={face.preview_crop_url || face.face_crop_url || face.person_crop_url} alt="face" />
                {face.low_quality && <div className="fg-lowq-note">⚠ Low-quality face — best available in this track</div>}
              </div>
              <div className="fg-view-info">
                <InfoRow k="Investigation" v={face.investigation || '—'} />
                <InfoRow k="Camera" v={face.camera_id || '—'} />
                <InfoRow k="Timestamp" v={fmtTs(face.timestamp)} />
                <InfoRow k="Age (est.)" v={face.age ?? '—'} />
                <InfoRow k="Quality score" v={face.confidence != null ? `${Math.round(face.confidence * 100)}%` : '—'} />
                <InfoRow k="Saved" v={fmtDate(face.created_at)} />
                {face.metrics && (
                  <>
                    <div className="vi-group-h" style={{ marginTop: 14 }}>Quality breakdown</div>
                    <div className="fg-metrics">
                      {[['det_score', 'Confidence'], ['sharpness', 'Sharpness'], ['frontal', 'Frontal pose'],
                        ['eyes', 'Eyes visible'], ['brightness', 'Brightness'], ['occlusion', 'Visibility'],
                        ['noise', 'Low noise']].map(([k, l]) => (
                          face.metrics[k] != null && <Meter key={k} label={l} v={face.metrics[k]} />))}
                      <div className="fg-mrow"><span>Face size</span><b>{face.metrics.face_size ?? '—'} px</b></div>
                      <div className="fg-mrow"><span>Resolution</span><b>{face.metrics.resolution ?? '—'} px²</b></div>
                      <div className="fg-mrow"><span>Frames inspected</span><b>{face.metrics.frames_seen ?? '—'}</b></div>
                      <div className="fg-mrow"><span>Faces ranked</span><b>{face.metrics.faces_seen ?? '—'}</b></div>
                    </div>
                  </>
                )}
                {face.person_crop_url && <img className="fg-person" src={face.person_crop_url} alt="person profile" title="Person profile image" />}
              </div>
            </div>
          ) : (
            <div className="fg-similar">
              {loading ? <div className="vi-msg">Searching all indexed faces…</div>
                : !sim || !sim.results?.length ? <div className="vi-msg">No similar person found in indexed footage.</div>
                  : (
                    <>
                      <div className="fg-sim-h">{sim.total} match{sim.total === 1 ? '' : 'es'} · sorted by similarity</div>
                      <div className="fg-sim-list">
                        {sim.results.map((r) => (
                          <div className="fg-sim" key={r.face_id}>
                            <div className="fg-sim-imgs">
                              {r.face_crop_url && <img src={r.face_crop_url} alt="" />}
                            </div>
                            <div className="fg-sim-info">
                              <div className="fg-sim-top">
                                <span className="cam">{r.camera_name || r.camera_id}</span>
                                <span className="sim">{Math.round(r.similarity * 100)}%</span>
                              </div>
                              <div className="fg-sim-ts mono">{fmtTs(r.timestamp)}</div>
                              <div className="fg-sim-actions">
                                {r.video_url && <button className="ws-btn-sm" onClick={() => setJump(r)}>⤿ Jump to Video</button>}
                                {r.detection_id != null && <button className="ws-btn-sm" onClick={() => setTrack(r)}>⤳ Track Person</button>}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </>)}
            </div>
          )}
        </div>
      </div>

      {jump && (
        <div className="vi-overlay" onMouseDown={() => setJump(null)} style={{ zIndex: 1100 }}>
          <div className="vi-modal" style={{ width: 'min(880px,96vw)' }} onMouseDown={(e) => e.stopPropagation()}>
            <div className="vi-head"><div className="vi-title">Jump to Video · {jump.camera_name || jump.camera_id}</div>
              <button className="vi-x" onClick={() => setJump(null)}>×</button></div>
            <div className="vi-body">
              <VideoPlayer key={jump.detection_id} src={jump.video_url} offset={jump.offset_seconds}
                           bbox={jump.bbox} frameW={jump.frame_width} frameH={jump.frame_height} autoPlay />
            </div>
          </div>
        </div>
      )}
      {track && <TrackingViewer detection={{ detection_id: track.detection_id, class_label: 'person', camera_id: track.camera_id, attributes: {} }} onClose={() => setTrack(null)} />}
    </div>
  )
}

function InfoRow({ k, v }) {
  return <div className="vi-row"><div className="vi-k">{k}</div><div className="vi-v">{v}</div></div>
}

function Meter({ label, v }) {
  const pct = Math.max(0, Math.min(100, Math.round((v || 0) * 100)))
  const col = pct >= 70 ? 'var(--fp-success)' : pct >= 40 ? 'var(--fp-warn)' : '#ff8a94'
  return (
    <div className="fg-mrow">
      <span>{label}</span>
      <span className="fg-bar"><i style={{ width: pct + '%', background: col }} /></span>
      <b>{pct}</b>
    </div>
  )
}
