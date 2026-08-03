// Case File - assemble the case, review its evidence, and export a sealed
// forensic package (SHA-256 chain of custody). Works on the SAME evidence set as
// the Workspace / Evidence Gallery via the investigation context, and lists
// previously created exports from the backend.
import { useEffect, useMemo, useState } from 'react'
import { useInvestigation } from '../context/investigation'
import { createExport, getExports, listSavedFaces } from '../api'
import { IcCase, IcClock } from '../components/icons'

const fmtTs = (t) => (t ? t.replace('T', ' ').slice(0, 19) : '—')
const fmtDate = (t) => { try { return new Date(t).toLocaleString() } catch { return t || '—' } }
const VEHICLES = new Set(['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'auto-rickshaw',
  'scooter', 'tempo', 'mini-truck', 'pickup', 'tractor', 'hcv', 'lcv'])

function attrText(a) {
  if (!a) return ''
  const p = []
  if (a.color) p.push(a.color)
  if (a.upper_color) p.push('top: ' + a.upper_color)
  if (a.lower_color) p.push('btm: ' + a.lower_color)
  if (a.vehicle_type) p.push(a.vehicle_type)
  if (Array.isArray(a.accessories) && a.accessories.length) p.push(a.accessories.join(', '))
  return p.join(' · ')
}

export default function CaseFile() {
  const { evidence, removeEvidence, caseInfo, setCaseInfo } = useInvestigation()
  const [exports, setExports] = useState([])
  const [faces, setFaces] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  async function load() {
    const [e, f] = await Promise.allSettled([getExports(), listSavedFaces()])
    setExports(e.status === 'fulfilled' ? (e.value || []) : [])
    setFaces(f.status === 'fulfilled' ? (f.value || []) : [])
  }
  useEffect(() => { load() }, [])

  const stats = useMemo(() => {
    const persons = evidence.filter((e) => e.class_label === 'person').length
    const vehicles = evidence.filter((e) => VEHICLES.has((e.class_label || '').toLowerCase())).length
    const plates = evidence.filter((e) => e.attributes?.plate_text).length
    const cams = new Set(evidence.map((e) => e.camera_id).filter(Boolean)).size
    return { persons, vehicles, plates, cams }
  }, [evidence])

  const ready = evidence.length > 0 && caseInfo.caseNumber.trim() && caseInfo.officer.trim()

  async function doExport() {
    setError(null); setResult(null)
    if (!evidence.length) { setError('Add evidence to the case first (use ＋ on search results).'); return }
    if (!caseInfo.caseNumber.trim() || !caseInfo.officer.trim()) {
      setError('Case number and lead officer are required for a sealed export.'); return
    }
    setBusy(true)
    try {
      const r = await createExport({
        detectionIds: evidence.map((e) => e.detection_id),
        caseNumber: caseInfo.caseNumber, officer: caseInfo.officer, notes: caseInfo.notes,
      })
      setResult(r); load()
    } catch (e) { setError(e?.response?.data?.detail || e.message || 'Export failed') }
    finally { setBusy(false) }
  }

  const set = (k) => (e) => setCaseInfo({ ...caseInfo, [k]: e.target.value })

  return (
    <div className="fp-page">
      <div className="fp-page-head">
        <div>
          <h1 className="fp-page-title">Case File</h1>
          <p className="fp-page-desc">Assemble evidence, add case metadata, and export a sealed forensic package.</p>
        </div>
        <button className="fp-btn primary" onClick={doExport} disabled={busy || !ready}>
          {busy ? 'Sealing…' : `Export Evidence (${evidence.length})`}
        </button>
      </div>

      {/* case summary */}
      <div className="fp-stats">
        <div className="fp-card fp-stat"><div className="v">{evidence.length}</div><div className="l">Evidence Items</div></div>
        <div className="fp-card fp-stat"><div className="v">{stats.persons}</div><div className="l">Persons</div></div>
        <div className="fp-card fp-stat"><div className="v">{stats.vehicles}</div><div className="l">Vehicles</div></div>
        <div className="fp-card fp-stat"><div className="v">{stats.cams}</div><div className="l">Cameras Involved</div></div>
      </div>

      {error && <div className="cf-alert err">{error}</div>}
      {result && (
        <div className="cf-alert ok">
          <b>Export sealed.</b> ID <code>{result.export_id}</code> · {result.file_count} files ·
          SHA-256 <code className="cf-hash">{result.manifest_hash}</code>
          {result.download_url && <a className="ws-btn-sm" style={{ marginLeft: 10 }} href={result.download_url} download>Download ZIP</a>}
        </div>
      )}

      <div className="fp-split">
        {/* evidence in this case */}
        <section className="fp-panel">
          <div className="fp-panel-title">
            <span>Evidence in this case</span>
            <span className="muted">{evidence.length} item{evidence.length === 1 ? '' : 's'}</span>
          </div>
          {evidence.length === 0 ? (
            <div className="cf-empty">
              No evidence yet. Run a search in the <b>Investigation Workspace</b> and add matches
              with the <b>＋</b> button — they appear here ready for a sealed export.
            </div>
          ) : (
            <div className="cf-list">
              {evidence.map((e, i) => (
                <div className="cf-row" key={e.detection_id}>
                  <span className="cf-idx">{i + 1}</span>
                  {e.crop_url ? <img className="cf-thumb" src={e.crop_url} alt={e.class_label} loading="lazy" />
                    : <div className="cf-thumb ph">{e.class_label}</div>}
                  <div className="cf-info">
                    <div className="cf-top">
                      <span className="lb">{e.class_label}</span>
                      <span className="cam">{e.camera_id}</span>
                    </div>
                    <div className="cf-sub">{attrText(e.attributes) || '—'}</div>
                    <div className="cf-ts">{fmtTs(e.timestamp)}
                      {e.attributes?.plate_text && <span className="cf-plate">{e.attributes.plate_text}</span>}
                    </div>
                  </div>
                  <button className="cf-x" title="Remove from case" onClick={() => removeEvidence(e.detection_id)}>×</button>
                </div>
              ))}
            </div>
          )}

          {faces.length > 0 && (
            <>
              <div className="fp-panel-title" style={{ marginTop: 18 }}>
                <span>Saved faces</span><span className="muted">{faces.length}</span>
              </div>
              <div className="cf-faces">
                {faces.slice(0, 12).map((f) => (
                  <img key={f.saved_id} src={f.preview_crop_url || f.face_crop_url}
                       title={`${f.investigation || 'Unassigned'} · ${f.camera_id || ''}`} alt="face" />
                ))}
              </div>
            </>
          )}
        </section>

        {/* case details + exports */}
        <aside>
          <section className="fp-panel">
            <div className="fp-panel-title"><span><IcCase size={16} /> Case details</span></div>
            <div className="ws-fld"><label>Case title</label>
              <input value={caseInfo.title} onChange={set('title')} placeholder="e.g. Diamond Market theft" /></div>
            <div className="ws-fld"><label>Case number <span className="cf-req">*</span></label>
              <input value={caseInfo.caseNumber} onChange={set('caseNumber')} placeholder="CASE-2026-001" /></div>
            <div className="ws-fld"><label>Lead officer <span className="cf-req">*</span></label>
              <input value={caseInfo.officer} onChange={set('officer')} placeholder="Insp. Name" /></div>
            <div className="ws-fld"><label>Notes</label>
              <textarea rows={4} value={caseInfo.notes} onChange={set('notes')}
                        placeholder="Observations, context, chain-of-custody remarks…" /></div>
            <div className="cf-note">
              Exports include every item's image, a <b>manifest.json</b> with per-file SHA-256
              hashes, a sealed <b>manifest.sha256</b>, and a PDF summary — packaged as a ZIP.
            </div>
          </section>

          <section className="fp-panel" style={{ marginTop: 16 }}>
            <div className="fp-panel-title"><span><IcClock size={16} /> Previous exports</span>
              <span className="muted">{exports.length}</span></div>
            {exports.length === 0 ? <div className="cf-empty small">No exports yet.</div>
              : (
                <div className="cf-exports">
                  {exports.map((x) => (
                    <div className="cf-exp" key={x.export_id}>
                      <div className="cf-exp-top">
                        <code>{x.export_id}</code>
                        <a className="ws-btn-sm" href={x.download_url} download>ZIP</a>
                      </div>
                      <div className="cf-exp-meta">{x.case_number} · {x.officer}</div>
                      <div className="cf-exp-meta mono">{fmtDate(x.created_at)}</div>
                      <div className="cf-exp-hash" title={x.manifest_hash}>
                        sha256 {(x.manifest_hash || '').slice(0, 24)}…</div>
                    </div>
                  ))}
                </div>)}
          </section>
        </aside>
      </div>
    </div>
  )
}
