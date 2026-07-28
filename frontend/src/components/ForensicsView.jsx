// Forensic export view: build a case file of detections, export a hashed,
// zipped evidence package (manifest + crops + PDF), and list past exports.
import { useEffect, useState } from 'react'
import { createExport, getExports } from '../api'

export default function ForensicsView({ items, onRemove, onClear }) {
  const [caseNumber, setCaseNumber] = useState('')
  const [officer, setOfficer] = useState('')
  const [notes, setNotes] = useState('')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [past, setPast] = useState([])

  async function refresh() {
    try { setPast(await getExports()) } catch { setPast([]) }
  }
  useEffect(() => { refresh() }, [])

  async function doExport() {
    setError(null)
    if (!items.length) { setError('Add detections to the case file first.'); return }
    if (!caseNumber.trim() || !officer.trim()) { setError('Case number and officer are required.'); return }
    setBusy(true)
    try {
      const r = await createExport({
        detectionIds: items.map((i) => i.detection_id), caseNumber, officer, notes,
      })
      setResult(r)
      refresh()
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || 'Export failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="forensics">
      <div className="forensics-grid">
        <div className="case-panel">
          <h3>Case file <span className="muted small">({items.length} item{items.length === 1 ? '' : 's'})</span></h3>
          <div className="case-items">
            {items.length === 0 && (
              <div className="muted small">
                Open a search result and choose “Add to case file” to build an evidence set.
              </div>
            )}
            {items.map((it) => (
              <div className="case-item" key={it.detection_id}>
                <div className="ti-thumb">
                  {it.crop_url ? <img src={it.crop_url} alt="" /> : <div className="thumb-empty">—</div>}
                </div>
                <div className="ti-body">
                  <div className="ti-top"><span className="label">{it.class_label}</span><span className="cam">{it.camera_id}</span></div>
                  <div className="ti-time mono">#{it.detection_id}</div>
                </div>
                <button className="chip" onClick={() => onRemove(it.detection_id)}>remove</button>
              </div>
            ))}
          </div>
          {items.length > 0 && <button className="btn ghost small" onClick={onClear}>Clear case file</button>}
        </div>

        <div className="export-panel">
          <h3>Export evidence</h3>
          <label className="fld"><span>Case number</span>
            <input value={caseNumber} onChange={(e) => setCaseNumber(e.target.value)} placeholder="CASE-2026-001" /></label>
          <label className="fld"><span>Officer</span>
            <input value={officer} onChange={(e) => setOfficer(e.target.value)} placeholder="Insp. Name" /></label>
          <label className="fld"><span>Notes</span>
            <textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={3} /></label>
          <button className="btn primary" onClick={doExport} disabled={busy}>
            {busy ? 'Exporting…' : `Export ${items.length} item(s)`}
          </button>
          {error && <div className="banner error">{error}</div>}
          {result && (
            <div className="export-result">
              <div className="kv"><span>Export ID</span><b>{result.export_id}</b></div>
              <div className="kv"><span>Files</span><b>{result.file_count}</b></div>
              <div className="kv"><span>Manifest SHA-256</span><b className="mono hash">{result.manifest_hash}</b></div>
              <a className="btn primary" href={result.download_url} download>Download evidence .zip</a>
            </div>
          )}
        </div>
      </div>

      <div className="past-exports">
        <div className="track-head">Previous exports</div>
        {past.length === 0 ? (
          <div className="muted small">No exports yet.</div>
        ) : (
          <table className="audit-table">
            <thead>
              <tr><th>Export</th><th>Case</th><th>Officer</th><th>Created</th><th>Items</th><th>SHA-256</th><th /></tr>
            </thead>
            <tbody>
              {past.map((e) => (
                <tr key={e.export_id}>
                  <td className="mono">{e.export_id}</td>
                  <td>{e.case_number}</td>
                  <td>{e.officer}</td>
                  <td className="mono">{e.created_at ? e.created_at.replace('T', ' ').slice(0, 19) : ''}</td>
                  <td>{Array.isArray(e.detection_ids) ? e.detection_ids.length : ''}</td>
                  <td className="mono hash">{e.manifest_hash ? e.manifest_hash.slice(0, 16) + '…' : ''}</td>
                  <td><a href={e.download_url} download>zip</a></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
