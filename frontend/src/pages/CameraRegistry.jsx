// Camera Registry - permanent siting record for every CCTV camera.
// Add / edit / delete / view / import / export. Coordinates entered here are
// stored permanently, so a camera is only located ONCE and reused by every
// future upload. Required before real journey reconstruction can run.
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  listCameraRegistry, getRegistryStatus, getRegistryCamera, saveRegistryCamera,
  deleteRegistryCamera, exportCameraRegistry, importCameraRegistry,
} from '../api'
import { IcSearch } from '../components/icons'

const FACINGS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
const BLANK = {
  camera_id: '', name: '', lat: '', lon: '', address: '', road_name: '',
  facing: '', fov_deg: '', coverage_m: '', description: '', active: true,
}

export default function CameraRegistry() {
  const [cams, setCams] = useState(null)
  const [status, setStatus] = useState(null)
  const [q, setQ] = useState('')
  const [form, setForm] = useState(null)          // null = closed; object = add/edit
  const [view, setView] = useState(null)          // camera detail
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState(null)
  const [err, setErr] = useState(null)
  const fileRef = useRef(null)

  async function load() {
    const [c, s] = await Promise.allSettled([listCameraRegistry(), getRegistryStatus()])
    setCams(c.status === 'fulfilled' ? (c.value || []) : [])
    setStatus(s.status === 'fulfilled' ? s.value : null)
  }
  useEffect(() => { load() }, [])

  const flash = (m) => { setMsg(m); setErr(null); setTimeout(() => setMsg(null), 3200) }
  const fail = (e) => { setErr(e?.response?.data?.detail || e?.message || 'Action failed'); setMsg(null) }

  const filtered = useMemo(() => {
    const list = cams || []
    const s = q.trim().toLowerCase()
    if (!s) return list
    return list.filter((c) => [c.camera_id, c.name, c.address, c.road_name]
      .some((x) => (x || '').toLowerCase().includes(s)))
  }, [cams, q])

  async function save() {
    if (!form.camera_id.trim()) { setErr('Camera ID is required.'); return }
    setBusy(true)
    try { await saveRegistryCamera(form); setForm(null); await load(); flash('Camera saved permanently') }
    catch (e) { fail(e) } finally { setBusy(false) }
  }
  async function remove(c) {
    const linked = c.video_count > 0
    const ok = window.confirm(linked
      ? `${c.camera_id} has ${c.video_count} linked video(s). Delete the camera anyway? (videos & detections are NOT deleted)`
      : `Delete camera ${c.camera_id}?`)
    if (!ok) return
    setBusy(true)
    try { await deleteRegistryCamera(c.camera_id, linked); await load(); flash('Camera deleted') }
    catch (e) { fail(e) } finally { setBusy(false) }
  }
  async function doExport() {
    try {
      const data = await exportCameraRegistry()
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob); a.download = 'camera_registry.json'; a.click()
      flash(`Exported ${data.length} cameras`)
    } catch (e) { fail(e) }
  }
  async function doImport(file) {
    if (!file) return
    setBusy(true)
    try {
      const items = JSON.parse(await file.text())
      const r = await importCameraRegistry(Array.isArray(items) ? items : items.cameras || [])
      await load(); flash(`Imported ${r.imported}/${r.total} cameras`)
    } catch (e) { fail(e) } finally { setBusy(false) }
  }
  async function openView(id) {
    try { setView(await getRegistryCamera(id)) } catch (e) { fail(e) }
  }

  return (
    <div className="fp-page">
      <input ref={fileRef} type="file" accept="application/json" hidden
             onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ''; doImport(f) }} />

      <div className="fp-page-head">
        <div>
          <h1 className="fp-page-title">Camera Registry</h1>
          <p className="fp-page-desc">Permanent camera locations &amp; siting details — required for journey reconstruction.</p>
        </div>
        <div className="jn-head-actions">
          <button className="fp-btn" onClick={() => fileRef.current?.click()} disabled={busy}>Import</button>
          <button className="fp-btn" onClick={doExport} disabled={busy}>Export</button>
          <button className="fp-btn primary" onClick={() => setForm({ ...BLANK })}>＋ Add Camera</button>
        </div>
      </div>

      {msg && <div className="cf-alert ok">{msg}</div>}
      {err && <div className="cf-alert err">{err}</div>}
      {status && !status.ready_for_journey && (
        <div className="cf-alert err">{status.notice} — at least 2 cameras need valid latitude/longitude.</div>
      )}

      <div className="fp-stats">
        <Stat v={status?.cameras ?? '—'} l="Cameras" />
        <Stat v={status?.with_gps ?? '—'} l="With Location" />
        <Stat v={status?.without_gps?.length ?? '—'} l="Missing Location" />
        <Stat v={status?.route_engine?.active ?? '—'} l="Route Engine" />
      </div>

      <div className="fp-quicksearch">
        <IcSearch size={20} />
        <input placeholder="Search cameras — ID, name, address, road…" value={q} onChange={(e) => setQ(e.target.value)} />
      </div>

      {cams === null ? <div className="cf-empty">Loading registry…</div>
        : filtered.length === 0 ? <div className="cf-empty">No cameras{q ? ' match this search' : ' registered yet'}.</div>
          : (
            <div className="cr-grid">
              {filtered.map((c) => (
                <div key={c.camera_id} className={'cr-card ' + (c.has_gps ? '' : 'nogps') + (c.active ? '' : ' off')}>
                  <div className="cr-top">
                    <span className="cr-id">{c.camera_id}</span>
                    {c.has_gps ? <span className="st-gps">GPS</span>
                      : <span className="cr-warn">NO LOCATION</span>}
                  </div>
                  <div className="cr-name">{c.name || '—'}</div>
                  <div className="cr-meta">{[c.road_name, c.address].filter(Boolean).join(' · ') || 'no address recorded'}</div>
                  <div className="cr-facts">
                    <span>{c.has_gps ? `${Number(c.lat).toFixed(5)}, ${Number(c.lon).toFixed(5)}` : 'lat/lon missing'}</span>
                    <span>{c.facing ? `facing ${c.facing} (${Math.round(c.facing_deg)}°)` : 'facing —'}</span>
                    <span>{c.fov_deg ? `FOV ${Math.round(c.fov_deg)}°` : 'FOV —'}</span>
                    <span>{c.coverage_m ? `${Math.round(c.coverage_m)} m range` : 'range —'}</span>
                  </div>
                  <div className="cr-links">
                    {c.video_count} video(s) · {Number(c.detection_count || 0).toLocaleString()} detections
                    {!c.active && <span className="cr-off">inactive</span>}
                  </div>
                  <div className="cr-acts">
                    <button className="ws-btn-sm" onClick={() => openView(c.camera_id)}>View</button>
                    <button className="ws-btn-sm" onClick={() => setForm({
                      ...BLANK, ...c, facing: c.facing || '',
                      lat: c.lat ?? '', lon: c.lon ?? '',
                      fov_deg: c.fov_deg ?? '', coverage_m: c.coverage_m ?? '',
                    })}>Edit</button>
                    <button className="ws-btn-sm" style={{ borderColor: 'var(--fp-danger)', color: '#ffb3bb' }}
                            onClick={() => remove(c)}>Delete</button>
                  </div>
                </div>
              ))}
            </div>
          )}

      {/* add / edit */}
      {form && (
        <div className="ws-overlay" onMouseDown={() => setForm(null)}>
          <div className="ws-modal" style={{ maxWidth: 620 }} onMouseDown={(e) => e.stopPropagation()}>
            <button className="ws-modal-x" onClick={() => setForm(null)}>×</button>
            <h3>{cams?.some((c) => c.camera_id === form.camera_id) ? 'Edit camera' : 'Add camera'}</h3>
            <div className="st-2col">
              <F l="Camera ID *" v={form.camera_id} on={(v) => setForm({ ...form, camera_id: v })} ph="CAM-07" />
              <F l="Camera name" v={form.name} on={(v) => setForm({ ...form, name: v })} ph="Station Road North" />
              <F l="Latitude" v={form.lat} on={(v) => setForm({ ...form, lat: v })} ph="21.1959" />
              <F l="Longitude" v={form.lon} on={(v) => setForm({ ...form, lon: v })} ph="72.8302" />
            </div>
            <F l="Address" v={form.address} on={(v) => setForm({ ...form, address: v })} ph="Nr. Delhi Gate, Ring Road" />
            <div className="st-2col">
              <F l="Road name" v={form.road_name} on={(v) => setForm({ ...form, road_name: v })} ph="Ring Road" />
              <div className="ws-fld">
                <label>Facing direction</label>
                <select className="eg-select" value={form.facing}
                        onChange={(e) => setForm({ ...form, facing: e.target.value })}>
                  <option value="">—</option>
                  {FACINGS.map((f) => <option key={f} value={f}>{f}</option>)}
                </select>
              </div>
              <F l="Field of view (°)" v={form.fov_deg} on={(v) => setForm({ ...form, fov_deg: v })} ph="90" />
              <F l="Coverage distance (m)" v={form.coverage_m} on={(v) => setForm({ ...form, coverage_m: v })} ph="45" />
            </div>
            <div className="ws-fld"><label>Description</label>
              <textarea rows={2} value={form.description || ''}
                        onChange={(e) => setForm({ ...form, description: e.target.value })}
                        placeholder="Mounted on the north pole, covers the junction approach…" /></div>
            <label className="jn-cam-pick" style={{ borderBottom: 'none' }}>
              <input type="checkbox" checked={!!form.active}
                     onChange={(e) => setForm({ ...form, active: e.target.checked })} />
              <span className="id">Active</span>
            </label>
            <div className="ws-modal-actions" style={{ marginTop: 12 }}>
              <button className="fp-btn" onClick={() => setForm(null)}>Cancel</button>
              <button className="fp-btn primary" onClick={save} disabled={busy}>Save permanently</button>
            </div>
          </div>
        </div>
      )}

      {/* view */}
      {view && (
        <div className="ws-overlay" onMouseDown={() => setView(null)}>
          <div className="ws-modal" style={{ maxWidth: 560 }} onMouseDown={(e) => e.stopPropagation()}>
            <button className="ws-modal-x" onClick={() => setView(null)}>×</button>
            <h3>{view.camera_id}</h3>
            {[['Name', view.name], ['Address', view.address], ['Road', view.road_name],
              ['Latitude', view.lat], ['Longitude', view.lon],
              ['Facing', view.facing ? `${view.facing} (${Math.round(view.facing_deg)}°)` : null],
              ['Field of view', view.fov_deg ? `${Math.round(view.fov_deg)}°` : null],
              ['Coverage', view.coverage_m ? `${Math.round(view.coverage_m)} m` : null],
              ['Active', view.active ? 'yes' : 'no'], ['Description', view.description]].map(([k, v]) => (
                <div className="st-row" key={k}><span className="st-k">{k}</span><span className="st-v">{v ?? '—'}</span></div>))}
            <div className="fp-panel-title" style={{ marginTop: 14 }}>
              <span>Linked videos</span><span className="muted">{(view.videos || []).length}</span></div>
            {(view.videos || []).length === 0 ? <div className="cf-empty small">No videos linked yet.</div>
              : <div className="ws-list" style={{ maxHeight: 180 }}>
                  {view.videos.map((v) => (
                    <div key={v.video_id} className="ws-cam">
                      <span className="id">#{v.video_id}</span>
                      <span className="nm">{v.filename}</span>
                      <span className="muted" style={{ fontSize: 10 }}>{v.status}</span>
                    </div>))}
                </div>}
          </div>
        </div>
      )}
    </div>
  )
}

function Stat({ v, l }) {
  return <div className="fp-card fp-stat"><div className="v">{v}</div><div className="l">{l}</div></div>
}
function F({ l, v, on, ph }) {
  return (
    <div className="ws-fld"><label>{l}</label>
      <input value={v ?? ''} onChange={(e) => on(e.target.value)} placeholder={ph} /></div>
  )
}
