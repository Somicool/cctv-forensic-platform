// Settings - live system configuration, models, cameras, storage and
// responsible-use controls. Values come from the backend (/system/info) so the
// page never hardcodes anything; toggles change allowlisted runtime flags.
import { useEffect, useState } from 'react'
import {
  getSystemInfo, updateSettings, getCameras, addCamera,
  recomputePlates, recomputeColors, backfillRegistry, clearActivity,
} from '../api'
import { IcSettings } from '../components/icons'

function Row({ k, v, mono }) {
  return (
    <div className="st-row">
      <span className="st-k">{k}</span>
      <span className={'st-v' + (mono ? ' mono' : '')}>{v ?? '—'}</span>
    </div>
  )
}

function Toggle({ label, hint, on, onChange, disabled }) {
  return (
    <div className="st-tog">
      <div className="st-tog-txt">
        <div className="lb">{label}</div>
        {hint && <div className="hint">{hint}</div>}
      </div>
      <button className={'st-switch ' + (on ? 'on' : '')} disabled={disabled}
              onClick={() => onChange(!on)} aria-pressed={on} aria-label={label}>
        <i />
      </button>
    </div>
  )
}

export default function Settings() {
  const [info, setInfo] = useState(null)
  const [cams, setCams] = useState([])
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState(null)
  const [err, setErr] = useState(null)
  const [nc, setNc] = useState({ cameraId: '', name: '', location: '', lat: '', lon: '' })

  async function load() {
    const [i, c] = await Promise.allSettled([getSystemInfo(), getCameras()])
    setInfo(i.status === 'fulfilled' ? i.value : null)
    setCams(c.status === 'fulfilled' ? (c.value || []) : [])
  }
  useEffect(() => { load() }, [])

  function flash(m) { setMsg(m); setErr(null); setTimeout(() => setMsg(null), 3500) }
  function fail(e) { setErr(e?.response?.data?.detail || e?.message || 'Action failed'); setMsg(null) }

  async function setFlag(key, val) {
    setBusy(true)
    try { await updateSettings({ [key]: val }); await load(); flash(`${key} → ${val}`) }
    catch (e) { fail(e) } finally { setBusy(false) }
  }
  async function setMode(m) {
    setBusy(true)
    try { await updateSettings({ processing_mode: m }); await load(); flash(`Processing mode → ${m}`) }
    catch (e) { fail(e) } finally { setBusy(false) }
  }
  async function run(fn, label) {
    if (!window.confirm(`${label}?`)) return
    setBusy(true)
    try { const r = await fn(); flash(`${label}: ${JSON.stringify(r)}`); await load() }
    catch (e) { fail(e) } finally { setBusy(false) }
  }
  async function saveCamera() {
    if (!nc.cameraId.trim()) { setErr('Camera ID is required.'); return }
    setBusy(true)
    try {
      await addCamera(nc)
      setNc({ cameraId: '', name: '', location: '', lat: '', lon: '' })
      await load(); flash('Camera saved')
    } catch (e) { fail(e) } finally { setBusy(false) }
  }

  if (!info) {
    return (
      <div className="fp-page">
        <div className="fp-page-head"><div>
          <h1 className="fp-page-title">Settings</h1>
          <p className="fp-page-desc">Processing mode, cameras, models, and responsible-use controls.</p>
        </div></div>
        <div className="cf-empty">Loading system information…</div>
      </div>
    )
  }

  const p = info.processing || {}, f = info.flags || {}
  const c = info.counts || {}, s = info.storage || {}, th = info.thresholds || {}
  const preset = p.preset || {}

  return (
    <div className="fp-page">
      <div className="fp-page-head">
        <div>
          <h1 className="fp-page-title">Settings</h1>
          <p className="fp-page-desc">Processing mode, cameras, models, and responsible-use controls.</p>
        </div>
        <button className="fp-btn" onClick={load} disabled={busy}>↻ Refresh</button>
      </div>

      {msg && <div className="cf-alert ok">{msg}</div>}
      {err && <div className="cf-alert err">{err}</div>}

      <div className="fp-grid cols-3">
        {/* ---- Processing Mode ---- */}
        <section className="fp-panel">
          <div className="fp-panel-title"><span>Processing Mode</span><span className="muted">{p.mode}</span></div>
          <div className="st-modes">
            {(p.modes || []).map((x) => (
              <button key={x} className={'st-mode ' + (p.mode === x ? 'on' : '')}
                      onClick={() => setMode(x)} disabled={busy}>{x}</button>
            ))}
          </div>
          <div className="st-hint">
            <b>Fast</b> indexes quickly (skips face + plate stages). <b>Accurate</b> runs the
            full forensic pipeline. Applies to new ingests.
          </div>
          <Row k="Sampling FPS" v={preset.fps ?? `${preset.fps_min ?? '?'}–${preset.fps_max ?? '?'} (adaptive)`} />
          <Row k="Detect resolution" v={preset.imgsz} />
          <Row k="Detect confidence" v={p.detect_conf} />
          <Row k="Faces / Plates" v={`${preset.do_faces ? 'on' : 'off'} / ${preset.do_plates ? 'on' : 'off'}`} />
          <Row k="Progressive chunk" v={`${p.progressive_chunk_frames} frames`} />
        </section>

        {/* ---- Hardware ---- */}
        <section className="fp-panel">
          <div className="fp-panel-title"><span>Hardware</span></div>
          <Row k="Device" v={(info.device || '').toUpperCase()} />
          <Row k="GPU" v={info.gpu} />
          <Row k="VRAM" v={info.gpu_vram_gb ? `${info.gpu_vram_gb} GB` : '—'} />
          <Row k="Low-VRAM mode" v={info.low_vram ? 'enabled' : 'disabled'} />
          <Row k="PyTorch / CUDA" v={info.torch ? `${info.torch} / ${info.cuda}` : '—'} mono />
          <div className="fp-panel-title" style={{ marginTop: 14 }}><span>Search index</span></div>
          {Object.entries(info.faiss || {}).map(([k, v]) =>
            <Row key={k} k={`FAISS ${k}`} v={Number(v).toLocaleString() + ' vectors'} />)}
        </section>

        {/* ---- Face Recognition / responsible use ---- */}
        <section className="fp-panel">
          <div className="fp-panel-title"><span>Face Recognition</span></div>
          <Toggle label="Face recognition" hint="Ethics-gated. Disables all face detection & search."
                  on={f.FACE_RECOGNITION_ENABLED} disabled={busy}
                  onChange={(v) => setFlag('face_recognition_enabled', v)} />
          <Toggle label="Face diagnostics log" hint="Logs why a frame was chosen for each saved face."
                  on={f.FACE_DIAG_LOG} disabled={busy}
                  onChange={(v) => setFlag('face_diag_log', v)} />
          <Row k="Accept quality ≥" v={th.face_accept_quality} />
          <Row k="Similar-face min" v={th.face_similar_min} />
          <Row k="Re-ID similarity" v={th.reid_sim_threshold} />
          <Row k="Indexed faces" v={Number(c.faces || 0).toLocaleString()} />
          <Row k="Saved faces" v={c.saved_faces} />
          <div className="st-hint warn">
            Face recognition is a bonus capability intended for authorised investigations only.
          </div>
        </section>

        {/* ---- Plate recognition / ANPR ---- */}
        <section className="fp-panel">
          <div className="fp-panel-title"><span>Plate Recognition (ANPR)</span></div>
          <Toggle label="Plate recognition" on={f.PLATE_RECOGNITION_ENABLED} disabled={busy}
                  onChange={(v) => setFlag('plate_recognition_enabled', v)} />
          <Toggle label="High-accuracy ANPR" hint="Plate-region detection, enhancement, multi-frame voting."
                  on={f.ANPR_ENABLED} disabled={busy} onChange={(v) => setFlag('anpr_enabled', v)} />
          <Toggle label="Adaptive sampling" hint="Dense re-sampling for two-wheelers & autos."
                  on={f.ANPR_ADAPTIVE_ENABLED} disabled={busy}
                  onChange={(v) => setFlag('anpr_adaptive_enabled', v)} />
          <Toggle label="Gemini vision fallback" hint={info.gemini_key_present
            ? 'Used only for low-confidence plates.' : 'No API key set — inactive.'}
                  on={f.GEMINI_ENABLED} disabled={busy || !info.gemini_key_present}
                  onChange={(v) => setFlag('gemini_enabled', v)} />
          <Row k="Fuzzy match ≥" v={th.plate_fuzzy_threshold} />
          <Row k="Single-read conf ≥" v={th.plate_single_conf} />
          <Row k="Plates stored" v={c.plates} />
        </section>

        {/* Cameras & GPS panel removed - camera siting and coordinates are managed
            in the Camera Registry, which is the single place that owns them. */}

        {/* ---- Data & Storage ---- */}
        <section className="fp-panel">
          <div className="fp-panel-title"><span>Data &amp; Storage</span><span className="muted">{s.db_mb} MB DB</span></div>
          <Row k="Videos" v={`${c.videos} · ${s.videos?.mb} MB`} />
          <Row k="Frames" v={`${s.frames?.files} files · ${s.frames?.mb} MB`} />
          <Row k="Crops" v={`${s.crops?.files} files · ${s.crops?.mb} MB`} />
          <Row k="Saved faces" v={`${s.saved_faces?.files} files · ${s.saved_faces?.mb} MB`} />
          <Row k="Exports" v={`${c.exports} · ${s.exports?.mb} MB`} />
          <Row k="Detections" v={Number(c.detections || 0).toLocaleString()} />
          <Row k="Vehicle registry" v={c.vehicle_registry} />
          <Row k="Activity history" v={c.activity} />
        </section>

        {/* ---- Maintenance ---- */}
        <section className="fp-panel">
          <div className="fp-panel-title"><span>Maintenance</span></div>
          <div className="st-actions">
            <button className="fp-btn" disabled={busy}
                    onClick={() => run(() => recomputePlates(), 'Re-run plate OCR on all footage')}>
              Recompute plates</button>
            <button className="fp-btn" disabled={busy}
                    onClick={() => run(() => recomputeColors(), 'Recompute clothing colours')}>
              Recompute colours</button>
            <button className="fp-btn" disabled={busy}
                    onClick={() => run(() => backfillRegistry(), 'Backfill vehicle registry from known plates')}>
              Backfill registry</button>
            <button className="fp-btn" disabled={busy} style={{ borderColor: 'var(--fp-danger)', color: '#ffb3bb' }}
                    onClick={() => run(() => clearActivity(), 'Clear dashboard activity history')}>
              Clear activity history</button>
          </div>
          <div className="st-hint">
            Recompute jobs are GPU/CPU heavy and run one at a time in the background;
            they never delete detections or search indexes.
          </div>
        </section>

      </div>
    </div>
  )
}
