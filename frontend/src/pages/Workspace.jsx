// Investigation Workspace - the primary investigator page.
// Single page, workflow: Upload -> Process -> Search -> Review -> Export.
// Uses only existing backend APIs (no backend changes).
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  getLibrary, getVideos, getCameras, ingestAll, getIngestJob, stopIngest, uploadVideo,
  deleteVideo, searchText, searchPlate, trackDetection, createExport, logActivity,
} from '../api'
import VideoPlayer from '../components/VideoPlayer'
import TrackingViewer from '../components/TrackingViewer'
import VehicleInfo from '../components/VehicleInfo'
import { IcSearch, IcUpload, IcPlus, IcClock } from '../components/icons'
import { useInvestigation } from '../context/investigation'

const STEPS = ['Upload', 'Process', 'Search', 'Review', 'Export']
const MODES = [{ id: 'text', label: 'Describe' }, { id: 'plate', label: 'Plate' }]
const LANGS = ['EN', 'HI', 'GU']
const EXAMPLES = ['a person carrying a backpack', 'a man in a white shirt', 'a white truck', 'a red car']

// Build a dashboard activity entry for a search result (person or vehicle).
function activityFromResult(r, action) {
  return {
    kind: r.class_label === 'person' ? 'person' : 'vehicle',
    action, ref: String(r.detection_id), label: r.class_label,
    camera_id: r.camera_id, timestamp: r.timestamp, crop_url: r.crop_url,
    plate: r.attributes?.plate_text || null,
  }
}

export default function Workspace() {
  const { evidence, inEvidence, toggleEvidence, removeEvidence, setMatches, caseInfo, setCaseInfo } = useInvestigation()
  const [cameras, setCameras] = useState([])
  const [library, setLibrary] = useState(null)
  const [job, setJob] = useState(null)
  const [uploadPct, setUploadPct] = useState(null)
  const autoOpenRef = useRef(false)

  const [scope, setScope] = useState('all')
  const [scopeVideo, setScopeVideo] = useState(null)

  const [mode, setMode] = useState('text')
  const [query, setQuery] = useState('')
  const [language, setLanguage] = useState('EN')
  const [plate, setPlate] = useState('')
  const [results, setResults] = useState(null)
  const [meta, setMeta] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const [current, setCurrent] = useState(null)
  const [activeId, setActiveId] = useState(null)
  const [playTime, setPlayTime] = useState(0)
  const [track, setTrack] = useState(null)
  const [tracing, setTracing] = useState(false)

  const [showExport, setShowExport] = useState(false)
  const [trackView, setTrackView] = useState(null)     // detection being replayed in the tracking viewer
  const [regPlate, setRegPlate] = useState(null)       // plate whose Vehicle Registry is open
  const [footageOpen, setFootageOpen] = useState(true)

  const fileRef = useRef(null)
  const playerRef = useRef(null)
  const pickScrollRef = useRef(false)                // true when a user View/open should scroll to the player (not the card)
  const [pickSeq, setPickSeq] = useState(0)          // bumped on user View / open, to scroll the player into view
  const currentRef = useRef(null); currentRef.current = current
  const resultsRef = useRef(null); resultsRef.current = results
  const activeIdRef = useRef(null); activeIdRef.current = activeId
  const cardRefs = useRef(new Map())

  const nameFor = (id) => { const n = (cameras || []).find((c) => c.camera_id === id)?.name; return (!n || n === id) ? '' : n }
  const camLabel = (id) => { const n = nameFor(id); return n ? `${id} · ${n}` : id }

  async function loadLibrary() {
    try { return await getLibrary().then((items) => { setLibrary(items); return items }) }
    catch {
      const vids = await getVideos().catch(() => [])
      const items = vids.map((v) => ({ ...v, processed: true })); setLibrary(items); return items
    }
  }
  useEffect(() => { loadLibrary(); getCameras().then(setCameras).catch(() => setCameras([])) /* eslint-disable-next-line */ }, [])

  const processed = (library || []).filter((v) => v.processed)
  const unprocessed = (library || []).filter((v) => !v.processed)
  const processing = job && job.status === 'processing'
  const uploading = uploadPct != null

  // ---- processing / upload ----
  async function startProcessing() {
    setError(null)
    try {
      const r = await ingestAll()
      if (!r.job_id) { await loadLibrary(); return }
      setJob({ job_id: r.job_id, status: 'processing', done: 0, total: r.total, current: null })
    } catch (e) { setError('Could not start processing. ' + (e?.message || '')) }
  }
  async function stopProcessing() { try { await stopIngest() } catch { /* polled */ } }
  async function handleDelete(v) {
    if (!window.confirm(`Permanently delete "${v.filename}"?\n\nThis removes the video file and all of its analysis from the system. This cannot be undone.`)) return
    setError(null)
    try {
      await deleteVideo(v.filename)
      if (scopeVideo && scopeVideo.video_id === v.video_id) { setScopeVideo(null); setScope('all'); setCurrent(null); setResults(null); setMeta(null); setActiveId(null) }
      await loadLibrary()
    } catch (e) { setError('Delete failed. ' + (e?.response?.data?.detail || e.message || '')) }
  }
  async function handleUpload(f) {
    if (!f) return
    setError(null); setUploadPct(0)
    try {
      const r = await uploadVideo({ file: f, onProgress: setUploadPct })
      setUploadPct(null)
      if (r.busy) { setError(r.message || 'Another job is running — wait for it to finish.'); return }
      if (r.job_id) { autoOpenRef.current = true; setJob({ job_id: r.job_id, status: 'processing', done: 0, total: 1, current: r.filename }) }
      else await loadLibrary()
    } catch (e) { setUploadPct(null); setError('Upload failed. ' + (e?.response?.data?.detail || e.message || '')) }
  }
  useEffect(() => {
    if (!processing) return
    const t = setInterval(async () => {
      try {
        const j = await getIngestJob(job.job_id); setJob(j)
        if (j.status !== 'processing') {
          clearInterval(t)
          const items = await loadLibrary()
          if (autoOpenRef.current && j.status === 'done') {
            autoOpenRef.current = false
            const newest = (items || []).filter((v) => v.processed).sort((a, b) => (b.video_id || 0) - (a.video_id || 0))[0]
            if (newest) openClip(newest)
          }
        }
      } catch { clearInterval(t) }
    }, 1500)
    return () => clearInterval(t)
    // eslint-disable-next-line
  }, [job?.job_id, job?.status])

  // ---- player selection ----
  function mediaFromVideo(v) {
    return { key: v.url, videoId: v.video_id, src: v.url, offset: 0, bbox: null, frameW: null, frameH: null,
             title: v.filename, item: null, sub: [camLabel(v.camera_id), fmtDur(v.duration)].filter(Boolean).join('  ·  ') }
  }
  function openClip(v) { pickScrollRef.current = true; setScope(v.video_id); setScopeVideo(v); setResults(null); setMeta(null); setTrack(null); setActiveId(null); setPlayTime(0); setCurrent(mediaFromVideo(v)); setPickSeq((n) => n + 1) }
  // Progressive: scope the search to the clip that's still being indexed. The
  // early portions are already searchable, so investigators don't have to wait.
  function openPartial() {
    const vp = job && job.video_progress
    if (!vp || !vp.video_id) return
    const fname = job.current || ''
    const v = { video_id: vp.video_id, filename: fname, camera_id: null, duration: null,
                url: fname ? `/media/videos/${fname}` : null }
    openClip(v)
  }
  function pickResult(r) {
    setCurrent({ key: r.video_url, videoId: r.video_id, src: r.video_url, offset: r.offset_seconds || 0,
      bbox: r.bbox, frameW: r.frame_width, frameH: r.frame_height, item: r,
      title: `${r.class_label}  ·  ${camLabel(r.camera_id)}`,
      sub: [fmtTs(r.timestamp), 'match ' + Math.round((r.score || 0) * 100) + '%', attrText(r.attributes)].filter(Boolean).join('  ·  ') })
    pickScrollRef.current = true
    setActiveId(r.detection_id); setPlayTime(r.offset_seconds || 0); setTrack(null); setPickSeq((n) => n + 1)
    logActivity(activityFromResult(r, 'found'))         // dashboard history: found
  }
  function trackObject(r) { logActivity(activityFromResult(r, 'tracked')); setTrackView(r) }
  // Keep the active card visible as playback moves through results - but NOT when
  // the user explicitly clicked View/opened a clip (that scrolls to the player).
  useEffect(() => { if (activeId == null || pickScrollRef.current) return; cardRefs.current.get(activeId)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }) }, [activeId])
  // Bring the player into view when the user clicks View / opens a clip, so the
  // selected result is visible instead of updating off-screen at the top.
  useEffect(() => { if (pickSeq === 0) return; playerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }); pickScrollRef.current = false }, [pickSeq])
  const onPlayTime = useCallback((t) => {
    setPlayTime(t)
    const cur = currentRef.current; if (!cur || cur.videoId == null) return
    let best = null
    for (const r of (resultsRef.current || [])) {
      if (r.video_id !== cur.videoId || r.offset_seconds == null) continue
      if (r.offset_seconds <= t + 0.3 && (!best || r.offset_seconds > best.offset_seconds)) best = r
    }
    if (best && best.detection_id !== activeIdRef.current) setActiveId(best.detection_id)
  }, [])

  // ---- search ----
  async function runSearch() {
    setError(null)
    if (mode === 'text' && !query.trim()) return
    if (mode === 'plate' && !plate.trim()) return
    setLoading(true); setTrack(null)
    const t0 = performance.now()
    // A plate is a global identifier - always search ALL footage so the vehicle
    // is found on any camera, regardless of which clip is currently open.
    const plateMode = mode === 'plate'
    const filters = (plateMode || scope === 'all') ? {} : { video_id: scope }
    try {
      let data
      if (mode === 'text') data = await searchText({ query, language: language.toLowerCase(), includeScenes: false, filters })
      else data = await searchPlate({ plate, filters })
      let res = data.results || []
      if (!plateMode && scope !== 'all') res = res.filter((r) => r.video_id === scope)
      setResults(res)
      setMatches(res)                                 // share with Evidence Gallery
      setMeta({ total: res.length, note: data.note, objectType: data.object_type, translated: data.translated_query, elapsed: Math.round(performance.now() - t0), mode })
      // dashboard history: record the search (person / vehicle / plate)
      const term = mode === 'plate' ? plate : query
      const skind = mode === 'plate' ? 'vehicle' : (data.object_type === 'person' ? 'person' : (data.object_type === 'vehicle' ? 'vehicle' : null))
      if ((term || '').trim()) logActivity({ kind: skind, action: 'searched', ref: `q:${mode}:${term}`.toLowerCase(), label: term, query: term, timestamp: new Date().toISOString() })
      if (res.length) pickResult(res[0])
      else { setActiveId(null); if (scopeVideo) setCurrent(mediaFromVideo(scopeVideo)) }
    } catch (e) { setError(e?.response?.data?.detail || e.message || 'Search failed'); setResults([]); setMeta(null) }
    finally { setLoading(false) }
  }
  async function traceCurrent() {
    if (!current?.item) return
    setTracing(true)
    try { setTrack(await trackDetection(current.item.detection_id)) } catch { setTrack({ appearances: [] }) } finally { setTracing(false) }
  }

  // ---- derived status + step ----
  const status = uploading ? ['Uploading', 'var(--fp-warn)']
    : processing ? ['Processing', 'var(--fp-warn)']
    : results !== null ? ['Reviewing', 'var(--fp-accent-2)']
    : processed.length ? ['Ready', 'var(--fp-success)']
    : ['Setup', 'var(--fp-muted)']
  let step = processed.length ? 3 : 1
  if (uploading || processing) step = 2
  if (results !== null) step = current?.item ? 4 : 3
  if (evidence.length) step = 5

  const onKey = (e) => { if (e.key === 'Enter') runSearch() }
  const cur = current
  const hasResults = Array.isArray(results) && results.length > 0
  const jobPct = job && job.total ? (vpActive(job.video_progress) ? job.video_progress.pct : Math.round(((job.done || 0) / job.total) * 100)) : 0
  const timeline = hasResults ? [...results].filter((r) => r.offset_seconds != null).sort((a, b) => a.offset_seconds - b.offset_seconds) : []

  return (
    <div className="fp-page">
      <input ref={fileRef} type="file" accept="video/*" hidden onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ''; handleUpload(f) }} />

      <div className="fp-page-head">
        <div className="ws-head-row">
          <div>
            <h1 className="fp-page-title">Investigation Workspace</h1>
            <p className="fp-page-desc">Upload footage, run AI search, review matches, and export evidence.</p>
          </div>
          <span className="ws-status"><span className="d" style={{ background: status[1], color: status[1] }} />{status[0]}</span>
        </div>
        <button className="fp-btn primary" disabled={!evidence.length} onClick={() => setShowExport(true)}>Export Evidence ({evidence.length})</button>
      </div>

      {/* workflow stepper */}
      <div className="ws-steps">
        {STEPS.map((label, i) => {
          const n = i + 1; const st = n < step ? 'done' : n === step ? 'active' : 'todo'
          return (
            <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div className={'ws-step ' + st}><span className="n">{n < step ? '✓' : n}</span><span className="t">{label}</span></div>
              {i < STEPS.length - 1 && <span className="ws-step-sep" />}
            </div>
          )
        })}
      </div>

      {error && <div className="ws-banner err">{error}</div>}

      {/* ===================== FOOTAGE (full-width, spacious) ===================== */}
      <section className="fp-card ws-foot">
        <div className="ws-foot-head">
          <div className="ws-foot-title"><IcUpload size={18} /> Footage <span className="muted">{processed.length}/{(library || []).length} analysed</span></div>
          <div className="ws-foot-actions">
            <button className="fp-btn primary" onClick={() => fileRef.current?.click()} disabled={processing || uploading}><IcUpload /> Upload video</button>
            {!processing && unprocessed.length > 0 && <button className="fp-btn" onClick={startProcessing} disabled={uploading}>Analyse {unprocessed.length}</button>}
            {processing && <button className="fp-btn" onClick={stopProcessing} style={{ borderColor: 'var(--fp-danger)', color: '#ffb3bb' }}>■ Stop</button>}
            {(library || []).length > 0 && <button className="fp-btn ghost" onClick={() => setFootageOpen((o) => !o)}>{footageOpen ? 'Hide' : 'Show'}</button>}
          </div>
        </div>

        {uploading && <div className="ws-prog"><div className="ws-prog-bar" style={{ width: uploadPct + '%' }} /><span className="ws-prog-l">Uploading {uploadPct}%</span></div>}
        {processing && (
          <>
            <div className="ws-prog"><div className="ws-prog-bar" style={{ width: jobPct + '%' }} /><span className="ws-prog-l">Analysing {job.current || ''} — {vpActive(job.video_progress) ? `${jobPct}% · ${stageText(job.video_progress)}` : `${jobPct}% · ${job.done || 0}/${job.total}`}</span></div>
            {job.video_progress?.searchable && (
              <div className="ws-partial">
                <span className="ws-badge ok">● Search Ready (Partial)</span>
                <span className="ws-partial-t">{Number(job.video_progress.indexed || 0).toLocaleString()} detections indexed so far — you can search while the rest processes.</span>
                {scopeVideo?.video_id !== job.video_progress.video_id &&
                  <button className="fp-btn ghost sm" onClick={openPartial}>Search this clip now</button>}
              </div>
            )}
          </>
        )}

        {footageOpen && (
          library === null ? <div className="ws-foot-empty">Loading footage…</div>
            : library.length === 0 ? <div className="ws-foot-empty">No footage yet — click <b>Upload video</b> to add a clip from your device.</div>
              : (
                <div className="ws-foot-grid">
                  {library.map((v) => {
                    const analysed = v.processed
                    return (
                      <div key={v.filename}
                        className={'ws-fc ' + (analysed ? 'analysed ' : '') + (scopeVideo?.video_id === v.video_id ? 'active' : '')}
                        role={analysed ? 'button' : undefined} tabIndex={analysed ? 0 : undefined}
                        onClick={analysed ? () => openClip(v) : undefined}
                        onKeyDown={analysed ? (e) => { if (e.key === 'Enter') openClip(v) } : undefined}
                        title={analysed ? 'Investigate this clip' : 'Not analysed yet'}>
                        <div className="ws-fc-thumb">
                          {analysed ? <video src={v.url + '#t=1'} muted preload="metadata" /> : <div className="ph">▶</div>}
                          <span className={'ws-badge ws-fc-badge ' + (analysed ? 'ok' : 'no')}>{analysed ? 'Analysed' : 'Pending'}</span>
                          {analysed && <div className="ws-fc-open">Investigate ›</div>}
                          <button className="ws-fc-del" title="Delete video permanently" disabled={processing || uploading}
                            onClick={(e) => { e.stopPropagation(); handleDelete(v) }}>🗑</button>
                        </div>
                        <div className="ws-fc-body">
                          <div className="ws-fc-name" title={v.filename}>{v.filename}</div>
                          <div className="ws-fc-sub">{analysed ? (camLabel(v.camera_id) + (v.duration ? '  ·  ' + fmtDur(v.duration) : '')) : (v.size_mb != null ? v.size_mb + ' MB · not analysed' : 'not analysed')}</div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )
        )}
      </section>

      <div className="ws-grid">
        {/* ===================== MAIN ===================== */}
        <div className="ws-main">
          {/* scope */}
          <div className="fp-card ws-scope">
            <IcSearch size={16} />
            <div className="ws-scope-mid">Investigating: <b>{scopeVideo ? scopeVideo.filename : 'All footage'}</b></div>
            {scopeVideo && (
              <div className="ws-toggle">
                <button className={scope !== 'all' ? 'on' : ''} onClick={() => setScope(scopeVideo.video_id)}>This clip</button>
                <button className={scope === 'all' ? 'on' : ''} onClick={() => setScope('all')}>All footage</button>
              </div>
            )}
          </div>

          {/* player */}
          {cur && (
            <section className="fp-card ws-player" ref={playerRef}>
              <VideoPlayer key={cur.key} src={cur.src} offset={cur.offset} bbox={cur.bbox} frameW={cur.frameW} frameH={cur.frameH} onTimeUpdate={onPlayTime} />
              <div className="ws-now">
                <div style={{ minWidth: 0 }}>
                  <div className="ws-now-title">{cur.title}<span className="ws-time">{fmtDur(playTime)}{cur.item?.offset_seconds != null ? ` / ${fmtDur(cur.item.offset_seconds)}` : ''}</span></div>
                  <div className="ws-now-sub">{cur.sub}</div>
                </div>
                {cur.item && (
                  <div className="ws-now-actions">
                    <button className={'ws-btn-sm ' + (inEvidence(cur.item.detection_id) ? '' : 'primary')} onClick={() => toggleEvidence(cur.item)}>
                      {inEvidence(cur.item.detection_id) ? '✓ In evidence' : '＋ Add to evidence'}
                    </button>
                    {cur.item.track_id != null && <button className="ws-btn-sm" onClick={() => trackObject(cur.item)} title="Follow this object in the video">⤳ Track object</button>}
                    {cur.item.attributes?.plate_text && <button className="ws-btn-sm" onClick={() => setRegPlate(cur.item.attributes.plate_text)} title="Demo vehicle registry lookup">ⓘ Vehicle Info</button>}
                    <button className="ws-btn-sm" onClick={traceCurrent} disabled={tracing}>{tracing ? 'Tracing…' : '⤳ Track across cameras'}</button>
                  </div>
                )}
              </div>
              {track && (
                <div className="ws-track">
                  <div className="ws-track-h">{track.appearances.length} appearance(s) across cameras</div>
                  {track.appearances.slice(0, 20).map((a, i) => (
                    <div className="ws-track-row" key={i}><span className="cam">{a.camera_id}</span><span className="mono">{fmtTs(a.timestamp).slice(11)}</span><span className="sim">{Math.round((a.similarity || 0) * 100)}%</span></div>
                  ))}
                </div>
              )}
            </section>
          )}

          {/* search */}
          <section className="fp-card" style={{ padding: 16 }}>
            <div className="ws-modes">
              {MODES.map((m) => <button key={m.id} className={'ws-mode ' + (mode === m.id ? 'on' : '')} onClick={() => setMode(m.id)}>{m.label}</button>)}
            </div>
            <div className="ws-search-row">
              {mode === 'text' && <input className="ws-input" autoFocus value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={onKey} placeholder='Describe who or what to find — e.g. "a man in a white shirt with a backpack"' />}
              {mode === 'plate' && <input className="ws-input" autoFocus value={plate} onKeyDown={onKey} onChange={(e) => setPlate(e.target.value.toUpperCase())} placeholder='Number plate — full or partial, e.g. "GJ05CU6120" or "GJ05"' />}
              {mode === 'text' && <div className="ws-lang">{LANGS.map((l) => <button key={l} className={language === l ? 'on' : ''} onClick={() => setLanguage(l)}>{l}</button>)}</div>}
              <button className="fp-btn primary" onClick={runSearch} disabled={loading}>{loading ? 'Searching…' : 'Search'}</button>
            </div>
            {mode === 'text' && results === null && <div className="ws-examples">{EXAMPLES.map((x) => <button key={x} className="ws-chip" onClick={() => setQuery(x)}>{x}</button>)}</div>}
          </section>

          {meta && !loading && (
            meta.mode === 'plate'
              ? <div className="ws-meta">{meta.total} vehicle{meta.total === 1 ? '' : 's'} with a matching plate across all footage · {meta.elapsed} ms</div>
              : <div className="ws-meta">{meta.total} match{meta.total === 1 ? '' : 'es'}{scope !== 'all' ? ' in this clip' : ' across all footage'} · {meta.elapsed} ms
                  {meta.objectType ? <> · focused on <em>{meta.objectType}s</em></> : null}
                  {meta.translated ? <> · translated to <em>“{meta.translated}”</em></> : null}</div>
          )}
          {meta?.note && !loading && <div className="ws-banner soft">{meta.note}</div>}

          {/* timeline */}
          {timeline.length > 0 && !loading && (
            <section className="fp-panel">
              <div className="fp-panel-title"><span><IcClock size={16} /> Result timeline</span><span className="muted">{timeline.length} in time order</span></div>
              <div className="ws-timeline">
                {timeline.map((r) => (
                  <div key={r.detection_id} className={'ws-tl ' + (activeId === r.detection_id ? 'active' : '')} onClick={() => pickResult(r)} title={fmtTs(r.timestamp)}>
                    {r.crop_url ? <img className="ws-tl-thumb" src={r.crop_url} alt="" loading="lazy" /> : <div className="ws-tl-thumb empty">{r.class_label}</div>}
                    <div className="ws-tl-t">⤿ {fmtDur(r.offset_seconds)}</div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* result cards */}
          {loading ? <div className="ws-loading">Searching footage…</div>
            : hasResults ? (
              <div className="ws-cards">
                {results.map((r) => (
                  <div key={r.detection_id} ref={(el) => { if (el) cardRefs.current.set(r.detection_id, el); else cardRefs.current.delete(r.detection_id) }}
                       className={'ws-rc ' + (activeId === r.detection_id ? 'active' : '')}>
                    <button className="ws-rc-hit" onClick={() => pickResult(r)} title="Click to view in the player">
                      <div className="ws-rc-thumb">
                        {r.crop_url ? <img src={r.crop_url} alt={r.class_label} loading="lazy" /> : <div className="empty">{r.class_label}</div>}
                        <span className={'ws-score ' + scoreTier(r.score)}>{Math.round((r.score || 0) * 100)}%</span>
                        <div className="ws-rc-play">▶ View</div>
                      </div>
                      <div className="ws-rc-body">
                        <div className="ws-rc-top"><span className="lb">{r.class_label}</span><span className="cam">{r.camera_id}</span></div>
                        {r.attributes?.plate_text && <div className="ws-rc-plate">{r.attributes.plate_text}</div>}
                        <div className="ws-rc-attr">{attrText(r.attributes)}</div>
                        <div className="ws-rc-ts">{fmtTs(r.timestamp).slice(11)}{r.offset_seconds != null ? '  ·  ⤿ ' + fmtDur(r.offset_seconds) : ''}</div>
                      </div>
                    </button>
                    <button className={'ws-rc-add ' + (inEvidence(r.detection_id) ? 'on' : '')} onClick={() => toggleEvidence(r)} title={inEvidence(r.detection_id) ? 'In evidence' : 'Add to evidence'}>{inEvidence(r.detection_id) ? '✓' : '＋'}</button>
                    <div className="ws-rc-foot">
                      {r.track_id != null && <button className="ws-rc-act track" onClick={() => trackObject(r)} title="Follow this object in the video">⤳ Track</button>}
                      {r.attributes?.plate_text && <button className="ws-rc-act info" onClick={() => setRegPlate(r.attributes.plate_text)} title="Demo vehicle registry lookup">ⓘ Vehicle Info</button>}
                    </div>
                  </div>
                ))}
              </div>
            ) : results !== null ? <div className="ws-empty">No matches{scope !== 'all' ? ' in this clip' : ''}. Try describing it differently.</div>
              : <div className="ws-empty">{scopeVideo ? 'Describe who or what to find in this clip, or watch it above.' : cur ? 'Describe who or what you are looking for.' : 'Upload or pick a clip on the right to begin.'}</div>}
        </div>

        {/* ===================== SIDE ===================== */}
        <div className="ws-side">
          {/* investigation details */}
          <section className="fp-panel">
            <div className="fp-panel-title"><span>Investigation details</span></div>
            <div className="ws-fld"><label>Title</label><input value={caseInfo.title} onChange={(e) => setCaseInfo({ ...caseInfo, title: e.target.value })} placeholder="e.g. Diamond Market theft" /></div>
            <div className="ws-fld"><label>Case number</label><input value={caseInfo.caseNumber} onChange={(e) => setCaseInfo({ ...caseInfo, caseNumber: e.target.value })} placeholder="CASE-2026-001" /></div>
            <div className="ws-fld"><label>Lead officer</label><input value={caseInfo.officer} onChange={(e) => setCaseInfo({ ...caseInfo, officer: e.target.value })} placeholder="Insp. Name" /></div>
          </section>

          {/* cameras */}
          <section className="fp-panel">
            <div className="fp-panel-title"><span>Cameras</span><span className="muted">{cameras.length}</span></div>
            <div className="ws-list" style={{ maxHeight: 200 }}>
              {cameras.length === 0 ? <div className="muted small">No cameras registered.</div>
                : cameras.map((c) => (
                  <div key={c.camera_id} className="ws-cam"><span className="id">{c.camera_id}</span><span className="nm">{c.name && c.name !== c.camera_id ? c.name : (c.location || '')}</span>{(c.lat != null && c.lon != null) && <span className="muted" style={{ fontSize: 10 }}>GPS</span>}</div>
                ))}
            </div>
          </section>

          {/* evidence */}
          <section className="fp-panel">
            <div className="fp-panel-title"><span>Evidence</span><span className="muted">{evidence.length} item{evidence.length === 1 ? '' : 's'}</span></div>
            {evidence.length === 0 ? <div className="muted small">Add matches with the ＋ button to build a case.</div>
              : <>
                  <div className="ws-ev-strip">
                    {evidence.map((it) => (
                      <div key={it.detection_id} className="ws-ev">
                        {it.crop_url ? <img src={it.crop_url} alt="" /> : <div className="empty">{it.class_label}</div>}
                        <button className="ws-ev-x" onClick={() => removeEvidence(it.detection_id)}>×</button>
                      </div>
                    ))}
                  </div>
                  <button className="fp-btn primary" style={{ width: '100%', justifyContent: 'center' }} onClick={() => setShowExport(true)}>Export Evidence</button>
                </>}
          </section>
        </div>
      </div>

      {showExport && <ExportModal items={evidence} caseInfo={caseInfo}
        onClose={() => setShowExport(false)} onRemove={removeEvidence} />}

      {trackView && <TrackingViewer detection={trackView} onClose={() => setTrackView(null)}
        onAddEvidence={toggleEvidence} inEvidence={inEvidence} />}

      {regPlate && <VehicleInfo plate={regPlate} onClose={() => setRegPlate(null)} />}
    </div>
  )
}

/* ---------------------------- Export modal ---------------------------- */
function ExportModal({ items, caseInfo, onClose, onRemove }) {
  const [caseNumber, setCaseNumber] = useState(caseInfo?.caseNumber || '')
  const [officer, setOfficer] = useState(caseInfo?.officer || '')
  const [notes, setNotes] = useState(caseInfo?.notes || '')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  useEffect(() => { const esc = (e) => { if (e.key === 'Escape') onClose() }; window.addEventListener('keydown', esc); return () => window.removeEventListener('keydown', esc) }, [onClose])

  async function doExport() {
    setError(null)
    if (!caseNumber.trim() || !officer.trim()) { setError('Case number and officer are required.'); return }
    setBusy(true)
    try { setResult(await createExport({ detectionIds: items.map((i) => i.detection_id), caseNumber, officer, notes })) }
    catch (e) { setError(e?.response?.data?.detail || e.message || 'Export failed') } finally { setBusy(false) }
  }

  return (
    <div className="ws-overlay" onClick={onClose}>
      <div className="ws-modal" onClick={(e) => e.stopPropagation()}>
        <button className="ws-modal-x" onClick={onClose}>×</button>
        <h3>Export evidence <span className="muted" style={{ fontSize: 13 }}>({items.length} item{items.length === 1 ? '' : 's'})</span></h3>
        <div className="ws-ev-strip">
          {items.map((it) => (
            <div key={it.detection_id} className="ws-ev">
              {it.crop_url ? <img src={it.crop_url} alt="" /> : <div className="empty">{it.class_label}</div>}
              <button className="ws-ev-x" onClick={() => onRemove(it.detection_id)}>×</button>
            </div>
          ))}
        </div>
        {result ? (
          <div>
            <div className="ws-kv"><span>Export ID</span><b>{result.export_id}</b></div>
            <div className="ws-kv"><span>Files</span><b>{result.file_count}</b></div>
            <div className="ws-kv"><span>SHA-256 seal</span><b className="ws-hash">{result.manifest_hash}</b></div>
            <a className="fp-btn primary" style={{ width: '100%', justifyContent: 'center', marginTop: 12 }} href={result.download_url} download>Download evidence .zip</a>
          </div>
        ) : (
          <>
            <div className="ws-fld"><label>Case number</label><input value={caseNumber} onChange={(e) => setCaseNumber(e.target.value)} placeholder="CASE-2026-001" /></div>
            <div className="ws-fld"><label>Officer</label><input value={officer} onChange={(e) => setOfficer(e.target.value)} placeholder="Insp. Name" /></div>
            <div className="ws-fld"><label>Notes (optional)</label><textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={2} /></div>
            {error && <div className="ws-banner err">{error}</div>}
            <button className="fp-btn primary" style={{ width: '100%', justifyContent: 'center' }} onClick={doExport} disabled={busy || !items.length}>{busy ? 'Sealing evidence…' : `Export ${items.length} item(s)`}</button>
          </>
        )}
      </div>
    </div>
  )
}

/* ------------------------------- helpers ------------------------------- */
function fmtDur(s) { if (s == null) return ''; s = Math.round(s); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0') }
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
  if (a.plate_text) p.push('plate: ' + a.plate_text)
  if (a.age) p.push('age ' + a.age)
  if (a.gender) p.push(a.gender)
  return p.join(' · ')
}
function vpActive(vp) { return !!vp && ((vp.pct || 0) > 0 || (vp.total || 0) > 0) }
function stageText(vp) {
  const s = vp.stage
  if (s === 'indexing' && (vp.indexed || 0) > 0) return `indexed ${Number(vp.indexed).toLocaleString()} detections`
  if ((!s || s === 'start' || s === 'detect+track') && (vp.total || 0) > 0) return `frame ${vp.frame}/${vp.total}`
  const map = { start: 'starting', 'detect+track': 'detecting', indexing: 'indexing', clip: 'embedding', reid: 're-identifying', store: 'saving', faces: 'faces', plates: 'plates', done: 'finishing' }
  return map[s] || s || 'processing'
}
