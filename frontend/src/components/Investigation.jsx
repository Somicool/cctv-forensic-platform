// Single-page investigation workflow, redesigned for speed and clarity.
//
//   Footage  ->  AI Processing  ->  Ready  ->  Pick a clip / all footage
//            ->  Describe & Search  ->  Jump to timestamp  ->  Export evidence
//
// Backend is unchanged: this screen only calls existing endpoints
//   GET  /library, /videos, /cameras          (footage)
//   POST /ingest/all, GET /ingest/job/{id}, POST /ingest/stop   (processing)
//   POST /search/text | /search/image | /search/plate          (search, scoped by video_id)
//   GET  /track/{id}                            (optional trace)
//   POST /export, GET /exports                  (evidence)
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  getLibrary, getVideos, ingestAll, getIngestJob, stopIngest, uploadVideo,
  searchText, searchImage, searchPlate, trackDetection,
  createExport,
} from '../api'
import VideoPlayer from './VideoPlayer'

const STEPS = [
  'Footage', 'AI processing', 'Investigation ready',
  'Describe & search', 'Jump to timestamp', 'Export evidence',
]
const MODES = [
  { id: 'text', label: 'Describe' },
  { id: 'image', label: 'Image' },
  { id: 'plate', label: 'Plate' },
]
const LANGS = ['EN', 'HI', 'GU']
const EXAMPLES = [
  'a person carrying a backpack',
  'a man in a white shirt',
  'a white truck',
  'a red car',
]

export default function Investigation({ cameras }) {
  // ---- footage + processing ----
  const [library, setLibrary] = useState(null)
  const [job, setJob] = useState(null)
  const [phase, setPhase] = useState('footage')      // 'footage' | 'investigate'
  const [uploadPct, setUploadPct] = useState(null)    // null = not uploading; 0-100 while transferring
  const autoOpenRef = useRef(false)                   // open the new clip once its job finishes

  // ---- investigation scope (one clip, or all footage) ----
  const [scope, setScope] = useState('all')          // 'all' | video_id
  const [scopeVideo, setScopeVideo] = useState(null)  // the clip picked from the library (if any)

  // ---- search ----
  const [mode, setMode] = useState('text')
  const [query, setQuery] = useState('')
  const [language, setLanguage] = useState('EN')
  const [plate, setPlate] = useState('')
  const [file, setFile] = useState(null)
  const [results, setResults] = useState(null)       // null = not searched yet
  const [meta, setMeta] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // ---- player (either the whole clip, or a specific match) ----
  const [current, setCurrent] = useState(null)
  const [activeId, setActiveId] = useState(null)     // highlighted result (follows playback)
  const [playTime, setPlayTime] = useState(0)        // live playback position (s)
  const [track, setTrack] = useState(null)
  const [tracing, setTracing] = useState(false)

  // ---- evidence + export ----
  const [evidence, setEvidence] = useState([])
  const [showExport, setShowExport] = useState(false)

  // refs so the (stable) timeupdate callback always sees fresh state
  const currentRef = useRef(null); currentRef.current = current
  const resultsRef = useRef(null); resultsRef.current = results
  const activeIdRef = useRef(null); activeIdRef.current = activeId
  const cardRefs = useRef(new Map())                 // detection_id -> card element

  const nameFor = (id) => {
    const n = (cameras || []).find((c) => c.camera_id === id)?.name
    return (!n || n === id) ? '' : n
  }
  const camLabel = (id) => { const n = nameFor(id); return n ? `${id} · ${n}` : id }

  // ---------- footage library ----------
  async function loadLibrary() {
    try {
      return await getLibrary().then((items) => { setLibrary(items); return items })
    } catch {
      const vids = await getVideos().catch(() => [])
      const items = vids.map((v) => ({ ...v, processed: true }))
      setLibrary(items)
      return items
    }
  }
  useEffect(() => { loadLibrary() /* eslint-disable-next-line */ }, [])

  const processed = (library || []).filter((v) => v.processed)
  const unprocessed = (library || []).filter((v) => !v.processed)
  const processing = job && job.status === 'processing'

  // ---------- processing ----------
  async function startProcessing() {
    setError(null)
    try {
      const r = await ingestAll()
      if (!r.job_id) { await loadLibrary(); return }
      setJob({ job_id: r.job_id, status: 'processing', done: 0, total: r.total, current: null })
    } catch (e) {
      setError('Could not start processing. ' + (e?.message || ''))
    }
  }
  async function stopProcessing() { try { await stopIngest() } catch { /* polled */ } }

  // Upload a clip from the device, then analyse + auto-open it.
  async function handleUpload(f) {
    if (!f) return
    setError(null); setUploadPct(0)
    try {
      const r = await uploadVideo({ file: f, onProgress: setUploadPct })
      setUploadPct(null)
      if (r.busy) { setError(r.message || 'Another job is running — wait for it to finish.'); return }
      if (r.job_id) {
        autoOpenRef.current = true
        setJob({ job_id: r.job_id, status: 'processing', done: 0, total: 1, current: r.filename })
      } else {
        await loadLibrary()
      }
    } catch (e) {
      setUploadPct(null)
      setError('Upload failed. ' + (e?.response?.data?.detail || e.message || ''))
    }
  }

  useEffect(() => {
    if (!processing) return
    const t = setInterval(async () => {
      try {
        const j = await getIngestJob(job.job_id)
        setJob(j)
        if (j.status !== 'processing') {
          clearInterval(t)
          const items = await loadLibrary()
          if (autoOpenRef.current && j.status === 'done') {
            autoOpenRef.current = false
            const newest = (items || []).filter((v) => v.processed)
              .sort((a, b) => (b.video_id || 0) - (a.video_id || 0))[0]
            if (newest) investigateClip(newest)
          }
        }
      } catch { clearInterval(t) }
    }, 1500)
    return () => clearInterval(t)
    // eslint-disable-next-line
  }, [job?.job_id, job?.status])

  // ---------- enter investigation ----------
  function mediaFromVideo(v) {
    return {
      key: v.url, videoId: v.video_id, src: v.url, offset: 0, bbox: null, frameW: null, frameH: null,
      title: v.filename, item: null,
      sub: [camLabel(v.camera_id), fmtDur(v.duration)].filter(Boolean).join('  ·  '),
    }
  }
  function investigateClip(v) {
    setScope(v.video_id); setScopeVideo(v)
    setResults(null); setMeta(null); setTrack(null); setActiveId(null); setPlayTime(0)
    setCurrent(mediaFromVideo(v))
    setPhase('investigate')
  }
  function investigateAll() {
    setScope('all'); setScopeVideo(null)
    setResults(null); setMeta(null); setTrack(null); setActiveId(null); setPlayTime(0); setCurrent(null)
    setPhase('investigate')
  }

  // ---------- selecting a result ----------
  function pickResult(r) {
    setCurrent({
      key: r.video_url, videoId: r.video_id, src: r.video_url, offset: r.offset_seconds || 0,
      bbox: r.bbox, frameW: r.frame_width, frameH: r.frame_height, item: r,
      title: `${r.class_label}  ·  ${camLabel(r.camera_id)}`,
      sub: [fmtTs(r.timestamp), 'match ' + Math.round((r.score || 0) * 100) + '%',
            attrText(r.attributes)].filter(Boolean).join('  ·  '),
    })
    setActiveId(r.detection_id)
    setPlayTime(r.offset_seconds || 0)
    setTrack(null)
  }

  // Smooth-scroll the active card into view whenever the active result changes
  // (whether clicked or advanced by playback).
  useEffect(() => {
    if (activeId == null) return
    const el = cardRefs.current.get(activeId)
    el?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [activeId])

  // Playback -> keep the active result synchronized with the video position.
  const onPlayTime = useCallback((t) => {
    setPlayTime(t)
    const cur = currentRef.current
    if (!cur || cur.videoId == null) return
    const res = resultsRef.current || []
    let best = null
    for (const r of res) {
      if (r.video_id !== cur.videoId || r.offset_seconds == null) continue
      if (r.offset_seconds <= t + 0.3 && (!best || r.offset_seconds > best.offset_seconds)) best = r
    }
    if (best && best.detection_id !== activeIdRef.current) setActiveId(best.detection_id)
  }, [])

  // ---------- search ----------
  async function runSearch() {
    setError(null)
    if (mode === 'text' && !query.trim()) return
    if (mode === 'plate' && !plate.trim()) return
    if (mode === 'image' && !file) { setError('Choose an image to search with.'); return }
    setLoading(true); setTrack(null)
    const t0 = performance.now()
    const filters = scope === 'all' ? {} : { video_id: scope }
    try {
      let data
      if (mode === 'text') data = await searchText({ query, language: language.toLowerCase(), includeScenes: false, filters })
      else if (mode === 'image') data = await searchImage({ file })
      else data = await searchPlate({ plate, filters })
      let res = data.results || []
      if (scope !== 'all') res = res.filter((r) => r.video_id === scope)   // scope client-side too (covers image)
      setResults(res)
      setMeta({
        total: res.length, note: data.note, objectType: data.object_type,
        translated: data.translated_query, elapsed: Math.round(performance.now() - t0),
      })
      if (res.length) pickResult(res[0])          // one search -> jump straight to top match
      else { setActiveId(null); if (scopeVideo) setCurrent(mediaFromVideo(scopeVideo)) }
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || 'Search failed')
      setResults([]); setMeta(null)
    } finally { setLoading(false) }
  }

  async function traceCurrent() {
    if (!current?.item) return
    setTracing(true)
    try { setTrack(await trackDetection(current.item.detection_id)) }
    catch { setTrack({ appearances: [] }) }
    finally { setTracing(false) }
  }

  // ---------- evidence ----------
  const inEvidence = (id) => evidence.some((e) => e.detection_id === id)
  function toggleEvidence(r) {
    setEvidence((prev) => prev.some((e) => e.detection_id === r.detection_id)
      ? prev.filter((e) => e.detection_id !== r.detection_id)
      : [...prev, r])
  }

  // ---------- workflow step (for the rail) ----------
  let step = 1
  if (phase === 'footage') {
    if (processing) step = 2
    else if (processed.length > 0) step = 3
    else step = 1
  } else {
    step = results === null ? 4 : (current?.item ? (evidence.length ? 6 : 5) : 4)
  }

  const onKey = (e) => { if (e.key === 'Enter') runSearch() }
  const cur = current
  const hasResults = Array.isArray(results) && results.length > 0

  const playerPanel = cur && (
    <section className="inv-player">
      <VideoPlayer key={cur.key} src={cur.src} offset={cur.offset}
                   bbox={cur.bbox} frameW={cur.frameW} frameH={cur.frameH}
                   onTimeUpdate={onPlayTime} />
      <div className="inv-now">
        <div className="inv-now-info">
          <div className="inv-now-title">
            {cur.title}
            <span className="inv-time mono">{fmtDur(playTime)}{cur.item?.offset_seconds != null ? ` / ${fmtDur(cur.item.offset_seconds)}` : ''}</span>
          </div>
          <div className="muted small">{cur.sub}</div>
        </div>
        {cur.item && (
          <div className="inv-now-actions">
            <button className={'btn small ' + (inEvidence(cur.item.detection_id) ? '' : 'primary')}
                    onClick={() => toggleEvidence(cur.item)}>
              {inEvidence(cur.item.detection_id) ? '✓ In evidence' : '＋ Add to evidence'}
            </button>
            <button className="btn small ghost" onClick={traceCurrent} disabled={tracing}>
              {tracing ? 'Tracing…' : '⤳ Track across cameras'}
            </button>
          </div>
        )}
      </div>
      {track && (
        <div className="track-list">
          <div className="track-head">{track.appearances.length} appearance(s) across cameras</div>
          {track.appearances.slice(0, 20).map((a, i) => (
            <div className="track-row" key={i}>
              <span className="cam">{a.camera_id}</span>
              <span className="mono">{fmtTs(a.timestamp).slice(11)}</span>
              <span className="sim">{Math.round((a.similarity || 0) * 100)}%</span>
            </div>
          ))}
        </div>
      )}
    </section>
  )

  const resultsGrid = hasResults && (
    <div className="results-grid inv-results">
      {results.map((r) => (
        <div key={r.detection_id}
             ref={(el) => { if (el) cardRefs.current.set(r.detection_id, el); else cardRefs.current.delete(r.detection_id) }}
             className={'result-card ' + (activeId === r.detection_id ? 'active' : '')}>
          <button className="rc-hit" onClick={() => pickResult(r)}>
            <div className="thumb">
              {r.crop_url ? <img src={r.crop_url} alt={r.class_label} loading="lazy" />
                          : <div className="thumb-empty">{r.class_label}</div>}
              <span className={'score ' + scoreTier(r.score)}>{Math.round((r.score || 0) * 100)}%</span>
            </div>
            <div className="result-body">
              <div className="result-top">
                <span className="label">{r.class_label}</span>
                <span className="cam">{r.camera_id}</span>
              </div>
              <div className="attrs">{attrText(r.attributes)}</div>
              <div className="ts">{fmtTs(r.timestamp).slice(11)}{r.offset_seconds != null ? '  ·  ⤿ ' + fmtDur(r.offset_seconds) : ''}</div>
            </div>
          </button>
          <button className={'rc-add ' + (inEvidence(r.detection_id) ? 'on' : '')}
                  title={inEvidence(r.detection_id) ? 'In evidence' : 'Add to evidence'}
                  onClick={() => toggleEvidence(r)}>
            {inEvidence(r.detection_id) ? '✓' : '＋'}
          </button>
        </div>
      ))}
    </div>
  )

  return (
    <div className="inv">
      <Stepper active={step} />

      {error && <div className="banner error inv-gap">{error}</div>}

      {phase === 'footage' ? (
        <Footage
          library={library} processed={processed} unprocessed={unprocessed}
          processing={processing} job={job} camLabel={camLabel} uploadPct={uploadPct}
          onStart={startProcessing} onStop={stopProcessing} onUpload={handleUpload}
          onInvestigateClip={investigateClip} onInvestigateAll={investigateAll}
        />
      ) : (
        <>
          {/* scope bar */}
          <div className="inv-scope">
            <button className="linklike" onClick={() => setPhase('footage')}>‹ Footage</button>
            <div className="inv-scope-mid">
              Investigating: <b>{scopeVideo ? scopeVideo.filename : 'All footage'}</b>
            </div>
            {scopeVideo && (
              <div className="scope-toggle">
                <button className={scope !== 'all' ? 'active' : ''} onClick={() => setScope(scopeVideo.video_id)}>This clip</button>
                <button className={scope === 'all' ? 'active' : ''} onClick={() => setScope('all')}>All footage</button>
              </div>
            )}
          </div>

          <section className="inv-search">
            <div className="mode-tabs">
              {MODES.map((m) => (
                <button key={m.id} className={'mode-tab ' + (mode === m.id ? 'active' : '')}
                        onClick={() => setMode(m.id)}>{m.label}</button>
              ))}
            </div>
            <div className="search-row">
              {mode === 'text' && (
                <input className="search-input" autoFocus value={query}
                       onChange={(e) => setQuery(e.target.value)} onKeyDown={onKey}
                       placeholder='Describe who or what to find — e.g. "a man in a white shirt with a backpack"' />
              )}
              {mode === 'image' && (
                <input type="file" accept="image/*" className="file-input"
                       onChange={(e) => setFile(e.target.files?.[0] || null)} />
              )}
              {mode === 'plate' && (
                <input className="search-input" value={plate} onKeyDown={onKey}
                       onChange={(e) => setPlate(e.target.value.toUpperCase())}
                       placeholder='Plate — full or partial, e.g. "GJ05" or "AB1234"' />
              )}
              {mode === 'text' && (
                <div className="lang-toggle">
                  {LANGS.map((l) => (
                    <button key={l} className={language === l ? 'active' : ''}
                            onClick={() => setLanguage(l)}>{l}</button>
                  ))}
                </div>
              )}
              <button className="btn primary" onClick={runSearch} disabled={loading}>
                {loading ? 'Searching…' : 'Search'}
              </button>
            </div>
            {mode === 'text' && results === null && (
              <div className="examples">
                {EXAMPLES.map((x) => <button key={x} className="chip" onClick={() => setQuery(x)}>{x}</button>)}
              </div>
            )}
          </section>

          {meta && !loading && (
            <div className="result-meta inv-gap">
              {meta.total} match{meta.total === 1 ? '' : 'es'}
              {scope !== 'all' ? ' in this clip' : ' across all footage'} · {meta.elapsed} ms
              {meta.objectType ? <> · focused on <em>{meta.objectType}s</em></> : null}
              {meta.translated ? <> · translated to <em>“{meta.translated}”</em></> : null}
            </div>
          )}
          {meta?.note && !loading && <div className="banner soft inv-gap">{meta.note}</div>}

          {loading ? (
            <div className="loading">Searching footage…</div>
          ) : hasResults ? (
            <div className="inv-work">
              <div className="inv-work-left">{playerPanel}</div>
              <div className="inv-work-right">{resultsGrid}</div>
            </div>
          ) : (
            <>
              {playerPanel}
              {results !== null ? (
                <div className="empty">No matches{scope !== 'all' ? ' in this clip' : ''}. Try describing it differently.</div>
              ) : (
                <div className="empty">
                  {scopeVideo
                    ? 'Describe who or what to find in this clip, or watch it above.'
                    : 'Describe who or what you are looking for to begin.'}
                </div>
              )}
            </>
          )}
        </>
      )}

      {/* sticky evidence bar */}
      {phase === 'investigate' && (
        <div className="inv-evidence">
          <button className="linklike" onClick={() => setPhase('footage')}>‹ Footage</button>
          <div className="inv-evidence-mid">
            {evidence.length === 0
              ? <span className="muted small">Add matches to build an evidence set</span>
              : <span>{evidence.length} item{evidence.length === 1 ? '' : 's'} in evidence</span>}
          </div>
          <button className="btn primary" disabled={!evidence.length} onClick={() => setShowExport(true)}>
            Export evidence
          </button>
        </div>
      )}

      {showExport && (
        <ExportModal items={evidence}
                     onClose={() => setShowExport(false)}
                     onRemove={(id) => setEvidence((p) => p.filter((e) => e.detection_id !== id))} />
      )}
    </div>
  )
}

/* ------------------------------- Stepper ------------------------------- */
function Stepper({ active }) {
  return (
    <div className="inv-steps">
      {STEPS.map((label, i) => {
        const n = i + 1
        const state = n < active ? 'done' : n === active ? 'active' : 'todo'
        return (
          <div className={'inv-step ' + state} key={label}>
            <span className="inv-step-dot">{n < active ? '✓' : n}</span>
            <span className="inv-step-label">{label}</span>
          </div>
        )
      })}
    </div>
  )
}

/* ------------------------------- Footage ------------------------------- */
function Footage({ library, processed, unprocessed, processing, job, camLabel, uploadPct,
                   onStart, onStop, onUpload, onInvestigateClip, onInvestigateAll }) {
  const fileRef = useRef(null)
  const uploading = uploadPct != null
  const busy = processing || uploading
  const pct = job && job.total
    ? (vpActive(job.video_progress)
        ? job.video_progress.pct
        : Math.round(((job.done || 0) / job.total) * 100))
    : 0

  return (
    <section className="inv-footage">
      <input ref={fileRef} type="file" accept="video/*" hidden
             onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ''; onUpload(f) }} />
      <div className="inv-head">
        <div>
          <h2>CCTV footage</h2>
          <p className="muted small">
            {library === null ? 'Loading…'
              : (processed.length > 0
                  ? `Upload a clip from your device, or pick one below — ${library.length} clip${library.length === 1 ? '' : 's'}, ${processed.length} analysed`
                  : `Upload a clip from your device to begin${library.length ? ` · ${library.length} clip${library.length === 1 ? '' : 's'} found` : ''}`)}
          </p>
        </div>
        <div className="inv-head-actions">
          <button className="btn primary" onClick={() => fileRef.current?.click()} disabled={busy}
                  title="Select a video file from this device and investigate it">
            ⬆ Upload video
          </button>
          {!processing && unprocessed.length > 0 && (
            <button className="btn" onClick={onStart} disabled={busy}>
              Analyse {unprocessed.length} existing
            </button>
          )}
          {processing && (
            <button className="btn danger" onClick={onStop}>■ Stop</button>
          )}
          {!processing && processed.length > 0 && (
            <button className="btn ghost" onClick={onInvestigateAll} disabled={busy}>Search all footage ›</button>
          )}
        </div>
      </div>

      {uploading && (
        <div className="banner soft proc">
          <div className="proc-head">Uploading video from device…</div>
          <div className="progress-wrap">
            <div className="progress-bar" style={{ width: uploadPct + '%' }} />
            <span className="progress-label">{uploadPct}%</span>
          </div>
        </div>
      )}

      {processing && (
        <div className="banner soft proc">
          <div className="proc-head">
            AI analysing <b>{job.current || ''}</b> — {(job.done || 0) + 1} of {job.total}
          </div>
          <div className="progress-wrap">
            <div className="progress-bar" style={{ width: pct + '%' }} />
            <span className="progress-label">
              {vpActive(job.video_progress)
                ? `${pct}% · ${stageText(job.video_progress)}`
                : `${pct}% · ${job.done || 0}/${job.total} clips`}
            </span>
          </div>
        </div>
      )}

      {library === null ? (
        <div className="muted">Loading footage…</div>
      ) : library.length === 0 ? (
        <div className="muted">No footage yet — use <b>Upload video</b> above to add a clip from your device.</div>
      ) : (
        <div className="lib-grid">
          {library.map((v) => {
            const clickable = v.processed
            return (
              <div key={v.filename}
                   className={'lib-card ' + (v.processed ? '' : 'unprocessed ') + (clickable ? 'clickable' : '')}
                   role={clickable ? 'button' : undefined} tabIndex={clickable ? 0 : undefined}
                   onClick={clickable ? () => onInvestigateClip(v) : undefined}
                   onKeyDown={clickable ? (e) => { if (e.key === 'Enter') onInvestigateClip(v) } : undefined}
                   title={clickable ? 'Investigate this clip' : 'Not analysed yet'}>
                <div className="lib-thumb">
                  {v.processed ? <video src={v.url + '#t=1'} muted preload="metadata" />
                               : <div className="lib-thumb-empty">▶</div>}
                  <span className={'lib-badge ' + (v.processed ? 'ok' : 'no')}>
                    {v.processed ? 'Analysed' : 'Not analysed'}
                  </span>
                  {clickable && <span className="lib-open">Investigate ›</span>}
                </div>
                <div className="lib-info">
                  <div className="lib-name" title={v.filename}>{v.filename}</div>
                  <div className="muted small">
                    {v.processed
                      ? (camLabel(v.camera_id) + (v.duration ? '  ·  ' + fmtDur(v.duration) : ''))
                      : (v.size_mb != null ? v.size_mb + ' MB' : '')}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </section>
  )
}

/* ---------------------------- Export modal ---------------------------- */
function ExportModal({ items, onClose, onRemove }) {
  const [caseNumber, setCaseNumber] = useState('')
  const [officer, setOfficer] = useState('')
  const [notes, setNotes] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  useEffect(() => {
    const esc = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [onClose])

  async function doExport() {
    setError(null)
    if (!caseNumber.trim() || !officer.trim()) { setError('Case number and officer are required.'); return }
    setBusy(true)
    try {
      const r = await createExport({
        detectionIds: items.map((i) => i.detection_id), caseNumber, officer, notes,
      })
      setResult(r)
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || 'Export failed')
    } finally { setBusy(false) }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal inv-export" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>×</button>
        <h3>Export evidence <span className="muted small">({items.length} item{items.length === 1 ? '' : 's'})</span></h3>

        <div className="inv-ev-strip">
          {items.map((it) => (
            <div className="inv-ev-item" key={it.detection_id}>
              {it.crop_url ? <img src={it.crop_url} alt="" /> : <div className="thumb-empty">{it.class_label}</div>}
              <button className="inv-ev-x" title="Remove" onClick={() => onRemove(it.detection_id)}>×</button>
            </div>
          ))}
        </div>

        {result ? (
          <div className="export-result">
            <div className="kv"><span>Export ID</span><b>{result.export_id}</b></div>
            <div className="kv"><span>Files</span><b>{result.file_count}</b></div>
            <div className="kv"><span>SHA-256 seal</span><b className="mono hash">{result.manifest_hash}</b></div>
            <a className="btn primary" href={result.download_url} download>Download evidence .zip</a>
          </div>
        ) : (
          <>
            <label className="fld"><span>Case number</span>
              <input value={caseNumber} onChange={(e) => setCaseNumber(e.target.value)} placeholder="CASE-2026-001" /></label>
            <label className="fld"><span>Officer</span>
              <input value={officer} onChange={(e) => setOfficer(e.target.value)} placeholder="Insp. Name" /></label>
            <label className="fld"><span>Notes (optional)</span>
              <textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={2} /></label>
            {error && <div className="banner error">{error}</div>}
            <button className="btn primary" onClick={doExport} disabled={busy || !items.length}>
              {busy ? 'Sealing evidence…' : `Export ${items.length} item(s)`}
            </button>
          </>
        )}
      </div>
    </div>
  )
}

/* ------------------------------- helpers ------------------------------- */
function fmtDur(s) {
  if (s == null) return ''
  s = Math.round(s)
  return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0')
}
function fmtTs(ts) { return ts ? ts.replace('T', ' ').slice(0, 19) : '—' }
function scoreTier(s) {
  if ((s || 0) >= 0.7) return 'high'
  if ((s || 0) >= 0.4) return 'mid'
  return 'low'
}
function attrText(a) {
  if (!a) return ''
  const parts = []
  if (a.color) parts.push(a.color)
  if (a.upper_color) parts.push('top: ' + a.upper_color)
  if (a.lower_color) parts.push('btm: ' + a.lower_color)
  if (a.vehicle_type) parts.push(a.vehicle_type)
  if (Array.isArray(a.accessories) && a.accessories.length) parts.push(a.accessories.join(', '))
  if (a.plate_text) parts.push('plate: ' + a.plate_text)
  if (a.age) parts.push('age ' + a.age)
  if (a.gender) parts.push(a.gender)
  return parts.join(' · ')
}
function vpActive(vp) { return !!vp && ((vp.pct || 0) > 0 || (vp.total || 0) > 0) }
function stageText(vp) {
  const s = vp.stage
  if ((!s || s === 'start' || s === 'detect+track') && (vp.total || 0) > 0) return `frame ${vp.frame}/${vp.total}`
  const map = {
    start: 'starting', 'detect+track': 'detecting & tracking', clip: 'embedding',
    reid: 're-identifying', store: 'saving', faces: 'reading faces',
    plates: 'reading plates', done: 'finishing',
  }
  return map[s] || s || 'processing'
}
