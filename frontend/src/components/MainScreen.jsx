// Main screen (tracking-enabled). Describe-and-filter search + cross-camera
// "Track" launch + Stop-processing button + Add/Edit-Camera (GPS) form + live
// ingest progress. Lives in its own file (SearchWatch.jsx is locked); App
// renders THIS component.
import { useEffect, useState } from 'react'
import {
  getVideos, getLibrary, ingestAll, getIngestJob, stopIngest, addCamera,
  describeSearch, searchImage, searchPlate,
} from '../api'
import VideoPlayer from './VideoPlayer'
import TrackingView from './TrackingView'

const MODES = [
  { id: 'text', label: 'Describe' },
  { id: 'image', label: 'Image' },
  { id: 'plate', label: 'Plate' },
]
const LANGS = ['EN', 'HI', 'GU']
const EXAMPLES = [
  'man in a blue shirt with a backpack',
  'woman in a red top with sunglasses',
  'white SUV',
  'person with a helmet on a motorcycle',
]

export default function MainScreen({ cameras }) {
  const [library, setLibrary] = useState(null)
  const [libMode, setLibMode] = useState('library')
  const [current, setCurrent] = useState(null)
  const [scope, setScope] = useState('all')
  const [mode, setMode] = useState('text')
  const [query, setQuery] = useState('')
  const [language, setLanguage] = useState('EN')
  const [plate, setPlate] = useState('')
  const [file, setFile] = useState(null)
  const [results, setResults] = useState(null)
  const [meta, setMeta] = useState(null)
  const [chips, setChips] = useState(null)         // parsed constraint chips (describe mode)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState(null)      // soft info (e.g. "all processed", "job running")
  const [job, setJob] = useState(null)
  const [tracking, setTracking] = useState(null)   // detection_id being traced, or null
  const [cams, setCams] = useState(cameras || [])  // local copy so the map refreshes after add
  const [camForm, setCamForm] = useState(null)     // null = closed; else {cameraId,name,location,lat,lon}
  const [stopping, setStopping] = useState(false)

  useEffect(() => { setCams(cameras || []) }, [cameras])

  const nameFor = (id) => {
    const n = (cams || []).find((c) => c.camera_id === id)?.name
    return (!n || n === id) ? '' : n
  }

  async function loadLibrary(selectFirst = false) {
    let items
    try {
      items = await getLibrary()
      setLibMode('library')
    } catch {
      const vids = await getVideos().catch(() => [])
      items = vids.map((v) => ({ ...v, processed: true }))
      setLibMode('videos')
    }
    setLibrary(items)
    if (selectFirst && !current) {
      const first = items.find((v) => v.processed)
      if (first) selectVideo(first, false)
    }
    return items
  }

  useEffect(() => { loadLibrary(true) /* eslint-disable-next-line */ }, [cameras])

  function selectVideo(v, alsoScope = true) {
    if (!v.processed) return
    setCurrent({
      key: 'rec-' + v.video_id, src: v.url, offset: 0, bbox: null, frameW: null, frameH: null,
      title: v.filename,
      sub: [labelFor(v.camera_id), fmtDur(v.duration)].filter(Boolean).join('  ·  '),
    })
    if (alsoScope) setScope(v.video_id)
  }

  function selectResult(r) {
    setCurrent({
      key: 'res-' + r.detection_id, src: r.video_url, offset: r.offset_seconds || 0,
      bbox: r.bbox, frameW: r.frame_width, frameH: r.frame_height,
      detectionId: r.detection_id, classLabel: r.class_label,
      title: (r.class_label + '  ·  ' + labelFor(r.camera_id)).trim(),
      sub: [r.timestamp ? r.timestamp.replace('T', ' ').slice(0, 19) : '',
            'match ' + Math.round((r.score || 0) * 100) + '%',
            r.offset_seconds != null ? 'at ' + fmtDur(r.offset_seconds) : '']
        .filter(Boolean).join('  ·  '),
    })
  }

  const labelFor = (id) => { const n = nameFor(id); return n ? (id + ' ' + n) : id }

  async function run() {
    setError(null)
    if (mode === 'text' && !query.trim()) return
    if (mode === 'plate' && !plate.trim()) return
    if (mode === 'image' && !file) { setError('Choose an image to search with.'); return }
    setLoading(true)
    const t0 = performance.now()
    const filters = scope === 'all' ? {} : { video_id: scope }
    try {
      let data
      if (mode === 'text') {
        data = await describeSearch({ query, filters })
        setChips(data.chips || [])
      } else if (mode === 'image') {
        data = await searchImage({ file }); setChips(null)
      } else {
        data = await searchPlate({ plate, filters }); setChips(null)
      }
      let res = data.results || []
      if (scope !== 'all') res = res.filter((r) => r.video_id === scope)
      setResults(res)
      setMeta({
        total: res.length, note: data.note,
        objectType: data.parsed?.object_type, strictTotal: data.strict_total,
        elapsed: Math.round(performance.now() - t0),
      })
      if (res.length) selectResult(res[0])
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || 'Search failed')
      setResults([]); setMeta(null); setChips(null)
    } finally {
      setLoading(false)
    }
  }

  function backToLibrary() { setResults(null); setMeta(null); setChips(null) }

  async function processAll() {
    setError(null)
    try {
      const r = await ingestAll()
      if (!r.job_id) { await loadLibrary(); return }
      setStopping(false)
      setJob({ job_id: r.job_id, status: 'processing', done: 0, total: r.total, current: null })
    } catch (e) {
      setError('Could not start processing. ' + (e?.response?.status === 404
        ? 'The processing endpoint is not enabled yet (see the 2-line backend note).'
        : (e.message || '')))
    }
  }

  async function handleStop() {
    setStopping(true)
    try { await stopIngest() } catch { /* ignore - polling will reflect it */ }
  }

  async function saveCamera() {
    if (!camForm?.cameraId?.trim()) { setError('Camera ID is required.'); return }
    setError(null)
    try {
      const updated = await addCamera(camForm)
      setCams(updated)
      setCamForm(null)
    } catch (e) {
      setError('Could not save camera. ' + (e?.response?.data?.detail || e.message || ''))
    }
  }

  useEffect(() => {
    if (!job || job.status !== 'processing') return
    const t = setInterval(async () => {
      try {
        const j = await getIngestJob(job.job_id)
        setJob(j)
        if (j.status === 'done' || j.status === 'error' || j.status === 'stopped') {
          clearInterval(t)
          setStopping(false)
          loadLibrary()
        }
      } catch { clearInterval(t) }
    }, 1500)
    return () => clearInterval(t)
    // eslint-disable-next-line
  }, [job?.job_id, job?.status])

  const processed = (library || []).filter((v) => v.processed)
  const unprocessed = (library || []).filter((v) => !v.processed)
  const processing = job && job.status === 'processing'
  const onKey = (e) => { if (e.key === 'Enter') run() }

  if (tracking != null) {
    return <TrackingView cameras={cams} initialDetectionId={tracking}
                         onBack={() => setTracking(null)} />
  }

  return (
    <div className="sw">
      <div className="sw-player">
        {current ? (
          <>
            <VideoPlayer key={current.key} src={current.src} offset={current.offset}
                         bbox={current.bbox} frameW={current.frameW} frameH={current.frameH} />
            <div className="sw-now">
              <div className="sw-now-title">{current.title}</div>
              <div className="muted small">{current.sub}</div>
              {current.detectionId != null && (
                <button className="btn primary sw-track-btn"
                        onClick={() => setTracking(current.detectionId)}
                        title="Find this same person/vehicle across all city cameras">
                  ⤳ Track {current.classLabel === 'person' ? 'this person' : 'this ' + (current.classLabel || 'object')} across cameras
                </button>
              )}
            </div>
          </>
        ) : (
          <div className="muted" style={{ padding: 40 }}>
            {library && processed.length === 0
              ? 'No processed videos yet. Use "Process" below to analyze your footage.'
              : 'Loading…'}
          </div>
        )}
      </div>

      <div className="sw-search">
        <div className="mode-tabs">
          {MODES.map((m) => (
            <button key={m.id} className={'mode-tab ' + (mode === m.id ? 'active' : '')}
                    onClick={() => setMode(m.id)}>{m.label}</button>
          ))}
        </div>

        <div className="search-row">
          <select className="scope-select" value={scope}
                  onChange={(e) => setScope(e.target.value === 'all' ? 'all' : Number(e.target.value))}>
            <option value="all">All videos</option>
            {processed.map((v) => <option key={v.video_id} value={v.video_id}>{v.filename}</option>)}
          </select>

          {mode === 'text' && (
            <input className="search-input" autoFocus value={query}
                   onChange={(e) => setQuery(e.target.value)} onKeyDown={onKey}
                   placeholder='Describe — e.g. "man in a blue shirt with a black backpack"' />
          )}
          {mode === 'image' && (
            <input type="file" accept="image/*" className="file-input"
                   onChange={(e) => setFile(e.target.files?.[0] || null)} />
          )}
          {mode === 'plate' && (
            <input className="search-input" value={plate}
                   onChange={(e) => setPlate(e.target.value.toUpperCase())} onKeyDown={onKey}
                   placeholder='Plate — e.g. "GJ05" or "AB1234"' />
          )}

          {mode === 'text' && (
            <div className="lang-toggle">
              {LANGS.map((l) => (
                <button key={l} className={language === l ? 'active' : ''} onClick={() => setLanguage(l)}>{l}</button>
              ))}
            </div>
          )}
          <button className="btn primary" onClick={run} disabled={loading}>{loading ? 'Searching…' : 'Search'}</button>
        </div>

        {mode === 'text' && results === null && (
          <div className="examples">
            {EXAMPLES.map((x) => <button key={x} className="chip" onClick={() => setQuery(x)}>{x}</button>)}
          </div>
        )}
      </div>

      {error && <div className="banner error">{error}</div>}

      {chips && chips.length > 0 && !loading && (
        <div className="parsed-chips">
          <span className="pc-label">Filtering for:</span>
          {chips.map((c, i) => (
            <span key={i} className={'pc ' + c.kind} title={c.kind === 'soft'
              ? 'Ranked visually (not a stored attribute we can filter exactly)'
              : 'Exact filter on a detected attribute'}>
              {c.kind === 'soft' ? '~ ' : ''}{c.label}
            </span>
          ))}
        </div>
      )}

      {meta && !loading && (
        <div className="result-meta">
          {meta.total} match{meta.total === 1 ? '' : 'es'}
          {scope !== 'all' ? ' in this video' : ' across all videos'} · {meta.elapsed} ms
          {meta.objectType ? <> · focused on <em>{meta.objectType}s</em></> : null}
          {'  ·  '}<button className="linklike" onClick={backToLibrary}>back to library</button>
        </div>
      )}
      {meta?.note && !loading && <div className="banner soft">{meta.note}</div>}

      {loading ? (
        <div className="loading">Searching footage…</div>
      ) : results !== null ? (
        results.length === 0 ? (
          <div className="empty">No matches{scope !== 'all' ? ' in this video' : ''}. Try different wording.</div>
        ) : (
          <div className="results-grid">
            {results.map((r) => (
              <button key={r.detection_id}
                      className={'result-card ' + (current?.key === 'res-' + r.detection_id ? 'active' : '')}
                      onClick={() => selectResult(r)}>
                <div className="thumb">
                  {r.crop_url ? <img src={r.crop_url} alt={r.class_label} loading="lazy" />
                              : <div className="thumb-empty">{r.class_label}</div>}
                  <span className={'score ' + scoreTier(r.score)}>{Math.round((r.score || 0) * 100)}%</span>
                </div>
                <div className="result-body">
                  <div className="result-top"><span className="label">{r.class_label}</span><span className="cam">{r.camera_id}</span></div>
                  <div className="attrs">{attrText(r.attributes)}</div>
                  {((r.matched && r.matched.length) || (r.soft && r.soft.length)) ? (
                    <div className="match-row">
                      {(r.matched || []).map((m, i) => <span key={'m' + i} className="mtag ok">✓ {m}</span>)}
                      {(r.soft || []).map((m, i) => <span key={'s' + i} className="mtag soft">~ {m}</span>)}
                    </div>
                  ) : null}
                  {r.track_appearances > 1 && (
                    <div className="seen">
                      👁 seen {r.track_appearances}× · {(r.visible_from || '').replace('T', ' ').slice(11, 19)}–{(r.visible_until || '').replace('T', ' ').slice(11, 19)}
                    </div>
                  )}
                  <div className="ts">
                    {r.timestamp ? r.timestamp.replace('T', ' ').slice(11, 19) : ''}
                    {r.offset_seconds != null ? '  ·  ⤿ ' + fmtDur(r.offset_seconds) : ''}
                  </div>
                </div>
              </button>
            ))}
          </div>
        )
      ) : (
        <div className="library">
          <div className="lib-head">
            <div className="section-label">
              Library — {(library || []).length} video{(library || []).length === 1 ? '' : 's'} · {processed.length} processed
            </div>
            <div className="lib-actions">
              <button className="btn ghost small"
                      onClick={() => setCamForm(camForm ? null : { cameraId: '', name: '', location: '', lat: '', lon: '' })}>
                {camForm ? 'Close' : '＋ Camera / GPS'}
              </button>
              {libMode === 'library' && unprocessed.length > 0 && !processing && (
                <button className="btn primary" onClick={processAll}>
                  Process {unprocessed.length} unprocessed
                </button>
              )}
              {processing && (
                <button className="btn danger" onClick={handleStop} disabled={stopping}>
                  {stopping ? 'Stopping…' : `■ Stop (${job.done}/${job.total})`}
                </button>
              )}
            </div>
          </div>

          {camForm && (
            <div className="cam-form">
              <input className="search-input" placeholder="Camera ID (e.g. CAM8)"
                     value={camForm.cameraId} onChange={(e) => setCamForm({ ...camForm, cameraId: e.target.value })} />
              <input className="search-input" placeholder="Name (e.g. Ring Road Jn)"
                     value={camForm.name} onChange={(e) => setCamForm({ ...camForm, name: e.target.value })} />
              <input className="search-input" placeholder="Location"
                     value={camForm.location} onChange={(e) => setCamForm({ ...camForm, location: e.target.value })} />
              <input className="search-input" placeholder="Lat (21.1959)"
                     value={camForm.lat} onChange={(e) => setCamForm({ ...camForm, lat: e.target.value })} />
              <input className="search-input" placeholder="Lon (72.8302)"
                     value={camForm.lon} onChange={(e) => setCamForm({ ...camForm, lon: e.target.value })} />
              <button className="btn primary" onClick={saveCamera}>Save</button>
            </div>
          )}

          {processing && (
            <div className="banner soft proc">
              <div className="proc-head">
                Processing <b>{job.current || ''}</b> — video {(job.done || 0) + 1} of {job.total}
              </div>
              <div className="progress-wrap">
                <div className="progress-bar"
                     style={{ width: (vpActive(job.video_progress)
                       ? job.video_progress.pct
                       : Math.round(((job.done || 0) / (job.total || 1)) * 100)) + '%' }} />
                <span className="progress-label">
                  {vpActive(job.video_progress)
                    ? `${job.video_progress.pct}% · ${stageText(job.video_progress)}`
                    : `${Math.round(((job.done || 0) / (job.total || 1)) * 100)}% · ${job.done || 0}/${job.total} videos done`}
                </span>
              </div>
            </div>
          )}

          {job && job.status === 'done' && <div className="banner soft">Done — processed {job.total} video(s).</div>}
          {job && job.status === 'stopped' && <div className="banner soft">Stopped — {job.done} of {job.total} video(s) were processed.</div>}

          {library === null ? (
            <div className="muted">Loading library…</div>
          ) : library.length === 0 ? (
            <div className="muted">No videos found in the folder. Add files to backend/data/videos.</div>
          ) : (
            <div className="lib-grid">
              {library.map((v) => (
                <button key={v.filename}
                        className={'lib-card ' + (v.processed ? '' : 'unprocessed ')
                          + (current?.key === 'rec-' + v.video_id ? 'active' : '')}
                        onClick={() => selectVideo(v)} disabled={!v.processed}
                        title={v.processed ? 'Watch / select' : 'Not processed yet'}>
                  <div className="lib-thumb">
                    {v.processed
                      ? <video src={v.url + '#t=1'} muted preload="metadata" />
                      : <div className="lib-thumb-empty">▶</div>}
                    <span className={'lib-badge ' + (v.processed ? 'ok' : 'no')}>
                      {v.processed ? 'Processed' : 'Not processed'}
                    </span>
                  </div>
                  <div className="lib-info">
                    <div className="lib-name" title={v.filename}>{v.filename}</div>
                    <div className="muted small">
                      {v.processed
                        ? (labelFor(v.camera_id) + (v.duration ? '  ·  ' + fmtDur(v.duration) : ''))
                        : (v.size_mb != null ? v.size_mb + ' MB' : '')}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function fmtDur(s) {
  if (s == null) return ''
  s = Math.round(s)
  return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0')
}
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

// Per-video progress helpers. The bar now covers the WHOLE pipeline (not just
// the detect+track pass), so it advances past ~40% into embed/re-id/faces/plates
// instead of freezing. Show the frame count while tracking, a stage name after.
function vpActive(vp) {
  return !!vp && ((vp.pct || 0) > 0 || (vp.total || 0) > 0)
}
function stageText(vp) {
  const s = vp.stage
  if ((!s || s === 'start' || s === 'detect+track') && (vp.total || 0) > 0) {
    return `frame ${vp.frame}/${vp.total}`
  }
  const map = {
    start: 'starting',
    'detect+track': 'detecting & tracking',
    clip: 'embedding crops',
    reid: 're-identifying people',
    store: 'saving detections',
    faces: 'reading faces',
    plates: 'reading plates',
    done: 'finishing',
  }
  return map[s] || s || 'processing'
}
