// Main screen: a library of all videos (Processed / Not-processed), a "Process
// all" button, a big player, and search that can be scoped to one video or run
// across all of them. Clicking a result jumps the player to that exact moment.
import { useEffect, useState } from 'react'
import {
  getVideos, getLibrary, ingestAll, getIngestJob,
  searchText, searchImage, searchPlate,
} from '../api'
import VideoPlayer from './VideoPlayer'
import TrackingView from './TrackingView'

const MODES = [
  { id: 'text', label: 'Describe' },
  { id: 'image', label: 'Image' },
  { id: 'plate', label: 'Plate' },
]
const LANGS = ['EN', 'HI', 'GU']
const EXAMPLES = ['a white truck', 'a red car', 'a person carrying a backpack', 'a man in a white shirt']

export default function SearchWatch({ cameras }) {
  const [library, setLibrary] = useState(null)   // [{filename, processed, video_id, url, duration, camera_id, size_mb}]
  const [libMode, setLibMode] = useState('library')  // 'library' (full folder) | 'videos' (processed only, fallback)
  const [current, setCurrent] = useState(null)
  const [scope, setScope] = useState('all')      // 'all' | video_id
  const [mode, setMode] = useState('text')
  const [query, setQuery] = useState('')
  const [language, setLanguage] = useState('EN')
  const [plate, setPlate] = useState('')
  const [file, setFile] = useState(null)
  const [results, setResults] = useState(null)   // null = idle (show library)
  const [meta, setMeta] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [job, setJob] = useState(null)
  const [tracking, setTracking] = useState(null)   // detection_id being traced, or null

  const nameFor = (id) => {
    const n = (cameras || []).find((c) => c.camera_id === id)?.name
    return (!n || n === id) ? '' : n
  }

  async function loadLibrary(selectFirst = false) {
    let items
    try {
      items = await getLibrary()
      setLibMode('library')
    } catch {
      // /api/library not registered yet -> show processed videos only
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
      if (mode === 'text') data = await searchText({ query, language: language.toLowerCase(), includeScenes: false, filters })
      else if (mode === 'image') data = await searchImage({ file })
      else data = await searchPlate({ plate, filters })
      let res = data.results || []
      if (scope !== 'all') res = res.filter((r) => r.video_id === scope)  // scope client-side too (covers image)
      setResults(res)
      setMeta({
        total: res.length, note: data.note, objectType: data.object_type,
        translated: data.translated_query, elapsed: Math.round(performance.now() - t0),
      })
      if (res.length) selectResult(res[0])
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || 'Search failed')
      setResults([]); setMeta(null)
    } finally {
      setLoading(false)
    }
  }

  async function processAll() {
    setError(null)
    try {
      const r = await ingestAll()
      if (!r.job_id) { await loadLibrary(); return }
      setJob({ job_id: r.job_id, status: 'processing', done: 0, total: r.total, current: null })
    } catch (e) {
      setError('Could not start processing. ' + (e?.response?.status === 404
        ? 'The processing endpoint is not enabled yet (see the 2-line backend note).'
        : (e.message || '')))
    }
  }

  useEffect(() => {
    if (!job || job.status !== 'processing') return
    const t = setInterval(async () => {
      try {
        const j = await getIngestJob(job.job_id)
        setJob(j)
        if (j.status === 'done' || j.status === 'error') {
          clearInterval(t)
          loadLibrary()
        }
      } catch { clearInterval(t) }
    }, 2500)
    return () => clearInterval(t)
    // eslint-disable-next-line
  }, [job?.job_id, job?.status])

  const processed = (library || []).filter((v) => v.processed)
  const unprocessed = (library || []).filter((v) => !v.processed)
  const processing = job && job.status === 'processing'
  const onKey = (e) => { if (e.key === 'Enter') run() }

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
                   placeholder='Search — e.g. "a white truck", "person with a red backpack"' />
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

      {meta && !loading && (
        <div className="result-meta">
          {meta.total} match{meta.total === 1 ? '' : 'es'}
          {scope !== 'all' ? ' in this video' : ' across all videos'} · {meta.elapsed} ms
          {meta.objectType ? <> · focused on <em>{meta.objectType}s</em></> : null}
          {meta.translated ? <> · translated to <em>“{meta.translated}”</em></> : null}
          {'  ·  '}<button className="linklike" onClick={() => { setResults(null); setMeta(null) }}>back to library</button>
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
            {libMode === 'library' && unprocessed.length > 0 && (
              <button className="btn primary" onClick={processAll} disabled={processing}>
                {processing ? `Processing ${job.done}/${job.total}…` : `Process ${unprocessed.length} unprocessed`}
              </button>
            )}
          </div>

          {processing && (
            <div className="banner soft">Processing {job.current || ''}…  ({job.done}/{job.total} done). This can take a bit per video.</div>
          )}
          {job && job.status === 'done' && <div className="banner soft">Done — processed {job.total} video(s).</div>}

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
