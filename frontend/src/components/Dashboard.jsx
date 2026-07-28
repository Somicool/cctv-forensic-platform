// Main search dashboard: Describe (text + EN/HI/GU) / Image upload / Plate,
// with filters and a results grid.
import { useState } from 'react'
import { searchText, searchImage, searchPlate } from '../api'
import Filters from './Filters'
import ResultsGrid from './ResultsGrid'
import ResultDetail from './ResultDetail'

const MODES = [
  { id: 'text', label: 'Describe' },
  { id: 'image', label: 'Image' },
  { id: 'plate', label: 'Plate' },
]
const LANGS = ['EN', 'HI', 'GU']
const EXAMPLES = [
  'a white truck',
  'red hatchback',
  'a person wearing a backpack',
  'silver SUV',
]

export default function Dashboard({ cameras, onOpenTrack, onAddToCase }) {
  const [mode, setMode] = useState('text')
  const [query, setQuery] = useState('')
  const [language, setLanguage] = useState('EN')
  const [plate, setPlate] = useState('')
  const [file, setFile] = useState(null)
  const [useReid, setUseReid] = useState(false)
  const [filters, setFilters] = useState({})
  const [showFilters, setShowFilters] = useState(false)
  const [results, setResults] = useState(null)
  const [meta, setMeta] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState(null)

  async function run() {
    setError(null)
    if (mode === 'text' && !query.trim()) return
    if (mode === 'plate' && !plate.trim()) return
    if (mode === 'image' && !file) { setError('Choose an image to search with.'); return }

    setLoading(true)
    const t0 = performance.now()
    try {
      let data
      if (mode === 'text') {
        data = await searchText({ query, language: language.toLowerCase(),
                                  includeScenes: false, filters })
      } else if (mode === 'image') {
        data = await searchImage({ file, useReid })
      } else {
        data = await searchPlate({ plate, filters })
      }
      setResults(data.results || [])
      setMeta({
        total: data.total,
        translated: data.translated_query,
        note: data.note,
        objectType: data.object_type,
        elapsed: Math.round(performance.now() - t0),
      })
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || 'Search failed')
      setResults([])
      setMeta(null)
    } finally {
      setLoading(false)
    }
  }

  const onKey = (e) => { if (e.key === 'Enter') run() }
  const nFilters = countFilters(filters)

  return (
    <div className="dash">
      <div className="search-panel">
        <div className="mode-tabs">
          {MODES.map((m) => (
            <button key={m.id}
                    className={`mode-tab ${mode === m.id ? 'active' : ''}`}
                    onClick={() => setMode(m.id)}>
              {m.label}
            </button>
          ))}
        </div>

        {mode === 'text' && (
          <div className="search-row">
            <input className="search-input" autoFocus value={query}
                   onChange={(e) => setQuery(e.target.value)} onKeyDown={onKey}
                   placeholder='Describe what to find — e.g. "a white truck" or "person with a red backpack"' />
            <div className="lang-toggle">
              {LANGS.map((l) => (
                <button key={l} className={language === l ? 'active' : ''}
                        onClick={() => setLanguage(l)}>{l}</button>
              ))}
            </div>
            <button className="btn primary" onClick={run} disabled={loading}>
              {loading ? 'Searching…' : 'Search'}
            </button>
          </div>
        )}

        {mode === 'image' && (
          <div className="search-row">
            <input type="file" accept="image/*" className="file-input"
                   onChange={(e) => setFile(e.target.files?.[0] || null)} />
            <label className="checkbox">
              <input type="checkbox" checked={useReid}
                     onChange={(e) => setUseReid(e.target.checked)} />
              Person re-ID (OSNet)
            </label>
            <button className="btn primary" onClick={run} disabled={loading}>
              {loading ? 'Searching…' : 'Search'}
            </button>
          </div>
        )}

        {mode === 'plate' && (
          <div className="search-row">
            <input className="search-input" value={plate}
                   onChange={(e) => setPlate(e.target.value.toUpperCase())} onKeyDown={onKey}
                   placeholder='Full or partial plate — e.g. "GJ05" or "AB1234"' />
            <button className="btn primary" onClick={run} disabled={loading}>
              {loading ? 'Searching…' : 'Search'}
            </button>
          </div>
        )}

        <div className="panel-row">
          <button className="btn ghost" onClick={() => setShowFilters((s) => !s)}>
            {showFilters ? 'Hide filters' : 'Filters'}{nFilters ? ` (${nFilters})` : ''}
          </button>
          {mode === 'text' && (
            <div className="examples">
              {EXAMPLES.map((x) => (
                <button key={x} className="chip" onClick={() => setQuery(x)}>{x}</button>
              ))}
            </div>
          )}
        </div>

        {showFilters && <Filters cameras={cameras} filters={filters} onChange={setFilters} />}
      </div>

      {error && <div className="banner error">{error}</div>}

      {meta && !loading && (
        <>
          <div className="result-meta">
            {meta.total} result{meta.total === 1 ? '' : 's'} · {meta.elapsed} ms
            {meta.objectType ? <> · focused on <em>{meta.objectType}s</em></> : null}
            {meta.translated ? <> · translated to <em>“{meta.translated}”</em></> : null}
          </div>
          {meta.note && <div className="banner soft">{meta.note}</div>}
        </>
      )}

      {loading ? (
        <div className="loading">Searching footage…</div>
      ) : results && results.length === 0 ? (
        <div className="empty">No matches. Try a different description or relax the filters.</div>
      ) : (
        <ResultsGrid results={results || []} onSelect={setSelected} />
      )}

      {selected && (
        <ResultDetail item={selected} onClose={() => setSelected(null)}
                      onOpenTrack={onOpenTrack} onAddToCase={onAddToCase} />
      )}
    </div>
  )
}

function countFilters(f) {
  let n = 0
  if (f.cameras?.length) n++
  if (f.start_time) n++
  if (f.end_time) n++
  if (f.object_type) n++
  if (f.colors?.length) n++
  if (f.vehicle_type) n++
  if (f.min_confidence) n++
  return n
}
