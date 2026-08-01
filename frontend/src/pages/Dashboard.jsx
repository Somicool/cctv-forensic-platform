// Dashboard - persistent investigation activity.
// Shows the persons AND vehicles that were searched, found, and tracked. The
// history is stored server-side (SQLite) so it survives reloads/restarts and is
// always accessible here. "Clear history" wipes it and recording starts fresh.
import { useEffect, useMemo, useState } from 'react'
import { getActivity, clearActivity, getVideos, getCameras } from '../api'
import VehicleInfo from '../components/VehicleInfo'
import { IcSearch } from '../components/icons'

const ACTION_LABEL = { searched: 'Searched', found: 'Found', tracked: 'Tracked' }

function StatCard({ n, label, warn }) {
  return (
    <div className="fp-card fp-stat">
      <div className="v" style={warn ? { color: 'var(--fp-warn)' } : null}>{n}</div>
      <div className="l">{label}</div>
    </div>
  )
}

function fmtWhen(iso) {
  if (!iso) return ''
  const d = new Date(iso); const diff = (Date.now() - d.getTime()) / 1000
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return d.toLocaleString()
}

function ActivityCard({ e, onOpen }) {
  const clickable = e.kind === 'vehicle' && e.plate
  return (
    <div className={'dash-act ' + (clickable ? 'clickable' : '')} onClick={clickable ? onOpen : undefined}
         title={clickable ? 'Open vehicle record' : ''}>
      <div className="dash-act-thumb">
        {e.crop_url ? <img src={e.crop_url} alt={e.label} loading="lazy" />
          : <div className="ph">{e.action === 'searched' ? '⌕' : (e.kind === 'person' ? '☺' : '▦')}</div>}
        <span className={'dash-act-badge ' + e.action}>{ACTION_LABEL[e.action] || e.action}</span>
      </div>
      <div className="dash-act-body">
        <div className="dash-act-top">
          <span className="lb">{e.label || (e.kind === 'person' ? 'person' : 'vehicle')}</span>
          <span className={'dash-kind ' + e.kind}>{e.kind || '—'}</span>
        </div>
        {e.plate && <div className="dash-act-plate">{e.plate}</div>}
        <div className="dash-act-meta">
          <span>{e.camera_id || (e.query ? `“${e.query}”` : '')}</span>
          <span className="t">{fmtWhen(e.created_at)}</span>
        </div>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [items, setItems] = useState(null)
  const [videos, setVideos] = useState([])
  const [cameras, setCameras] = useState([])
  const [q, setQ] = useState('')
  const [kind, setKind] = useState('all')       // all | person | vehicle
  const [act, setAct] = useState('all')         // all | searched | found | tracked
  const [regPlate, setRegPlate] = useState(null)
  const [busy, setBusy] = useState(false)

  async function load() {
    const [a, v, c] = await Promise.allSettled([getActivity(400), getVideos(), getCameras()])
    setItems(a.status === 'fulfilled' ? (a.value || []) : [])
    setVideos(v.status === 'fulfilled' ? (v.value || []) : [])
    setCameras(c.status === 'fulfilled' ? (c.value || []) : [])
  }
  useEffect(() => { load() }, [])

  async function clearAll() {
    if (!window.confirm('Clear all dashboard activity history? This cannot be undone — recording starts fresh from now.')) return
    setBusy(true)
    try { await clearActivity(); await load() } finally { setBusy(false) }
  }

  const filtered = useMemo(() => {
    let list = items || []
    if (kind !== 'all') list = list.filter((e) => e.kind === kind)
    if (act !== 'all') list = list.filter((e) => e.action === act)
    const s = q.trim().toLowerCase()
    if (s) list = list.filter((e) => [e.label, e.plate, e.camera_id, e.query, e.kind]
      .some((x) => (x || '').toLowerCase().includes(s)))
    return list
  }, [items, kind, act, q])

  const all = items || []
  const persons = all.filter((e) => e.kind === 'person').length
  const vehicles = all.filter((e) => e.kind === 'vehicle').length
  const tracked = all.filter((e) => e.action === 'tracked').length

  return (
    <div className="fp-page">
      <div className="fp-page-head">
        <div>
          <h1 className="fp-page-title">Dashboard</h1>
          <p className="fp-page-desc">Persons &amp; vehicles you searched, found, and tracked. History is saved and persists across sessions.</p>
        </div>
        <button className="fp-btn" onClick={clearAll} disabled={busy || all.length === 0}
                style={{ borderColor: 'var(--fp-danger)', color: '#ffb3bb' }}>🗑 Clear history</button>
      </div>

      <div className="fp-quicksearch">
        <IcSearch size={20} />
        <input placeholder="Search activity — person, vehicle, plate, camera, query…"
               value={q} onChange={(e) => setQ(e.target.value)} />
      </div>

      <div className="fp-stats">
        <StatCard n={all.length} label="Activity Records" />
        <StatCard n={persons} label="Persons" />
        <StatCard n={vehicles} label="Vehicles" />
        <StatCard n={tracked} label="Tracked" />
      </div>

      <section>
        <div className="fp-section-h">
          <h3>Investigation Activity</h3>
          <div className="dash-filters">
            {[['all', 'All'], ['person', 'Persons'], ['vehicle', 'Vehicles']].map(([id, l]) => (
              <button key={id} className={'dash-chip ' + (kind === id ? 'on' : '')} onClick={() => setKind(id)}>{l}</button>
            ))}
            <span className="dash-sep" />
            {[['all', 'All'], ['searched', 'Searched'], ['found', 'Found'], ['tracked', 'Tracked']].map(([id, l]) => (
              <button key={id} className={'dash-chip ' + (act === id ? 'on' : '')} onClick={() => setAct(id)}>{l}</button>
            ))}
          </div>
        </div>

        {items === null ? <div className="dash-empty">Loading activity…</div>
          : all.length === 0 ? (
            <div className="dash-empty">
              No activity yet. Go to the <b>Investigation Workspace</b>, run a search, and view or track results —
              every person and vehicle you search, find, and track is saved here <b>permanently</b>.
            </div>)
            : filtered.length === 0 ? <div className="dash-empty">No activity matches these filters.</div>
              : (
                <div className="dash-grid">
                  {filtered.map((e) => (
                    <ActivityCard key={e.id} e={e} onOpen={() => setRegPlate(e.plate)} />
                  ))}
                </div>)}
      </section>

      {regPlate && <VehicleInfo plate={regPlate} onClose={() => setRegPlate(null)} />}
    </div>
  )
}
