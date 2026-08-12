// Command Dashboard — the investigating officer's overview of the whole system.
//
// This is deliberately NOT a media page: no uploaded clips, no video thumbnails,
// no gallery. Footage is reached by opening an investigation. What it answers at
// a glance is operational — what the system is doing, which investigations exist,
// what needs attention, and where to start.
//
// Every figure comes from an EXISTING endpoint (health, system info, journeys,
// camera registry, saved case, exports, reports, saved faces, activity history,
// videos). Nothing is fabricated: each number is derived from real records, and
// where there is no data the panel shows an explicit empty state.
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  getHealth, getSystemInfo, listJourneys, listCameraRegistry, getRegistryStatus,
  loadCase, getExports, getCaseReports, listSavedFaces, getActivity, getVideos,
} from '../api'
import CameraMap from '../components/CameraMap'
import {
  IcShield, IcSettings, IcSearch, IcCamera, IcEvidence, IcFace, IcJourney,
  IcCase, IcClock, IcPlus, IcWorkspace,
} from '../components/icons'
import '../styles/command-dashboard.css'

/* ------------------------------------------------------------------ helpers */
const pad3 = (n) => String(n ?? '').padStart(3, '0')

function useClock() {
  const [now, setNow] = useState(() => new Date())
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 30000)
    return () => clearInterval(t)
  }, [])
  return now
}

function ago(iso) {
  if (!iso) return '—'
  const t = new Date(iso).getTime()
  if (Number.isNaN(t)) return '—'
  const s = (Date.now() - t) / 1000
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)} min ago`
  if (s < 86400) return `${Math.floor(s / 3600)} hr ago`
  if (s < 604800) return `${Math.floor(s / 86400)} d ago`
  return new Date(iso).toLocaleDateString()
}

function clockTime(iso) {
  if (!iso) return '--:--'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '--:--'
    : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })
}

const within = (iso, hours) => {
  const t = new Date(iso || 0).getTime()
  return !Number.isNaN(t) && (Date.now() - t) < hours * 3600 * 1000
}

// Journey confidence -> the tier an officer should act on. The real percentage is
// always shown beside it, so the badge never hides the underlying number.
function tierOf(conf) {
  const c = Number(conf) || 0
  if (c >= 0.75) return { key: 'confirmed', label: 'Confirmed' }
  if (c >= 0.5) return { key: 'review', label: 'Review' }
  return { key: 'weak', label: 'Weak' }
}

const fmtSpan = (s) => {
  const v = Number(s)
  if (!Number.isFinite(v) || v <= 0) return '—'
  if (v < 60) return `${v.toFixed(1)}s`
  const mn = Math.floor(v / 60)
  return mn < 60 ? `${mn}m ${Math.round(v % 60)}s` : `${Math.floor(mn / 60)}h ${mn % 60}m`
}

/* ------------------------------------------------------------- small pieces */
function Stat({ icon: Icon, label, value, unit, sub, subTone, tone }) {
  return (
    <div className={'cd-stat' + (tone ? ' ' + tone : '')}>
      <div className="cd-stat-h"><Icon /><span>{label}</span></div>
      <div className="cd-stat-v">{value}{unit ? <small>{unit}</small> : null}</div>
      <div className={'cd-stat-sub' + (subTone ? ' ' + subTone : '')}>{sub}</div>
    </div>
  )
}

function Panel({ icon: Icon, title, count, link, onLink, children, bodyPad }) {
  return (
    <section className="cd-panel">
      <div className="cd-panel-h">
        {Icon ? <Icon /> : null}
        <h3>{title}</h3>
        {link
          ? <a className="cd-link" href="#" onClick={(e) => { e.preventDefault(); onLink?.() }}>{link}</a>
          : (count != null ? <span className="cd-count">{count}</span> : null)}
      </div>
      <div className={'cd-panel-b' + (bodyPad ? ' pad' : '')}>{children}</div>
    </section>
  )
}

function SysRow({ label, state, value }) {
  return (
    <div className="cd-sysrow">
      <span className="k">{label}</span>
      <span className={'cd-dot ' + (state || '')} />
      <span className="v">{value}</span>
    </div>
  )
}

/* ================================================================ dashboard */
export default function Dashboard() {
  const nav = useNavigate()
  const now = useClock()
  const [d, setD] = useState(null)          // null until the first load resolves
  const [err, setErr] = useState(false)

  const load = useCallback(async () => {
    const r = await Promise.allSettled([
      getHealth(), getSystemInfo(), listJourneys(), listCameraRegistry(),
      getRegistryStatus(), loadCase(), getExports(), getCaseReports(),
      listSavedFaces(), getActivity(60), getVideos(),
    ])
    const v = (i, fb) => (r[i].status === 'fulfilled' && r[i].value != null ? r[i].value : fb)
    setErr(r[0].status !== 'fulfilled')
    setD({
      health: v(0, {}), info: v(1, {}), journeys: v(2, []), registry: v(3, []),
      regStatus: v(4, {}), caseData: v(5, {}), exports: v(6, []), reports: v(7, []),
      faces: v(8, []), activity: v(9, []), videos: v(10, []),
    })
  }, [])
  useEffect(() => { load() }, [load])

  /* ---------------------------------------------------- derived, real values */
  const m = useMemo(() => {
    if (!d) return null
    const cams = d.registry || []
    const camsTotal = cams.length
    const camsOnline = cams.filter((c) => c.active !== false).length
    const camsGps = cams.filter((c) => c.has_gps).length

    const journeys = d.journeys || []
    const st = d.info?.storage || {}
    const caseInfo = d.caseData?.case_info || {}
    const openEvidence = (d.caseData?.evidence || []).length
    const sealed = d.exports || []
    const acts = d.activity || []

    return {
      camsTotal, camsOnline, camsGps,
      camsOffline: camsTotal - camsOnline,
      noGps: (d.regStatus?.without_gps || []).length,
      journeys,
      journeys24: journeys.filter((j) => within(j.created_at, 24)).length,
      needReview: journeys.filter((j) => (Number(j.confidence) || 0) < 0.75).length,
      openEvidence,
      sealed,
      sealedEvidence: sealed.reduce((a, x) => a + (x.detection_ids || []).length, 0),
      caseInfo,
      caseOpen: !!(openEvidence || (caseInfo.caseNumber || '').trim() || (caseInfo.title || '').trim()),
      tracked: acts.filter((a) => a.action === 'tracked').length,
      found: acts.filter((a) => a.action === 'found').length,
      pending: (d.videos || []).filter((x) => x.status && x.status !== 'done').length,
      storageMb: ['videos', 'frames', 'crops', 'saved_faces', 'exports']
        .reduce((a, k) => a + (st[k]?.mb || 0), 0) + (st.db_mb || 0),
      dbMb: st.db_mb || 0,
      faiss: d.info?.faiss || {},
      geminiKey: d.info?.gemini_key_present,
      reports: d.reports || [],
      faces: d.faces || [],
      acts,
    }
  }, [d])

  /* --------------------------------------------- attention items (all real) */
  const attention = useMemo(() => {
    if (!m) return []
    const out = []
    if (m.camsOffline > 0) out.push({
      tone: 'crit', ic: '⚠', to: '/cameras',
      text: `${m.camsOffline} camera${m.camsOffline > 1 ? 's' : ''} marked inactive`,
      sub: 'Inactive cameras are excluded from search scope' })
    if (m.noGps > 0) out.push({
      tone: 'warn', ic: '⚠', to: '/cameras',
      text: `${m.noGps} camera${m.noGps > 1 ? 's' : ''} without geolocation`,
      sub: 'Journey reconstruction needs coordinates for every camera on the route' })
    if (m.needReview > 0) out.push({
      tone: 'warn', ic: '⚠', to: '/journey',
      text: `${m.needReview} investigation${m.needReview > 1 ? 's' : ''} below confirmed confidence`,
      sub: 'Cross-camera matches under 75% need an officer decision' })
    if (m.pending > 0) out.push({
      tone: 'warn', ic: '⚠', to: '/workspace',
      text: `${m.pending} recording${m.pending > 1 ? 's' : ''} awaiting processing`,
      sub: 'Not yet searchable' })
    if (m.openEvidence > 0) out.push({
      tone: 'info', ic: '◆', to: '/case',
      text: `${m.openEvidence} exhibit${m.openEvidence > 1 ? 's' : ''} in the open case not yet sealed`,
      sub: 'Seal the case to fix its SHA-256 chain of custody' })
    if (m.geminiKey === false) out.push({
      tone: 'info', ic: '◆', to: '/settings',
      text: 'Report narration key not configured',
      sub: 'Evidence reports build from recorded attributes only' })
    if (m.sealed.length > 0) out.push({
      tone: 'ok', ic: '✓', to: '/case',
      text: `${m.sealed.length} case${m.sealed.length > 1 ? 's' : ''} sealed`,
      sub: `${m.reports.length} evidence report${m.reports.length === 1 ? '' : 's'} issued` })
    if (!out.length) out.push({ tone: 'ok', ic: '✓', text: 'No outstanding items', sub: 'All checks clear' })
    return out
  }, [m])

  const alerts = attention.filter((a) => a.tone === 'crit' || a.tone === 'warn').length
  const online = !!d && !err && (d.health?.status === 'ok' || d.health?.state === 'online')
  const gpsCams = useMemo(() => (d?.registry || []).filter((c) => c.has_gps), [d])

  const go = (to) => (e) => { e.preventDefault(); nav(to) }

  /* --------------------------------------------------------------- rendering */
  return (
    <div className="fp-page cd">
      {/* ---------------------------------------------- command header */}
      <header className="cd-head">
        <div className="cd-head-mark">
          <span className="cd-shield"><IcShield /></span>
          <span className="cd-wordmark">
            <b>NIRIXAN AI</b>
            <span>AI Forensic Investigation Platform</span>
          </span>
        </div>
        <div className="cd-head-spacer" />
        <div className="cd-head-meta">
          <div className="cd-clock">
            <b>{now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })}</b>
            <span>{now.toLocaleDateString([], { weekday: 'short', day: '2-digit', month: 'short', year: 'numeric' })}</span>
          </div>
          <span className={'cd-sys ' + (d === null ? 'wait' : online ? '' : 'off')}>
            <i />{d === null ? 'Connecting' : online ? 'System Operational' : 'Engine Offline'}
          </span>
          <div className="cd-officer">
            <span className="cd-officer-badge">
              {(m?.caseInfo?.officer || 'IN').trim().slice(0, 2).toUpperCase() || 'IN'}
            </span>
            <span className="cd-officer-txt">
              <b>{m?.caseInfo?.officer?.trim() || 'Investigating Officer'}</b>
              <span>{m?.caseInfo?.caseNumber?.trim() ? `Case ${m.caseInfo.caseNumber}` : 'No case assigned'}</span>
            </span>
          </div>
          <button className="cd-head-btn" title="Settings" onClick={() => nav('/settings')}>
            <IcSettings />
          </button>
        </div>
      </header>

      {/* ------------------------------- section 1 · command overview */}
      <div className="cd-stats">
        <Stat icon={IcJourney} label="Investigations" value={m ? m.journeys.length : '—'}
              sub={m ? (m.journeys24 ? `+${m.journeys24} in last 24 h` : 'none in last 24 h') : ' '}
              subTone={m?.journeys24 ? 'up' : ''} />
        <Stat icon={IcCase} label="Cases" value={m ? m.sealed.length : '—'}
              sub={m ? (m.caseOpen ? '1 open · rest sealed' : 'all sealed') : ' '} />
        <Stat icon={IcCamera} label="Cameras Online" value={m ? m.camsOnline : '—'}
              unit={m ? ` / ${m.camsTotal}` : ''} tone={m && m.camsOffline ? 'warn' : 'ok'}
              sub={m ? `${m.camsGps} geolocated` : ' '} />
        <Stat icon={IcEvidence} label="Evidence Items"
              value={m ? m.openEvidence + m.sealedEvidence : '—'}
              sub={m ? `${m.faces.length} saved face${m.faces.length === 1 ? '' : 's'}` : ' '} />
        <Stat icon={IcFace} label="Identity Matches" value={m ? m.tracked : '—'}
              sub={m ? `${m.found} sighting${m.found === 1 ? '' : 's'} confirmed` : ' '} />
        <Stat icon={IcShield} label="Alerts" value={m ? alerts : '—'}
              tone={alerts ? (m && m.camsOffline ? 'crit' : 'warn') : 'ok'}
              sub={m ? (alerts ? 'requires attention' : 'all clear') : ' '}
              subTone={alerts ? 'attn' : 'up'} />
      </div>

      {/* --------------- primary band: investigations + how to start ------- */}
      <div className="cd-grid">
        {/* ---------------------- section 2 · active investigations */}
        <Panel icon={IcJourney} title="Active Investigations"
               link="Open Journey →" onLink={() => nav('/journey')}>
          {d === null ? <div className="cd-empty">Loading investigations…</div>
            : m.journeys.length === 0 ? (
              <div className="cd-empty">
                No investigations reconstructed yet.<br />
                Search for a person in the <b>Investigation Workspace</b>, then run
                <b> Journey Reconstruction</b> to trace them across cameras.
              </div>
            ) : (
              <div className="cd-tbl-wrap">
                <table className="cd-tbl">
                  <thead>
                    <tr>
                      <th>Case ID</th><th>Investigation</th><th>Status</th>
                      <th>Cameras</th><th>Span</th><th>Distance</th><th>Last activity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {/* Render them all — the panel shows as many as fit the
                        space it has and scrolls for the rest, so no room is
                        left empty. */}
                    {m.journeys.map((j) => {
                      const t = tierOf(j.confidence)
                      return (
                        <tr key={j.journey_id} onClick={() => nav('/journey')}
                            title="Open in Journey Reconstruction">
                          <td className="cd-cid">VS-{pad3(j.journey_id)}</td>
                          <td className="cd-name">
                            {j.investigation?.trim() || `Subject #${j.reference_detection_id}`}
                          </td>
                          <td>
                            <span className={'cd-badge ' + t.key}>{t.label}</span>
                            <span className="dim num"> {Math.round((Number(j.confidence) || 0) * 100)}%</span>
                          </td>
                          <td className="num">{j.camera_count ?? '—'}</td>
                          <td className="num dim">{fmtSpan(j.span_seconds)}</td>
                          <td className="num dim">
                            {j.distance_km != null ? `${Number(j.distance_km).toFixed(2)} km` : '—'}
                          </td>
                          <td className="dim">{ago(j.created_at)}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
        </Panel>

        <div className="cd-col">
          {/* ------------------- section 3 · start investigation */}
          <Panel icon={IcSearch} title="Start Investigation" bodyPad>
            <p className="cd-start-p">
              Search CCTV footage by natural-language description, number plate,
              appearance attributes or a saved identity.
            </p>
            <button className="cd-cta" onClick={() => nav('/workspace')}>
              <IcPlus /> New Investigation
            </button>
            {/* All three open the unified search surface in the Investigation
                Workspace, which is where person, vehicle and plate search live.
                No new workflow is introduced. */}
            <div className="cd-actions">
              <a className="cd-act" href="/workspace" onClick={go('/workspace')}>
                <IcWorkspace /><span>Search Person</span></a>
              <a className="cd-act" href="/workspace" onClick={go('/workspace')}>
                <IcSearch /><span>Search Vehicle</span></a>
              <a className="cd-act" href="/workspace" onClick={go('/workspace')}>
                <IcSearch /><span>Search Plate</span></a>
            </div>
            <div className="cd-jump">
              <a href="/faces" onClick={go('/faces')}><IcFace /> Face Gallery</a>
              <a href="/evidence" onClick={go('/evidence')}><IcEvidence /> Evidence</a>
              <a href="/case" onClick={go('/case')}><IcCase /> Case File</a>
            </div>
          </Panel>

          {/* ---------------------- section 6 · requires attention */}
          <Panel icon={IcShield} title="Requires Attention"
                 count={m ? `${alerts} alert${alerts === 1 ? '' : 's'}` : null}>
            <div className="cd-attn">
              {d === null ? <div className="cd-empty">Running checks…</div>
                : attention.map((a, i) => {
                  const Row = a.to ? 'a' : 'div'
                  return (
                    <Row key={i} className={'cd-attn-row ' + a.tone}
                         {...(a.to ? { href: a.to, onClick: go(a.to) } : {})}>
                      <span className="cd-attn-ic">{a.ic}</span>
                      <span className="cd-attn-txt"><b>{a.text}</b><span>{a.sub}</span></span>
                    </Row>
                  )
                })}
            </div>
          </Panel>
        </div>
      </div>

      {/* ------------ operational band: activity · status · coverage ------- */}
      <div className="cd-grid3">
        {/* --------------- section 5 · recent investigative activity */}
        <Panel icon={IcClock} title="Recent Activity"
               count={m ? `${m.acts.length} records` : null}>
          {d === null ? <div className="cd-empty">Loading activity…</div>
            : m.acts.length === 0 ? (
              <div className="cd-empty">
                No investigative activity recorded yet. Searches, identifications and
                cross-camera tracking appear here as they happen.
              </div>
            ) : (
              <div className="cd-feed">
                {/* same here: fill the panel, scroll for the remainder */}
                {m.acts.slice(0, 40).map((a) => (
                  <div className="cd-feed-row" key={a.id}>
                    <span className="cd-feed-t">{clockTime(a.created_at)}</span>
                    <span className={'cd-feed-ic ' + (a.action || '')}>
                      {a.action === 'tracked' ? '⤳' : a.action === 'found' ? '◉' : '⌕'}
                    </span>
                    <span className="cd-feed-txt">
                      {a.action === 'tracked'
                        ? <><b>{a.kind === 'person' ? 'Person' : 'Vehicle'}</b> tracked across cameras</>
                        : a.action === 'found'
                          ? <><b>{a.kind === 'person' ? 'Person' : 'Vehicle'}</b> identified in footage</>
                          : <>Search executed{a.query ? <> — “{a.query}”</> : null}</>}
                      {a.camera_id ? <span className="dim"> · {a.camera_id}</span> : null}
                    </span>
                    <span className="cd-feed-ref">{a.plate || (a.ref ? `#${a.ref}` : '—')}</span>
                  </div>
                ))}
              </div>
            )}
        </Panel>

        {/* ------------------------ section 4 · live system status */}
        <Panel icon={IcSettings} title="System Status">
          <div className="cd-sysgrid">
            {d === null ? <div className="cd-empty">Reading system state…</div> : (
              <>
                <SysRow label="Cameras" state={m.camsOffline ? 'warn' : 'ok'}
                        value={`${m.camsOnline} / ${m.camsTotal} online`} />
                <SysRow label="Processing" state={m.pending ? 'warn' : 'ok'}
                        value={m.pending ? `${m.pending} queued` : 'Idle'} />
                <SysRow label="Database" state="ok" value={`Healthy · ${m.dbMb} MB`} />
                <SysRow label="AI Engine" state={online ? 'ok' : 'crit'}
                        value={online
                          ? `${(d.health.device || '').toUpperCase()}${d.health.gpu_vram_gb ? ` · ${d.health.gpu_vram_gb} GB` : ''}`
                          : 'Offline'} />
                <SysRow label="Search Index" state="ok"
                        value={`${(m.faiss.clip || 0).toLocaleString()} vectors`} />
                <SysRow label="Storage" state="ok"
                        value={`${(m.storageMb / 1024).toFixed(2)} GB used`} />
              </>
            )}
          </div>
        </Panel>

        {/* ----------------------------- optional · camera coverage */}
        <Panel icon={IcCamera} title="Camera Coverage"
               link="Registry →" onLink={() => nav('/cameras')}>
          {d === null ? <div className="cd-empty">Loading camera registry…</div>
            : gpsCams.length === 0 ? (
              <div className="cd-empty">
                <b>Camera geolocation not configured.</b><br />
                Add coordinates in the{' '}
                <a href="/cameras" onClick={go('/cameras')}>Camera Registry</a> to place
                cameras on the map and enable journey reconstruction.
              </div>
            ) : (
              <div className="cd-map">
                <CameraMap cameras={gpsCams} height={150} />
                <div className="cd-map-foot">
                  {gpsCams.length} of {m.camsTotal} cameras geolocated
                  {m.noGps ? ` · ${m.noGps} awaiting coordinates` : ''}
                </div>
              </div>
            )}
        </Panel>
      </div>
    </div>
  )
}
