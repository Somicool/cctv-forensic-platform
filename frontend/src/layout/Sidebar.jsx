import { NavLink } from 'react-router-dom'
import {
  IcDashboard, IcWorkspace, IcEvidence, IcCase, IcSettings, IcShield, IcFace, IcJourney,
  IcCamera,
} from '../components/icons'

const NAV = [
  { to: '/dashboard', label: 'Dashboard', icon: IcDashboard },
  { to: '/workspace', label: 'Investigation Workspace', icon: IcWorkspace },
  { to: '/evidence', label: 'Evidence Gallery', icon: IcEvidence },
  { to: '/faces', label: 'Face Gallery', icon: IcFace },
  { to: '/journey', label: 'Journey', icon: IcJourney },
  { to: '/cameras', label: 'Camera Registry', icon: IcCamera },
  { to: '/case', label: 'Case File', icon: IcCase },
  { to: '/settings', label: 'Settings', icon: IcSettings },
]

const STATUS = {
  checking: ['var(--fp-warn)', 'Connecting…', ''],
  online: ['var(--fp-success)', 'AI engine online', ''],
  offline: ['var(--fp-danger)', 'Engine offline', ''],
}

export default function Sidebar({ health, onNavigate }) {
  const [color, label] = STATUS[health.state] || STATUS.offline
  const sub = health.state === 'online'
    ? `${(health.device || '').toUpperCase()}${health.gpu_vram_gb ? ` · ${health.gpu_vram_gb}GB` : ''}`
    : 'Smart City CCTV'

  return (
    <aside className="fp-sidebar">
      {/* NiriXan AI mark: shield badge + split wordmark. "NIRI" carries the
          weight, "XAN" the accent, and "AI" sits as a small suffix, with a thin
          rule underneath so the lockup reads as one logo rather than plain text. */}
      <div className="fp-brand vs-brand">
        <span className="vs-badge"><IcShield /></span>
        <span className="vs-mark">
          <span className="vs-word"><b>NIRI</b><i>XAN</i><u>AI</u></span>
          <span className="vs-rule" />
        </span>
      </div>

      <nav className="fp-nav">
        <div className="fp-nav-label">Investigation</div>
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={to} onClick={onNavigate}
                   className={({ isActive }) => 'fp-nav-item' + (isActive ? ' active' : '')}>
            <Icon />
            <span className="fp-nav-txt">{label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="fp-sidebar-foot">
        <div className="fp-status">
          <span className="fp-status-dot" style={{ background: color, color }} />
          <span className="fp-status-txt">
            <b>{label}</b>
            <span>{sub}</span>
          </span>
        </div>
      </div>
    </aside>
  )
}
