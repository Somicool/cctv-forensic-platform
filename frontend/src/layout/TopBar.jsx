import { useLocation } from 'react-router-dom'
import { IcMenu, IcSearch } from '../components/icons'

const TITLES = {
  '/dashboard': ['Overview', 'Dashboard'],
  '/workspace': ['Analysis', 'Investigation Workspace'],
  '/evidence': ['Media', 'Evidence Gallery'],
  '/case': ['Records', 'Case File'],
  '/settings': ['System', 'Settings'],
}

export default function TopBar({ onMenu }) {
  const { pathname } = useLocation()
  const [crumb, title] = TITLES[pathname] || ['', 'Sentinel']

  return (
    <header className="fp-topbar">
      <button className="fp-icon-btn" onClick={onMenu} aria-label="Toggle navigation">
        <IcMenu />
      </button>
      <div className="fp-crumb">
        <span className="fp-crumb-top">{crumb}</span>
        <span className="fp-crumb-title">{title}</span>
      </div>

      <div className="fp-topbar-spacer" />

      <div className="fp-topsearch">
        <IcSearch />
        <input placeholder="Search evidence, plates, descriptions…" />
      </div>
      <div className="fp-avatar" title="Investigator">IN</div>
    </header>
  )
}
