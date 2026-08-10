import { useLocation } from 'react-router-dom'
import { IcMenu } from '../components/icons'

const TITLES = {
  '/dashboard': ['Overview', 'Command Dashboard'],
  '/workspace': ['Analysis', 'Investigation Workspace'],
  '/evidence': ['Evidence', 'Evidence Gallery'],
  '/faces': ['Identities', 'Face Gallery'],
  '/journey': ['Analysis', 'Journey Reconstruction'],
  '/cameras': ['Infrastructure', 'Camera Registry'],
  '/case': ['Records', 'Case File'],
  '/settings': ['System', 'Settings'],
}

// The global search box and the avatar chip were removed: search belongs to the
// Investigation Workspace, where it has filters and scope, and a decorative
// initials chip carried no information. The bar now only breadcrumbs the page and
// exposes the navigation toggle.
export default function TopBar({ onMenu }) {
  const { pathname } = useLocation()
  const [crumb, title] = TITLES[pathname] || ['', 'VigilSense']

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
    </header>
  )
}
