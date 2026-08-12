import { useLocation } from 'react-router-dom'

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

// The bar only breadcrumbs the current page. The global search box, the avatar
// chip and the hamburger toggle have all been removed: search belongs to the
// Investigation Workspace where it has filters and scope, the initials chip
// carried no information, and the toggle did nothing on a desktop layout where
// the sidebar is always visible.
export default function TopBar() {
  const { pathname } = useLocation()
  const [crumb, title] = TITLES[pathname] || ['', 'NiriXan AI']

  return (
    <header className="fp-topbar">
      <div className="fp-crumb">
        <span className="fp-crumb-top">{crumb}</span>
        <span className="fp-crumb-title">{title}</span>
      </div>
      <div className="fp-topbar-spacer" />
    </header>
  )
}
