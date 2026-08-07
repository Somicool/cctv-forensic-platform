import { useEffect, useState } from 'react'
import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { getHealth } from './api'
import Sidebar from './layout/Sidebar'
import TopBar from './layout/TopBar'
import Dashboard from './pages/Dashboard'
import Workspace from './pages/Workspace'
import Evidence from './pages/Evidence'
import FaceGallery from './pages/FaceGallery'
import Journey from './pages/Journey'
import CameraRegistry from './pages/CameraRegistry'
import CaseFile from './pages/CaseFile'
import Settings from './pages/Settings'

export default function App() {
  const [health, setHealth] = useState({ state: 'checking' })
  const [navOpen, setNavOpen] = useState(false)   // mobile off-canvas sidebar
  const location = useLocation()

  useEffect(() => {
    getHealth().then((d) => setHealth({ state: 'online', ...d }))
               .catch(() => setHealth({ state: 'offline' }))
  }, [])

  // close the mobile drawer whenever the route changes
  useEffect(() => { setNavOpen(false) }, [location.pathname])

  return (
    <div className={'fp-app' + (navOpen ? ' nav-open' : '')}>
      <div className="fp-scrim" onClick={() => setNavOpen(false)} />
      <Sidebar health={health} onNavigate={() => setNavOpen(false)} />
      <div className="fp-main">
        <TopBar onMenu={() => setNavOpen((v) => !v)} />
        <main className="fp-content">
          {/* keyed by path so each page replays its enter transition */}
          <div key={location.pathname}>
            <Routes location={location}>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/workspace" element={<Workspace />} />
              <Route path="/evidence" element={<Evidence />} />
              <Route path="/faces" element={<FaceGallery />} />
              <Route path="/journey" element={<Journey />} />
              <Route path="/cameras" element={<CameraRegistry />} />
              <Route path="/case" element={<CaseFile />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </div>
        </main>
      </div>
    </div>
  )
}
