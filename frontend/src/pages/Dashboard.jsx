// Dashboard skeleton — layout only (no data/logic yet).
// Sections: Quick Search bar, Create New Investigation, Active Investigations
// (investigation cards), and Recent Activity.
import { IcSearch, IcPlus, IcFolder, IcClock } from '../components/icons'

function InvestigationCardSkeleton() {
  return (
    <div className="fp-card" style={{ padding: 16 }}>
      <div className="sk sk-thumb" />
      <div className="sk sk-line w60" style={{ marginTop: 14 }} />
      <div className="sk sk-line w40" />
      <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
        <div className="sk" style={{ width: 64, height: 22, borderRadius: 20 }} />
        <div className="sk" style={{ width: 84, height: 22, borderRadius: 20 }} />
      </div>
    </div>
  )
}

export default function Dashboard() {
  return (
    <div className="fp-page">
      {/* Quick Search bar */}
      <div className="fp-quicksearch">
        <IcSearch size={20} />
        <input placeholder="Quick search across all investigations — people, vehicles, plates, or a description…" />
        <button className="fp-btn" disabled>Search</button>
      </div>

      {/* Header + Create New Investigation */}
      <div className="fp-page-head">
        <div>
          <h1 className="fp-page-title">Dashboard</h1>
          <p className="fp-page-desc">Overview of active investigations and recent activity.</p>
        </div>
        <button className="fp-btn primary" disabled><IcPlus /> Create New Investigation</button>
      </div>

      {/* Stat chips */}
      <div className="fp-stats">
        {['Active Investigations', 'Evidence Items', 'Cameras Online', 'Exports'].map((l) => (
          <div key={l} className="fp-card fp-stat">
            <div className="sk sk-line w40" style={{ height: 24, marginBottom: 8 }} />
            <div className="l">{l}</div>
          </div>
        ))}
      </div>

      <div className="fp-split">
        {/* Active Investigations — investigation cards */}
        <section>
          <div className="fp-section-h">
            <h3><IcFolder /> Active Investigations</h3>
            <button className="fp-btn ghost" disabled>View all</button>
          </div>
          <div className="fp-grid cols-3">
            {[0, 1, 2, 3].map((i) => <InvestigationCardSkeleton key={i} />)}
          </div>
        </section>

        {/* Recent Activity */}
        <aside className="fp-panel">
          <div className="fp-panel-title"><span><IcClock /> Recent Activity</span></div>
          <div className="fp-activity">
            {[0, 1, 2, 3, 4].map((i) => (
              <div className="fp-activity-row" key={i}>
                <span className="fp-dot-ic"><IcClock size={16} /></span>
                <div style={{ flex: 1 }}>
                  <div className="sk sk-line w80" />
                  <div className="sk sk-line w40" />
                </div>
              </div>
            ))}
          </div>
        </aside>
      </div>
    </div>
  )
}
