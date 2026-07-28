import PageSkeleton from './PageSkeleton'
import { IcCase } from '../components/icons'

export default function CaseFile() {
  return (
    <PageSkeleton
      title="Case File"
      desc="Assemble evidence, add case metadata, and export a sealed forensic package."
      icon={<IcCase size={26} />}
      action={<button className="fp-btn primary" disabled>Export Evidence</button>}
    >
      <div className="fp-split">
        <div className="fp-panel">
          <div className="fp-panel-title"><span>Evidence in this case</span><span className="muted">0 items</span></div>
          {[0, 1, 2].map((i) => (
            <div key={i} className="fp-activity-row">
              <span className="sk" style={{ width: 46, height: 46, borderRadius: 8 }} />
              <div style={{ flex: 1 }}>
                <div className="sk sk-line w60" />
                <div className="sk sk-line w40" />
              </div>
            </div>
          ))}
        </div>
        <aside className="fp-panel">
          <div className="fp-panel-title"><span>Case details</span></div>
          <div className="sk sk-line w80" style={{ height: 38 }} />
          <div className="sk sk-line w80" style={{ height: 38, marginTop: 12 }} />
          <div className="sk sk-line w60" style={{ height: 64, marginTop: 12 }} />
        </aside>
      </div>
    </PageSkeleton>
  )
}
