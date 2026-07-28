import PageSkeleton from './PageSkeleton'
import { IcSettings } from '../components/icons'

export default function Settings() {
  return (
    <PageSkeleton
      title="Settings"
      desc="Processing mode, cameras, models, and responsible-use controls."
      icon={<IcSettings size={26} />}
    >
      <div className="fp-grid cols-3">
        {['Processing Mode', 'Cameras & GPS', 'AI Models', 'Face Recognition', 'Data & Storage', 'About'].map((s) => (
          <div key={s} className="fp-panel">
            <div className="fp-panel-title"><span>{s}</span></div>
            <div className="sk sk-line w80" />
            <div className="sk sk-line w60" />
          </div>
        ))}
      </div>
    </PageSkeleton>
  )
}
