// Grid of detection result cards (crop thumbnail + attributes + score).
export default function ResultsGrid({ results, onSelect }) {
  if (!results?.length) return null
  return (
    <div className="results-grid">
      {results.map((r) => (
        <ResultCard key={r.detection_id} r={r} onSelect={onSelect} />
      ))}
    </div>
  )
}

function ResultCard({ r, onSelect }) {
  const t = r.timestamp ? r.timestamp.replace('T', ' ').slice(0, 19) : ''
  return (
    <button className="result-card" onClick={() => onSelect(r)}>
      <div className="thumb">
        {r.crop_url
          ? <img src={r.crop_url} alt={r.class_label} loading="lazy" />
          : <div className="thumb-empty">{r.class_label}</div>}
        <span className={`score ${scoreTier(r.score)}`}>{Math.round((r.score || 0) * 100)}%</span>
      </div>
      <div className="result-body">
        <div className="result-top">
          <span className="label">{r.class_label}</span>
          <span className="cam">{r.camera_id}</span>
        </div>
        <div className="attrs">{attrText(r.attributes)}</div>
        <div className="ts">{t}</div>
      </div>
    </button>
  )
}

function scoreTier(s) {
  if ((s || 0) >= 0.7) return 'high'
  if ((s || 0) >= 0.4) return 'mid'
  return 'low'
}

export function attrText(a) {
  if (!a) return ''
  const parts = []
  if (a.color) parts.push(a.color)
  if (a.upper_color) parts.push(`top: ${a.upper_color}`)
  if (a.lower_color) parts.push(`btm: ${a.lower_color}`)
  if (a.vehicle_type) parts.push(a.vehicle_type)
  if (Array.isArray(a.accessories) && a.accessories.length) parts.push(a.accessories.join(', '))
  if (a.plate_text) parts.push(`plate: ${a.plate_text}`)
  if (a.age) parts.push(`age ${a.age}`)
  if (a.gender) parts.push(a.gender)
  return parts.join(' · ')
}
