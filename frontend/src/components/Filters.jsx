// Metadata filter controls, mapped 1:1 to the backend SearchFilters schema.
const COLORS = ['red', 'blue', 'white', 'black', 'silver', 'grey',
                'yellow', 'green', 'brown', 'orange', 'maroon']
const VEHICLE_TYPES = ['sedan', 'hatchback', 'SUV', 'pickup truck', 'van',
                       'auto-rickshaw', 'bus', 'truck', 'motorcycle', 'bicycle']

export default function Filters({ cameras, filters, onChange }) {
  const set = (key, val) => onChange({ ...filters, [key]: val })
  const toggleArr = (key, val) => {
    const cur = filters[key] || []
    set(key, cur.includes(val) ? cur.filter((x) => x !== val) : [...cur, val])
  }

  return (
    <div className="filters">
      <div className="filter-group wide">
        <label>Cameras</label>
        <div className="chips">
          {(cameras || []).map((c) => (
            <button
              key={c.camera_id}
              className={`chip ${filters.cameras?.includes(c.camera_id) ? 'on' : ''}`}
              onClick={() => toggleArr('cameras', c.camera_id)}
            >
              {c.camera_id}
            </button>
          ))}
        </div>
      </div>

      <div className="filter-group">
        <label>Object type</label>
        <select value={filters.object_type || ''}
                onChange={(e) => set('object_type', e.target.value || undefined)}>
          <option value="">Any</option>
          <option value="person">Person</option>
          <option value="vehicle">Vehicle</option>
        </select>
      </div>

      <div className="filter-group">
        <label>Vehicle type</label>
        <select value={filters.vehicle_type || ''}
                onChange={(e) => set('vehicle_type', e.target.value || undefined)}>
          <option value="">Any</option>
          {VEHICLE_TYPES.map((v) => <option key={v} value={v}>{v}</option>)}
        </select>
      </div>

      <div className="filter-group wide">
        <label>Colours</label>
        <div className="chips">
          {COLORS.map((c) => (
            <button key={c}
                    className={`chip ${filters.colors?.includes(c) ? 'on' : ''}`}
                    onClick={() => toggleArr('colors', c)}>
              {c}
            </button>
          ))}
        </div>
      </div>

      <div className="filter-group">
        <label>Min confidence: {Math.round((filters.min_confidence || 0) * 100)}%</label>
        <input type="range" min="0" max="1" step="0.05"
               value={filters.min_confidence || 0}
               onChange={(e) => set('min_confidence', parseFloat(e.target.value) || undefined)} />
      </div>

      <div className="filter-group">
        <label>From</label>
        <input type="datetime-local" value={filters.start_time || ''}
               onChange={(e) => set('start_time', e.target.value || undefined)} />
      </div>

      <div className="filter-group">
        <label>To</label>
        <input type="datetime-local" value={filters.end_time || ''}
               onChange={(e) => set('end_time', e.target.value || undefined)} />
      </div>

      <button className="btn ghost small" onClick={() => onChange({})}>Clear all</button>
    </div>
  )
}
