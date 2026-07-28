// Simple camera registry view (from GET /api/cameras).
export default function CamerasView({ cameras }) {
  if (!cameras?.length) {
    return <div className="muted">No cameras registered yet.</div>
  }
  return (
    <div className="cards">
      {cameras.map((c) => (
        <div className="card" key={c.camera_id}>
          <div className="card-label">{c.camera_id}</div>
          <div className="card-body">
            <div className="cam-name">{c.name || '—'}</div>
            <div className="muted small">{c.location || ''}</div>
            {c.lat != null && c.lon != null && (
              <div className="muted small mono">{Number(c.lat).toFixed(4)}, {Number(c.lon).toFixed(4)}</div>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}
