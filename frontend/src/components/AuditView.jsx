// Audit trail view (from GET /api/audit) - every search/export is logged.
import { useEffect, useState } from 'react'
import { getAudit } from '../api'

export default function AuditView() {
  const [rows, setRows] = useState([])
  const [error, setError] = useState(false)

  useEffect(() => {
    getAudit(100).then(setRows).catch(() => setError(true))
  }, [])

  if (error) return <div className="muted">Could not load the audit log.</div>
  if (!rows.length) return <div className="muted">No audit entries yet.</div>

  return (
    <table className="audit-table">
      <thead>
        <tr><th>Time</th><th>Action</th><th>Type</th><th>Query</th><th>Results</th></tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.log_id}>
            <td className="mono">{r.timestamp ? r.timestamp.replace('T', ' ').slice(0, 19) : ''}</td>
            <td>{r.action}</td>
            <td>{r.query_type || ''}</td>
            <td>{r.query_text || ''}</td>
            <td>{r.result_count != null ? r.result_count : ''}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
