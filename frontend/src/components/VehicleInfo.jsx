// Demo Vehicle Registry viewer (offline, synthetic - NOT a real police database).
// Opens when "Vehicle Info" is clicked on a plate result: fetches the permanent
// registry record for the plate and shows all stored fields, grouped. Supports
// manual editing that permanently overwrites the record.
//
// The backend provider is swappable (demo -> real police API) with no change
// here; this component only talks to /vehicle-registry/{plate}.
import { useEffect, useState } from 'react'
import { getVehicleRegistry, updateVehicleRegistry } from '../api'

// grouped field layout: [group title, [ [key, label, editable], ... ] ]
const GROUPS = [
  ['Owner', [
    ['owner_name', 'Owner Name', true], ['father_name', "Father's Name", true],
    ['gender', 'Gender', true], ['mobile_number', 'Mobile', true],
    ['alternate_mobile', 'Alternate Mobile', true], ['driving_license_no', 'Driving License', true],
    ['emergency_contact', 'Emergency Contact', true],
  ]],
  ['Vehicle', [
    ['vehicle_number', 'Vehicle Number', true], ['vehicle_type', 'Type', true],
    ['vehicle_brand', 'Brand', true], ['vehicle_model', 'Model', true],
    ['vehicle_color', 'Colour', true], ['fuel_type', 'Fuel', true],
    ['vehicle_class', 'Class', true], ['chassis_number', 'Chassis No. (masked)', false],
    ['engine_number', 'Engine No. (masked)', false],
  ]],
  ['Registration', [
    ['registration_date', 'Registered On', true], ['registration_state', 'Reg. State', true],
    ['registration_office', 'RTO', true], ['rc_status', 'RC Status', true],
    ['insurance_status', 'Insurance', true], ['puc_status', 'PUC', true],
  ]],
  ['Address', [
    ['address', 'Address', true], ['city', 'City', true],
    ['district', 'District', true], ['state', 'State', true], ['pin_code', 'PIN', true],
  ]],
  ['Status & Flags', [
    ['blacklist_status', 'Blacklist', true], ['stolen_status', 'Stolen (Demo)', true],
    ['previous_violations', 'Previous Violations', true],
    ['previous_investigation_count', 'Investigations (Demo)', true],
    ['notes', 'Notes', true],
  ]],
]

const asText = (v) => Array.isArray(v) ? v.join(', ') : (v == null ? '' : String(v))
const isFlag = (k, v) => (k === 'stolen_status' && /stolen/i.test(v)) ||
                          (k === 'blacklist_status' && /blacklist/i.test(v)) ||
                          (k === 'rc_status' && /suspend/i.test(v)) ||
                          (k === 'insurance_status' && /expired/i.test(v)) ||
                          (k === 'puc_status' && /expired/i.test(v))

export default function VehicleInfo({ plate, onClose }) {
  const [rec, setRec] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState({})
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    let alive = true
    setLoading(true); setError(null)
    getVehicleRegistry(plate)
      .then((r) => { if (alive) { setRec(r); setDraft(r); setLoading(false) } })
      .catch((e) => { if (alive) { setError(e?.response?.data?.detail || e.message || 'Lookup failed'); setLoading(false) } })
    return () => { alive = false }
  }, [plate])

  useEffect(() => { const esc = (e) => { if (e.key === 'Escape') onClose() }; window.addEventListener('keydown', esc); return () => window.removeEventListener('keydown', esc) }, [onClose])

  async function save() {
    setSaving(true); setError(null)
    // build the update payload (arrays parsed back from comma strings)
    const updates = {}
    for (const [, fields] of GROUPS) {
      for (const [k] of fields) {
        let v = draft[k]
        if (k === 'previous_violations') v = asText(v).split(',').map((s) => s.trim()).filter(Boolean)
        if (k === 'previous_investigation_count') v = Number(v) || 0
        updates[k] = v
      }
    }
    try {
      const r = await updateVehicleRegistry(plate, updates)
      setRec(r); setDraft(r); setEditing(false)
    } catch (e) { setError(e?.response?.data?.detail || e.message || 'Save failed') }
    finally { setSaving(false) }
  }

  return (
    <div className="vi-overlay" onMouseDown={onClose}>
      <div className="vi-modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="vi-head">
          <div className="vi-title">
            <span className="vi-badge">DEMO</span> Vehicle Registry
            <span className="vi-plate">{rec?.vehicle_number || plate}</span>
          </div>
          <div className="vi-head-actions">
            {!loading && !error && (editing
              ? <><button className="fp-btn sm" onClick={() => { setDraft(rec); setEditing(false) }}>Cancel</button>
                  <button className="fp-btn primary sm" onClick={save} disabled={saving}>{saving ? 'Saving…' : 'Save'}</button></>
              : <button className="fp-btn sm" onClick={() => setEditing(true)}>✎ Edit</button>)}
            <button className="vi-x" onClick={onClose} aria-label="Close">×</button>
          </div>
        </div>

        <div className="vi-body">
          {loading ? <div className="vi-msg">Looking up registry…</div>
            : error ? <div className="vi-msg err">{error}</div>
              : (
                <>
                  {(rec.stolen_status && /stolen/i.test(rec.stolen_status)) || (rec.blacklist_status && /blacklist/i.test(rec.blacklist_status))
                    ? <div className="vi-alert">⚠ {[/stolen/i.test(rec.stolen_status) ? 'Reported stolen' : null, /blacklist/i.test(rec.blacklist_status) ? 'Blacklisted' : null].filter(Boolean).join(' · ')} (demo flag)</div>
                    : null}
                  {GROUPS.map(([title, fields]) => (
                    <section className="vi-group" key={title}>
                      <div className="vi-group-h">{title}</div>
                      <div className="vi-rows">
                        {fields.map(([k, label, editable]) => (
                          <div className="vi-row" key={k}>
                            <div className="vi-k">{label}</div>
                            {editing && editable
                              ? <input className="vi-input" value={asText(draft[k])}
                                       onChange={(e) => setDraft({ ...draft, [k]: e.target.value })} />
                              : <div className={'vi-v ' + (isFlag(k, asText(rec[k])) ? 'flag' : '')}>{asText(rec[k]) || '—'}</div>}
                          </div>
                        ))}
                      </div>
                    </section>
                  ))}
                  <div className="vi-note">{rec.notes || 'Synthetic demo record.'}</div>
                </>
              )}
        </div>
      </div>
    </div>
  )
}
