// Result detail: play the source recording from the exact detected moment (box
// overlaid), with metadata, cross-camera trace, map link, and case-file add.
import { useEffect, useState } from 'react'
import { trackDetection } from '../api'
import VideoPlayer from './VideoPlayer'

export default function ResultDetail({ item, onClose, onOpenTrack, onAddToCase }) {
  const [track, setTrack] = useState(null)
  const [loading, setLoading] = useState(false)
  const [added, setAdded] = useState(false)

  useEffect(() => {
    const esc = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [onClose])

  async function doTrack() {
    setLoading(true)
    try { setTrack(await trackDetection(item.detection_id)) }
    catch { setTrack({ appearances: [] }) }
    finally { setLoading(false) }
  }

  const t = item.timestamp ? item.timestamp.replace('T', ' ').slice(0, 19) : '—'
  const attrs = item.attributes || {}
  const attrKeys = Object.keys(attrs)

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal wide" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>×</button>

        <div className="modal-player">
          <VideoPlayer src={item.video_url} offset={item.offset_seconds}
                       bbox={item.bbox} frameW={item.frame_width} frameH={item.frame_height} />
        </div>

        <div className="modal-meta">
          <div className="modal-meta-head">
            <h3>{item.class_label} <span className="cam">{item.camera_name || item.camera_id}</span></h3>
            {item.crop_url && <img className="matched-crop" src={item.crop_url} alt="matched" />}
          </div>

          <div className="kv-row">
            <span>#{item.detection_id}</span>
            <span>{t}</span>
            <span>match {((item.score || 0) * 100).toFixed(1)}%</span>
            <span>conf {((item.confidence || 0) * 100).toFixed(1)}%</span>
            {item.offset_seconds != null && <span>@ {fmtTime(item.offset_seconds)} in clip</span>}
          </div>

          {attrKeys.length > 0 && (
            <div className="attr-list">
              {attrKeys.map((k) => (
                <span className="attr-pill" key={k}>
                  {k}: {String(Array.isArray(attrs[k]) ? attrs[k].join(', ') : attrs[k])}
                </span>
              ))}
            </div>
          )}

          <div className="modal-actions">
            <button className="btn primary" onClick={doTrack} disabled={loading}>
              {loading ? 'Tracing…' : 'Track across cameras'}
            </button>
            {onOpenTrack && (
              <button className="btn ghost" onClick={() => { onClose(); onOpenTrack(item.detection_id) }}>
                Open in map view
              </button>
            )}
            {onAddToCase && (
              <button className="btn ghost" onClick={() => { onAddToCase(item); setAdded(true) }} disabled={added}>
                {added ? 'Added to case file ✓' : 'Add to case file'}
              </button>
            )}
          </div>

          {track && (
            <div className="track-list">
              <div className="track-head">{track.appearances.length} appearance(s)</div>
              {track.appearances.slice(0, 25).map((a, i) => (
                <div className="track-row" key={i}>
                  <span className="cam">{a.camera_id}</span>
                  <span className="mono">{a.timestamp ? a.timestamp.replace('T', ' ').slice(11, 19) : ''}</span>
                  <span className="sim">{Math.round((a.similarity || 0) * 100)}%</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function fmtTime(s) {
  s = Math.max(0, Math.round(s || 0))
  const m = Math.floor(s / 60)
  const ss = s % 60
  return `${m}:${String(ss).padStart(2, '0')}`
}
