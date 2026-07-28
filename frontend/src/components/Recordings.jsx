// The stored recordings, shown as playable clips. Pick one to watch it in the
// big player; each thumbnail is the actual first frame of that recording.
import { useEffect, useState } from 'react'
import { getVideos } from '../api'
import VideoPlayer from './VideoPlayer'

export default function Recordings({ cameras }) {
  const [videos, setVideos] = useState(null)
  const [sel, setSel] = useState(0)

  useEffect(() => { getVideos().then(setVideos).catch(() => setVideos([])) }, [])

  const nameFor = (id) => (cameras || []).find((c) => c.camera_id === id)?.name || ''

  if (videos === null) return <div className="muted">Loading recordings…</div>
  if (!videos.length) return <div className="muted">No recordings ingested yet.</div>

  const cur = videos[sel] || videos[0]
  return (
    <div className="recordings">
      <div className="rec-main">
        <div className="rec-head">
          <span className="cam">{cur.camera_id}</span> {nameFor(cur.camera_id)}
          <span className="muted small">
            {' · '}{fmtDur(cur.duration)}
            {cur.start_time ? ` · ${cur.start_time.replace('T', ' ').slice(0, 19)}` : ''}
          </span>
        </div>
        <VideoPlayer key={cur.url} src={cur.url} offset={0} />
      </div>

      <div className="rec-strip">
        {videos.map((v, i) => (
          <button key={v.video_id} className={`rec-card ${i === sel ? 'active' : ''}`}
                  onClick={() => setSel(i)}>
            <video className="rec-thumb" src={`${v.url}#t=1`} muted preload="metadata" />
            <div className="rec-label">
              <div><span className="cam">{v.camera_id}</span> {nameFor(v.camera_id)}</div>
              <div className="muted small">{fmtDur(v.duration)}</div>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}

function fmtDur(s) {
  if (!s) return ''
  s = Math.round(s)
  const m = Math.floor(s / 60)
  const ss = s % 60
  return `${m}:${String(ss).padStart(2, '0')}`
}
