// Video player that jumps to a detection's exact moment and overlays its box.
// The box is only accurate at the seeked frame, so it hides once you press play
// and reappears when you click "Jump to match".
import { useEffect, useRef, useState } from 'react'

export default function VideoPlayer({ src, offset = 0, bbox, frameW, frameH, autoPlay = false, onTimeUpdate }) {
  const videoRef = useRef(null)
  const [showBox, setShowBox] = useState(true)

  // Keep the latest callback in a ref so the timeupdate listener never goes stale
  // and we don't have to re-bind it on every render.
  const timeCb = useRef(onTimeUpdate)
  timeCb.current = onTimeUpdate

  const playable = !!src && /\.(mp4|webm|ogg)(\?|#|$)/i.test(src)

  // Seek to the detection's exact moment. When the src is unchanged (same clip,
  // different match) React keeps the <video> mounted, so this just re-seeks
  // instantly instead of reloading the whole video. We pause + show the box so
  // the user lands on the precise frame; pressing play hides the box.
  useEffect(() => {
    const v = videoRef.current
    if (!v || !playable) return
    const seek = () => {
      try { v.currentTime = Math.max(0, offset || 0); v.pause() } catch (e) { /* noop */ }
      setShowBox(true)
    }
    const onPlay = () => setShowBox(false)
    v.addEventListener('loadedmetadata', seek)
    v.addEventListener('play', onPlay)
    if (v.readyState >= 1) seek()
    setShowBox(true)
    return () => {
      v.removeEventListener('loadedmetadata', seek)
      v.removeEventListener('play', onPlay)
    }
  }, [src, offset, playable])

  // Report playback position so the parent can keep the active result in sync.
  useEffect(() => {
    const v = videoRef.current
    if (!v || !playable) return
    const onTime = () => { if (timeCb.current) timeCb.current(v.currentTime) }
    v.addEventListener('timeupdate', onTime)
    return () => v.removeEventListener('timeupdate', onTime)
  }, [src, playable])

  function jumpToMoment() {
    const v = videoRef.current
    if (!v) return
    v.pause()
    v.currentTime = Math.max(0, offset || 0)
    setShowBox(true)
  }

  if (!src) return <div className="muted small">No source recording linked to this detection.</div>
  if (!playable) {
    return (
      <div className="vp-unplayable muted small">
        This recording's format isn't playable in the browser. Production recordings
        are stored as H.264 MP4 (searchable + seekable).
      </div>
    )
  }

  const boxStyle = (bbox && frameW && frameH) ? {
    left: `${(bbox[0] / frameW) * 100}%`,
    top: `${(bbox[1] / frameH) * 100}%`,
    width: `${(bbox[2] / frameW) * 100}%`,
    height: `${(bbox[3] / frameH) * 100}%`,
  } : null

  return (
    <div className="vp">
      <div className="vp-stage">
        <video ref={videoRef} src={src} controls autoPlay={autoPlay} preload="metadata" className="vp-video" />
        {showBox && boxStyle && <div className="vp-box" style={boxStyle} />}
      </div>
      <div className="vp-controls">
        <button className="btn small" onClick={jumpToMoment}>⤿ Jump to match · {fmtTime(offset)}</button>
        {boxStyle && (
          <label className="checkbox">
            <input type="checkbox" checked={showBox} onChange={(e) => setShowBox(e.target.checked)} /> highlight box
          </label>
        )}
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
