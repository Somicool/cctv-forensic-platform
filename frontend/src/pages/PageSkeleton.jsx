// Shared skeleton scaffold for pages whose contents aren't built yet.
export default function PageSkeleton({ title, desc, action, icon, children }) {
  return (
    <div className="fp-page">
      <div className="fp-page-head">
        <div>
          <h1 className="fp-page-title">{title}</h1>
          {desc && <p className="fp-page-desc">{desc}</p>}
        </div>
        {action}
      </div>
      {children || (
        <div className="fp-panel">
          <div className="fp-empty">
            <div className="ic">{icon}</div>
            <h4>Coming up next</h4>
            <p>This section's layout is ready. Content and interactions will be wired in the next step.</p>
          </div>
        </div>
      )}
    </div>
  )
}
