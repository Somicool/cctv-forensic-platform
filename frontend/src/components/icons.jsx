// Lightweight inline SVG icons (no icon-library dependency). Stroke-based,
// inherit currentColor, sized 20 by default.
const base = (size) => ({
  width: size, height: size, viewBox: '0 0 24 24', fill: 'none',
  stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round',
})

export const IcDashboard = ({ size = 20 }) => (
  <svg {...base(size)}><rect x="3" y="3" width="7" height="9" rx="1.5" /><rect x="14" y="3" width="7" height="5" rx="1.5" /><rect x="14" y="12" width="7" height="9" rx="1.5" /><rect x="3" y="16" width="7" height="5" rx="1.5" /></svg>
)
export const IcWorkspace = ({ size = 20 }) => (
  <svg {...base(size)}><circle cx="11" cy="11" r="7" /><path d="M21 21l-4.3-4.3" /></svg>
)
export const IcEvidence = ({ size = 20 }) => (
  <svg {...base(size)}><rect x="3" y="3" width="8" height="8" rx="1.5" /><rect x="13" y="3" width="8" height="8" rx="1.5" /><rect x="3" y="13" width="8" height="8" rx="1.5" /><rect x="13" y="13" width="8" height="8" rx="1.5" /></svg>
)
export const IcCase = ({ size = 20 }) => (
  <svg {...base(size)}><path d="M4 7h16v13H4z" /><path d="M9 7V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2" /><path d="M4 12h16" /></svg>
)
export const IcSettings = ({ size = 20 }) => (
  <svg {...base(size)}><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-2.7 1.1V21a2 2 0 1 1-4 0v-.1A1.6 1.6 0 0 0 7 19.4a1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0-1.1-2.7H1a2 2 0 1 1 0-4h.1A1.6 1.6 0 0 0 2.6 7a1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1A1.6 1.6 0 0 0 7 2.6h.1A1.6 1.6 0 0 0 8 1.1V1a2 2 0 1 1 4 0v.1A1.6 1.6 0 0 0 15 2.6a1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V7a1.6 1.6 0 0 0 1.1 1.5H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z" transform="scale(0.82) translate(2.6 2.6)" /></svg>
)
export const IcSearch = ({ size = 18 }) => (
  <svg {...base(size)}><circle cx="11" cy="11" r="7" /><path d="M21 21l-4.3-4.3" /></svg>
)
export const IcPlus = ({ size = 18 }) => (
  <svg {...base(size)}><path d="M12 5v14M5 12h14" /></svg>
)
export const IcMenu = ({ size = 20 }) => (
  <svg {...base(size)}><path d="M4 6h16M4 12h16M4 18h16" /></svg>
)
export const IcFolder = ({ size = 18 }) => (
  <svg {...base(size)}><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /></svg>
)
export const IcUpload = ({ size = 18 }) => (
  <svg {...base(size)}><path d="M12 16V4M7 9l5-5 5 5" /><path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" /></svg>
)
export const IcClock = ({ size = 18 }) => (
  <svg {...base(size)}><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></svg>
)
export const IcShield = ({ size = 22 }) => (
  <svg {...base(size)} strokeWidth="1.6"><path d="M12 3l7 3v5c0 4.5-3 8.5-7 10-4-1.5-7-5.5-7-10V6z" /><path d="M9 12l2 2 4-4" /></svg>
)
