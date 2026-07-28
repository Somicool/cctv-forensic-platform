// Shared state for the active investigation so the Workspace, Evidence Gallery
// and Case File pages operate on the SAME evidence - never on "every extracted
// frame". Holds the last search matches, the bookmarked/selected evidence set,
// and lightweight case metadata. No backend calls here.
import { createContext, useCallback, useContext, useState } from 'react'

const Ctx = createContext(null)

export function InvestigationProvider({ children }) {
  const [matches, setMatches] = useState([])          // last search result set
  const [evidence, setEvidence] = useState([])        // bookmarked / selected items
  const [caseInfo, setCaseInfo] = useState({ title: '', caseNumber: '', officer: '', notes: '' })

  const inEvidence = useCallback((id) => evidence.some((e) => e.detection_id === id), [evidence])
  const toggleEvidence = useCallback((r) => setEvidence((p) =>
    p.some((e) => e.detection_id === r.detection_id)
      ? p.filter((e) => e.detection_id !== r.detection_id)
      : [...p, r]), [])
  const removeEvidence = useCallback((id) => setEvidence((p) => p.filter((e) => e.detection_id !== id)), [])

  const value = {
    matches, setMatches,
    evidence, setEvidence, inEvidence, toggleEvidence, removeEvidence,
    caseInfo, setCaseInfo,
  }
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useInvestigation() {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useInvestigation must be used within InvestigationProvider')
  return ctx
}
