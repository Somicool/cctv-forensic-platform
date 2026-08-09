// Shared state for the active investigation so the Workspace, Evidence Gallery
// and Case File pages operate on the SAME evidence - never on "every extracted
// frame". Holds the last search matches, the bookmarked/selected evidence set,
// and lightweight case metadata.
//
// Evidence and case details are PERSISTED server-side (SQLite, see
// app/case_store.py). They used to live only in React state, so everything an
// investigator saved was lost on a refresh or a backend restart - unacceptable
// for a forensic case file. The context API below is unchanged; it now loads the
// saved case on startup and writes changes straight through.
//
// `matches` stays in memory on purpose: it is the last search result set, not
// collected evidence, and is reproduced by re-running the search.
import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import { loadCase, saveCaseEvidence, saveCaseInfo } from '../api'

const Ctx = createContext(null)
const EMPTY_CASE = { title: '', caseNumber: '', officer: '', notes: '' }

export function InvestigationProvider({ children }) {
  const [matches, setMatches] = useState([])          // last search result set (in-memory)
  const [evidence, setEvidence] = useState([])        // bookmarked / selected items
  const [caseInfo, setCaseInfo] = useState(EMPTY_CASE)
  const [restored, setRestored] = useState(false)     // saved case loaded from the backend
  const [persistError, setPersistError] = useState(null)

  // --- restore the saved case once on startup -------------------------------
  useEffect(() => {
    let alive = true
    ;(async () => {
      try {
        const data = await loadCase()
        if (!alive) return
        if (Array.isArray(data?.evidence)) setEvidence(data.evidence)
        if (data?.case_info) setCaseInfo({ ...EMPTY_CASE, ...data.case_info })
      } catch (e) {
        if (alive) setPersistError('Could not load the saved case from the server.')
      } finally {
        if (alive) setRestored(true)
      }
    })()
    return () => { alive = false }
  }, [])

  // --- write evidence through on every change ------------------------------
  // Guarded by `restored` so the initial empty array can never overwrite a
  // stored case before it has been loaded.
  useEffect(() => {
    if (!restored) return
    saveCaseEvidence(evidence).then(
      () => setPersistError(null),
      () => setPersistError('Evidence could not be saved to the server.'),
    )
  }, [evidence, restored])

  // --- persist case details, debounced (these fields are typed into) -------
  const infoTimer = useRef(null)
  useEffect(() => {
    if (!restored) return
    clearTimeout(infoTimer.current)
    infoTimer.current = setTimeout(() => {
      saveCaseInfo(caseInfo).then(
        () => setPersistError(null),
        () => setPersistError('Case details could not be saved to the server.'),
      )
    }, 500)
    return () => clearTimeout(infoTimer.current)
  }, [caseInfo, restored])

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
    restored, persistError,
  }
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useInvestigation() {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useInvestigation must be used within InvestigationProvider')
  return ctx
}
