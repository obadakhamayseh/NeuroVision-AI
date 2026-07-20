
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { PredictResponse } from '../api/predict'

export interface AnalysisEntry {
  id: string
  mrn: string
  timestamp: string
  status: 'completed' | 'failed'
  result: PredictResponse
  filename: string
}

interface AnalysisState {
  latest: PredictResponse | null
  history: AnalysisEntry[]
  setLatest: (r: PredictResponse, filename: string) => void
  clearLatest: () => void
  clearHistory: () => void
}

export const useAnalysisStore = create<AnalysisState>()(
  persist(
    (set) => ({
      latest: null,
      history: [],
      setLatest: (result, filename) => {
        const entry: AnalysisEntry = {
          id: result.report_id,
          mrn: `SYN-${Math.floor(Math.random() * 9000 + 1000)}`,
          timestamp: result.timestamp || new Date().toISOString(),
          status: 'completed',
          result,
          filename,
        }
        set((s) => ({
          latest: result,
          history: [entry, ...s.history].slice(0, 50),
        }))
      },
      clearLatest: () => set({ latest: null }),
      clearHistory: () => set({ history: [] }),
    }),
    { name: 'neurovision-analysis' }
  )
)
