
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { ArrowRight, X } from 'lucide-react'
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts'
import { fetchHealth } from '../api/health'
import { useAnalysisStore } from '../store/analysisStore'
import { motion, AnimatePresence } from 'framer-motion'

function getPredictionBadgeClass(pred: string): string {
  const p = pred.toLowerCase()
  if (p.includes('glioma')) return 'badge badge-glioma'
  if (p.includes('notumor') || p.includes('no tumor')) return 'badge badge-notumor'
  if (p.includes('meningioma')) return 'badge badge-meningioma'
  if (p.includes('pituitary')) return 'badge badge-pituitary'
  return 'badge badge-processing'
}

function formatTimestamp(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins} min${mins > 1 ? 's' : ''} ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs} hour${hrs > 1 ? 's' : ''} ago`
  return new Date(iso).toLocaleDateString()
}

const ACCURACY_DATA = [
  { value: 98.2, color: '#00d4e8' },
  { value: 1.8, color: '#1a2d42' },
]

export default function Dashboard() {
  const navigate = useNavigate()
  const { history } = useAnalysisStore()
  const [showStatus, setShowStatus] = useState(true)

  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: 30000,
    retry: false,
  })

  const displayStudies =
    history.length > 0
      ? history.slice(0, 8).map((e) => ({
          mrn: e.mrn,
          time: formatTimestamp(e.timestamp),
          status: 'completed' as const,
          pred: `${e.result.prediction} (${e.result.confidence}%)`,
          conf: e.result.confidence,
        }))
      : []

  return (
    <div className="page-content" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {}
      <motion.div
        style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: 16, flex: 1 }}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        {}
        <div className="card" style={{ overflow: 'hidden' }}>
          <div
            style={{
              padding: '14px 20px',
              borderBottom: '1px solid var(--border)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <span
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 12,
                fontWeight: 700,
                color: 'var(--text-primary)',
              }}
            >
              Recent Studies
            </span>
            <button
              className="btn-ghost"
              style={{ fontSize: 10, gap: 4 }}
              onClick={() => navigate('/history')}
            >
              View Archive <ArrowRight size={11} />
            </button>
          </div>

          {displayStudies.length === 0 ? (
            <div
              style={{
                padding: 40,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 10,
                opacity: 0.5,
              }}
            >
              <span
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 12,
                  color: 'var(--text-muted)',
                  textAlign: 'center',
                }}
              >
                No analyses yet.
                <br />
                Go to Upload & Analyze to run your first scan.
              </span>
              <button
                className="btn-secondary"
                style={{ marginTop: 8, gap: 6 }}
                onClick={() => navigate('/upload')}
              >
                Start Analysis
              </button>
            </div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Patient MRN</th>
                  <th>Timestamp</th>
                  <th>Status</th>
                  <th>Prediction</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {displayStudies.map((s, i) => (
                  <tr
                    key={i}
                    style={{ cursor: 'pointer' }}
                    onClick={() => navigate('/history')}
                  >
                    <td>
                      <span
                        style={{
                          color: 'var(--accent)',
                          fontFamily: 'var(--font-mono)',
                          fontSize: 12,
                          fontWeight: 700,
                        }}
                      >
                        {s.mrn}
                      </span>
                    </td>
                    <td
                      style={{
                        color: 'var(--text-secondary)',
                        fontFamily: 'var(--font-mono)',
                        fontSize: 12,
                      }}
                    >
                      {s.time}
                    </td>
                    <td>
                      <span
                        style={{ display: 'flex', alignItems: 'center', gap: 6 }}
                      >
                        <span className="dot dot-green" />
                        <span
                          style={{
                            fontFamily: 'var(--font-mono)',
                            fontSize: 11,
                            color: 'var(--accent-green)',
                          }}
                        >
                          Completed
                        </span>
                      </span>
                    </td>
                    <td>
                      <span className={getPredictionBadgeClass(s.pred)}>
                        {s.pred}
                      </span>
                    </td>
                    <td>
                      <span
                        style={{
                          fontFamily: 'var(--font-mono)',
                          fontSize: 11,
                          color: 'var(--text-secondary)',
                        }}
                      >
                        {s.conf}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {}
        <div
          className="card"
          style={{
            padding: '18px 20px',
            display: 'flex',
            flexDirection: 'column',
            gap: 16,
          }}
        >
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 12,
              fontWeight: 700,
              color: 'var(--text-primary)',
            }}
          >
            Predictive Accuracy
          </span>

          {}
          <div style={{ height: 160, position: 'relative' }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={ACCURACY_DATA}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={72}
                  startAngle={90}
                  endAngle={-270}
                  dataKey="value"
                  strokeWidth={0}
                >
                  {ACCURACY_DATA.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
            <div
              style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <div
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 20,
                  fontWeight: 700,
                  color: 'var(--text-primary)',
                }}
              >
                98.2%
              </div>
              <div
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 8,
                  color: 'var(--text-muted)',
                  letterSpacing: '0.06em',
                  textTransform: 'uppercase',
                }}
              >
                B0 Model Precision
              </div>
            </div>
          </div>

          {}
          {[
            { label: 'Sensitivity', value: 99.1 },
            { label: 'Specificity', value: 97.4 },
          ].map(({ label, value }) => (
            <div key={label} style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: 11,
                    color: 'var(--text-secondary)',
                  }}
                >
                  {label}
                </span>
                <span
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: 11,
                    color: 'var(--accent)',
                    fontWeight: 700,
                  }}
                >
                  {value}%
                </span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${value}%` }} />
              </div>
            </div>
          ))}

          {}
          {history.length > 0 && (
            <div
              style={{
                paddingTop: 12,
                borderTop: '1px solid var(--border)',
                fontFamily: 'var(--font-mono)',
                fontSize: 10,
                color: 'var(--text-muted)',
              }}
            >
              <span style={{ color: 'var(--accent)', fontWeight: 700 }}>
                {history.length}
              </span>{' '}
              analyses completed in this session
            </div>
          )}
        </div>
      </motion.div>

      {}
      <AnimatePresence>
        {showStatus && (
          <motion.div
            className="status-bar"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
          >
            <span className={`dot ${health ? 'dot-green' : 'dot-red'}`} />
            System Status:{' '}
            {health ? 'Optimal' : 'Connecting'} | Backend:{' '}
            {health ? `v${health.version}` : '...'}
            <button
              onClick={() => setShowStatus(false)}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                color: 'var(--text-muted)',
                marginLeft: 8,
              }}
            >
              <X size={13} />
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
