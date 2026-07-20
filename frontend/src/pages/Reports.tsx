
import { useState } from 'react'
import {
  FileText, Download, ExternalLink, Search,
  ChevronDown, Copy, Check, AlertTriangle, ShieldCheck,
} from 'lucide-react'
import { motion } from 'framer-motion'
import { useAnalysisStore } from '../store/analysisStore'

const BASE_URL = 'http://127.0.0.1:8000'

function getPredBadgeClass(pred: string) {
  const p = pred.toLowerCase()
  if (p.includes('glioma')) return 'badge badge-glioma'
  if (p.includes('no tumor') || p.includes('notumor')) return 'badge badge-notumor'
  if (p.includes('meningioma')) return 'badge badge-meningioma'
  if (p.includes('pituitary')) return 'badge badge-pituitary'
  return 'badge'
}

function isNoTumor(pred: string) {
  return pred.toLowerCase().includes('no tumor') || pred.toLowerCase().includes('notumor')
}

function normPct(v: number): number {
  return typeof v === 'number' && v <= 1 ? v * 100 : v
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }
  return (
    <button
      className="btn-ghost"
      style={{ padding: '2px 7px', fontSize: 9, gap: 4 }}
      onClick={copy}
      title="Copy Report ID"
    >
      {copied ? <Check size={10} color="var(--accent-green)" /> : <Copy size={10} />}
      {copied ? 'Copied' : 'Copy'}
    </button>
  )
}

export default function Reports() {
  const { history } = useAnalysisStore()
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('All')

  const entries = history.length > 0 ? history : []

  const filtered = entries.filter((e) => {
    const matchSearch =
      e.mrn.toLowerCase().includes(search.toLowerCase()) ||
      e.id.toLowerCase().includes(search.toLowerCase()) ||
      e.result.report_id?.toLowerCase().includes(search.toLowerCase())
    const matchFilter =
      filter === 'All' || e.result.prediction.toLowerCase().includes(filter.toLowerCase())
    return matchSearch && matchFilter
  })

  return (
    <div className="page-content" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{ position: 'relative', flex: 1 }}>
          <Search size={13} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
          <input
            className="input-field"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by MRN or Report ID..."
            style={{ paddingLeft: 32 }}
          />
        </div>
        <div style={{ position: 'relative' }}>
          <select
            className="input-field"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            style={{ paddingRight: 32, minWidth: 150 }}
          >
            <option>All</option>
            <option>Glioma</option>
            <option>Meningioma</option>
            <option>Pituitary</option>
            <option>No Tumor</option>
          </select>
          <ChevronDown size={12} style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)', pointerEvents: 'none' }} />
        </div>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
          {filtered.length} report{filtered.length !== 1 ? 's' : ''}
        </span>
      </div>

      {}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        {filtered.map((entry, i) => {
          const r = entry.result
          const tumor = !isNoTumor(r.prediction)
          return (
            <motion.div
              key={entry.id}
              className="card"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.04 }}
              style={{
                padding: '18px 20px',
                borderLeft: `3px solid ${tumor ? 'var(--accent-red)' : 'var(--accent-green)'}`,
              }}
            >
              {}
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 14 }}>
                {}
                <div style={{
                  width: 40, height: 40, borderRadius: 8, flexShrink: 0,
                  background: tumor ? 'rgba(239,68,68,0.08)' : 'rgba(16,185,129,0.08)',
                  border: `1px solid ${tumor ? 'rgba(239,68,68,0.2)' : 'rgba(16,185,129,0.2)'}`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  {tumor
                    ? <AlertTriangle size={18} color="var(--accent-red)" />
                    : <ShieldCheck size={18} color="var(--accent-green)" />}
                </div>

                {}
                <div style={{ flex: 1, display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
                  <div>
                    <div className="section-label" style={{ marginBottom: 4 }}>Prediction</div>
                    <span className={getPredBadgeClass(r.prediction)}>{r.prediction}</span>
                  </div>
                  <div>
                    <div className="section-label" style={{ marginBottom: 4 }}>Confidence</div>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 16, fontWeight: 700, color: 'var(--accent)' }}>
                      {r.confidence}%
                    </span>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 8, color: 'var(--text-muted)', marginTop: 2 }}>
                      {r.confidence_level}
                    </div>
                  </div>
                  <div>
                    <div className="section-label" style={{ marginBottom: 4 }}>Patient MRN</div>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-primary)', fontWeight: 700 }}>
                      {entry.mrn}
                    </span>
                  </div>
                  <div>
                    <div className="section-label" style={{ marginBottom: 4 }}>Analyzed</div>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-secondary)' }}>
                      {new Date(entry.timestamp).toLocaleString()}
                    </span>
                  </div>
                </div>

                {}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, flexShrink: 0 }}>
                  {r.report_pdf ? (
                    <a href={`${BASE_URL}/${r.report_pdf}`} target="_blank" rel="noopener noreferrer"
                      className="btn-secondary" style={{ padding: '5px 12px', fontSize: 10, textDecoration: 'none', gap: 5 }}>
                      <Download size={11} /> PDF
                    </a>
                  ) : (
                    <button className="btn-secondary" style={{ padding: '5px 12px', fontSize: 10, gap: 5, opacity: 0.35 }} disabled>
                      <Download size={11} /> PDF
                    </button>
                  )}
                  {r.report_html ? (
                    <a href={`${BASE_URL}/${r.report_html}`} target="_blank" rel="noopener noreferrer"
                      className="btn-ghost" style={{ padding: '5px 12px', fontSize: 10, textDecoration: 'none', gap: 5 }}>
                      <ExternalLink size={11} /> HTML
                    </a>
                  ) : (
                    <button className="btn-ghost" style={{ padding: '5px 12px', fontSize: 10, gap: 5, opacity: 0.35 }} disabled>
                      <ExternalLink size={11} /> HTML
                    </button>
                  )}
                  {r.report_json ? (
                    <a href={`${BASE_URL}/${r.report_json}`} target="_blank" rel="noopener noreferrer"
                      className="btn-ghost" style={{ padding: '5px 12px', fontSize: 10, textDecoration: 'none', gap: 5 }}>
                      <Download size={11} /> JSON
                    </a>
                  ) : (
                    <button className="btn-ghost" style={{ padding: '5px 12px', fontSize: 10, gap: 5, opacity: 0.35 }} disabled>
                      <Download size={11} /> JSON
                    </button>
                  )}
                </div>
              </div>

              {}
              {r.probabilities && Object.keys(r.probabilities).length > 0 && (
                <div style={{ marginTop: 14, paddingTop: 12, borderTop: '1px solid var(--border)' }}>
                  <div className="section-label" style={{ marginBottom: 8 }}>Posterior Probabilities</div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
                    {Object.entries(r.probabilities)
                      .sort(([, a], [, b]) => normPct(b) - normPct(a))
                      .map(([cls, val]) => {
                        const pct = normPct(val)
                        const isTop = cls === r.prediction
                        return (
                          <div key={cls}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                              <span style={{
                                fontFamily: 'var(--font-mono)', fontSize: 9, textTransform: 'uppercase',
                                color: isTop ? 'var(--text-primary)' : 'var(--text-muted)',
                                fontWeight: isTop ? 700 : 400, letterSpacing: '0.05em',
                              }}>
                                {cls}
                              </span>
                              <span style={{
                                fontFamily: 'var(--font-mono)', fontSize: 9,
                                color: isTop ? 'var(--accent)' : 'var(--text-secondary)',
                                fontWeight: isTop ? 700 : 400,
                              }}>
                                {pct.toFixed(1)}%
                              </span>
                            </div>
                            <div className="progress-bar">
                              <div className="progress-fill" style={{
                                width: `${pct}%`,
                                background: isTop ? 'var(--accent)' : 'rgba(0,212,232,0.25)',
                              }} />
                            </div>
                          </div>
                        )
                      })}
                  </div>
                </div>
              )}

              {}
              <div style={{
                marginTop: 12, paddingTop: 10,
                borderTop: '1px solid var(--border)',
                display: 'flex', alignItems: 'center', gap: 20, flexWrap: 'wrap',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <FileText size={11} color="var(--text-muted)" />
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)' }}>
                    {r.report_id}
                  </span>
                  <CopyButton text={r.report_id || ''} />
                </div>

                {[
                  { label: 'Model', value: r.model },
                  { label: 'Version', value: r.model_version },
                  { label: 'Inference', value: `${r.inference_time_ms}ms` },
                ].map(({ label, value }) => (
                  <div key={label}>
                    <span className="section-label">{label}: </span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-secondary)' }}>
                      {value}
                    </span>
                  </div>
                ))}

                {}
                {tumor && r.confidence > 80 && (
                  <span className="badge badge-critical" style={{ fontSize: 9, marginLeft: 'auto' }}>
                    URGENT REVIEW
                  </span>
                )}
              </div>
            </motion.div>
          )
        })}
      </div>

      {}
      {filtered.length === 0 && (
        <div className="card" style={{ padding: 48, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, opacity: 0.55 }}>
          <FileText size={44} color="var(--text-muted)" strokeWidth={1} />
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-muted)', textAlign: 'center' }}>
            {history.length === 0
              ? 'No analyses yet. Go to Upload & Analyze to run your first scan.'
              : 'No reports match your search.'}
          </span>
        </div>
      )}
    </div>
  )
}
