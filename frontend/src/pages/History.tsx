
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, ArrowRight, Trash2, Clock } from 'lucide-react'
import { motion } from 'framer-motion'
import { useAnalysisStore } from '../store/analysisStore'

function getPredBadgeClass(pred: string) {
  const p = pred.toLowerCase()
  if (p.includes('glioma')) return 'badge badge-glioma'
  if (p.includes('no tumor') || p.includes('notumor')) return 'badge badge-notumor'
  if (p.includes('meningioma')) return 'badge badge-meningioma'
  if (p.includes('pituitary')) return 'badge badge-pituitary'
  return 'badge'
}

export default function History() {
  const navigate = useNavigate()
  const { history, clearHistory } = useAnalysisStore()
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const PER_PAGE = 10

  const displayEntries = history.map((e) => ({
    mrn: e.mrn,
    ts: new Date(e.timestamp).toLocaleString(),
    status: e.status,
    pred: e.result.prediction,
    conf: e.result.confidence,
    ms: e.result.inference_time_ms,
  }))

  const filtered = displayEntries.filter(
    (e) =>
      e.mrn.toLowerCase().includes(search.toLowerCase()) ||
      e.pred.toLowerCase().includes(search.toLowerCase())
  )

  const totalPages = Math.ceil(filtered.length / PER_PAGE)
  const pageData = filtered.slice((page - 1) * PER_PAGE, page * PER_PAGE)

  return (
    <div className="page-content" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{ position: 'relative', flex: 1 }}>
          <Search size={13} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
          <input
            className="input-field"
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(1) }}
            placeholder="Search by MRN or prediction..."
            style={{ paddingLeft: 32 }}
          />
        </div>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
          {filtered.length} entries
        </span>
        {history.length > 0 && (
          <button
            className="btn-ghost"
            style={{ gap: 5, color: 'var(--accent-red)', borderColor: 'rgba(239,68,68,0.2)' }}
            onClick={clearHistory}
          >
            <Trash2 size={12} /> Clear
          </button>
        )}
      </div>

      {}
      <div className="card" style={{ overflow: 'hidden' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Patient MRN</th>
              <th>Timestamp</th>
              <th>Status</th>
              <th>Prediction</th>
              <th>Confidence</th>
              <th>Latency</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {pageData.map((row, i) => (
              <motion.tr
                key={i}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: i * 0.03 }}
              >
                <td>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--accent)', fontWeight: 700, cursor: 'pointer' }}
                    onClick={() => navigate('/reports')}
                  >
                    {row.mrn}
                  </span>
                </td>
                <td>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 5, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>
                    <Clock size={11} />
                    {row.ts}
                  </div>
                </td>
                <td>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span className={`dot ${row.status === 'completed' ? 'dot-green' : 'dot-cyan'}`} />
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: row.status === 'completed' ? 'var(--accent-green)' : 'var(--accent)' }}>
                      {row.status === 'completed' ? 'Completed' : 'In-Progress'}
                    </span>
                  </div>
                </td>
                <td>
                  {row.pred === 'Processing...' ? (
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>Processing...</span>
                  ) : (
                    <span className={getPredBadgeClass(row.pred)}>{row.pred}</span>
                  )}
                </td>
                <td>
                  {row.conf != null ? (
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: row.conf > 85 ? 'var(--accent)' : 'var(--text-secondary)' }}>
                      {row.conf}%
                    </span>
                  ) : (
                    <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>—</span>
                  )}
                </td>
                <td>
                  {row.ms != null ? (
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-secondary)' }}>
                      {row.ms}ms
                    </span>
                  ) : (
                    <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>—</span>
                  )}
                </td>
                <td>
                  <button
                    className="btn-ghost"
                    style={{ padding: '4px 8px', fontSize: 10, gap: 4 }}
                    onClick={() => navigate('/reports')}
                  >
                    View <ArrowRight size={10} />
                  </button>
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      {}
      {totalPages > 1 && (
        <div style={{ display: 'flex', justifyContent: 'center', gap: 6 }}>
          {Array.from({ length: totalPages }).map((_, i) => (
            <button
              key={i}
              className={i + 1 === page ? 'btn-secondary' : 'btn-ghost'}
              style={{ width: 32, height: 32, padding: 0, justifyContent: 'center', fontFamily: 'var(--font-mono)', fontSize: 11 }}
              onClick={() => setPage(i + 1)}
            >
              {i + 1}
            </button>
          ))}
        </div>
      )}

      {}
      {filtered.length === 0 && (
        <div className="card" style={{ padding: 40, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, opacity: 0.6 }}>
          <Clock size={40} color="var(--text-muted)" strokeWidth={1} />
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-muted)', textAlign: 'center' }}>
            {history.length === 0
              ? 'No analyses yet. Go to Upload & Analyze to run your first scan.'
              : 'No entries match your search.'}
          </span>
        </div>
      )}
    </div>
  )
}
