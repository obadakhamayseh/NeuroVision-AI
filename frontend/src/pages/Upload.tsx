
import { useState, useCallback, useRef } from 'react'
import {
  Upload as UploadIcon,
  Zap,
  CheckCircle,
  Circle,
  Loader,
  Maximize2,
  X,
  Download,
  ExternalLink,
  ScanLine,
  Brain,
  Copy,
  Check,
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { runPrediction } from '../api/predict'
import { useAnalysisStore } from '../store/analysisStore'

const API_BASE = 'http://127.0.0.1:8000'

const PIPELINE_STEPS = [
  'Queued',
  'Preprocessing',
  'Normalization (EfficientNet-B0)',
  'Feature Extraction',
  'Classification',
  'Grad-CAM Attribution',
]

function isNoTumor(pred: string) {
  return pred.toLowerCase().includes('no tumor') || pred.toLowerCase().includes('notumor')
}

function getPredBadgeClass(pred: string) {
  const p = pred.toLowerCase()
  if (p.includes('glioma')) return 'badge badge-glioma'
  if (p.includes('no tumor') || p.includes('notumor')) return 'badge badge-notumor'
  if (p.includes('meningioma')) return 'badge badge-meningioma'
  if (p.includes('pituitary')) return 'badge badge-pituitary'
  return 'badge'
}

export default function Upload() {
  const { latest, setLatest } = useAnalysisStore()

  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const [isRunning, setIsRunning] = useState(false)
  const [currentStep, setCurrentStep] = useState(-1)
  const [error, setError] = useState<string | null>(null)

  const [viewMode, setViewMode] = useState<'raw' | 'ai'>('raw')
  const [fullscreen, setFullscreen] = useState(false)

  const [copied, setCopied] = useState(false)

  const handleFile = (f: File) => {
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setCurrentStep(-1)
    setError(null)
    setViewMode('raw')
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }, [])

  const simulateSteps = async () => {
    for (let i = 0; i < PIPELINE_STEPS.length; i++) {
      setCurrentStep(i)
      await new Promise((r) => setTimeout(r, 600 + Math.random() * 500))
    }
  }

  const runAnalysis = async () => {
    if (!file) return
    setIsRunning(true)
    setError(null)
    setCurrentStep(0)

    const stepPromise = simulateSteps()
    try {
      const result = await runPrediction(file)
      await stepPromise
      setCurrentStep(PIPELINE_STEPS.length)
      setLatest(result, file.name)
      
      if (!isNoTumor(result.prediction)) {
        setViewMode('ai')
      } else {
        setViewMode('raw')
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Analysis failed. Is the backend running?'
      setError(msg)
      setCurrentStep(-1)
    } finally {
      setIsRunning(false)
    }
  }

  const clearFile = () => {
    setFile(null)
    setPreview(null)
    setCurrentStep(-1)
    setError(null)
    setViewMode('raw')
    if (fileRef.current) fileRef.current.value = ''
  }

  const copyReportId = () => {
    if (result?.report_id) {
      navigator.clipboard.writeText(result.report_id).then(() => {
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
      })
    }
  }

  const result = latest

  const hasTumor = result && !isNoTumor(result.prediction)
  const rawSrc = preview
  const aiSrc = hasTumor && result?.overlay_image
    ? `${API_BASE}/${result.overlay_image}`
    : hasTumor && result?.heatmap_image
      ? `${API_BASE}/${result.heatmap_image}`
      : null

  const currentSrc = viewMode === 'ai' && aiSrc ? aiSrc : rawSrc

  return (
    <>
      {}
      <AnimatePresence>
        {fullscreen && currentSrc && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.96)',
              zIndex: 1000, display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center',
            }}
          >
            <button
              className="btn-ghost"
              style={{ position: 'absolute', top: 20, right: 20, gap: 6 }}
              onClick={() => setFullscreen(false)}
            >
              <X size={16} /> Close
            </button>
            <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
              <button
                className={viewMode === 'raw' ? 'btn-secondary' : 'btn-ghost'}
                onClick={() => setViewMode('raw')}
              >
                Raw Scan
              </button>
              {aiSrc && (
                <button
                  className={viewMode === 'ai' ? 'btn-secondary' : 'btn-ghost'}
                  onClick={() => setViewMode('ai')}
                >
                  <Brain size={13} /> Grad-CAM
                </button>
              )}
            </div>
            <img
              src={currentSrc}
              alt="MRI viewer"
              style={{
                maxWidth: '90vw', maxHeight: '80vh',
                objectFit: 'contain', borderRadius: 8,
                border: '1px solid var(--border)',
              }}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {}
      <div
        className="page-content"
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 20,
          alignItems: 'start',
          minHeight: 0,
        }}
      >
        {}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

          {}
          <div className="card" style={{ padding: 20 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
              <div style={{
                width: 24, height: 24, borderRadius: 5,
                background: 'rgba(0,212,232,0.15)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <ScanLine size={13} color="var(--accent)" />
              </div>
              <span style={{
                fontFamily: 'var(--font-mono)', fontSize: 12,
                color: 'var(--accent)', fontWeight: 700,
              }}>
                MRI Scan Upload
              </span>
            </div>

            {}
            <div
              className={`drop-zone ${dragging ? 'active' : ''}`}
              style={{ height: 160, marginBottom: 14, position: 'relative' }}
              onClick={() => fileRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
            >
              <input
                ref={fileRef}
                type="file"
                accept="image/*,.dcm"
                style={{ display: 'none' }}
                onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f) }}
              />
              {file ? (
                <>
                  <CheckCircle size={28} color="var(--accent-green)" />
                  <span style={{
                    fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--accent-green)',
                    textAlign: 'center', padding: '0 12px', wordBreak: 'break-all',
                  }}>
                    {file.name}
                  </span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)' }}>
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </span>
                  <button
                    className="btn-ghost"
                    style={{ position: 'absolute', top: 8, right: 8, padding: '3px 6px' }}
                    onClick={(e) => { e.stopPropagation(); clearFile() }}
                  >
                    <X size={11} />
                  </button>
                </>
              ) : (
                <>
                  <UploadIcon size={28} color="var(--text-muted)" strokeWidth={1.5} />
                  <span style={{
                    fontFamily: 'var(--font-mono)', fontSize: 11,
                    color: 'var(--text-secondary)', letterSpacing: '0.08em', textTransform: 'uppercase',
                  }}>
                    Drag & Drop MRI Scan
                  </span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)' }}>
                    JPG · PNG · BMP · TIFF &nbsp;|&nbsp; Max 15 MB
                  </span>
                </>
              )}
            </div>

            {}
            <button
              className="btn-primary"
              style={{
                width: '100%', justifyContent: 'center',
                padding: '13px 20px', fontSize: 13,
                opacity: !file || isRunning ? 0.6 : 1,
              }}
              onClick={runAnalysis}
              disabled={!file || isRunning}
            >
              {isRunning
                ? <Loader size={16} style={{ animation: 'spin-slow 1s linear infinite' }} />
                : <Zap size={16} />}
              {isRunning ? 'ANALYZING...' : 'RUN ANALYSIS'}
            </button>

            {}
            {error && (
              <div style={{
                marginTop: 12, padding: '10px 14px',
                background: 'rgba(239,68,68,0.08)',
                border: '1px solid rgba(239,68,68,0.25)',
                borderRadius: 6, display: 'flex', alignItems: 'flex-start', gap: 8,
              }}>
                <X size={13} color="var(--accent-red)" style={{ flexShrink: 0, marginTop: 1 }} />
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--accent-red)', lineHeight: 1.5 }}>
                  {error}
                </span>
              </div>
            )}
          </div>

          {}
          <div className="grid-2">
            <div className="card" style={{ padding: '12px 16px' }}>
              <div className="section-label" style={{ marginBottom: 5 }}>Network</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--accent)', fontWeight: 700 }}>
                EfficientNet-B0
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)', marginTop: 2 }}>
                Quantized INT8 Engine
              </div>
            </div>
            <div className="card" style={{ padding: '12px 16px' }}>
              <div className="section-label" style={{ marginBottom: 5 }}>Classes</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--accent)', fontWeight: 700 }}>
                4-Class
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)', marginTop: 2 }}>
                Glioma · Meningioma · Pituitary · None
              </div>
            </div>
          </div>

          {}
          <div className="card" style={{ padding: '16px 20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
              <div className="section-label">Inference Stream</div>
              {isRunning && <span className="badge badge-live">LIVE-PROCESS</span>}
              {currentStep >= PIPELINE_STEPS.length && !isRunning && (
                <span className="badge badge-notumor">COMPLETE</span>
              )}
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {PIPELINE_STEPS.map((step, i) => {
                const done = currentStep > i
                const active = currentStep === i
                return (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <span className={`dot ${done ? 'dot-cyan' : active ? 'dot-orange' : 'dot-dim'}`} />
                      <span style={{
                        fontFamily: 'var(--font-mono)', fontSize: 11,
                        color: done ? 'var(--text-primary)' : active ? 'var(--accent-orange)' : 'var(--text-muted)',
                        transition: 'color 0.3s',
                      }}>
                        {step}
                      </span>
                    </div>
                    {done && <CheckCircle size={13} color="var(--accent)" />}
                    {active && <Loader size={13} color="var(--accent-orange)" style={{ animation: 'spin-slow 1s linear infinite' }} />}
                  </div>
                )
              })}
            </div>

            {(isRunning || currentStep >= PIPELINE_STEPS.length) && currentStep >= 0 && (
              <div className="progress-bar" style={{ marginTop: 14 }}>
                <div
                  className="progress-fill"
                  style={{
                    width: `${currentStep >= PIPELINE_STEPS.length
                      ? 100
                      : ((currentStep + 1) / PIPELINE_STEPS.length) * 100}%`,
                  }}
                />
              </div>
            )}
          </div>
        </div>

        {}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

          {}
          <div className="card" style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            {}
            <div style={{
              padding: '10px 14px',
              borderBottom: '1px solid var(--border)',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            }}>
              <span style={{
                fontFamily: 'var(--font-mono)', fontSize: 9,
                color: 'var(--text-muted)', letterSpacing: '0.1em', textTransform: 'uppercase',
              }}>
                {viewMode === 'ai' ? 'Grad-CAM Heatmap Overlay' : 'Raw MRI Scan'}
              </span>
              <div style={{ display: 'flex', gap: 6 }}>
                <button
                  className={viewMode === 'raw' ? 'btn-secondary' : 'btn-ghost'}
                  style={{ padding: '4px 10px', fontSize: 10 }}
                  onClick={() => setViewMode('raw')}
                  disabled={!rawSrc}
                >
                  Raw Scan
                </button>
                <button
                  className={viewMode === 'ai' && aiSrc ? 'btn-secondary' : 'btn-ghost'}
                  style={{ padding: '4px 10px', fontSize: 10, gap: 5, opacity: aiSrc ? 1 : 0.35 }}
                  onClick={() => { if (aiSrc) setViewMode('ai') }}
                  disabled={!aiSrc}
                  title={
                    !result
                      ? 'Run analysis first'
                      : isNoTumor(result.prediction)
                        ? 'Grad-CAM not available for No Tumor prediction'
                        : 'Show Grad-CAM heatmap'
                  }
                >
                  <Brain size={11} /> Grad-CAM
                </button>
                <button
                  className="btn-ghost"
                  style={{ padding: '4px 8px', opacity: currentSrc ? 1 : 0.35 }}
                  onClick={() => setFullscreen(true)}
                  disabled={!currentSrc}
                >
                  <Maximize2 size={11} />
                </button>
              </div>
            </div>

            {}
            <div style={{
              background: '#000',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              minHeight: 260, position: 'relative',
            }}>
              <AnimatePresence mode="wait">
                {currentSrc ? (
                  <motion.img
                    key={currentSrc}
                    src={currentSrc}
                    alt={viewMode === 'ai' ? 'Grad-CAM overlay' : 'MRI scan'}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.25 }}
                    style={{ maxWidth: '100%', maxHeight: 320, objectFit: 'contain' }}
                    onError={(e) => {
                      if (viewMode === 'ai' && rawSrc) {
                        ; (e.target as HTMLImageElement).src = rawSrc
                        setViewMode('raw')
                      }
                    }}
                  />
                ) : (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10, opacity: 0.4, padding: 40 }}
                  >
                    <Circle size={36} color="var(--text-muted)" strokeWidth={1} />
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
                      Upload a scan to preview
                    </span>
                  </motion.div>
                )}
              </AnimatePresence>

              {}
              {viewMode === 'ai' && aiSrc && (
                <div style={{
                  position: 'absolute', bottom: 10, left: 10,
                  background: 'rgba(0,0,0,0.75)',
                  border: '1px solid rgba(0,212,232,0.3)',
                  borderRadius: 4, padding: '4px 8px',
                  fontFamily: 'var(--font-mono)', fontSize: 9,
                  color: 'var(--accent)', letterSpacing: '0.06em',
                }}>
                  AI ATTRIBUTION · GRAD-CAM
                </div>
              )}

              {}
              {result && isNoTumor(result.prediction) && (
                <div style={{
                  position: 'absolute', bottom: 10, left: 10,
                  background: 'rgba(0,0,0,0.75)',
                  border: '1px solid rgba(16,185,129,0.3)',
                  borderRadius: 4, padding: '4px 8px',
                  fontFamily: 'var(--font-mono)', fontSize: 9,
                  color: 'var(--accent-green)', letterSpacing: '0.06em',
                }}>
                  NO TUMOR DETECTED · GRAD-CAM N/A
                </div>
              )}
            </div>
          </div>

          {}
          <AnimatePresence>
            {result && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
                style={{ display: 'flex', flexDirection: 'column', gap: 12 }}
              >
                {}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  {}
                  <div className="card" style={{
                    padding: '16px 18px',
                    background: isNoTumor(result.prediction)
                      ? 'rgba(16,185,129,0.05)' : 'rgba(239,68,68,0.05)',
                    borderColor: isNoTumor(result.prediction)
                      ? 'rgba(16,185,129,0.2)' : 'rgba(239,68,68,0.2)',
                  }}>
                    <div className="section-label" style={{ marginBottom: 8 }}>Prediction</div>
                    <div style={{
                      fontFamily: 'var(--font-mono)', fontSize: 16, fontWeight: 700,
                      color: 'var(--text-primary)', marginBottom: 8,
                    }}>
                      {result.prediction}
                    </div>
                    <span className={getPredBadgeClass(result.prediction)}>
                      {result.prediction}
                    </span>
                    {!isNoTumor(result.prediction) && result.confidence > 80 && (
                      <div style={{ marginTop: 6 }}>
                        <span className="badge badge-critical" style={{ fontSize: 9 }}>URGENT REVIEW</span>
                      </div>
                    )}
                  </div>

                  {}
                  <div className="card" style={{ padding: '16px 18px' }}>
                    <div className="section-label" style={{ marginBottom: 8 }}>Confidence</div>
                    <div style={{
                      fontFamily: 'var(--font-mono)', fontSize: 28, fontWeight: 700,
                      color: 'var(--accent)', lineHeight: 1,
                    }}>
                      {result.confidence}%
                    </div>
                    <div className="progress-bar" style={{ marginTop: 10 }}>
                      <div className="progress-fill" style={{ width: `${result.confidence}%` }} />
                    </div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)', marginTop: 6 }}>
                      {result.confidence_level}
                    </div>
                  </div>
                </div>

                {}
                <div className="card" style={{ padding: '16px 18px' }}>
                  <div className="section-label" style={{ marginBottom: 12 }}>Posterior Probabilities</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                    {Object.entries(result.probabilities)
                      .sort(([, a], [, b]) => b - a)
                      .map(([k, v]) => {
                        const pctVal = typeof v === 'number' && v <= 1 ? v * 100 : v
                        const isTop = k === result.prediction
                        return (
                          <div key={k}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                              <span style={{
                                fontFamily: 'var(--font-mono)', fontSize: 11,
                                color: isTop ? 'var(--text-primary)' : 'var(--text-muted)',
                                fontWeight: isTop ? 700 : 400,
                              }}>
                                {k}
                              </span>
                              <span style={{
                                fontFamily: 'var(--font-mono)', fontSize: 11,
                                color: isTop ? 'var(--accent)' : 'var(--text-secondary)',
                                fontWeight: isTop ? 700 : 400,
                              }}>
                                {pctVal.toFixed(2)}%
                              </span>
                            </div>
                            <div className="progress-bar">
                              <div
                                className="progress-fill"
                                style={{
                                  width: `${pctVal}%`,
                                  background: isTop ? 'var(--accent)' : 'rgba(0,212,232,0.3)',
                                }}
                              />
                            </div>
                          </div>
                        )
                      })}
                  </div>
                </div>

                {}
                <div className="card" style={{ padding: '12px 16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                    <div className="section-label" style={{ flex: 1 }}>Reports</div>
                    {}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)' }}>
                        ID: {result.report_id?.substring(0, 18)}...
                      </span>
                      <button
                        className="btn-ghost"
                        style={{ padding: '3px 6px', gap: 4, fontSize: 9 }}
                        onClick={copyReportId}
                        title="Copy full Report ID"
                      >
                        {copied ? <Check size={11} color="var(--accent-green)" /> : <Copy size={11} />}
                        {copied ? 'Copied!' : 'Copy ID'}
                      </button>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {result.report_pdf ? (
                      <a
                        href={`${API_BASE}/${result.report_pdf}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="btn-secondary"
                        style={{ padding: '6px 14px', fontSize: 10, textDecoration: 'none', gap: 5 }}
                      >
                        <Download size={11} /> PDF Report
                      </a>
                    ) : (
                      <button className="btn-secondary" style={{ padding: '6px 14px', fontSize: 10, gap: 5, opacity: 0.35 }} disabled>
                        <Download size={11} /> PDF Report
                      </button>
                    )}

                    {result.report_html ? (
                      <a
                        href={`${API_BASE}/${result.report_html}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="btn-ghost"
                        style={{ padding: '6px 14px', fontSize: 10, textDecoration: 'none', gap: 5 }}
                      >
                        <ExternalLink size={11} /> HTML Report
                      </a>
                    ) : (
                      <button className="btn-ghost" style={{ padding: '6px 14px', fontSize: 10, gap: 5, opacity: 0.35 }} disabled>
                        <ExternalLink size={11} /> HTML Report
                      </button>
                    )}

                    {result.report_json ? (
                      <a
                        href={`${API_BASE}/${result.report_json}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="btn-ghost"
                        style={{ padding: '6px 14px', fontSize: 10, textDecoration: 'none', gap: 5 }}
                      >
                        <Download size={11} /> JSON
                      </a>
                    ) : (
                      <button className="btn-ghost" style={{ padding: '6px 14px', fontSize: 10, gap: 5, opacity: 0.35 }} disabled>
                        <Download size={11} /> JSON
                      </button>
                    )}
                  </div>

                  {}
                  <div style={{
                    marginTop: 10, paddingTop: 10,
                    borderTop: '1px solid var(--border)',
                    display: 'flex', gap: 20, flexWrap: 'wrap',
                  }}>
                    {[
                      { label: 'Model', value: result.model },
                      { label: 'Version', value: result.model_version },
                      { label: 'Inference', value: `${result.inference_time_ms}ms` },
                      { label: 'Confidence Level', value: result.confidence_level },
                    ].map(({ label, value }) => (
                      <div key={label}>
                        <span className="section-label">{label}: </span>
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-secondary)' }}>
                          {value}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {}
          {!result && !isRunning && (
            <div className="card" style={{
              padding: 32, display: 'flex', flexDirection: 'column',
              alignItems: 'center', gap: 10, opacity: 0.5,
            }}>
              <Circle size={36} color="var(--text-muted)" strokeWidth={1} />
              <span style={{
                fontFamily: 'var(--font-mono)', fontSize: 11,
                color: 'var(--text-muted)', textAlign: 'center',
              }}>
                Upload an MRI scan and click RUN ANALYSIS<br />
                to see prediction results and Grad-CAM heatmap
              </span>
            </div>
          )}
        </div>
      </div>
    </>
  )
}
