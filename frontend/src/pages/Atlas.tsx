
import { useState } from 'react'
import { Eye, Printer, Download, ZoomIn, ZoomOut, RotateCcw, Move, ChevronDown } from 'lucide-react'
import { motion } from 'framer-motion'

const REGIONS = [
  { name: 'Frontal Lobe', sub: 'Executive Function', color: '#00d4e8', active: true },
  { name: 'Parietal Lobe', sub: 'Sensory Processing', color: '#7a9bb5' },
  { name: 'Temporal Lobe', sub: 'Auditory / Memory', color: '#7a9bb5' },
  { name: 'Occipital Lobe', sub: 'Visual Processing', color: '#7a9bb5' },
  { name: 'Cerebellum', sub: 'Coordination', color: '#7a9bb5' },
  { name: 'Brainstem', sub: 'Vitals Control', color: '#7a9bb5' },
]

const SESSIONS = [
  'Historical Session: 24-05-23',
  'Historical Session: 23-05-23',
  'Historical Session: 22-05-23',
  'Live Session',
]

export default function Atlas() {
  const [sagittal, setSagittal] = useState(45)
  const [axial, setAxial] = useState(62)
  const [coronal, setCoronal] = useState(28)
  const [ghostIsolation, setGhostIsolation] = useState(true)
  const [session, setSession] = useState(SESSIONS[0])
  const [activeRegion, setActiveRegion] = useState(REGIONS[0])
  const [zoom, setZoom] = useState(100)

  return (
    <div
      style={{
        height: '100%',
        display: 'grid',
        gridTemplateColumns: '260px 1fr 240px',
        gridTemplateRows: '1fr',
        overflow: 'hidden',
      }}
    >
      {}
      <div
        style={{
          borderRight: '1px solid var(--border)',
          padding: 20,
          display: 'flex',
          flexDirection: 'column',
          gap: 16,
          overflowY: 'auto',
        }}
      >
        {}
        <div className="card" style={{ padding: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 700, color: 'var(--accent)' }}>
              Cross-section
            </span>
            <button className="btn-ghost" style={{ padding: '3px 7px' }}>
              <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
                <rect x="1" y="1" width="6" height="6" stroke="currentColor" strokeWidth="1.5" rx="1"/>
                <rect x="9" y="1" width="6" height="6" stroke="currentColor" strokeWidth="1.5" rx="1"/>
                <rect x="1" y="9" width="6" height="6" stroke="currentColor" strokeWidth="1.5" rx="1"/>
                <rect x="9" y="9" width="6" height="6" stroke="currentColor" strokeWidth="1.5" rx="1"/>
              </svg>
            </button>
          </div>

          {[
            { label: 'SAGITTAL (X)', value: sagittal, set: setSagittal },
            { label: 'AXIAL (Y)', value: axial, set: setAxial },
            { label: 'CORONAL (Z)', value: coronal, set: setCoronal },
          ].map(({ label, value, set }) => (
            <div key={label} style={{ marginBottom: 18 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
                  {label}
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-secondary)' }}>
                  {value}%
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={100}
                value={value}
                onChange={(e) => set(Number(e.target.value))}
              />
            </div>
          ))}
        </div>

        {}
        <div className="card" style={{ padding: 16 }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700, color: 'var(--text-muted)', letterSpacing: '0.12em', textTransform: 'uppercase', display: 'block', marginBottom: 14 }}>
            Visualization Modes
          </span>

          {}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-secondary)' }}>
              Ghost Isolation
            </span>
            <label className="toggle">
              <input
                type="checkbox"
                checked={ghostIsolation}
                onChange={(e) => setGhostIsolation(e.target.checked)}
              />
              <span className="toggle-slider" />
            </label>
          </div>

          {}
          <div>
            <div className="section-label" style={{ marginBottom: 6 }}>Session Projection Overlay</div>
            <div style={{ position: 'relative' }}>
              <select
                className="input-field"
                value={session}
                onChange={(e) => setSession(e.target.value)}
                style={{ paddingRight: 32 }}
              >
                {SESSIONS.map((s) => <option key={s}>{s}</option>)}
              </select>
              <ChevronDown size={12} style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)', pointerEvents: 'none' }} />
            </div>
          </div>
        </div>
      </div>

      {}
      <div style={{ position: 'relative', overflow: 'hidden', background: 'var(--bg-primary)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>

        {}
        <motion.div
          animate={{ rotate: [0, 1, -1, 0] }}
          transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }}
          style={{ position: 'relative' }}
        >
          <svg width="380" height="380" viewBox="0 0 380 380" fill="none" xmlns="http://www.w3.org/2000/svg">
            <ellipse cx="190" cy="190" rx="150" ry="130" stroke="rgba(0,212,232,0.08)" strokeWidth="1" />
            <ellipse cx="190" cy="190" rx="120" ry="104" stroke="rgba(0,212,232,0.12)" strokeWidth="1" />
            <ellipse cx="190" cy="190" rx="90" ry="78" stroke="rgba(0,212,232,0.18)" strokeWidth="1" />

            {}
            <path
              d="M130 120 C100 130, 70 160, 75 200 C80 240, 110 260, 150 265 C170 268, 185 265, 190 260"
              stroke="rgba(0,212,232,0.5)" strokeWidth="1.5" fill="none" strokeLinecap="round"
            />
            {}
            <path
              d="M250 120 C280 130, 310 160, 305 200 C300 240, 270 260, 230 265 C210 268, 195 265, 190 260"
              stroke="rgba(0,212,232,0.5)" strokeWidth="1.5" fill="none" strokeLinecap="round"
            />
            {}
            <path
              d="M130 120 C140 80, 170 65, 190 65 C210 65, 240 80, 250 120"
              stroke="rgba(0,212,232,0.5)" strokeWidth="1.5" fill="none" strokeLinecap="round"
            />
            {}
            <path
              d="M155 175 C170 165, 210 165, 225 175"
              stroke="rgba(0,212,232,0.3)" strokeWidth="1" fill="none"
            />
            {}
            <path d="M120 160 C130 155, 145 158, 150 165" stroke="rgba(0,212,232,0.2)" strokeWidth="1" fill="none" />
            <path d="M105 185 C115 180, 130 183, 135 190" stroke="rgba(0,212,232,0.2)" strokeWidth="1" fill="none" />
            <path d="M255 165 C250 158, 235 156, 230 163" stroke="rgba(0,212,232,0.2)" strokeWidth="1" fill="none" />
            <path d="M270 190 C265 183, 250 181, 245 188" stroke="rgba(0,212,232,0.2)" strokeWidth="1" fill="none" />
            <path d="M160 100 C165 93, 178 92, 182 98" stroke="rgba(0,212,232,0.2)" strokeWidth="1" fill="none" />
            <path d="M200 95 C205 89, 218 90, 222 96" stroke="rgba(0,212,232,0.2)" strokeWidth="1" fill="none" />

            {}
            <ellipse
              cx="190" cy="110"
              rx="52" ry="38"
              fill="rgba(0,212,232,0.06)"
              stroke="rgba(0,212,232,0.4)"
              strokeWidth="1.5"
              strokeDasharray="4 3"
            />

            {}
            <line
              x1={80 + sagittal * 2.2}
              y1="75"
              x2={80 + sagittal * 2.2}
              y2="305"
              stroke="rgba(0,212,232,0.25)"
              strokeWidth="0.8"
              strokeDasharray="3 3"
            />
            <line
              x1="75"
              y1={75 + axial * 2.1}
              x2="305"
              y2={75 + axial * 2.1}
              stroke="rgba(245,158,11,0.25)"
              strokeWidth="0.8"
              strokeDasharray="3 3"
            />

            {}
            {ghostIsolation && (
              <rect x="75" y="75" width="230" height="230" fill="url(#scanGrad)" opacity="0.3" />
            )}
            <defs>
              <linearGradient id="scanGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="rgba(0,212,232,0)" />
                <stop offset="50%" stopColor="rgba(0,212,232,0.04)" />
                <stop offset="100%" stopColor="rgba(0,212,232,0)" />
              </linearGradient>
            </defs>

            {}
            <text x="82" y="300" fill="rgba(0,212,232,0.5)" fontSize="9" fontFamily="Space Mono">
              X: {sagittal.toFixed(2)} | Y: {axial.toFixed(2)} | Z: {coronal.toFixed(2)}
            </text>
          </svg>
        </motion.div>

        {}
        <div
          style={{
            position: 'absolute', bottom: 20, left: '50%', transform: 'translateX(-50%)',
            display: 'flex', gap: 6,
          }}
        >
          {[
            { icon: ZoomIn, action: () => setZoom(Math.min(200, zoom + 10)) },
            { icon: ZoomOut, action: () => setZoom(Math.max(50, zoom - 10)) },
            { icon: RotateCcw, action: () => { setSagittal(45); setAxial(62); setCoronal(28) } },
            { icon: Move, action: () => {} },
          ].map(({ icon: Icon, action }, i) => (
            <button
              key={i}
              className="btn-ghost"
              style={{ padding: '8px', width: 36, height: 36, justifyContent: 'center' }}
              onClick={action}
            >
              <Icon size={14} />
            </button>
          ))}
        </div>

        {}
        <motion.div
          className="card"
          animate={{ opacity: 1 }}
          style={{
            position: 'absolute', bottom: 20, left: 20,
            padding: '14px 16px', minWidth: 260,
            border: '1px solid rgba(0,212,232,0.25)',
            background: 'rgba(14,22,33,0.92)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 700, color: 'var(--accent)' }}>
              {activeRegion.name}
            </span>
            <span className="badge badge-notumor" style={{ fontSize: 9 }}>98.2% Clarity</span>
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)', letterSpacing: '0.06em', marginBottom: 8, textTransform: 'uppercase' }}>
            Active Region
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 12 }}>
            Primary center for executive function, motor control, and personality.
            Currently displaying axial cross-section through the precentral gyrus.
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div>
              <div className="section-label" style={{ marginBottom: 3 }}>Volume</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-primary)' }}>352.4 cm³</div>
            </div>
            <div>
              <div className="section-label" style={{ marginBottom: 3 }}>Status</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--accent-green)' }}>Normal</div>
            </div>
          </div>
          <div style={{ marginTop: 6, fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--accent)', letterSpacing: '0.06em' }}>
            AI SIGNAL LOCKED
          </div>
        </motion.div>
      </div>

      {}
      <div
        style={{
          borderLeft: '1px solid var(--border)',
          padding: 20,
          display: 'flex',
          flexDirection: 'column',
          gap: 16,
        }}
      >
        <div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>
            Region Directory
          </div>
          <div className="section-label">Select Anatomical Group</div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {REGIONS.map((region) => (
            <button
              key={region.name}
              onClick={() => setActiveRegion(region)}
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '12px 14px', borderRadius: 6,
                background: activeRegion.name === region.name ? 'rgba(0,212,232,0.08)' : 'transparent',
                border: `1px solid ${activeRegion.name === region.name ? 'rgba(0,212,232,0.3)' : 'var(--border)'}`,
                cursor: 'pointer', textAlign: 'left',
                transition: 'all 0.15s',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div
                  style={{
                    width: 3, height: 32, borderRadius: 2,
                    background: activeRegion.name === region.name ? 'var(--accent)' : 'var(--border)',
                  }}
                />
                <div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: activeRegion.name === region.name ? 'var(--accent)' : 'var(--text-secondary)', fontWeight: activeRegion.name === region.name ? 700 : 400 }}>
                    {region.name}
                  </div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)' }}>
                    {region.sub}
                  </div>
                </div>
              </div>
              {activeRegion.name === region.name && <Eye size={13} color="var(--accent)" />}
            </button>
          ))}
        </div>

        {}
        <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: 8 }}>
          <button className="btn-secondary" style={{ width: '100%', justifyContent: 'center', gap: 6 }}>
            <Printer size={13} /> Generate Snapshot
          </button>
          <button className="btn-ghost" style={{ width: '100%', justifyContent: 'center', gap: 6 }}>
            <Download size={13} /> Export Coordinates
          </button>
        </div>
      </div>
    </div>
  )
}
