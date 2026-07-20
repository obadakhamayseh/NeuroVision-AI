
import { NavLink, useNavigate } from 'react-router-dom'
import {
  LayoutDashboard,
  Upload,
  FileText,
  History,
  Plus,
  Activity,
} from 'lucide-react'

const NAV_ITEMS = [
  { to: '/', label: 'Overview', icon: LayoutDashboard },
  { to: '/upload', label: 'Upload & Analyze', icon: Upload },
  { to: '/reports', label: 'Reports', icon: FileText },
  { to: '/history', label: 'History', icon: History },
]

export default function Sidebar() {
  const navigate = useNavigate()

  return (
    <aside
      style={{
        width: 208,
        minWidth: 208,
        height: '100%',
        background: 'var(--bg-surface)',
        borderRight: '1px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
        padding: '20px 0',
        flexShrink: 0,
      }}
    >
      {}
      <div style={{ padding: '0 20px 24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div
            style={{
              width: 32,
              height: 32,
              background: 'linear-gradient(135deg, var(--accent), #0090a0)',
              borderRadius: 6,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 0 16px rgba(0,212,232,0.4)',
            }}
          >
            <Activity size={18} color="#000" strokeWidth={2.5} />
          </div>
          <div>
            <div
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 14,
                fontWeight: 700,
                color: 'var(--text-primary)',
                lineHeight: 1.1,
              }}
            >
              BrainCircuit
            </div>
            <div
              style={{
                fontSize: 9,
                color: 'var(--text-muted)',
                letterSpacing: '0.12em',
                textTransform: 'uppercase',
                fontFamily: 'var(--font-mono)',
              }}
            >
              NEUROVISION AI
            </div>
          </div>
        </div>
      </div>

      {}
      <nav
        style={{
          flex: 1,
          padding: '0 12px',
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
        }}
      >
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            style={({ isActive }) => ({
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              padding: '10px 12px',
              borderRadius: 6,
              textDecoration: 'none',
              color: isActive ? 'var(--accent)' : 'var(--text-secondary)',
              background: isActive ? 'rgba(0,212,232,0.08)' : 'transparent',
              borderLeft: isActive
                ? '2px solid var(--accent)'
                : '2px solid transparent',
              fontFamily: 'var(--font-mono)',
              fontSize: 12,
              fontWeight: isActive ? 700 : 400,
              letterSpacing: '0.02em',
              transition: 'all 0.15s',
            })}
          >
            <Icon size={15} strokeWidth={1.5} />
            {label}
          </NavLink>
        ))}
      </nav>

      {}
      <div style={{ padding: '16px 16px 16px' }}>
        <button
          className="btn-primary"
          style={{ width: '100%', justifyContent: 'center' }}
          onClick={() => navigate('/upload')}
        >
          <Plus size={14} />
          New Analysis
        </button>
      </div>
    </aside>
  )
}
