
import { useState } from 'react'
import { useLocation } from 'react-router-dom'
import { Search, RefreshCw } from 'lucide-react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { fetchHealth } from '../../api/health'

const PAGE_TITLES: Record<string, string> = {
  '/': 'Overview',
  '/upload': 'Upload & Analyze',
  '/reports': 'Reports',
  '/history': 'History',
}

export default function TopBar() {
  const location = useLocation()
  const title = PAGE_TITLES[location.pathname] || 'NeuroVision AI'
  const [search, setSearch] = useState('')
  const queryClient = useQueryClient()

  const { data: health, isFetching } = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: 30000,
    retry: false,
  })

  const handleSync = () => {
    queryClient.invalidateQueries({ queryKey: ['health'] })
  }

  return (
    <header
      style={{
        height: 56,
        background: 'var(--bg-surface)',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        padding: '0 24px',
        gap: 16,
        flexShrink: 0,
      }}
    >
      {}
      <h1
        style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 18,
          fontWeight: 700,
          color: 'var(--accent)',
          letterSpacing: '-0.01em',
          minWidth: 200,
          margin: 0,
        }}
      >
        {title}
      </h1>

      {}
      <div style={{ flex: 1 }} />

      {}
      <div style={{ position: 'relative' }}>
        <Search
          size={13}
          style={{
            position: 'absolute',
            left: 10,
            top: '50%',
            transform: 'translateY(-50%)',
            color: 'var(--text-muted)',
          }}
        />
        <input
          className="search-input"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search Patient ID..."
        />
      </div>

      {}
      <button
        className="btn-secondary"
        style={{ padding: '7px 14px', gap: 6 }}
        onClick={handleSync}
        title="Refresh backend connection"
      >
        <RefreshCw
          size={13}
          style={{
            animation: isFetching ? 'spin-slow 1s linear infinite' : 'none',
          }}
        />
        Sync Data
      </button>

      {}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          fontFamily: 'var(--font-mono)',
          fontSize: 10,
          color: 'var(--text-muted)',
          minWidth: 80,
        }}
      >
        <span
          className={`dot ${health ? 'dot-green' : 'dot-red'}`}
          style={{ width: 7, height: 7 }}
        />
        {health ? (
          <span>v{health.version}</span>
        ) : (
          <span style={{ color: 'var(--accent-red)' }}>Offline</span>
        )}
      </div>
    </header>
  )
}
