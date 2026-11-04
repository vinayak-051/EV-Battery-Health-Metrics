import SohGauge from './SohGauge.jsx'

const STATUS_COLORS = {
  Excellent: '#34D399',
  Good: '#34D399',
  Fair: '#FBBF24',
  Poor: '#F87171',
  Critical: '#F87171',
}

const STATUS_MSG = {
  Excellent: 'Battery is in peak condition. No action needed.',
  Good: 'Battery is healthy. Continue regular monitoring.',
  Fair: 'Battery is degrading. Plan for replacement within next maintenance cycle.',
  Poor: 'Battery health is low. Schedule replacement soon.',
  Critical: 'Battery has reached end-of-life. Replace immediately.',
}

function EmptyState() {
  return (
    <div className="empty-state">
      <div className="empty-icon">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <rect x="2" y="7" width="16" height="10" rx="2" />
          <path d="M18 11h2a1 1 0 0 1 1 1v0a1 1 0 0 1-1 1h-2" />
          <path d="M7 11h6M7 14h3" />
        </svg>
      </div>
      <p className="empty-text">Enter battery data to analyse health</p>
    </div>
  )
}

export default function ResultPanel({ result }) {
  if (!result) return (
    <div className="result-panel">
      <EmptyState />
    </div>
  )

  const { soh, rul, status, discharge_duration_min, inputs } = result
  const color = STATUS_COLORS[status] || '#22D3EE'
  const totalLife = (inputs?.cycle || 0) + rul
  const usedPct = totalLife > 0 ? Math.min(100, Math.round((inputs?.cycle / totalLife) * 100)) : 0

  return (
    <div className="result-panel">
      <div>
        <p className="panel-label">Health Analysis</p>
      </div>

      {/* SOH Gauge */}
      <SohGauge soh={soh} />

      {/* Status badge */}
      <div className="status-row">
        <span
          className="status-dot"
          style={{ background: color, boxShadow: `0 0 8px ${color}` }}
        />
        <span className="status-text" style={{ color }}>{status}</span>
      </div>

      <div className="divider" />

      {/* RUL */}
      <div className="rul-section">
        <p className="panel-label">Remaining Useful Life</p>
        <div className="rul-header">
          <span className="rul-number">{rul}</span>
          <span className="rul-unit">cycles remaining</span>
        </div>

        {/* Life timeline bar */}
        <div>
          <p className="timeline-label">Battery Life Timeline</p>
          <div className="timeline-track">
            <div
              className="timeline-used"
              style={{ width: `${usedPct}%` }}
            />
            <div
              className="timeline-remaining-fill"
              style={{ left: `${usedPct}%`, right: '0%' }}
            />
            <div
              className="timeline-marker"
              style={{ left: `${usedPct}%` }}
            />
          </div>
          <div className="timeline-legend">
            <span>Cycle {inputs?.cycle ?? '—'} used ({usedPct}%)</span>
            <span>{rul} remaining ({100 - usedPct}%)</span>
          </div>
        </div>
      </div>

      <div className="divider" />

      {/* Detail card */}
      <div className="details-card">
        <div className="detail-row">
          <span className="detail-key">SOH</span>
          <span className="detail-val">{soh.toFixed(2)} %</span>
        </div>
        <div className="detail-row">
          <span className="detail-key">RUL</span>
          <span className="detail-val">{rul} cycles</span>
        </div>
        <div className="detail-row">
          <span className="detail-key">Est. discharge duration</span>
          <span className="detail-val">{discharge_duration_min.toFixed(1)} min</span>
        </div>
        <div className="detail-row">
          <span className="detail-key">Cycle</span>
          <span className="detail-val">{inputs?.cycle}</span>
        </div>
        <div className="detail-row">
          <span className="detail-key">Voltage</span>
          <span className="detail-val">{inputs?.voltage_mean} V</span>
        </div>
        <div className="detail-row">
          <span className="detail-key">Temperature</span>
          <span className="detail-val">{inputs?.temperature_mean} °C</span>
        </div>
        <p className="interpretation">{STATUS_MSG[status]}</p>
      </div>
    </div>
  )
}
