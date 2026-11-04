const HALF_CIRC = Math.PI * 100  // radius=100, half circle
const FULL_CIRC = HALF_CIRC * 2

function getColor(soh) {
  if (soh >= 85) return '#34D399'
  if (soh >= 70) return '#FBBF24'
  return '#F87171'
}

export default function SohGauge({ soh }) {
  const color = getColor(soh)
  const filled = (HALF_CIRC * soh / 100).toFixed(2)

  return (
    <div className="gauge-wrapper">
      <svg
        className="gauge-svg"
        viewBox="0 0 220 120"
        aria-label={`State of Health: ${soh}%`}
      >
        {/* Endpoint caps */}
        <circle cx="10" cy="115" r="5" fill="#1D2E47" />
        <circle cx="210" cy="115" r="5" fill="#1D2E47" />

        {/* Background track */}
        <circle
          cx="110" cy="115" r="100"
          fill="none"
          stroke="#1D2E47"
          strokeWidth="13"
          strokeDasharray={`${HALF_CIRC.toFixed(2)} ${FULL_CIRC.toFixed(2)}`}
          transform="rotate(180 110 115)"
          strokeLinecap="round"
        />

        {/* Progress arc */}
        <circle
          cx="110" cy="115" r="100"
          fill="none"
          stroke={color}
          strokeWidth="13"
          strokeLinecap="round"
          strokeDasharray={`${filled} ${FULL_CIRC.toFixed(2)}`}
          transform="rotate(180 110 115)"
          style={{ transition: 'stroke-dasharray 0.8s ease, stroke 0.4s ease' }}
        />

        {/* Glow on filled arc tip — subtle */}
        <circle
          cx="110" cy="115" r="100"
          fill="none"
          stroke={color}
          strokeWidth="4"
          strokeLinecap="round"
          strokeDasharray={`${filled} ${FULL_CIRC.toFixed(2)}`}
          transform="rotate(180 110 115)"
          opacity="0.35"
          style={{ filter: 'blur(3px)', transition: 'stroke-dasharray 0.8s ease, stroke 0.4s ease' }}
        />

        {/* SOH number */}
        <text
          x="110"
          y="85"
          className="gauge-number"
          fill={color}
          style={{ transition: 'fill 0.4s ease' }}
        >
          {soh.toFixed(1)}
          <tspan className="gauge-unit" fill={color}>%</tspan>
        </text>

        {/* Caption */}
        <text x="110" y="106" className="gauge-caption">
          STATE OF HEALTH
        </text>
      </svg>
    </div>
  )
}
