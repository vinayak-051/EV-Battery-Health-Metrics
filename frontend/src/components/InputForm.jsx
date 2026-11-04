import { useState } from 'react'

const FIELDS = [
  {
    key: 'cycle',
    label: 'Cycle Number',
    hint: 'Discharge cycle count (e.g. 1 – 500)',
    placeholder: '150',
    step: '1',
    min: '1',
  },
  {
    key: 'energy_Wh',
    label: 'Discharge Energy (Wh)',
    hint: 'Energy released per cycle (e.g. 1.5 – 5.0 Wh)',
    placeholder: '4.2',
    step: '0.01',
    min: '0.01',
  },
  {
    key: 'voltage_mean',
    label: 'Mean Voltage (V)',
    hint: 'Average discharge voltage (e.g. 3.0 – 4.2 V)',
    placeholder: '3.85',
    step: '0.01',
    min: '0.1',
  },
  {
    key: 'current_mean',
    label: 'Mean Current (A)',
    hint: 'Average discharge current (e.g. 0.5 – 3.0 A)',
    placeholder: '2.0',
    step: '0.01',
    min: '0.01',
  },
  {
    key: 'temperature_mean',
    label: 'Temperature (°C)',
    hint: 'Average battery temperature (e.g. 20 – 45 °C)',
    placeholder: '32',
    step: '0.1',
  },
]

export default function InputForm({ onResult, onError, onLoading }) {
  const [values, setValues] = useState({
    cycle: '',
    energy_Wh: '',
    voltage_mean: '',
    current_mean: '',
    temperature_mean: '',
  })
  const [loading, setLoading] = useState(false)
  const [localError, setLocalError] = useState('')

  function handleChange(e) {
    setValues(v => ({ ...v, [e.target.name]: e.target.value }))
    setLocalError('')
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setLocalError('')

    const payload = {}
    for (const f of FIELDS) {
      const val = parseFloat(values[f.key])
      if (isNaN(val)) {
        setLocalError(`Please enter a valid number for "${f.label}".`)
        return
      }
      payload[f.key] = val
    }

    setLoading(true)
    onLoading(true)
    try {
      let res
      try {
        res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        })
      } catch {
        throw new Error('Cannot reach the backend. Start it with: uvicorn main:app --reload (port 8000)')
      }

      const text = await res.text()
      if (!text) {
        throw new Error(`Empty response from server (status ${res.status}). Is the backend running?`)
      }

      let data
      try {
        data = JSON.parse(text)
      } catch {
        throw new Error(`Server returned non-JSON (status ${res.status}). Check backend logs.`)
      }

      if (!res.ok) {
        throw new Error(data?.detail || `Server error ${res.status}`)
      }

      onResult({ ...data, inputs: payload })
      onError('')
    } catch (err) {
      const msg = err.message || 'Unknown error. Is the backend running on port 8000?'
      setLocalError(msg)
      onError(msg)
    } finally {
      setLoading(false)
      onLoading(false)
    }
  }

  const allFilled = FIELDS.every(f => values[f.key].trim() !== '')

  return (
    <form className="input-panel" onSubmit={handleSubmit} noValidate>
      <div>
        <p className="panel-label">Battery Data Input</p>
      </div>

      <div className="fields-grid">
        {FIELDS.map(f => (
          <div className="field" key={f.key}>
            <label className="field-label" htmlFor={f.key}>{f.label}</label>
            <input
              id={f.key}
              name={f.key}
              type="number"
              step={f.step}
              min={f.min}
              placeholder={f.placeholder}
              value={values[f.key]}
              onChange={handleChange}
              autoComplete="off"
            />
            <span className="field-hint">{f.hint}</span>
          </div>
        ))}
      </div>

      {localError && (
        <div className="error-msg" role="alert">{localError}</div>
      )}

      <button
        type="submit"
        className="btn-analyze"
        disabled={loading || !allFilled}
      >
        {loading ? <><span className="spinner" />Analyzing…</> : 'Analyze Battery'}
      </button>
    </form>
  )
}
