import { useState } from 'react'
import InputForm from './components/InputForm.jsx'
import ResultPanel from './components/ResultPanel.jsx'

export default function App() {
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  return (
    <div className="page">
      <header className="header">
        <span className="header-dot" />
        <span className="header-title">EV Battery Health Monitor</span>
        <span className="header-sub">/ SOH &amp; RUL Prediction</span>
        {loading && <span className="spinner" style={{ marginLeft: 'auto' }} />}
      </header>

      <div className="layout">
        <InputForm
          onResult={setResult}
          onError={setError}
          onLoading={setLoading}
        />
        <ResultPanel result={result} />
      </div>
    </div>
  )
}
