import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/chat'

function App() {
  const [view, setView] = useState('chat') // 'chat' | 'upload'

  return (
    <div className="shell">
      <aside className="sidebar">
        <h2 className="logo">PolicyPilot</h2>
        <nav className="nav">
          <button className={view === 'chat' ? 'active' : ''} onClick={() => setView('chat')}>Chat</button>
          <button className={view === 'upload' ? 'active' : ''} onClick={() => setView('upload')}>Upload</button>
        </nav>
      </aside>
      <main className="main">
        {view === 'chat' ? <Chat /> : <Upload />}
      </main>
    </div>
  )
}

// ── Chat with WebSocket streaming ──

function Chat() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const wsRef = useRef(null)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)
  const streamBuf = useRef('')

  useEffect(() => {
    let cancelled = false
    let reconnectTimer = null

    function connect() {
      if (cancelled) return
      const ws = new WebSocket(WS_URL)
      ws.onopen = () => { if (!cancelled) wsRef.current = ws }
      ws.onclose = () => {
        if (wsRef.current === ws) wsRef.current = null
        if (!cancelled) reconnectTimer = setTimeout(connect, 1000)
      }
      ws.onerror = () => ws.close()
      ws.onmessage = (e) => {
        const data = JSON.parse(e.data)
        if (data.type === 'start') {
          streamBuf.current = ''
          setStreaming(true)
          setMessages((prev) => [...prev, { role: 'assistant', content: '' }])
        } else if (data.type === 'token') {
          streamBuf.current += data.content
          setMessages((prev) => {
            const copy = [...prev]
            copy[copy.length - 1] = { role: 'assistant', content: streamBuf.current }
            return copy
          })
        } else if (data.type === 'end') {
          setStreaming(false)
          setTimeout(() => inputRef.current?.focus(), 0)
        } else if (data.type === 'error') {
          setStreaming(false)
          setMessages((prev) => [...prev, { role: 'error', content: data.content }])
        }
      }
    }

    connect()
    return () => { cancelled = true; clearTimeout(reconnectTimer); wsRef.current?.close() }
  }, [])
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const send = (e) => {
    e.preventDefault()
    const text = input.trim()
    if (!text || streaming) return
    setMessages((prev) => [...prev, { role: 'user', content: text }])
    wsRef.current?.send(JSON.stringify({ message: text }))
    setInput('')
    setTimeout(() => inputRef.current?.focus(), 0)
  }

  return (
    <div className="chat">
      <div className="chat-messages">
        {messages.length === 0 && (
          <p className="empty">Upload a policy PDF then ask questions about it.</p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <span className="msg-label">{m.role === 'user' ? 'You' : m.role === 'error' ? 'Error' : 'PolicyPilot'}</span>
            <div className="msg-body">{m.content}{streaming && i === messages.length - 1 && m.role === 'assistant' && <span className="cursor" />}</div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      <form onSubmit={send} className="chat-input">
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about your policies…"
          disabled={streaming}
          autoFocus
        />
        <button type="submit" disabled={streaming || !input.trim()}>Send</button>
      </form>
    </div>
  )
}

// ── Upload panel ──

function Upload() {
  const [categories, setCategories] = useState([])
  const [category, setCategory] = useState('')
  const [file, setFile] = useState(null)
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetch(`${API_BASE}/api/categories`)
      .then((r) => r.json())
      .then((d) => {
        setCategories(d.categories || [])
        if (d.categories?.length) setCategory(d.categories[0])
      })
      .catch(() => setStatus({ error: 'Failed to load categories' }))
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file || !category) return
    setStatus(null)
    setLoading(true)
    const fd = new FormData()
    fd.append('file', file)
    fd.append('category', category)
    try {
      const res = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: fd })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        const msg = Array.isArray(data.detail) ? data.detail.map((d) => d.msg || d).join(', ') : (data.detail || res.statusText)
        throw new Error(msg)
      }
      setStatus({ success: data.message || 'Uploaded. Ingestion started.' })
      setFile(null)
      e.target.reset()
    } catch (err) {
      setStatus({ error: err.message || 'Upload failed' })
    } finally {
      setLoading(false)
    }
  }

  const label = (c) => c.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())

  return (
    <div className="upload-panel">
      <h2>Upload Policy PDF</h2>
      <p className="subtitle">Choose the type and upload a PDF. It will be stored and indexed for search.</p>
      <form onSubmit={handleSubmit} className="upload-form">
        <div className="field">
          <label htmlFor="category">PDF type</label>
          <select id="category" value={category} onChange={(e) => setCategory(e.target.value)} required>
            {categories.map((c) => <option key={c} value={c}>{label(c)}</option>)}
          </select>
        </div>
        <div className="field">
          <label htmlFor="file">PDF file</label>
          <input id="file" type="file" accept=".pdf,application/pdf" onChange={(e) => setFile(e.target.files?.[0] || null)} required />
        </div>
        <button type="submit" disabled={loading || !file}>{loading ? 'Uploading…' : 'Upload & ingest'}</button>
      </form>
      {status?.success && <p className="status success">{status.success}</p>}
      {status?.error && <p className="status error">{status.error}</p>}
    </div>
  )
}

export default App
