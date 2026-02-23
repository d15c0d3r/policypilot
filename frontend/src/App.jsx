import { useState, useEffect, useRef } from 'react'
import './App.css'

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/chat'

function App() {
  const [view, setView] = useState('chat')

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
        {view === 'chat' ? <Chat /> : <p style={{padding: '2rem', color: '#888'}}>Upload coming soon.</p>}
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

export default App
