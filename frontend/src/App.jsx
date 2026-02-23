import { useState } from 'react'
import './App.css'

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
        <p style={{padding: '2rem', color: '#888'}}>Select a view from the sidebar.</p>
      </main>
    </div>
  )
}

export default App
