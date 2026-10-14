import React, { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './style.css'

type Document = { id: string; filename: string }
type Answer = { answer: string; citations: { filename: string; file_id: string; index: number }[] }
type Exchange = Answer & { question: string; document: string }

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response
  try { response = await fetch(`/api/${path}`, options) }
  catch { throw new Error('Cannot reach the backend. Start the FastAPI server and retry.') }
  const data = await response.json().catch(() => null)
  if (!response.ok) throw new Error(typeof data?.detail === 'string' ? data.detail : 'The request failed. Check the backend and retry.')
  return data as T
}

function App() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [selected, setSelected] = useState('')
  const [question, setQuestion] = useState('')
  const [exchanges, setExchanges] = useState<Exchange[]>([])
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const fileInput = useRef<HTMLInputElement>(null)

  async function loadDocuments() {
    setError('')
    try {
      const items = await request<Document[]>('documents')
      setDocuments(items)
      setSelected(current => items.some(item => item.id === current) ? current : items[0]?.id || '')
    } catch (error) { setError((error as Error).message) }
  }
  useEffect(() => { void loadDocuments() }, [])

  async function upload(file?: File) {
    if (!file) return
    setError('')
    if (file.size > 20 * 1024 * 1024 || file.size === 0) { setError('Choose a nonempty document up to 20 MB.'); return }
    setBusy('Uploading and preparing your document… This can take a few minutes.')
    try {
      const body = new FormData(); body.append('file', file)
      const document = await request<Document>('documents', { method: 'POST', body })
      setDocuments(items => [...items.filter(item => item.id !== document.id), document])
      setSelected(document.id)
    } catch (error) { setError((error as Error).message) }
    finally { setBusy(''); if (fileInput.current) fileInput.current.value = '' }
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    if (!selected || !question.trim() || busy) return
    const prompt = question.trim()
    const name = documents.find(item => item.id === selected)?.filename || 'Document'
    setBusy('Searching your document and writing an answer…'); setError('')
    try {
      const answer = await request<Answer>('questions', { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ document_id: selected, question: prompt }) })
      setExchanges(items => [...items, { ...answer, question: prompt, document: name }]); setQuestion('')
    } catch (error) { setError((error as Error).message) }
    finally { setBusy('') }
  }

  return <div className="shell">
    <header><a className="brand" href="/">▤ <span>Document Q&A</span></a><span className="badge">LOCAL WORKSPACE</span></header>
    <main>
      <section className="intro"><span className="eyebrow">ANSWERS, GROUNDED IN YOUR FILES</span><h1>Your documents.<br />A clearer answer.</h1><p>Bring a document, ask a question, and see where the answer comes from.</p></section>
      <div className="workspace">
        <aside className="panel"><div className="section-heading"><h2>Your documents</h2><button className="text-button" disabled={!!busy} onClick={() => void loadDocuments()}>Refresh</button></div>
          <label className={`upload ${busy ? 'disabled' : ''}`}><span className="upload-icon">↑</span><strong>Add a document</strong><span>PDF, DOCX, TXT, or Markdown · up to 20 MB</span><input ref={fileInput} type="file" accept=".pdf,.docx,.txt,.md" disabled={!!busy} onChange={event => void upload(event.target.files?.[0])} /></label>
          <p className="privacy">Files are sent to OpenAI for processing. Your API key stays on the backend.</p>
          <div className="document-list" aria-label="Select a document">{documents.map(document => <button key={document.id} disabled={!!busy} className={`document ${selected === document.id ? 'selected' : ''}`} onClick={() => setSelected(document.id)}><span>▤</span><span>{document.filename}</span><span>{selected === document.id ? '✓' : ''}</span></button>)}</div>
          {!documents.length && <p className="muted">Your processed documents will appear here.</p>}
        </aside>
        <section className="panel conversation"><div className="section-heading"><h2>Ask your document</h2><span className="status">{selected ? 'Ready' : 'Add a file to begin'}</span></div>
          <div className="answers" aria-live="polite">{!exchanges.length && <div className="empty"><span className="empty-icon">↗</span><h3>Start with a question</h3><p>Try “What are the key points?” or ask about a specific detail. Each question searches only your selected document.</p></div>}
            {exchanges.map((exchange, index) => <article className="exchange" key={index}><span className="eyebrow">{exchange.document}</span><h3>{exchange.question}</h3><p className="answer">{exchange.answer}</p><details><summary>Sources · {exchange.citations.length}</summary>{exchange.citations.length ? <ul>{exchange.citations.map((citation, index) => <li key={index}><strong>{citation.filename}</strong><small>Answer text index {citation.index} · {citation.file_id}</small></li>)}</ul> : <p className="muted">No file citations returned.</p>}</details></article>)}
          </div>
          {busy && <p className="progress" role="status"><span className="spinner" />{busy}</p>}
          {error && <p className="error" role="alert">{error}</p>}
          <form onSubmit={submit}><label htmlFor="question">Your question</label><textarea id="question" placeholder="What would you like to know?" value={question} onChange={event => setQuestion(event.target.value)} maxLength={10000} disabled={!!busy || !selected} rows={3} /><div className="form-footer"><span>Answers are based on the selected document.</span><button className="primary" disabled={!!busy || !selected || !question.trim()}>Ask question <span>↗</span></button></div></form>
        </section>
      </div>
    </main><footer>Document Q&A · Backend proof of concept</footer>
  </div>
}

createRoot(document.getElementById('root')!).render(<App />)
