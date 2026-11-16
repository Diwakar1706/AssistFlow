import React, { useEffect, useRef, useState } from 'react'
import {
  addUrl, askInConversationStream, createConversation, deleteConversation, deleteDocument,
  fetchConversation, fetchConversations, fetchDocument, fetchDocuments, fetchHealth, uploadPdf
} from './api'
import PipelineModal from './PipelineModal.jsx'

function fmtSize(bytes) {
  if (!bytes && bytes !== 0) return ''
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
function timeAgo(ts) {
  if (!ts) return ''
  const s = Math.max(1, Math.floor(Date.now() / 1000 - ts))
  if (s < 60) return 'Just now'
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  return `${Math.floor(s / 86400)}d ago`
}
function statusOf(doc) {
  if (doc.steps?.vectorization === 'done') return { label: 'Ready', cls: 'st-ready' }
  if (doc.steps?.vectorization === 'failed') return { label: 'Embedding failed', cls: 'st-fail' }
  return { label: 'Chunks ready · add API key for chat', cls: 'st-pending' }
}

export default function App() {
  const [docs, setDocs] = useState([])
  const [backendOk, setBackendOk] = useState(null)
  const [keySet, setKeySet] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [pipelineDoc, setPipelineDoc] = useState(null)
  const [focusChunk, setFocusChunk] = useState(null) // chunk id to highlight when the pipeline opens from a source click
  const [activeDoc, setActiveDoc] = useState(null)       // the PDF selected in the centre panel
  const [conversations, setConversations] = useState([]) // its chats (summaries)
  const [history, setHistory] = useState({})            // docId -> chat summaries, for the left History sidebar
  const [activeConv, setActiveConv] = useState(null)     // open conversation (with messages)
  const [question, setQuestion] = useState('')
  const [urlValue, setUrlValue] = useState('')
  const [asking, setAsking] = useState(false)
  const [liveMsg, setLiveMsg] = useState(null) // assistant answer currently streaming in
  const [drag, setDrag] = useState(false)
  const fileRef = useRef(null)
  const chatEndRef = useRef(null)

  async function refreshDocs() {
    try {
      const h = await fetchHealth()
      setBackendOk(true)
      setKeySet(!!h.openai_key_set)
      const list = await fetchDocuments()
      setDocs(list)
      loadHistory(list)
      return list
    } catch {
      setBackendOk(false)
      return []
    }
  }
  useEffect(() => { refreshDocs() }, [])
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [activeConv, liveMsg])
  useEffect(() => {
    if (activeDoc) setHistory(h => ({ ...h, [activeDoc.id]: conversations }))
  }, [conversations, activeDoc])

  // Load every source's chat list once, so the left sidebar can show the full history grouped by source
  async function loadHistory(list) {
    const entries = await Promise.all((list || []).map(async d => {
      try { return [d.id, await fetchConversations(d.id)] } catch { return [d.id, []] }
    }))
    setHistory(Object.fromEntries(entries))
  }

  // Select a PDF -> load ITS conversations, open the most recent chat (history is kept per PDF)
  async function selectDocument(docSummary, { openLatest = true } = {}) {
    setActiveDoc(docSummary)
    setActiveConv(null)
    try {
      const convs = await fetchConversations(docSummary.id)
      setConversations(convs)
      setHistory(h => ({ ...h, [docSummary.id]: convs }))
      if (openLatest && convs.length > 0) {
        setActiveConv(await fetchConversation(convs[0].id))
      }
    } catch {
      setConversations([])
    }
  }

  async function handleUpload(file) {
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.pdf')) { setError('Please choose a PDF file.'); return }
    setError('')
    setUploading(true)
    try {
      const doc = await uploadPdf(file)
      await refreshDocs()
      await selectDocument(doc, { openLatest: false })
      setConversations(await fetchConversations(doc.id))
      setPipelineDoc(doc) // pipeline popup opens straight away, like before
    } catch (e) {
      setError(e.message || 'Upload failed — is the backend running on port 8000?')
    } finally {
      setUploading(false)
    }
  }

  async function handleAddUrl() {
    const url = urlValue.trim()
    if (!url) return
    setError('')
    setUploading(true)
    try {
      const doc = await addUrl(url)
      setUrlValue('')
      await refreshDocs()
      await selectDocument(doc, { openLatest: false })
      setConversations(await fetchConversations(doc.id))
      setPipelineDoc(doc)
    } catch (e) {
      setError(e.message || 'Could not load that URL — is the backend running?')
    } finally {
      setUploading(false)
    }
  }

  async function openPipeline(summary, e) {
    e?.stopPropagation()
    setFocusChunk(null)
    try {
      setPipelineDoc(await fetchDocument(summary.id))
    } catch { setError('Could not load document details.') }
  }

  // Click a source under an answer -> open the pipeline's View Chunks with that chunk highlighted
  async function openSource(chunkId) {
    if (!activeDoc) return
    try {
      setFocusChunk(chunkId)
      setPipelineDoc(await fetchDocument(activeDoc.id))
    } catch { setError('Could not open the source chunk.') }
  }

  async function newChat() {
    if (!activeDoc) return
    try {
      const conv = await createConversation(activeDoc.id)
      setConversations(cs => [{ id: conv.id, document_id: conv.document_id, title: conv.title, created_at: conv.created_at, updated_at: conv.updated_at, message_count: 0 }, ...cs])
      setActiveConv(conv)
    } catch { setError('Could not start a new chat.') }
  }

  async function openChat(summary) {
    try {
      setActiveConv(await fetchConversation(summary.id))
    } catch { setError('Could not open that chat.') }
  }

  async function removeChat(summary, e) {
    e.stopPropagation()
    if (!window.confirm(`Delete chat "${summary.title}"? This cannot be undone.`)) return
    try {
      await deleteConversation(summary.id)
      const left = conversations.filter(c => c.id !== summary.id)
      setConversations(left)
      if (activeConv?.id === summary.id) {
        setActiveConv(left.length ? await fetchConversation(left[0].id) : null)
      }
    } catch { setError('Could not delete the chat.') }
  }

  async function openChatFromHistory(docSummary, summary) {
    if (activeDoc?.id !== docSummary.id) await selectDocument(docSummary, { openLatest: false })
    await openChat(summary)
  }

  async function removeChatFor(docSummary, summary, e) {
    e.stopPropagation()
    if (!window.confirm(`Delete chat "${summary.title}"? This cannot be undone.`)) return
    try {
      await deleteConversation(summary.id)
      setHistory(h => ({ ...h, [docSummary.id]: (h[docSummary.id] || []).filter(c => c.id !== summary.id) }))
      if (activeDoc?.id === docSummary.id) {
        const left = conversations.filter(c => c.id !== summary.id)
        setConversations(left)
        if (activeConv?.id === summary.id) {
          setActiveConv(left.length ? await fetchConversation(left[0].id) : null)
        }
      } else if (activeConv?.id === summary.id) {
        setActiveConv(null)
      }
    } catch { setError('Could not delete the chat.') }
  }

  async function removePdf(docSummary, e) {
    e.stopPropagation()
    if (!window.confirm(`Remove "${docSummary.name}" and ALL its chats? This deletes the PDF, its index and its chat history.`)) return
    try {
      await deleteDocument(docSummary.id)
      const list = await refreshDocs()
      if (activeDoc?.id === docSummary.id) {
        setActiveDoc(null); setConversations([]); setActiveConv(null)
      }
      if (pipelineDoc?.id === docSummary.id) setPipelineDoc(null)
      if (list.length && activeDoc?.id === docSummary.id) { /* stay cleared */ }
    } catch { setError('Could not remove the PDF.') }
  }

  function startChatFromPipeline(doc) {
    setPipelineDoc(null)
    selectDocument(doc)
  }

  async function sendQuestion(e) {
    e.preventDefault()
    const q = question.trim()
    if (!q || !activeDoc || asking) return
    setQuestion('')
    setAsking(true)
    let streamError = null
    try {
      let conv = activeConv
      if (!conv) { // no chat yet for this PDF -> create one automatically
        conv = await createConversation(activeDoc.id)
        setActiveConv(conv)
      }
      setActiveConv(c => ({ ...c, messages: [...(c?.messages || []), { role: 'user', content: q }] }))
      setLiveMsg({ role: 'assistant', content: '' })
      await askInConversationStream(conv.id, q, (ev) => {
        if (ev.type === 'meta') {
          setLiveMsg(m => ({ ...m, pages: ev.pages, retrieval: ev.retrieval, sources: ev.sources, searched_as: ev.searched_as }))
        } else if (ev.type === 'token') {
          setLiveMsg(m => ({ ...m, content: (m?.content || '') + ev.text }))
        } else if (ev.type === 'error') {
          streamError = ev.detail || 'Streaming failed'
        }
      })
      if (streamError) throw new Error(streamError)
      const full = await fetchConversation(conv.id)
      setActiveConv(full)
      setConversations(cs => {
        const others = cs.filter(c => c.id !== full.id)
        return [{ id: full.id, document_id: full.document_id, title: full.title, created_at: full.created_at, updated_at: full.updated_at, message_count: full.messages.length }, ...others]
      })
    } catch (err) {
      setActiveConv(c => c ? { ...c, messages: [...c.messages, { role: 'assistant', content: `⚠️ ${err.message}` }] } : c)
    } finally {
      setLiveMsg(null)
      setAsking(false)
    }
  }

  const messages = activeConv?.messages || []
  const displayMessages = liveMsg ? [...messages, liveMsg] : messages

  return (
    <div className="layout">
      {/* ── Left sidebar ─────────────────────────────── */}
      <aside className="sidebar">
        <div className="brand"> AssistFlow</div>
        <div className="side-label row">
          <span>🕘 History</span>
          <button className="mini-btn" onClick={newChat} disabled={!activeDoc} title="Start a new chat for the selected source">＋ New chat</button>
        </div>
        <div className="history">
          {docs.map(d => (
            <div key={d.id} className="hist-doc">
              <button className={activeDoc?.id === d.id ? 'hist-src active' : 'hist-src'} onClick={() => selectDocument(d)} title={d.source_url || d.name}>
                <span className="hist-src-ico">{d.kind === 'url' ? '🌐' : '📄'}</span>
                <span className="hist-src-name">{d.name}</span>
              </button>
              <div className="hist-chats">
                {(history[d.id] || []).map(c => (
                  <div key={c.id} className={activeConv?.id === c.id ? 'hist-chat active' : 'hist-chat'} onClick={() => openChatFromHistory(d, c)}>
                    <span className="hist-chat-title">💬 {c.title}</span>
                    <button className="icon-btn danger" title="Delete chat" onClick={e => removeChatFor(d, c, e)}>🗑</button>
                  </div>
                ))}
                {(history[d.id] || []).length === 0 && <div className="hist-empty muted">No chats yet</div>}
              </div>
            </div>
          ))}
          {docs.length === 0 && <div className="muted pad">No sources yet. Add a PDF or a website URL in the Knowledge Base →</div>}
        </div>
        <div className="side-foot muted">
          Backend: {backendOk === null ? 'checking…' : backendOk ? '🟢 running' : '🔴 not running'}
          <br />OpenAI key: {keySet ? '🟢 set' : '⚪ not set'}
        </div>
      </aside>

      {/* ── Centre: conversations + chat ─────────────── */}
      <main className="main">
        <header className="main-head">
          <h1 title={activeDoc ? (activeDoc.source_url || activeDoc.name) : ''}>{activeDoc ? activeDoc.name : 'No source selected'}</h1>
        </header>

        <div className="chat-area">
          {messages.length === 0 && (
            <div className="empty-chat">
              <div className="empty-ico">💬</div>
              <h3>{activeDoc ? `Ask questions about ${activeDoc.name}` : 'Ask questions about your PDF'}</h3>
              <p className="muted">Upload a PDF in the Knowledge Base — even a 100+ page policy — then chat here in plain English. Your chats for every source live in the History panel on the left.</p>
            </div>
          )}
          {displayMessages.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              <div className={liveMsg && i === displayMessages.length - 1 ? 'msg-bubble streaming' : 'msg-bubble'}>
                {m.content || (liveMsg && i === displayMessages.length - 1 ? <span className="muted">Searching your PDF…</span> : null)}
                {m.searched_as && <div className="searched-as">🔎 Searched as: “{m.searched_as}”</div>}
                {m.pages && <div className="msg-pages">Based on page(s): {m.pages.join(', ')}{m.retrieval ? ` · Retrieval: ${m.retrieval}` : ''}</div>}
                {m.sources && m.sources.length > 0 && (
                  <div className="msg-sources">
                    <div className="sources-title">Sources used — click to view the chunk</div>
                    <div className="msg-sources-grid">
                      {m.sources.map(src => {
                        const clean = (src.preview || '').replace(/\s+/g, ' ').trim()
                        return (
                          <button key={src.chunk_id} className="source-chip" onClick={() => openSource(src.chunk_id)} title={clean}>
                            <span className="source-chip-head">📄 Page {src.page} · Chunk {src.chunk_id}</span>
                            <span className="source-chip-text">{clean}</span>
                          </button>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        <footer className="main-foot">
          {backendOk === false && (
            <div className="backend-banner">Backend is not running. Start it with: <code>cd python &amp;&amp; uvicorn api_server:app --port 8000</code></div>
          )}
          {error && <div className="error-banner">{error}</div>}
          <form className="chat-input" onSubmit={sendQuestion}>
            <input
              value={question}
              onChange={e => setQuestion(e.target.value)}
              placeholder={activeDoc ? `Ask about ${activeDoc.name}…` : 'Pick a PDF from the Knowledge Base to start chatting…'}
              disabled={!activeDoc}
            />
            <button className="primary-btn" disabled={!activeDoc || asking}>Send</button>
          </form>
        </footer>
      </main>

      {/* ── Right: Knowledge Base ─────────────────────── */}
      <aside className="kb">
        <div className="kb-head"><span>📘 Knowledge Base</span></div>
        <div className="kb-tabs">
          <span className="kb-tab active">📄 Documents <span className="count">{docs.length}</span></span>
        </div>

        <div className="kb-section">Add Sources</div>
        <div
          className={drag ? 'dropzone drag' : 'dropzone'}
          onClick={() => fileRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={e => { e.preventDefault(); setDrag(false); handleUpload(e.dataTransfer.files[0]) }}
        >
          <div className="dz-ico">⬆</div>
          <div className="dz-title">{uploading ? 'Processing your PDF…' : 'Drop files or click to upload'}</div>
          <div className="muted">PDF · Max 50MB</div>
          <input ref={fileRef} type="file" accept="application/pdf" hidden onChange={e => handleUpload(e.target.files[0])} />
        </div>

        <div className="or-row"><span>OR</span></div>
        <input className="url-input" placeholder="🌐 Paste website URL — article, docs, or a PDF link" value={urlValue}
          onChange={e => setUrlValue(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') handleAddUrl() }} />
        <button className="add-site" onClick={handleAddUrl} disabled={uploading || !urlValue.trim()}>{uploading ? 'Loading…' : '＋ Add website'}</button>

        <div className="kb-section row"><span>Sources</span><span className="count">{docs.length}</span></div>
        <div className="sources">
          {docs.map(d => {
            const st = statusOf(d)
            return (
              <div key={d.id} className={activeDoc?.id === d.id ? 'source-card active' : 'source-card'} onClick={() => { selectDocument(d); openPipeline(d) }}>
                <span className="file-ico">{d.kind === 'url' ? '🌐' : '📄'}</span>
                <span className="source-info">
                  <span className="source-name" title={d.source_url || d.name}>{d.name}</span>
                  <span className="source-meta">{fmtSize(d.size_bytes)} · {timeAgo(d.created_at)} · <span className={st.cls}>{st.label}</span></span>
                </span>
                <button className="icon-btn" title="View processing pipeline" onClick={e => openPipeline(d, e)}>👁</button>
                <button className="icon-btn danger" title="Remove PDF and all its chats" onClick={e => removePdf(d, e)}>🗑</button>
              </div>
            )
          })}
          {docs.length === 0 && <p className="muted pad">No sources yet. Upload your first PDF above.</p>}
        </div>
      </aside>

      {pipelineDoc && (
        <PipelineModal doc={pipelineDoc} focusChunkId={focusChunk} onClose={() => { setPipelineDoc(null); setFocusChunk(null) }} onChat={startChatFromPipeline} />
      )}
    </div>
  )
}
