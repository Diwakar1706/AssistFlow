// Thin wrapper over the local FastAPI backend (python/api_server.py)
export async function fetchHealth() {
  const r = await fetch('/api/health')
  return r.json()
}
export async function fetchDocuments() {
  const r = await fetch('/api/documents')
  return r.json()
}
export async function fetchDocument(id) {
  const r = await fetch(`/api/documents/${id}`)
  if (!r.ok) throw new Error('Document not found')
  return r.json()
}
export async function uploadPdf(file) {
  const fd = new FormData()
  fd.append('file', file)
  const r = await fetch('/api/upload', { method: 'POST', body: fd })
  if (!r.ok) {
    const e = await r.json().catch(() => ({}))
    throw new Error(e.detail || 'Upload failed')
  }
  return r.json()
}
export async function askQuestion(documentId, question) {
  const r = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document_id: documentId, question })
  })
  const data = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(data.detail || 'Chat failed')
  return data
}

// ── Conversations (per-document chat history) ──────────────────────
export async function fetchConversations(docId) {
  const r = await fetch(`/api/documents/${docId}/conversations`)
  return r.json()
}
export async function createConversation(docId) {
  const r = await fetch(`/api/documents/${docId}/conversations`, { method: 'POST' })
  if (!r.ok) throw new Error('Could not create a chat')
  return r.json()
}
export async function fetchConversation(convId) {
  const r = await fetch(`/api/conversations/${convId}`)
  if (!r.ok) throw new Error('Chat not found')
  return r.json()
}
export async function deleteConversation(convId) {
  const r = await fetch(`/api/conversations/${convId}`, { method: 'DELETE' })
  if (!r.ok) throw new Error('Could not delete the chat')
  return r.json()
}
export async function askInConversation(convId, question) {
  const r = await fetch(`/api/conversations/${convId}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  })
  const data = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(data.detail || 'Chat failed')
  return data
}
export async function deleteDocument(docId) {
  const r = await fetch(`/api/documents/${docId}`, { method: 'DELETE' })
  if (!r.ok) throw new Error('Could not remove the PDF')
  return r.json()
}

// Streaming chat (Server-Sent Events over fetch): onEvent receives
// {type:'meta'|'token'|'done'|'error', ...} as the answer is generated.
export async function askInConversationStream(convId, question, onEvent) {
  const r = await fetch(`/api/conversations/${convId}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  })
  if (!r.ok) {
    const data = await r.json().catch(() => ({}))
    throw new Error(data.detail || 'Chat failed')
  }
  const reader = r.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    const parts = buf.split('\n\n')
    buf = parts.pop()
    for (const raw of parts) {
      const line = raw.trim()
      if (line.startsWith('data:')) {
        try { onEvent(JSON.parse(line.slice(5).trim())) } catch { /* ignore partial */ }
      }
    }
  }
}

export async function addUrl(url) {
  const r = await fetch('/api/upload-url', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url })
  })
  const data = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(data.detail || 'Could not load that URL')
  return data
}
