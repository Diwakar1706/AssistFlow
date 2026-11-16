import React, { useEffect, useMemo, useState } from 'react'

const STEPS = [
  { key: 'upload', label: 'Upload' },
  { key: 'partitioning', label: 'Partitioning' },
  { key: 'chunking', label: 'Chunking' },
  { key: 'summarisation', label: 'Summarisation' },
  { key: 'vectorization', label: 'Vectorization & Storage' },
  { key: 'chunks', label: 'View Chunks' }
]

function fmtSize(bytes) {
  if (!bytes && bytes !== 0) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function DoneBanner({ text = 'Step completed successfully' }) {
  return <div className="done-banner">✓ {text}</div>
}

export default function PipelineModal({ doc, onClose, onChat, focusChunkId = null }) {
  const [tab, setTab] = useState('partitioning')
  const [filter, setFilter] = useState('all')
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState(null)
  const [inspTab, setInspTab] = useState('summary')

  const chunks = doc.chunks || []
  const filtered = useMemo(() => {
    return chunks.filter(c => {
      const okFilter = filter === 'all' || c.types.includes(filter)
      const okSearch = !search || c.content.toLowerCase().includes(search.toLowerCase())
      return okFilter && okSearch
    })
  }, [chunks, filter, search])

  // Opened from a source chip under an answer: jump straight to that chunk, highlighted
  useEffect(() => {
    if (focusChunkId == null) return
    const ch = chunks.find(c => c.id === focusChunkId)
    if (!ch) return
    setTab('chunks')
    setFilter('all')
    setSearch('')
    setSelected(ch)
    const t = setTimeout(() => {
      document.getElementById(`chunk-card-${focusChunkId}`)?.scrollIntoView({ block: 'center', behavior: 'smooth' })
    }, 120)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusChunkId, doc])

  const sel = selected || filtered[0] || chunks[0]
  const p = doc.partitioning || {}
  const ck = doc.chunking || {}
  const vecDone = doc.steps?.vectorization === 'done'

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal-head">
          <div className="modal-title">
            <span className="file-ico">{doc.kind === 'url' ? '🌐' : '📄'}</span>
            <div>
              <div className="modal-name">{doc.name}</div>
              <div className="modal-sub">Processing Pipeline · {fmtSize(doc.size_bytes)} · local processing (no S3, no queue)</div>
            </div>
          </div>
          <button className="x-btn" onClick={onClose}>✕</button>
        </div>

        <div className="step-tabs">
          {STEPS.map(s => (
            <button key={s.key} className={tab === s.key ? 'step-tab active' : 'step-tab'} onClick={() => setTab(s.key)}>
              {s.label}
              {doc.steps?.[s.key] === 'done' && <span className="dot-done"> ✓</span>}
              {doc.steps?.[s.key] === 'skipped' && <span className="dot-skip"> –</span>}
            </button>
          ))}
        </div>

        <div className="modal-body">
          <div className="step-panel">
            {tab === 'upload' && (
              <div className="center-step">
                <div className="big-check">✓</div>
                <h3>Uploaded</h3>
                <p className="muted">File saved locally on this machine — no S3, no queue. Processing starts instantly.</p>
                <div className="kv-card">
                  <div><span>File</span><b>{doc.name}</b></div>
                  <div><span>Size</span><b>{fmtSize(doc.size_bytes)}</b></div>
                  <div><span>Pages</span><b>{p.pages ?? '—'}</b></div>
                </div>
                <DoneBanner />
              </div>
            )}

            {tab === 'partitioning' && (
              <div className="center-step">
                <h3>Partitioning</h3>
                <p className="muted">Processing and extracting text, images, and tables</p>
                <div className="elements-card">
                  <div className="elements-head">📊 Elements Discovered</div>
                  <div className="elements-grid">
                    <div className="el-row"><span>Text sections</span><b>{p.text_sections ?? 0}</b></div>
                    <div className="el-row"><span>Tables</span><b>{p.tables ?? 0}</b></div>
                    <div className="el-row"><span>Images</span><b>{p.images ?? 0}</b></div>
                    <div className="el-row"><span>Titles / Headers</span><b>{p.titles_headers ?? 0}</b></div>
                    <div className="el-row"><span>Other elements</span><b>{p.other_elements ?? 0}</b></div>
                    <div className="el-row"><span>Pages</span><b>{p.pages ?? 0}</b></div>
                  </div>
                  <p className="tiny muted">Text is read with pypdf; tables are extracted as structured chunks and images counted with PyMuPDF. Image contents are described by GPT-4o vision in the Summarisation step, and scanned pages are OCR'd the same way.</p>
                </div>
                <DoneBanner />
              </div>
            )}

            {tab === 'chunking' && (
              <div className="center-step">
                <h3>Chunking</h3>
                <p className="muted">Creating chunks with RecursiveCharacterTextSplitter (1000 chars, 200 overlap)</p>
                <div className="chunk-result">
                  <div className="chunk-nums">
                    <div><div className="big-num">{ck.total_elements ?? p.total_elements ?? 0}</div><div className="muted">atomic elements</div></div>
                    <div className="arrow">→</div>
                    <div><div className="big-num green">{ck.chunks_created ?? chunks.length}</div><div className="muted">chunks created</div></div>
                  </div>
                  <div className="el-row"><span>Average chunk size</span><b>{ck.avg_chunk_chars ?? 0} characters</b></div>
                  <p className="tiny muted">{ck.total_elements ?? 0} atomic elements have been chunked to produce {ck.chunks_created ?? chunks.length} chunks. Every chunk keeps its page number, so answers can cite the exact page.</p>
                </div>
                <DoneBanner />
              </div>
            )}

            {tab === 'summarisation' && (
              doc.multimodal && (doc.multimodal.images_described > 0 || doc.multimodal.ocr_pages > 0) ? (
                <div className="center-step">
                  <DoneBanner text="Vision processing completed" />
                  <h3>Summarisation — GPT-4o Vision</h3>
                  <p className="muted">Non-text content was converted into searchable text (lesson 08's idea, done with GPT-4o vision + PyMuPDF instead of the heavy Unstructured stack).</p>
                  <div className="elements-card">
                    <div className="elements-grid">
                      <div className="el-row"><span>Images described</span><b>{doc.multimodal.images_described} of {doc.multimodal.images_found || p.images || 0} found</b></div>
                      <div className="el-row"><span>Tables extracted</span><b>{doc.multimodal.tables ?? 0}</b></div>
                      <div className="el-row"><span>Scanned pages OCR'd</span><b>{doc.multimodal.ocr_pages}</b></div>
                      {doc.multimodal.scanned_pages > 0 && <div className="el-row"><span>Scanned pages not read</span><b>{doc.multimodal.scanned_pages}</b></div>}
                    </div>
                    <p className="tiny muted">Each image description became its own chunk (View Chunks → Image filter), each table a structured chunk (Table filter), and OCR text was added to its page before chunking — so chat answers can use all of it.</p>
                  </div>
                </div>
              ) : (
                <div className="center-step">
                  <div className="skip-banner">⏭ Step skipped for this document</div>
                  <h3>Summarisation skipped</h3>
                  <p className="muted">Nothing here needed vision processing: no images were described and no scanned pages were found{doc.kind === 'pdf' ? '' : ' (web sources are text-only)'}. When a PDF contains images or scanned pages, GPT-4o vision describes/OCRs them here so they become searchable — add your OpenAI key and re-upload if this PDF has image-only content.</p>
                </div>
              )
            )}

            {tab === 'vectorization' && (
              <div className="center-step">
                {vecDone ? (
                  <>
                    <div className="big-check">✓</div>
                    <h3>Vectorization & Storage</h3>
                    <p className="muted">Every chunk was embedded with OpenAI <b>text-embedding-3-small</b> and stored in <b>ChromaDB</b> on disk. Chat is now enabled for this PDF.</p>
                    <DoneBanner />
                  </>
                ) : (
                  <>
                    <div className="big-warn">!</div>
                    <h3>Waiting for OpenAI API key</h3>
                    <p className="muted">{doc.vector_note || 'Add OPENAI_API_KEY to python/.env and re-upload this PDF to embed the chunks into ChromaDB and enable chat.'}</p>
                    <div className="warn-banner">Chunks are ready — only this step is pending</div>
                  </>
                )}
              </div>
            )}

            {tab === 'chunks' && (
              <div className="chunks-layout">
                <div className="chunks-list">
                  <div className="chunks-list-head">
                    <h3>Content Chunks</h3>
                    <span className="muted">{filtered.length} of {chunks.length} chunks</span>
                  </div>
                  <div className="chunk-filters">
                    {['all', 'text', 'image', 'table'].map(f => (
                      <button key={f} className={filter === f ? 'chip active' : 'chip'} onClick={() => setFilter(f)}>
                        {f[0].toUpperCase() + f.slice(1)}
                      </button>
                    ))}
                    <input className="chunk-search" placeholder="🔍 Search chunks..." value={search} onChange={e => setSearch(e.target.value)} />
                  </div>
                  <div className="chunk-scroll">
                    {filtered.map(c => (
                      <button key={c.id} id={`chunk-card-${c.id}`} className={(sel && sel.id === c.id ? 'chunk-card active' : 'chunk-card') + (focusChunkId === c.id ? ' focus' : '')} onClick={() => setSelected(c)}>
                        <div className="chunk-top">
                          <span>
                            {c.types.map(t => <span key={t} className={`badge badge-${t}`}>{t}</span>)}
                            <span className="chunk-page"> Page {c.page}</span>
                          </span>
                          <span className="muted">{c.chars} chars</span>
                        </div>
                        <div className="chunk-preview">{c.content.slice(0, 180)}...</div>
                      </button>
                    ))}
                    {filtered.length === 0 && <p className="muted pad">No chunks match.</p>}
                  </div>
                </div>

                <div className="inspector">
                  <h3>Detail Inspector</h3>
                  {sel ? (
                    <>
                      <div className="insp-tabs">
                        <button className={inspTab === 'summary' ? 'chip active' : 'chip'} onClick={() => setInspTab('summary')}>📄 Summary</button>
                        <button className={inspTab === 'original' ? 'chip active' : 'chip'} onClick={() => setInspTab('original')}>📊 Original</button>
                      </div>
                      <div className="insp-tags">
                        {sel.types.map(t => <span key={t} className={`badge badge-${t}`}>{t.toUpperCase()}</span>)}
                        <span className="badge badge-page">PAGE {sel.page}</span>
                      </div>
                      <div className="insp-label">{inspTab === 'summary' ? 'Content preview' : 'Original Text'}</div>
                      <div className="insp-content">
                        {inspTab === 'summary'
                          ? sel.content.split(/(?<=[.?!])\s+/).slice(0, 2).join(' ')
                          : sel.content}
                      </div>
                      <div className="insp-meta">{sel.chars} characters · Chunk #{sel.id} · from page {sel.page}</div>
                    </>
                  ) : <p className="muted">Select a chunk to inspect it.</p>}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="modal-foot">
          <span className="muted">Pipeline: Upload → Partitioning → Chunking → Vectorization & Storage → Chat</span>
          <button className="primary-btn" onClick={() => onChat(doc)}>💬 Chat with this PDF</button>
        </div>
      </div>
    </div>
  )
}
