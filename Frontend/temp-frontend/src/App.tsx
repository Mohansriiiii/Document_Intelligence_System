import { useState, useRef } from "react";
import "./App.css";

interface Source {
  source: string;
  page: string | number;
  snippet: string;
}

interface ChatMessage {
  question: string;
  answer: string;
  sources: Source[];
}

const API_URL = "http://localhost:8000";

function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [uploadedDocs, setUploadedDocs] = useState<string[]>([]);
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploading, setUploading] = useState(false);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUpload = async () => {
    if (files.length === 0) return;
    setUploading(true);
    setUploadStatus("Indexing documents...");

    for (const file of files) {
      const formData = new FormData();
      formData.append("file", file);
      try {
        const res = await fetch(`${API_URL}/upload`, {
          method: "POST",
          body: formData,
        });
        if (!res.ok) throw new Error(await res.text());
        setUploadedDocs((prev) => [...prev, file.name]);
      } catch (err) {
        setUploadStatus(`Failed: ${file.name} — ${err}`);
      }
    }

    setUploadStatus("");
    setFiles([]);
    setUploading(false);
  };

  const handleAsk = async () => {
    if (!question.trim()) return;
    setLoading(true);
    const askedQuestion = question;
    setQuestion("");
    try {
      const historyPayload = messages.map((m) => ({
        question: m.question,
        answer: m.answer,
      }));

      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: askedQuestion,
          history: historyPayload,
          doc_filter: selectedDocs.length > 0 ? selectedDocs : null,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { question: askedQuestion, answer: data.answer, sources: data.sources },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { question: askedQuestion, answer: `Something went wrong: ${err}`, sources: [] },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <aside className="shelf">
        <div className="shelf-header">
          <span className="mark">§</span>
          <h1>Archive</h1>
        </div>
        <p className="shelf-subtitle">Ask questions grounded in your own documents</p>

        <div className="upload-zone">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            id="file-input"
            className="file-input-hidden"
            onChange={(e) => {
              const newFiles = e.target.files ? Array.from(e.target.files) : [];
              setFiles((prev) => [...prev, ...newFiles]);
              e.target.value = "";
            }}
          />
          <label htmlFor="file-input" className="upload-label">
            + Add documents
          </label>

          {files.length > 0 && (
            <ul className="pending-list">
              {files.map((f, i) => (
                <li key={i}>
                  <span>{f.name}</span>
                  <button
                    className="remove-btn"
                    onClick={() => setFiles((prev) => prev.filter((_, idx) => idx !== i))}
                    aria-label={`Remove ${f.name}`}
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}

          {files.length > 0 && (
            <button className="index-btn" onClick={handleUpload} disabled={uploading}>
              {uploading ? "Indexing…" : `Index ${files.length} file${files.length > 1 ? "s" : ""}`}
            </button>
          )}
          {uploadStatus && <p className="upload-status">{uploadStatus}</p>}
        </div>

        <div className="doc-list">
          <div className="doc-list-header">
            <span className="doc-list-label">In this archive — {uploadedDocs.length}</span>
            {selectedDocs.length > 0 && (
              <button className="clear-filter-btn" onClick={() => setSelectedDocs([])}>
                Clear filter
              </button>
            )}
          </div>
          {uploadedDocs.length === 0 ? (
            <p className="empty-note">No documents indexed yet.</p>
          ) : (
            <>
              <p className="filter-hint">
                {selectedDocs.length === 0
                  ? "Searching all documents. Check any to narrow scope."
                  : `Searching only ${selectedDocs.length} selected document${selectedDocs.length > 1 ? "s" : ""}.`}
              </p>
              <ul>
                {uploadedDocs.map((name, i) => {
                  const checked = selectedDocs.includes(name);
                  return (
                    <li key={i} className="doc-item">
                      <label className="doc-checkbox-label">
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={() =>
                            setSelectedDocs((prev) =>
                              checked ? prev.filter((n) => n !== name) : [...prev, name]
                            )
                          }
                        />
                        <span className="doc-dot" />
                        <span className="doc-name">{name}</span>
                      </label>
                    </li>
                  );
                })}
              </ul>
            </>
          )}
        </div>
      </aside>

      <main className="research-pane">
        {messages.length === 0 ? (
          <div className="empty-state">
            <span className="mark large">§</span>
            <h2>Nothing asked yet</h2>
            <p>Index a document on the left, then ask a question below.</p>
          </div>
        ) : (
          <div className="thread">
            {messages.map((msg, i) => (
              <div className="entry" key={i}>
                <p className="entry-question">{msg.question}</p>
                <div className="entry-answer">
                  <p>
                    {msg.answer}
                    {msg.sources.length > 0 && (
                      <span className="inline-cites">
                        {msg.sources.map((_, j) => (
                          <sup key={j} className="cite-marker">
                            [{j + 1}]
                          </sup>
                        ))}
                      </span>
                    )}
                  </p>
                </div>
                {msg.sources.length > 0 && (
                  <div className="citations">
                    {msg.sources.map((s, j) => (
                      <div className="citation-card" key={j}>
                        <span className="citation-number">{j + 1}</span>
                        <div className="citation-body">
                          <span className="citation-source">
                            {s.source} <span className="citation-page">p.{s.page}</span>
                          </span>
                          <p className="citation-snippet">{s.snippet}…</p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        <div className="ask-bar">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAsk()}
            placeholder="Ask something about your documents…"
          />
          <button onClick={handleAsk} disabled={loading || !question.trim()}>
            {loading ? "…" : "Ask"}
          </button>
        </div>
      </main>
    </div>
  );
}

export default App;