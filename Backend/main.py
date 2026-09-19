import os
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredHTMLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2")
vectorstore = Chroma(
    collection_name="docs",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".csv": CSVLoader,
    ".html": UnstructuredHTMLLoader,
}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in LOADER_MAP:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        loader = LOADER_MAP[ext](str(save_path))
        docs = loader.load()
    except Exception as e:
        raise HTTPException(400, f"Failed to read {file.filename}: {e}")

    for doc in docs:
        doc.metadata["source"] = file.filename

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    if len(chunks) == 0:
        raise HTTPException(
            400,
            f"No extractable text found in {file.filename}. "
            "This usually means it's a scanned/image-based PDF (needs OCR, not supported) "
            "or the file has no readable text content."
        )

    vectorstore.add_documents(chunks)

    return {"filename": file.filename, "chunks_added": len(chunks)}


class ChatTurn(BaseModel):
    question: str
    answer: str


class QueryRequest(BaseModel):
    question: str
    history: Optional[List[ChatTurn]] = None
    doc_filter: Optional[List[str]] = None  # filenames to restrict search to


def condense_question(question: str, history: List[ChatTurn]) -> str:
    """Rewrite a follow-up question into a standalone question using chat history."""
    if not history:
        return question

    history_text = "\n".join(f"Q: {t.question}\nA: {t.answer}" for t in history[-5:])

    prompt = f"""Given this conversation history and a follow-up question, rewrite the follow-up
into a standalone question that includes all necessary context. If the follow-up is already
standalone, return it unchanged. Reply with ONLY the rewritten question, nothing else.

Conversation history:
{history_text}

Follow-up question: {question}

Standalone question:"""

    result = llm.invoke(prompt)
    return result.content.strip()


def rerank_results(question: str, results: list, top_n: int = 4) -> list:
    """Re-score retrieved chunks against the question using a cross-encoder,
    and return only the top_n most relevant ones."""
    if not results:
        return results

    pairs = [(question, r.page_content) for r in results]
    scores = reranker.predict(pairs)

    scored = list(zip(results, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored[:top_n]]


@app.get("/documents")
def list_documents():
    """Return the distinct list of source filenames currently indexed."""
    data = vectorstore.get(include=["metadatas"])
    sources = sorted({m.get("source") for m in data["metadatas"] if m.get("source")})
    return {"documents": sources}


@app.post("/query")
async def query(req: QueryRequest):
    history = req.history or []
    standalone_question = condense_question(req.question, history)

    where_filter = None
    if req.doc_filter:
        if len(req.doc_filter) == 1:
            where_filter = {"source": req.doc_filter[0]}
        else:
            where_filter = {"source": {"$in": req.doc_filter}}

    # Cast a wider net (12 candidates) so reranking has more to work with
    candidates = vectorstore.similarity_search(
        standalone_question, k=12, filter=where_filter
    )

    if len(candidates) == 0:
        return {
            "answer": "No indexed documents match this question yet. Upload a document first, "
                      "or check your document filter.",
            "sources": [],
        }

    results = rerank_results(standalone_question, candidates, top_n=4)

    context = "\n\n".join(
        f"[Source: {r.metadata.get('source')}, page {r.metadata.get('page', 'N/A')}]\n{r.page_content}"
        for r in results
    )

    history_text = ""
    if history:
        history_text = "\n".join(f"Q: {t.question}\nA: {t.answer}" for t in history[-5:])
        history_text = f"\nPrevious conversation:\n{history_text}\n"

    prompt = f"""Answer the question using ONLY the context below. If the context doesn't
contain the answer, say so. Use the previous conversation only to understand what the
question is referring to, not as a source of facts.
{history_text}
Context:
{context}

Question: {req.question}

Answer:"""

    response = llm.invoke(prompt)

    sources = [
        {
            "source": r.metadata.get("source"),
            "page": r.metadata.get("page", "N/A"),
            "snippet": r.page_content[:200],
        }
        for r in results
    ]

    return {"answer": response.content, "sources": sources}