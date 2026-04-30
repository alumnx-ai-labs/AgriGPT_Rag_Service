"""
AgriGPT RAG Service  v5.2
Embeddings  → Ollama native  /api/embed  (host root)
Chat        → Ollama OpenAI  /v1/chat/completions
Vector DB   → Pinecone (auto-recreates on dim mismatch)
"""

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os, io, time, json
from urllib.parse import urlparse

import httpx
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY")
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
OLLAMA_API_KEY    = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set")

# Derive host root for native Ollama embedding endpoints
_parsed    = urlparse(OLLAMA_BASE_URL)
OLLAMA_HOST = f"{_parsed.scheme}://{_parsed.netloc}"   # e.g. http://3.109.63.164

INDEX_NAME    = os.getenv("PINECONE_INDEX_NAME", "agriculture-knowledge-base")
ALLOWED_TYPES = {"pests", "schemes"}

CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 200
EMBED_BATCH    = 5       # small batches — large model, avoid timeouts
UPSERT_BATCH   = 100
OLLAMA_TIMEOUT = 300.0

_HEADERS = {
    "Authorization": f"Bearer {OLLAMA_API_KEY}",
    "Content-Type":  "application/json",
}

# ── Pinecone ───────────────────────────────────────────────────────────────────

pc  = Pinecone(api_key=PINECONE_API_KEY)

app = FastAPI(
    title="AgriGPT RAG Service",
    description="Ollama embeddings + chat, Pinecone vector store",
    version="5.2.0",
)

# ── Embedding helpers ──────────────────────────────────────────────────────────

def _ollama_embed(texts) -> List[List[float]]:
    """
    Try native Ollama endpoints on the host root.
    /api/embed  (Ollama >= 0.3, batch)  → {"embeddings": [[...]]}
    /api/embeddings  (older, single)    → {"embedding": [...]}
    """
    if isinstance(texts, str):
        texts = [texts]

    with httpx.Client(timeout=OLLAMA_TIMEOUT) as http:
        # Try newer batch endpoint first
        r = http.post(
            f"{OLLAMA_HOST}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": texts},
            headers=_HEADERS,
        )
        if r.status_code == 200:
            return r.json()["embeddings"]
        print(f"[EMBED] /api/embed {r.status_code}: {r.text[:200]}")

        # Fall back to older single-text endpoint
        results = []
        for t in texts:
            r2 = http.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={"model": OLLAMA_EMBED_MODEL, "prompt": t},
                headers=_HEADERS,
            )
            if r2.status_code == 200:
                results.append(r2.json()["embedding"])
            else:
                print(f"[EMBED] /api/embeddings {r2.status_code}: {r2.text[:200]}")
                r2.raise_for_status()
        return results


def _detect_dim() -> int:
    try:
        vec = _ollama_embed("test")[0]
        print(f"[EMBED] model={OLLAMA_EMBED_MODEL} dim={len(vec)}")
        return len(vec)
    except Exception as e:
        fallback = int(os.getenv("EMBED_DIM", "768"))
        print(f"[EMBED] detection failed: {e} — using {fallback}")
        return fallback


# ── Pinecone index (auto-recreate on dim mismatch) ────────────────────────────

def _init_index(dim: int):
    existing = {idx["name"] for idx in pc.list_indexes()}
    if INDEX_NAME in existing:
        desc = pc.describe_index(INDEX_NAME)
        if desc.dimension != dim:
            print(f"[PINECONE] dim mismatch {desc.dimension}→{dim}, recreating…")
            pc.delete_index(INDEX_NAME)
            time.sleep(10)
        else:
            print(f"[PINECONE] connected (dim={dim})")
            return pc.Index(INDEX_NAME)
    print(f"[PINECONE] creating index (dim={dim})")
    pc.create_index(
        name=INDEX_NAME,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    return pc.Index(INDEX_NAME)


EMBED_DIM = _detect_dim()
index     = _init_index(EMBED_DIM)

# ── Pydantic models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class SourceChunk(BaseModel):
    chunk_id: str
    filename: str
    type: str
    score: float
    text: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    tools_used: List[str]

class UploadResponse(BaseModel):
    message: str
    filename: str
    type: str
    chunks_added: int

# ── Text helpers ───────────────────────────────────────────────────────────────

def _extract_text(file: UploadFile) -> str:
    data = file.file.read()
    if file.filename.endswith(".txt"):
        return data.decode("utf-8")
    if file.filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if file.filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    raise HTTPException(400, "Unsupported file type. Use .pdf, .txt or .docx")

def _chunk(text: str) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE]
        if chunk.strip():
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# ── Embed wrappers ─────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
def _embed_batch(texts: List[str]) -> List[List[float]]:
    return _ollama_embed(texts)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
def _embed_query(text: str) -> List[float]:
    return _ollama_embed(text)[0]

# ── Chat helper ────────────────────────────────────────────────────────────────

def _parse_sse(text: str) -> Dict:
    content_parts  = []
    tool_calls_map: Dict[int, Dict] = {}
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                content_parts.append(delta["content"])
            for tc in delta.get("tool_calls", []):
                idx = tc.get("index", 0)
                if idx not in tool_calls_map:
                    tool_calls_map[idx] = {
                        "id": tc.get("id", f"call_{idx}"),
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc.get("id"):
                    tool_calls_map[idx]["id"] = tc["id"]
                fn = tc.get("function", {})
                tool_calls_map[idx]["function"]["name"]      += fn.get("name", "")
                tool_calls_map[idx]["function"]["arguments"] += fn.get("arguments", "")
        except json.JSONDecodeError:
            pass
    msg: Dict = {"role": "assistant", "content": "".join(content_parts)}
    if tool_calls_map:
        msg["tool_calls"] = [tool_calls_map[i] for i in sorted(tool_calls_map)]
    return {"choices": [{"message": msg}]}


def _chat(messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
    payload: Dict = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    with httpx.Client(timeout=OLLAMA_TIMEOUT) as http:
        r = http.post(
            f"{OLLAMA_BASE_URL}/chat/completions",
            json=payload,
            headers=_HEADERS,
        )
        if r.status_code != 200:
            print(f"[CHAT] {r.status_code}: {r.text[:300]}")
            r.raise_for_status()
        text = r.text.strip()
        if text.startswith("data:"):
            return _parse_sse(text)
        return json.loads(text)

# ── Pinecone search ────────────────────────────────────────────────────────────

def _search(doc_type: str, query: str, top_k: int) -> List[Dict]:
    vec = _embed_query(query)
    res = index.query(
        vector=vec, top_k=top_k, include_metadata=True,
        filter={"type": {"$eq": doc_type}},
    )
    return [
        {
            "chunk_id": m["id"],
            "filename": m["metadata"]["filename"],
            "type":     m["metadata"]["type"],
            "score":    float(m["score"]),
            "text":     m["metadata"]["text"],
        }
        for m in res["matches"]
    ]

# ── Tool definitions ───────────────────────────────────────────────────────────

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_pests",
            "description": (
                "Search the pests and diseases knowledge base for questions about "
                "crop diseases, pest identification, symptoms, treatments and prevention."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_schemes",
            "description": (
                "Search the government schemes knowledge base for questions about "
                "agricultural subsidies, programs, farmer benefits and financial aid."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
]

_TOOL_FN = {
    "search_pests":   lambda q, k: _search("pests", q, k),
    "search_schemes": lambda q, k: _search("schemes", q, k),
}

# ── Upload ─────────────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse, tags=["Upload"])
async def upload(
    file: UploadFile = File(...),
    type: str = Form(...),
):
    if type not in ALLOWED_TYPES:
        raise HTTPException(400, f"type must be one of: {', '.join(sorted(ALLOWED_TYPES))}")
    text = _extract_text(file)
    if not text.strip():
        raise HTTPException(400, "File is empty or unreadable")
    chunks = _chunk(text)
    if not chunks:
        raise HTTPException(400, "No valid chunks extracted")

    print(f"[UPLOAD] {file.filename} ({type}): {len(chunks)} chunks")
    vectors = []
    for i in range(0, len(chunks), EMBED_BATCH):
        batch      = chunks[i : i + EMBED_BATCH]
        embeddings = _embed_batch(batch)
        for j, (chunk, emb) in enumerate(zip(batch, embeddings)):
            chunk_idx = i + j
            vectors.append({
                "id":     f"{file.filename}_{chunk_idx}",
                "values": emb,
                "metadata": {
                    "text":        chunk,
                    "filename":    file.filename,
                    "chunk_index": chunk_idx,
                    "type":        type,
                },
            })
        time.sleep(0.5)

    for i in range(0, len(vectors), UPSERT_BATCH):
        index.upsert(vectors=vectors[i : i + UPSERT_BATCH])

    return UploadResponse(
        message="Indexed successfully",
        filename=file.filename,
        type=type,
        chunks_added=len(chunks),
    )

# ── Query ──────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    messages: List[Dict] = [{"role": "user", "content": request.question}]
    resp = _chat(messages, tools=_TOOLS)
    msg  = resp["choices"][0]["message"]

    all_sources: List[Dict] = []
    tools_used:  List[str]  = []
    tool_calls = msg.get("tool_calls") or []

    print(f"[QUERY] tool_calls={bool(tool_calls)}  content={str(msg.get('content',''))[:80]!r}")

    if tool_calls:
        messages.append(msg)
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            args    = tc["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            top_k  = int(args.get("top_k", request.top_k))
            q_text = args["query"]
            if fn_name not in _TOOL_FN:
                raise HTTPException(500, f"Unknown tool: {fn_name}")
            print(f"[TOOL] {fn_name}(query={q_text!r}, top_k={top_k})")
            chunks = _TOOL_FN[fn_name](q_text, top_k)
            all_sources.extend(chunks)
            tools_used.append(fn_name)
            messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "content":      json.dumps({
                    "chunks": [{"text": c["text"], "source": c["filename"]} for c in chunks]
                }),
            })
        resp = _chat(messages)
        msg  = resp["choices"][0]["message"]

    else:
        # Fallback: model skipped tools — search directly
        print("[QUERY] no tool calls — direct search fallback")
        for doc_type in ("pests", "schemes"):
            chunks = _search(doc_type, request.question, request.top_k)
            if chunks:
                all_sources.extend(chunks)
                tools_used.append(f"search_{doc_type}")
        if all_sources:
            context = "\n\n".join(f"[{c['filename']}]: {c['text']}" for c in all_sources)
            messages = [{"role": "user", "content": (
                f"Answer using only the context below.\n\nContext:\n{context}\n\nQuestion: {request.question}"
            )}]
            resp = _chat(messages)
            msg  = resp["choices"][0]["message"]

    return QueryResponse(
        answer=msg.get("content") or "No answer could be generated.",
        sources=[SourceChunk(**s) for s in all_sources],
        tools_used=tools_used,
    )

# ── System endpoints ───────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    try:
        stats = index.describe_index_stats()
        return {
            "status":        "healthy",
            "index":         INDEX_NAME,
            "total_vectors": stats.total_vector_count,
            "embed_dim":     EMBED_DIM,
            "model":         OLLAMA_MODEL,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/stats", tags=["System"])
def stats():
    s = index.describe_index_stats()
    return {
        "index_name":     INDEX_NAME,
        "total_vectors":  s.total_vector_count,
        "dimension":      s.dimension,
        "index_fullness": s.index_fullness,
    }

@app.get("/", tags=["System"])
def root():
    return {
        "service":   "AgriGPT RAG Service",
        "version":   "5.2.0",
        "model":     OLLAMA_MODEL,
        "embed_dim": EMBED_DIM,
        "docs":      "/docs",
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8010))
    uvicorn.run(app, host="0.0.0.0", port=port)
