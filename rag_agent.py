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

def _chat(messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
    payload: Dict = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    with httpx.Client(timeout=OLLAMA_TIMEOUT) as http:
        r = http.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            headers=_HEADERS,
        )
        if r.status_code != 200:
            print(f"[CHAT] {r.status_code}: {r.text[:300]}")
            r.raise_for_status()
        # Ollama may stream NDJSON even with stream=False — accumulate all content
        lines  = [l for l in r.text.strip().splitlines() if l.strip()]
        parsed = [json.loads(l) for l in lines]
        if len(parsed) == 1:
            msg = parsed[0].get("message", {})
        else:
            # streaming: each line has a content delta; last line is empty with done=true
            content = "".join(p.get("message", {}).get("content", "") for p in parsed)
            tc_list = next(
                (p["message"]["tool_calls"] for p in parsed
                 if p.get("message", {}).get("tool_calls")), None
            )
            msg = {"role": "assistant", "content": content}
            if tc_list:
                msg["tool_calls"] = tc_list
        normalized: Dict = {"role": "assistant", "content": msg.get("content") or ""}
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for i, tc in enumerate(tool_calls):
                if "id" not in tc:
                    tc["id"] = f"call_{i}"
                if "type" not in tc:
                    tc["type"] = "function"
                # Native Ollama returns arguments as a dict; serialize for consistency
                fn_args = tc.get("function", {}).get("arguments")
                if isinstance(fn_args, dict):
                    tc["function"]["arguments"] = json.dumps(fn_args)
            normalized["tool_calls"] = tool_calls
        return {"choices": [{"message": normalized}]}

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
            "name": "search_agriculture",
            "description": (
                "Search the agriculture knowledge base. "
                "Set type='pests' for questions about crop pests, diseases, symptoms, or treatments. "
                "Set type='schemes' for questions about government schemes, subsidies, farmer benefits, or financial aid."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "type":  {"type": "string", "enum": ["pests", "schemes"],
                              "description": "Knowledge base to search"},
                    "top_k": {"type": "integer", "description": "Number of results"},
                },
                "required": ["query", "type"],
            },
        },
    },
]

_TOOL_FN = {
    "search_agriculture": lambda q, t, k: _search(t, q, k),
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
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            args    = tc["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            top_k   = int(args.get("top_k", request.top_k))
            q_text  = args["query"]
            q_type  = args.get("type", "pests")   # model must pass 'pests' or 'schemes'
            if fn_name not in _TOOL_FN:
                raise HTTPException(500, f"Unknown tool: {fn_name}")
            print(f"[TOOL] {fn_name}(type={q_type!r}, query={q_text!r}, top_k={top_k})")
            chunks = _TOOL_FN[fn_name](q_text, q_type, top_k)
            all_sources.extend(chunks)
            tools_used.append(f"search_{q_type}")

        if all_sources:
            context = "\n\n".join(f"[{c['filename']}]: {c['text']}" for c in all_sources)
            synthesis = [{"role": "user", "content": (
                f"Answer the question using only the context below.\n\n"
                f"Context:\n{context}\n\nQuestion: {request.question}"
            )}]
            resp = _chat(synthesis)
            msg  = resp["choices"][0]["message"]

    else:
        # Fallback: model skipped tools — infer type from answer content keywords
        print("[QUERY] no tool calls — direct search fallback")
        q = request.question.lower()
        schemes_kw = {"scheme", "yojana", "kisan", "subsidy", "pension", "insurance",
                      "pmfby", "rkvy", "enam", "msp", "benefit", "loan", "bima"}
        doc_types = ["schemes"] if any(k in q for k in schemes_kw) else ["pests"]
        for doc_type in doc_types:
            chunks = _search(doc_type, request.question, request.top_k)
            if chunks:
                all_sources.extend(chunks)
                tools_used.append(f"search_{doc_type}")
        if not all_sources:
            # nothing found in guessed type — try the other
            other = "pests" if doc_types[0] == "schemes" else "schemes"
            chunks = _search(other, request.question, request.top_k)
            all_sources.extend(chunks)
            if chunks:
                tools_used.append(f"search_{other}")
        if all_sources:
            context = "\n\n".join(f"[{c['filename']}]: {c['text']}" for c in all_sources)
            synthesis = [{"role": "user", "content": (
                f"Answer using only the context below.\n\nContext:\n{context}\n\nQuestion: {request.question}"
            )}]
            resp = _chat(synthesis)
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
