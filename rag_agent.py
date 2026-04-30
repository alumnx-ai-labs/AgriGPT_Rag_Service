"""
AgriGPT RAG Service  v4.0
Single Pinecone index, metadata-filtered retrieval, Ollama tool calling.
"""

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os, io, time, json

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY   = os.getenv("OLLAMA_API_KEY", "ollama")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL", "gemma4:26b")
RAG_BASE_URL     = os.getenv("RAG_BASE_URL", "http://localhost:8010")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set")

INDEX_NAME    = os.getenv("PINECONE_INDEX_NAME", "agriculture-knowledge-base")
ALLOWED_TYPES = {"pests", "schemes"}

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
EMBED_DIM     = int(os.getenv("EMBED_DIM", "3072"))
EMBED_BATCH   = 20
UPSERT_BATCH  = 100

# ── Clients ────────────────────────────────────────────────────────────────────

pc     = Pinecone(api_key=PINECONE_API_KEY)
client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
    timeout=300.0,  # 5 min — 26B models need time to generate
)

app = FastAPI(
    title="AgriGPT RAG Service",
    description="Single-index RAG with metadata filtering and Ollama tool calling",
    version="4.0.0",
)

# ── Pinecone setup ─────────────────────────────────────────────────────────────

def _init_index():
    existing = {idx["name"] for idx in pc.list_indexes()}
    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index: {INDEX_NAME} (dim={EMBED_DIM})")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        print(f"Connected to existing index: {INDEX_NAME}")
    return pc.Index(INDEX_NAME)

index = _init_index()

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

# ── Text extraction ────────────────────────────────────────────────────────────

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

# ── Embeddings ─────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _embed_batch(texts: List[str]) -> List[List[float]]:
    res = client.embeddings.create(model=OLLAMA_MODEL, input=texts)
    return [e.embedding for e in res.data]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _embed_query(text: str) -> List[float]:
    res = client.embeddings.create(model=OLLAMA_MODEL, input=text)
    return res.data[0].embedding

# ── Pinecone search (metadata-filtered) ───────────────────────────────────────

def _search(doc_type: str, query: str, top_k: int) -> List[Dict]:
    vec = _embed_query(query)
    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
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

# ── Tool definitions (OpenAI format) ──────────────────────────────────────────

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_pests",
            "description": (
                "Search the pests and diseases knowledge base. Use this for questions about "
                "crop diseases, pest identification, symptoms, treatments, prevention, and "
                "agricultural pest management."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific search query for pests or diseases",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to retrieve (default 5)",
                    },
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
                "Search the government schemes knowledge base. Use this for questions about "
                "agricultural subsidies, government programs, farmer benefits, financial aid, "
                "and agricultural policy schemes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific search query for government schemes",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to retrieve (default 5)",
                    },
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

# ── Upload endpoint ────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse, tags=["Upload"])
async def upload(
    file: UploadFile = File(...),
    type: str = Form(..., description="Document type: 'pests' or 'schemes'"),
):
    """
    Upload a document and index it.

    - **file**: .pdf, .txt, or .docx
    - **type**: `pests` or `schemes` — stored as metadata on every chunk
    """
    if type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Invalid type '{type}'. Must be one of: {', '.join(sorted(ALLOWED_TYPES))}")

    text = _extract_text(file)
    if not text.strip():
        raise HTTPException(400, "File is empty or unreadable")

    chunks = _chunk(text)
    if not chunks:
        raise HTTPException(400, "No valid chunks extracted from file")

    print(f"[{type}] {file.filename}: {len(chunks)} chunks")

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
        if i + EMBED_BATCH < len(chunks):
            time.sleep(0.5)

    for i in range(0, len(vectors), UPSERT_BATCH):
        index.upsert(vectors=vectors[i : i + UPSERT_BATCH])

    return UploadResponse(
        message="Indexed successfully",
        filename=file.filename,
        type=type,
        chunks_added=len(chunks),
    )

# ── Query endpoint ─────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Ask a question. The model picks the right tool (`search_pests` or `search_schemes`),
    queries the single index with a metadata filter, and returns a grounded answer.
    """
    messages = [{"role": "user", "content": request.question}]

    resp = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=messages,
        tools=_TOOLS,
    )

    all_sources: List[Dict] = []
    tools_used:  List[str]  = []

    msg = resp.choices[0].message

    if msg.tool_calls:
        messages.append(msg)

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            args    = json.loads(tc.function.arguments)
            top_k   = int(args.get("top_k", request.top_k))
            q_text  = args["query"]

            if fn_name not in _TOOL_FN:
                raise HTTPException(500, f"Unknown tool requested by model: {fn_name}")

            print(f"Tool → {fn_name}(query={q_text!r}, top_k={top_k})")
            chunks = _TOOL_FN[fn_name](q_text, top_k)
            all_sources.extend(chunks)
            tools_used.append(fn_name)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps({
                    "chunks": [
                        {"text": c["text"], "source": c["filename"]}
                        for c in chunks
                    ]
                }),
            })

        # Round 2 — generate final grounded answer
        resp = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=_TOOLS,
        )

    answer = resp.choices[0].message.content or "No answer could be generated."

    return QueryResponse(
        answer=answer,
        sources=[SourceChunk(**s) for s in all_sources],
        tools_used=tools_used,
    )

# ── Health & stats ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    try:
        stats = index.describe_index_stats()
        return {
            "status":        "healthy",
            "index":         INDEX_NAME,
            "total_vectors": stats.total_vector_count,
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
        "service": "AgriGPT RAG Service",
        "version": "4.0.0",
        "model":   OLLAMA_MODEL,
        "docs":    "/docs",
        "endpoints": {
            "POST /upload": "Index a document — pass file + type ('pests' or 'schemes')",
            "POST /query":  "Ask a question — model picks the right tool automatically",
            "GET  /health": "Health check with total vector count",
            "GET  /stats":  "Index statistics",
        },
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8010))
    uvicorn.run(app, host="0.0.0.0", port=port)
