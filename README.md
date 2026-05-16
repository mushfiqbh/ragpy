# RAG Backend

Simple, production-lean RAG backend with:
- document parsing (`markitdown`)
- chunking with overlap
- embedding (Hugging Face API with local hash fallback)
- cosine similarity retrieval
- FastAPI endpoints for upload and Q&A
- persisted vector index in `uploads/index.json`

## Setup

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

## API

### `POST /upload`
Multipart form data:
- `file`: document to index
- `replace_index` (optional query bool): replace current index before indexing

Example:

```bash
curl -X POST "http://127.0.0.1:8000/upload?replace_index=true" \
  -F "file=@./mydoc.pdf"
```

### `POST /ask`
JSON body:

```json
{
  "question": "What is this document about?",
  "top_k": 3
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What is this document about?\",\"top_k\":3}"
```

### `GET /health`
Health status and indexed chunk count.

### `GET /stats`
Index file path and current chunk count.

## Optional Environment Variables

- `RAG_EMBEDDING_BACKEND=auto|hf|hash` (default: `auto`)
- `RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `RAG_EMBEDDING_DIM=384`
- `RAG_USE_GENERATION=true|false` (default: `false`)
- `RAG_GENERATION_MODEL=google/flan-t5-base`

When external model calls fail or are unavailable, the system falls back to local hash embeddings and extractive answers.
