# Persistent Knowledge Base Design

## Goal

Build a production-shaped document ingestion and retrieval path for DataCenter-HVAC Copilot. Operators should be able to upload PDF, DOCX, TXT, or Markdown manuals and SOP documents, have them parsed and chunked, persist metadata in SQLite, persist embeddings in FAISS, and use the uploaded knowledge as grounded RAG evidence in `/ask`.

## Current Gap

The current RAG corpus is static. `src/api/demo_factory.py` loads `.md` and `.txt` files from `data/documents/` at application startup, chunks them in memory, and builds a `HybridRetriever`. There is no upload API, no PDF/DOCX parsing, no document metadata store, and no persisted vector index for user-provided documents.

## Architecture

Add a new `src/knowledge/` package that owns dynamic document ingestion and persistent retrieval. SQLite is the source of truth for document and chunk metadata. FAISS stores dense vectors for retrieval, with a sidecar `chunks.jsonl` mapping FAISS row positions back to chunk metadata and text.

The first version rebuilds the full FAISS index whenever documents change. It writes `index.faiss.tmp` and `chunks.jsonl.tmp`, then atomically replaces `index.faiss` and `chunks.jsonl` only after the full rebuild succeeds. This avoids partial-index corruption and keeps delete/update semantics simple.

## Storage Layout

```text
data/knowledge/
  uploads/
    <document_id>_<safe_filename>
  parsed/
    <document_id>.json
  faiss/
    index.faiss
    chunks.jsonl
  knowledge.db
```

## SQLite Model

`documents` stores one row per uploaded document:

- `document_id`
- `filename`
- `file_type`
- `file_hash`
- `source_path`
- `parsed_path`
- `status`
- `chunk_count`
- `error_message`
- `created_at`
- `updated_at`
- `metadata_json`

`chunks` stores one row per chunk:

- `chunk_id`
- `document_id`
- `chunk_index`
- `text`
- `page_number`
- `section_title`
- `token_count`
- `metadata_json`
- `created_at`

`index_versions` stores rebuild metadata:

- `index_id`
- `faiss_path`
- `chunks_path`
- `embedding_provider`
- `embedding_model`
- `chunk_count`
- `created_at`
- `metadata_json`

## Parsing

Supported first-version formats:

- `.md`
- `.txt`
- `.pdf`
- `.docx`

TXT and Markdown use UTF-8 text loading. PDF uses PyMuPDF if available, then pypdf fallback if installed. DOCX uses `python-docx`. If an optional parser dependency is missing, the upload should fail with a clear status and error message rather than silently ingesting an empty document.

Parsed text is normalized into pages or logical sections before chunking. Each chunk keeps source metadata such as filename, document id, page number, and section title when available.

## Chunking

Use a knowledge-specific chunker rather than reusing the current simple word chunker directly. The first version should support:

- configurable `chunk_size_words`, default `220`
- configurable `overlap_words`, default `40`
- stable chunk ids: `<document_id>::chunk_<0000>`
- metadata-preserving chunks

## Embedding And FAISS

Use the existing embedding interfaces from `src/retrieval/embeddings.py`.

Default high-quality configuration:

```text
KNOWLEDGE_EMBEDDING_PROVIDER=sentence-transformers
KNOWLEDGE_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

Tests can use the deterministic hash embedding provider for speed and offline reproducibility.

The persistent retriever loads:

- `data/knowledge/faiss/index.faiss`
- `data/knowledge/faiss/chunks.jsonl`

It returns retrieval results with the same shape expected by existing RAG components:

```python
{
    "chunk_id": "...",
    "score": 0.87,
    "text": "...",
    "citation": {
        "source_id": document_id,
        "title": filename,
        "source_path": "...",
        "page_number": 12,
        "section": "Alarm Handling",
    },
    "retrieval_mode": "persistent_faiss",
}
```

## API

Add knowledge endpoints to FastAPI:

```text
POST   /knowledge/documents/upload
GET    /knowledge/documents
GET    /knowledge/documents/{document_id}
DELETE /knowledge/documents/{document_id}
POST   /knowledge/reindex
GET    /knowledge/status
```

Upload returns the document record plus index rebuild status. Delete marks or removes the document and triggers a full FAISS rebuild. The first version can perform indexing synchronously because the expected corpus size is small.

## RAG Integration

`build_demo_orchestrator()` should prefer the persistent FAISS knowledge base when an index exists. Static `data/documents/` can remain as seed/fallback knowledge, but uploaded documents should be available to `/ask` without restarting the API after upload.

The simplest first-version integration is:

1. Build a persistent retriever at app startup.
2. Let upload/reindex refresh the orchestrator's RAG pipeline or retriever instance.
3. If no persistent index exists, fall back to the current static `HybridRetriever`.

## Streamlit

Add a `Knowledge Base` tab:

- upload PDF/DOCX/TXT/MD
- show document list
- show status, chunk count, upload time, and error message
- show FAISS index status
- provide a `Reindex` button

The Copilot tab should continue to work normally. After upload succeeds, the user can ask questions grounded in the uploaded document.

## Error Handling

- Unsupported suffix: return 400 with supported suffix list.
- Empty parsed text: store failed status and return a clear error.
- Parser dependency missing: store failed status and include install hint.
- FAISS rebuild failure: keep the previous `index.faiss` and `chunks.jsonl`.
- Duplicate upload: deduplicate by file hash and return the existing document unless the user later asks for versioning.

## Testing

Use TDD. Tests should cover:

- parser behavior for TXT/MD and dependency-gated PDF/DOCX
- SQLite document/chunk persistence
- chunk metadata preservation
- FAISS save/load retrieval consistency using deterministic embeddings
- atomic rebuild keeps old index on failure
- API upload/list/status/reindex/delete behavior
- `/ask` retrieval from uploaded knowledge
- Streamlit API client wrappers for knowledge endpoints

## Out Of Scope For First Version

- Multi-user permissions
- async task queue
- Qdrant/Milvus
- OCR
- table extraction
- document version history beyond hash deduplication
- cross-encoder reranking

