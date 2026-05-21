# FAISS Dense Retrieval Design

## Goal

Add an optional dense retrieval path so the project can compare keyword, BM25-style hybrid, dense embedding, and rerank retrieval on the same HVAC evaluation set.

## Scope

The default project must still run without network access, API keys, `faiss-cpu`, or `sentence-transformers`. Dense retrieval is an optional RAG enhancement. Tests use deterministic local embeddings and do not download models.

The first version adds a dense retriever baseline and optional FAISS backend. Qdrant remains roadmap because it adds service deployment and collection management that are not needed for the current local evaluation loop.

## Architecture

- `src/retrieval/embeddings.py` defines a small `EmbeddingProvider` protocol.
- `DeterministicHashEmbeddingProvider` provides stable local vectors for tests and deterministic smoke runs.
- `SentenceTransformerEmbeddingProvider` wraps `sentence-transformers` when installed.
- `src/retrieval/dense.py` provides `DenseRetriever`, a pure-Python cosine similarity retriever over in-memory vectors. This makes dense behavior testable without FAISS.
- `src/retrieval/faiss_retriever.py` provides `FaissDenseRetriever`, an optional FAISS-backed retriever. It raises a clear `ImportError` if `faiss` is missing.
- `src/evaluation/runner.py` adds `rag_dense` to baseline comparison using the deterministic embedding provider by default. A later command-line flag can opt into the real sentence-transformer provider.

## Retrieval Output Contract

Dense retrievers return the same structure as existing retrievers:

```json
{
  "chunk_id": "...",
  "score": 0.91,
  "text": "...",
  "citation": {...},
  "retrieval_mode": "dense_hash" 
}
```

FAISS-backed results use `retrieval_mode = "dense_faiss"`.

## Dependencies

Default dependencies do not change. Add optional extra:

```toml
[project.optional-dependencies]
dense = [
  "faiss-cpu>=1.8",
  "sentence-transformers>=3.0"
]
```

This keeps `pip install -e ".[dev]"` fast and deterministic. Users who want real dense retrieval can run:

```bash
pip install -e ".[dev,dense]"
```

## Evaluation

The default comparison gains `rag_dense` using deterministic hash embeddings. This is not a semantic model benchmark, but it proves the dense retriever contract and keeps the report shape stable without optional dependencies.

If real dense dependencies are installed later, the project can add a CLI flag such as `--dense-provider sentence-transformers` to run FAISS + real embeddings.

## Documentation

README and system design should explain:

- FAISS is a local vector index and does not require an API.
- API cost only appears if an API embedding provider is chosen.
- This project starts with optional local `faiss-cpu + sentence-transformers`.
- Qdrant is kept as production-style vector DB roadmap.

## Testing

Tests cover:

- deterministic embeddings are stable and normalized;
- dense retriever returns expected nearest chunks and preserves citations;
- FAISS retriever reports missing optional dependency clearly when FAISS is absent;
- baseline comparison includes `rag_dense`;
- report and docs mention dense retrieval without claiming Qdrant is implemented.
