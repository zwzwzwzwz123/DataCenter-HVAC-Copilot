from __future__ import annotations

from typing import Iterable

import numpy as np

from src.retrieval.embeddings import DeterministicHashEmbeddingProvider, EmbeddingProvider
from src.retrieval.schemas import DocumentChunk


class FaissDenseRetriever:
    """Optional FAISS-backed dense retriever."""

    def __init__(
        self,
        chunks: Iterable[DocumentChunk],
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                'faiss-cpu is required for FAISS dense retrieval. Install it with: pip install -e ".[dev,dense]"'
            ) from exc
        self._faiss = faiss
        self.chunks = list(chunks)
        self.embedding_provider = embedding_provider or DeterministicHashEmbeddingProvider()
        vectors = self.embedding_provider.embed_texts([chunk.text for chunk in self.chunks])
        self._vectors = np.asarray(vectors, dtype="float32")
        dimension = self._vectors.shape[1] if len(self._vectors) else 1
        self._index = faiss.IndexFlatIP(dimension)
        if len(self._vectors):
            self._index.add(self._vectors)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not query.strip() or not self.chunks:
            return []
        query_vector = np.asarray(self.embedding_provider.embed_texts([query]), dtype="float32")
        scores, indices = self._index.search(query_vector, min(top_k, len(self.chunks)))
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or float(score) <= 0:
                continue
            chunk = self.chunks[int(index)]
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "score": float(score),
                    "text": chunk.text,
                    "citation": chunk.citation,
                    "retrieval_mode": "dense_faiss",
                }
            )
        return results
