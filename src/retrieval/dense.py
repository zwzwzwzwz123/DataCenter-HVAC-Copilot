from __future__ import annotations

from typing import Iterable

from src.retrieval.embeddings import DeterministicHashEmbeddingProvider, EmbeddingProvider
from src.retrieval.schemas import DocumentChunk


class DenseRetriever:
    """In-memory dense retriever using normalized embedding dot products."""

    def __init__(
        self,
        chunks: Iterable[DocumentChunk],
        *,
        embedding_provider: EmbeddingProvider | None = None,
        retrieval_mode: str = "dense_hash",
    ) -> None:
        self.chunks = list(chunks)
        self.embedding_provider = embedding_provider or DeterministicHashEmbeddingProvider()
        self.retrieval_mode = retrieval_mode
        self._chunk_vectors = self.embedding_provider.embed_texts(
            [chunk.text for chunk in self.chunks]
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not query.strip() or not self.chunks:
            return []
        query_vector = self.embedding_provider.embed_texts([query])[0]
        scored = []
        for chunk, vector in zip(self.chunks, self._chunk_vectors):
            score = _dot(query_vector, vector)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            {
                "chunk_id": chunk.chunk_id,
                "score": score,
                "text": chunk.text,
                "citation": chunk.citation,
                "retrieval_mode": self.retrieval_mode,
            }
            for score, chunk in scored[:top_k]
        ]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
