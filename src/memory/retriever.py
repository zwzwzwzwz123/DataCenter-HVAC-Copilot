from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from src.memory.schemas import MemoryChunk
from src.retrieval.dense import DenseRetriever
from src.retrieval.embeddings import (
    DeterministicHashEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from src.retrieval.faiss_retriever import FaissDenseRetriever
from src.retrieval.retriever import HybridRetriever, RerankingRetriever
from src.retrieval.schemas import DocumentChunk, DocumentMetadata


@dataclass(frozen=True)
class MemoryRetrieverConfig:
    backend: str = "faiss_dense"
    allow_fallback: bool = False
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"


class MemoryRetriever(Protocol):
    status: dict[str, Any]

    def search(self, query: str, *, session_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        ...


def build_memory_retriever(
    config: MemoryRetrieverConfig,
    chunks: list[MemoryChunk],
) -> MemoryRetriever:
    try:
        return _build_available_retriever(config, chunks)
    except Exception as exc:
        if config.backend == "faiss_dense" and config.allow_fallback:
            fallback = MemoryRetrieverConfig(
                backend="hybrid_rerank",
                allow_fallback=False,
                embedding_provider=config.embedding_provider,
                embedding_model=config.embedding_model,
            )
            retriever = _build_available_retriever(fallback, chunks)
            retriever.status.update(
                {
                    "fallback_used": True,
                    "fallback_from": "faiss_dense",
                    "fallback_error": str(exc),
                }
            )
            return retriever
        return UnavailableMemoryRetriever(config.backend, str(exc))


def _build_available_retriever(
    config: MemoryRetrieverConfig,
    chunks: list[MemoryChunk],
) -> MemoryRetriever:
    documents = [_to_document_chunk(chunk) for chunk in chunks]
    if config.backend == "dense_memory":
        base = DenseRetriever(
            documents,
            embedding_provider=DeterministicHashEmbeddingProvider(),
            retrieval_mode="dense_memory",
        )
    elif config.backend == "hybrid":
        base = HybridRetriever(documents)
    elif config.backend == "hybrid_rerank":
        base = RerankingRetriever(HybridRetriever(documents), candidate_k=max(10, len(documents), 1))
    elif config.backend == "faiss_dense":
        base = FaissDenseRetriever(
            documents,
            embedding_provider=_embedding_provider(config),
        )
    else:
        raise ValueError(f"Unsupported memory retriever backend: {config.backend}")
    return FilteringMemoryRetriever(
        base_retriever=base,
        chunks=chunks,
        status={
            "available": True,
            "backend": config.backend,
            "fallback_used": False,
            "embedding_provider": config.embedding_provider,
            "embedding_model": config.embedding_model,
        },
    )


class FilteringMemoryRetriever:
    def __init__(
        self,
        *,
        base_retriever: Any,
        chunks: list[MemoryChunk],
        status: dict[str, Any],
    ) -> None:
        self.base_retriever = base_retriever
        self.status = status
        self._chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks if chunk.chunk_id}

    def search(self, query: str, *, session_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        candidate_k = max(top_k, len(self._chunk_by_id))
        candidates = self.base_retriever.search(query, top_k=candidate_k)
        results: list[dict[str, Any]] = []
        for candidate in candidates:
            chunk = self._chunk_by_id.get(candidate.get("chunk_id"))
            if chunk is None or chunk.session_id != session_id:
                continue
            results.append(
                {
                    **candidate,
                    "session_id": chunk.session_id,
                    "turn_id": chunk.turn_id,
                    "metadata": chunk.metadata,
                }
            )
            if len(results) >= top_k:
                break
        return results


class UnavailableMemoryRetriever:
    def __init__(self, backend: str, error: str) -> None:
        self.status = {
            "available": False,
            "backend": backend,
            "fallback_used": False,
            "error": error,
            "hint": 'Install dense extras with: pip install -e ".[dense]" or enable fallback.',
        }

    def search(self, query: str, *, session_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        return []


def _embedding_provider(config: MemoryRetrieverConfig):
    if config.embedding_provider == "deterministic":
        return DeterministicHashEmbeddingProvider()
    if config.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(config.embedding_model)
    raise ValueError(f"Unsupported memory embedding provider: {config.embedding_provider}")


def _to_document_chunk(chunk: MemoryChunk) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=str(chunk.chunk_id),
        text=chunk.text,
        metadata=DocumentMetadata(
            source_id=str(chunk.chunk_id),
            title=f"Conversation turn {chunk.turn_id}",
            source_path=f"memory://{chunk.session_id}/{chunk.turn_id}",
            category="conversation_memory",
        ),
        section=str(chunk.metadata.get("route", "conversation")),
        start_word=0,
        end_word=len(chunk.text.split()),
    )
