from __future__ import annotations

import math

import pytest

from src.retrieval.dense import DenseRetriever
from src.retrieval.embeddings import DeterministicHashEmbeddingProvider
from src.retrieval.faiss_retriever import FaissDenseRetriever
from src.retrieval.schemas import DocumentMetadata


def test_deterministic_hash_embeddings_are_stable_and_unit_norm() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=16)

    first = provider.embed_texts(["cooling airflow", "rack alarm"])
    second = provider.embed_texts(["cooling airflow", "rack alarm"])

    assert first == second
    assert len(first) == 2
    assert len(first[0]) == 16
    assert math.isclose(sum(value * value for value in first[0]), 1.0, rel_tol=1e-6)


def test_deterministic_hash_embedding_rejects_invalid_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        DeterministicHashEmbeddingProvider(dimension=0)


def test_dense_retriever_returns_ranked_chunks_with_citations() -> None:
    metadata = DocumentMetadata(
        source_id="dense_doc",
        title="Dense Doc",
        source_path="memory",
        published_at="2026",
        category="note",
    )
    chunks = [
        metadata.to_chunk(
            chunk_id="dense_doc::cooling",
            text="cooling airflow containment reduces hot spots",
            section="Cooling",
            start_word=0,
            end_word=6,
        ),
        metadata.to_chunk(
            chunk_id="dense_doc::battery",
            text="battery maintenance schedule and electrical inspection",
            section="Maintenance",
            start_word=7,
            end_word=12,
        ),
    ]

    retriever = DenseRetriever(
        chunks,
        embedding_provider=DeterministicHashEmbeddingProvider(dimension=64),
    )
    results = retriever.search("cooling airflow hot spots", top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "dense_doc::cooling"
    assert results[0]["citation"]["title"] == "Dense Doc"
    assert results[0]["retrieval_mode"] == "dense_hash"
    assert results[0]["score"] > 0


def test_dense_retriever_returns_nearest_chunks_even_when_scores_are_non_positive() -> None:
    metadata = DocumentMetadata(
        source_id="tiny_dense_doc",
        title="Tiny Dense Doc",
        source_path="memory",
    )
    chunks = [
        metadata.to_chunk(
            chunk_id="tiny_dense_doc::chunk_0000",
            text="cooling procedure",
            section="Ops",
            start_word=0,
            end_word=2,
        )
    ]

    retriever = DenseRetriever(
        chunks,
        embedding_provider=DeterministicHashEmbeddingProvider(dimension=1),
    )
    results = retriever.search("unrelated query", top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "tiny_dense_doc::chunk_0000"


def test_faiss_dense_retriever_reports_missing_optional_dependency() -> None:
    metadata = DocumentMetadata(
        source_id="faiss_doc",
        title="FAISS Doc",
        source_path="memory",
    )
    chunks = [
        metadata.to_chunk(
            chunk_id="faiss_doc::chunk_0000",
            text="cooling airflow",
            section=None,
            start_word=0,
            end_word=2,
        )
    ]

    try:
        import faiss  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"\.\[dev,dense\]"):
            FaissDenseRetriever(
                chunks,
                embedding_provider=DeterministicHashEmbeddingProvider(dimension=8),
            )
    else:
        retriever = FaissDenseRetriever(
            chunks,
            embedding_provider=DeterministicHashEmbeddingProvider(dimension=8),
        )
        results = retriever.search("cooling", top_k=1)
        assert results[0]["retrieval_mode"] == "dense_faiss"
