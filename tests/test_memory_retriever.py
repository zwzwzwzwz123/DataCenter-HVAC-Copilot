from src.memory.retriever import MemoryRetrieverConfig, build_memory_retriever
from src.memory.schemas import MemoryChunk


def test_dense_memory_retriever_enforces_session_isolation():
    chunks = [
        MemoryChunk(
            chunk_id="chunk-1",
            session_id="session-a",
            turn_id="turn-1",
            chunk_index=0,
            text="Question: zone temperature. Answer summary: fan power was stable.",
            metadata={"route": "timeseries_query"},
        ),
        MemoryChunk(
            chunk_id="chunk-2",
            session_id="session-b",
            turn_id="turn-2",
            chunk_index=0,
            text="Question: zone temperature. Answer summary: policy should reset setpoint.",
            metadata={"route": "policy_recommendation"},
        ),
    ]
    retriever = build_memory_retriever(MemoryRetrieverConfig(backend="dense_memory"), chunks)

    results = retriever.search("fan power", session_id="session-a", top_k=5)

    assert results
    assert {result["session_id"] for result in results} == {"session-a"}
    assert results[0]["chunk_id"] == "chunk-1"


def test_default_faiss_retriever_reports_unavailable_without_fallback(monkeypatch):
    chunks = [
        MemoryChunk(
            chunk_id="chunk-1",
            session_id="session-a",
            turn_id="turn-1",
            chunk_index=0,
            text="Question: temperature",
        )
    ]

    def raise_import_error(*args, **kwargs):
        raise ImportError("faiss missing")

    monkeypatch.setattr("src.memory.retriever.FaissDenseRetriever", raise_import_error)
    retriever = build_memory_retriever(
        MemoryRetrieverConfig(backend="faiss_dense", allow_fallback=False),
        chunks,
    )

    results = retriever.search("temperature", session_id="session-a")

    assert results == []
    assert retriever.status["available"] is False
    assert retriever.status["backend"] == "faiss_dense"
    assert "faiss missing" in retriever.status["error"]


def test_fallback_only_occurs_when_explicitly_allowed(monkeypatch):
    chunks = [
        MemoryChunk(
            chunk_id="chunk-1",
            session_id="session-a",
            turn_id="turn-1",
            chunk_index=0,
            text="Question: policy setpoint",
        )
    ]

    def raise_import_error(*args, **kwargs):
        raise ImportError("faiss missing")

    monkeypatch.setattr("src.memory.retriever.FaissDenseRetriever", raise_import_error)
    retriever = build_memory_retriever(
        MemoryRetrieverConfig(backend="faiss_dense", allow_fallback=True),
        chunks,
    )

    results = retriever.search("policy", session_id="session-a")

    assert results
    assert retriever.status["available"] is True
    assert retriever.status["fallback_used"] is True
    assert retriever.status["backend"] == "hybrid_rerank"
