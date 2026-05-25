from pathlib import Path

from src.memory.context_manager import ContextManager
from src.memory.retriever import MemoryRetrieverConfig
from src.memory.schemas import ConversationTurn
from src.memory.storage import ConversationMemoryStore


def test_context_manager_builds_context_with_recent_and_retrieved_memory(tmp_path: Path):
    store = ConversationMemoryStore(tmp_path / "conversations.db")
    manager = ContextManager(
        store=store,
        retriever_config=MemoryRetrieverConfig(backend="dense_memory"),
    )
    session = manager.create_session(title="Investigation")
    manager.save_turn(
        ConversationTurn(
            session_id=session.session_id,
            question="What was fan power?",
            answer="Fan power stayed near 20 kW.",
            route="timeseries_query",
            tools=["query_metric"],
        )
    )

    context = manager.load_context(session.session_id, "What did we see last time?")

    assert context.session_id == session.session_id
    assert context.recent_turns[0]["question"] == "What was fan power?"
    assert context.relevant_memory
    assert context.stable_context["version"]
    assert context.budget["max_chars"] == 6000
    assert context.memory_status["storage"]["available"] is True
    assert context.memory_status["retrieval"]["available"] is True


def test_context_manager_budget_marks_truncation(tmp_path: Path):
    store = ConversationMemoryStore(tmp_path / "conversations.db")
    manager = ContextManager(
        store=store,
        retriever_config=MemoryRetrieverConfig(backend="dense_memory"),
        max_context_chars=900,
    )
    session = manager.create_session()
    manager.save_turn(
        ConversationTurn(
            session_id=session.session_id,
            question="Explain the long result",
            answer="A" * 3000,
            route="document_qa",
        )
    )

    context = manager.load_context(session.session_id, "Summarize prior result")

    assert context.budget["truncated"] is True
    assert len(str(context.to_dict())) < 2500


def test_context_manager_preserves_storage_when_retrieval_unavailable(tmp_path: Path, monkeypatch):
    store = ConversationMemoryStore(tmp_path / "conversations.db")

    def raise_import_error(*args, **kwargs):
        raise ImportError("faiss missing")

    monkeypatch.setattr("src.memory.retriever.FaissDenseRetriever", raise_import_error)
    manager = ContextManager(
        store=store,
        retriever_config=MemoryRetrieverConfig(backend="faiss_dense", allow_fallback=False),
    )
    session = manager.create_session()
    saved = manager.save_turn(
        ConversationTurn(
            session_id=session.session_id,
            question="q",
            answer="a",
            route="document_qa",
        )
    )
    context = manager.load_context(session.session_id, "q")

    assert saved.turn_id
    assert context.memory_status["storage"]["available"] is True
    assert context.memory_status["retrieval"]["available"] is False
    assert context.relevant_memory == []
