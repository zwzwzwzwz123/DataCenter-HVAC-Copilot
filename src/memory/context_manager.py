from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.memory.budget import ContextBudgetManager
from src.memory.indexer import TurnMemoryIndexer
from src.memory.retriever import MemoryRetrieverConfig, build_memory_retriever
from src.memory.schemas import ConversationContext, ConversationSession, ConversationTurn
from src.memory.stable_context import get_stable_context
from src.memory.storage import ConversationMemoryStore


class ContextManager:
    """High-level memory boundary used by API and agent layers."""

    def __init__(
        self,
        *,
        store: ConversationMemoryStore | None = None,
        retriever_config: MemoryRetrieverConfig | None = None,
        indexer: TurnMemoryIndexer | None = None,
        max_context_chars: int = 6000,
    ) -> None:
        self.store = store or ConversationMemoryStore(default_memory_db_path())
        self.retriever_config = retriever_config or memory_retriever_config_from_env()
        self.indexer = indexer or TurnMemoryIndexer()
        self.budget_manager = ContextBudgetManager(max_chars=max_context_chars)

    def create_session(self, *, title: str = "New HVAC analysis session") -> ConversationSession:
        return self.store.create_session(title=title)

    def load_context(self, session_id: str, question: str) -> ConversationContext:
        session = self.store.require_session(session_id)
        recent_turns = [_turn_context(turn) for turn in self.store.load_recent_turns(session_id, limit=3)]
        chunks = self.store.load_chunks(session_id)
        retriever = build_memory_retriever(self.retriever_config, chunks)
        relevant_memory = retriever.search(question, session_id=session_id, top_k=5)
        budgeted = self.budget_manager.apply(
            summary=session.summary,
            recent_turns=recent_turns,
            relevant_memory=relevant_memory,
            reusable_evidence_refs=_reusable_evidence_refs(recent_turns),
            stable_context=get_stable_context(),
        )
        return ConversationContext(
            session_id=session_id,
            summary=budgeted["summary"],
            recent_turns=budgeted["recent_turns"],
            relevant_memory=budgeted["relevant_memory"],
            reusable_evidence_refs=budgeted["reusable_evidence_refs"],
            stable_context=budgeted["stable_context"],
            budget=budgeted["budget"],
            memory_status={
                "storage": {"available": True, "db_path": str(self.store.db_path)},
                "retrieval": retriever.status,
            },
        )

    def save_turn(self, turn: ConversationTurn) -> ConversationTurn:
        saved = self.store.save_turn(turn)
        chunks = self.indexer.chunks_from_turn(saved)
        self.store.save_chunks(chunks)
        return saved


def default_memory_db_path() -> Path:
    return Path(os.getenv("HVAC_COPILOT_MEMORY_DB_PATH", "data/memory/conversations.db"))


def memory_enabled_from_env() -> bool:
    return os.getenv("HVAC_COPILOT_MEMORY_ENABLED", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def memory_retriever_config_from_env() -> MemoryRetrieverConfig:
    return MemoryRetrieverConfig(
        backend=os.getenv("HVAC_COPILOT_MEMORY_RETRIEVER", "faiss_dense"),
        allow_fallback=os.getenv("HVAC_COPILOT_MEMORY_ALLOW_FALLBACK", "false").lower()
        in {"1", "true", "yes", "on"},
        embedding_provider=os.getenv(
            "HVAC_COPILOT_MEMORY_EMBEDDING_PROVIDER",
            "sentence-transformers",
        ),
        embedding_model=os.getenv(
            "HVAC_COPILOT_MEMORY_EMBEDDING_MODEL",
            "BAAI/bge-small-zh-v1.5",
        ),
    )


def _turn_context(turn: ConversationTurn) -> dict[str, Any]:
    return {
        "turn_id": turn.turn_id,
        "turn_index": turn.turn_index,
        "question": turn.question,
        "answer": turn.answer,
        "route": turn.route,
        "tools": turn.tools,
        "citation_source_ids": [
            str(citation.get("source_id"))
            for citation in turn.citations
            if citation.get("source_id") is not None
        ],
        "policy_result": turn.policy_result,
    }


def _reusable_evidence_refs(recent_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for turn in recent_turns:
        if turn.get("citation_source_ids") or turn.get("tools"):
            refs.append(
                {
                    "turn_id": turn.get("turn_id"),
                    "route": turn.get("route"),
                    "tools": turn.get("tools", []),
                    "citation_source_ids": turn.get("citation_source_ids", []),
                }
            )
    return refs
