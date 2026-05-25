from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConversationSession:
    session_id: str
    title: str
    created_at: str
    updated_at: str
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConversationTurn:
    session_id: str
    question: str
    answer: str
    route: str
    turn_id: str | None = None
    turn_index: int | None = None
    tools: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    retrieved_contexts: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    policy_result: dict[str, Any] = field(default_factory=dict)
    workflow_trace: list[dict[str, Any]] = field(default_factory=list)
    answer_audit: dict[str, Any] = field(default_factory=dict)
    data_source: dict[str, Any] = field(default_factory=dict)
    memory_context: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryChunk:
    session_id: str
    turn_id: str
    chunk_index: int
    text: str
    chunk_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding_status: str = "pending"
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConversationContext:
    session_id: str
    summary: str
    recent_turns: list[dict[str, Any]]
    relevant_memory: list[dict[str, Any]]
    reusable_evidence_refs: list[dict[str, Any]]
    stable_context: dict[str, Any]
    budget: dict[str, Any]
    memory_status: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
