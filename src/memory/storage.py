from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from src.memory.schemas import ConversationSession, ConversationTurn, MemoryChunk


class UnknownSessionError(KeyError):
    """Raised when a caller references a session that does not exist."""


class ConversationMemoryStore:
    """SQLite source of truth for conversation sessions, turns, and chunks."""

    def __init__(self, db_path: str | Path = "data/memory/conversations.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def create_session(
        self,
        *,
        title: str = "New HVAC analysis session",
        summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ConversationSession:
        now = _utc_now()
        session_id = f"session_{uuid.uuid4().hex}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_sessions (
                  session_id, title, created_at, updated_at, summary, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, title, now, now, summary, _json_dumps(metadata or {})),
            )
        return ConversationSession(
            session_id=session_id,
            title=title,
            created_at=now,
            updated_at=now,
            summary=summary,
            metadata=metadata or {},
        )

    def get_session(self, session_id: str) -> ConversationSession | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversation_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return _session_from_row(row) if row else None

    def require_session(self, session_id: str) -> ConversationSession:
        session = self.get_session(session_id)
        if session is None:
            raise UnknownSessionError(session_id)
        return session

    def save_turn(self, turn: ConversationTurn) -> ConversationTurn:
        self.require_session(turn.session_id)
        now = turn.created_at or _utc_now()
        turn_id = turn.turn_id or f"turn_{uuid.uuid4().hex}"
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT COALESCE(MAX(turn_index), 0) FROM conversation_turns WHERE session_id = ?",
                (turn.session_id,),
            ).fetchone()[0]
            turn_index = turn.turn_index if turn.turn_index is not None else int(current) + 1
            conn.execute(
                """
                INSERT INTO conversation_turns (
                  turn_id, session_id, turn_index, question, answer, route,
                  tools_json, citations_json, retrieved_contexts_json, tool_results_json,
                  policy_result_json, workflow_trace_json, answer_audit_json,
                  data_source_json, memory_context_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    turn.session_id,
                    turn_index,
                    turn.question,
                    turn.answer,
                    turn.route,
                    _json_dumps(turn.tools),
                    _json_dumps(turn.citations),
                    _json_dumps(turn.retrieved_contexts),
                    _json_dumps(turn.tool_results),
                    _json_dumps(turn.policy_result),
                    _json_dumps(turn.workflow_trace),
                    _json_dumps(turn.answer_audit),
                    _json_dumps(turn.data_source),
                    _json_dumps(turn.memory_context),
                    now,
                ),
            )
            conn.execute(
                "UPDATE conversation_sessions SET updated_at = ? WHERE session_id = ?",
                (now, turn.session_id),
            )
        return ConversationTurn(
            **{
                **turn.to_dict(),
                "turn_id": turn_id,
                "turn_index": turn_index,
                "created_at": now,
            }
        )

    def load_recent_turns(self, session_id: str, limit: int = 3) -> list[ConversationTurn]:
        self.require_session(session_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM conversation_turns
                WHERE session_id = ?
                ORDER BY turn_index DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [_turn_from_row(row) for row in reversed(rows)]

    def save_chunks(self, chunks: list[MemoryChunk | dict[str, Any]]) -> list[MemoryChunk]:
        saved: list[MemoryChunk] = []
        with self._connect() as conn:
            for raw_chunk in chunks:
                chunk = _coerce_chunk(raw_chunk)
                self.require_session(chunk.session_id)
                created_at = chunk.created_at or _utc_now()
                chunk_id = chunk.chunk_id or f"{chunk.turn_id}::memory_{chunk.chunk_index:04d}"
                conn.execute(
                    """
                    INSERT OR REPLACE INTO memory_chunks (
                      chunk_id, session_id, turn_id, chunk_index, text,
                      metadata_json, embedding_status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        chunk.session_id,
                        chunk.turn_id,
                        chunk.chunk_index,
                        chunk.text,
                        _json_dumps(chunk.metadata),
                        chunk.embedding_status,
                        created_at,
                    ),
                )
                saved.append(
                    MemoryChunk(
                        **{
                            **chunk.to_dict(),
                            "chunk_id": chunk_id,
                            "created_at": created_at,
                        }
                    )
                )
        return saved

    def load_chunks(self, session_id: str) -> list[MemoryChunk]:
        self.require_session(session_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memory_chunks
                WHERE session_id = ?
                ORDER BY created_at, chunk_index
                """,
                (session_id,),
            ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def _initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                  session_id TEXT PRIMARY KEY,
                  title TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  summary TEXT NOT NULL DEFAULT '',
                  metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS conversation_turns (
                  turn_id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  turn_index INTEGER NOT NULL,
                  question TEXT NOT NULL,
                  answer TEXT NOT NULL,
                  route TEXT NOT NULL,
                  tools_json TEXT NOT NULL DEFAULT '[]',
                  citations_json TEXT NOT NULL DEFAULT '[]',
                  retrieved_contexts_json TEXT NOT NULL DEFAULT '[]',
                  tool_results_json TEXT NOT NULL DEFAULT '[]',
                  policy_result_json TEXT NOT NULL DEFAULT '{}',
                  workflow_trace_json TEXT NOT NULL DEFAULT '[]',
                  answer_audit_json TEXT NOT NULL DEFAULT '{}',
                  data_source_json TEXT NOT NULL DEFAULT '{}',
                  memory_context_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL,
                  UNIQUE(session_id, turn_index),
                  FOREIGN KEY(session_id) REFERENCES conversation_sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS memory_chunks (
                  chunk_id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  turn_id TEXT NOT NULL,
                  chunk_index INTEGER NOT NULL,
                  text TEXT NOT NULL,
                  metadata_json TEXT NOT NULL DEFAULT '{}',
                  embedding_status TEXT NOT NULL DEFAULT 'pending',
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(session_id) REFERENCES conversation_sessions(session_id),
                  FOREIGN KEY(turn_id) REFERENCES conversation_turns(turn_id)
                );

                CREATE TABLE IF NOT EXISTS memory_index_metadata (
                  index_id TEXT PRIMARY KEY,
                  session_id TEXT,
                  backend TEXT NOT NULL,
                  embedding_provider TEXT NOT NULL,
                  embedding_model TEXT NOT NULL,
                  index_path TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_conversation_turns_session_turn_index
                ON conversation_turns(session_id, turn_index);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


def _session_from_row(row: sqlite3.Row) -> ConversationSession:
    return ConversationSession(
        session_id=row["session_id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        summary=row["summary"],
        metadata=_json_loads(row["metadata_json"], {}),
    )


def _turn_from_row(row: sqlite3.Row) -> ConversationTurn:
    return ConversationTurn(
        turn_id=row["turn_id"],
        session_id=row["session_id"],
        turn_index=int(row["turn_index"]),
        question=row["question"],
        answer=row["answer"],
        route=row["route"],
        tools=_json_loads(row["tools_json"], []),
        citations=_json_loads(row["citations_json"], []),
        retrieved_contexts=_json_loads(row["retrieved_contexts_json"], []),
        tool_results=_json_loads(row["tool_results_json"], []),
        policy_result=_json_loads(row["policy_result_json"], {}),
        workflow_trace=_json_loads(row["workflow_trace_json"], []),
        answer_audit=_json_loads(row["answer_audit_json"], {}),
        data_source=_json_loads(row["data_source_json"], {}),
        memory_context=_json_loads(row["memory_context_json"], {}),
        created_at=row["created_at"],
    )


def _chunk_from_row(row: sqlite3.Row) -> MemoryChunk:
    return MemoryChunk(
        chunk_id=row["chunk_id"],
        session_id=row["session_id"],
        turn_id=row["turn_id"],
        chunk_index=int(row["chunk_index"]),
        text=row["text"],
        metadata=_json_loads(row["metadata_json"], {}),
        embedding_status=row["embedding_status"],
        created_at=row["created_at"],
    )


def _coerce_chunk(raw_chunk: MemoryChunk | dict[str, Any]) -> MemoryChunk:
    if isinstance(raw_chunk, MemoryChunk):
        return raw_chunk
    return MemoryChunk(
        session_id=str(raw_chunk["session_id"]),
        turn_id=str(raw_chunk["turn_id"]),
        chunk_index=int(raw_chunk["chunk_index"]),
        text=str(raw_chunk["text"]),
        chunk_id=raw_chunk.get("chunk_id"),
        metadata=dict(raw_chunk.get("metadata", {})),
        embedding_status=str(raw_chunk.get("embedding_status", "pending")),
        created_at=raw_chunk.get("created_at"),
    )


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
