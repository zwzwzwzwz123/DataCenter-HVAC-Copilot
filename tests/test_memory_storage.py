from pathlib import Path
import sqlite3

import pytest

from src.memory.schemas import ConversationTurn
from src.memory.storage import ConversationMemoryStore, UnknownSessionError


def test_storage_initializes_schema_and_creates_session(tmp_path: Path):
    store = ConversationMemoryStore(tmp_path / "conversations.db")

    session = store.create_session(title="Cooling investigation")

    assert session.session_id
    assert session.title == "Cooling investigation"
    assert session.summary == ""
    assert (tmp_path / "conversations.db").exists()


def test_storage_saves_turns_with_monotonic_turn_index(tmp_path: Path):
    store = ConversationMemoryStore(tmp_path / "conversations.db")
    session = store.create_session()

    first = store.save_turn(
        ConversationTurn(
            session_id=session.session_id,
            question="What was zone_a temperature?",
            answer="24 C",
            route="timeseries_query",
            tools=["query_metric"],
            tool_results=[{"metric_name": "zone_temperature"}],
            data_source={"kind": "mock"},
        )
    )
    second = store.save_turn(
        ConversationTurn(
            session_id=session.session_id,
            question="And the policy?",
            answer="Use bounded policy output.",
            route="policy_recommendation",
            tools=["rule_based_policy"],
            policy_result={"policy_name": "rule_based"},
        )
    )

    assert first.turn_index == 1
    assert second.turn_index == 2
    assert first.turn_id != second.turn_id


def test_storage_enforces_unique_turn_index_per_session(tmp_path: Path):
    store = ConversationMemoryStore(tmp_path / "conversations.db")
    session = store.create_session()
    store.save_turn(
        ConversationTurn(
            session_id=session.session_id,
            question="q1",
            answer="a1",
            route="document_qa",
            turn_index=1,
        )
    )

    with pytest.raises(sqlite3.IntegrityError):
        store.save_turn(
            ConversationTurn(
                session_id=session.session_id,
                question="q2",
                answer="a2",
                route="document_qa",
                turn_index=1,
            )
        )


def test_storage_adds_unique_turn_index_index_to_existing_schema(tmp_path: Path):
    db_path = tmp_path / "conversations.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE conversation_sessions (
              session_id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              summary TEXT NOT NULL DEFAULT '',
              metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE conversation_turns (
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
              created_at TEXT NOT NULL
            );
            CREATE TABLE memory_chunks (
              chunk_id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL,
              turn_id TEXT NOT NULL,
              chunk_index INTEGER NOT NULL,
              text TEXT NOT NULL,
              metadata_json TEXT NOT NULL DEFAULT '{}',
              embedding_status TEXT NOT NULL DEFAULT 'pending',
              created_at TEXT NOT NULL
            );
            CREATE TABLE memory_index_metadata (
              index_id TEXT PRIMARY KEY,
              session_id TEXT,
              backend TEXT NOT NULL,
              embedding_provider TEXT NOT NULL,
              embedding_model TEXT NOT NULL,
              index_path TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            """
        )

    ConversationMemoryStore(db_path)

    with sqlite3.connect(db_path) as conn:
        indexes = conn.execute("PRAGMA index_list(conversation_turns)").fetchall()
    assert any(index[1] == "idx_conversation_turns_session_turn_index" and index[2] for index in indexes)


def test_storage_duplicate_turn_index_migration_error_has_repair_hint(tmp_path: Path):
    db_path = tmp_path / "conversations.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE conversation_sessions (
              session_id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              summary TEXT NOT NULL DEFAULT '',
              metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE conversation_turns (
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
              created_at TEXT NOT NULL
            );
            CREATE TABLE memory_chunks (
              chunk_id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL,
              turn_id TEXT NOT NULL,
              chunk_index INTEGER NOT NULL,
              text TEXT NOT NULL,
              metadata_json TEXT NOT NULL DEFAULT '{}',
              embedding_status TEXT NOT NULL DEFAULT 'pending',
              created_at TEXT NOT NULL
            );
            CREATE TABLE memory_index_metadata (
              index_id TEXT PRIMARY KEY,
              session_id TEXT,
              backend TEXT NOT NULL,
              embedding_provider TEXT NOT NULL,
              embedding_model TEXT NOT NULL,
              index_path TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            INSERT INTO conversation_turns (
              turn_id, session_id, turn_index, question, answer, route, created_at
            ) VALUES
              ('turn_1', 'session_1', 1, 'q1', 'a1', 'document_qa', '2026-01-01T00:00:00Z'),
              ('turn_2', 'session_1', 1, 'q2', 'a2', 'document_qa', '2026-01-01T00:00:01Z');
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="duplicate conversation turn_index"):
        ConversationMemoryStore(db_path)


def test_storage_loads_recent_turns_in_chronological_order(tmp_path: Path):
    store = ConversationMemoryStore(tmp_path / "conversations.db")
    session = store.create_session()
    for index in range(4):
        store.save_turn(
            ConversationTurn(
                session_id=session.session_id,
                question=f"question {index}",
                answer=f"answer {index}",
                route="document_qa",
            )
        )

    turns = store.load_recent_turns(session.session_id, limit=3)

    assert [turn.question for turn in turns] == ["question 1", "question 2", "question 3"]


def test_storage_updates_turn_workflow_trace(tmp_path: Path):
    store = ConversationMemoryStore(tmp_path / "conversations.db")
    session = store.create_session()
    turn = store.save_turn(
        ConversationTurn(
            session_id=session.session_id,
            question="q",
            answer="a",
            route="document_qa",
            workflow_trace=[{"node": "answer_generator"}],
        )
    )

    store.update_turn_workflow_trace(
        turn.turn_id,
        [
            {"node": "answer_generator"},
            {"node": "memory_turn_saved", "turn_id": turn.turn_id},
        ],
    )

    saved = store.load_recent_turns(session.session_id, limit=1)[0]
    assert saved.workflow_trace[-1] == {
        "node": "memory_turn_saved",
        "turn_id": turn.turn_id,
    }


def test_storage_update_turn_workflow_trace_rejects_unknown_turn_id(tmp_path: Path):
    store = ConversationMemoryStore(tmp_path / "conversations.db")

    with pytest.raises(KeyError, match="missing_turn"):
        store.update_turn_workflow_trace("missing_turn", [{"node": "memory_turn_saved"}])


def test_storage_rejects_turn_for_unknown_session(tmp_path: Path):
    store = ConversationMemoryStore(tmp_path / "conversations.db")

    with pytest.raises(UnknownSessionError):
        store.save_turn(
            ConversationTurn(
                session_id="missing",
                question="q",
                answer="a",
                route="document_qa",
            )
        )


def test_storage_saves_and_loads_memory_chunks_by_session(tmp_path: Path):
    store = ConversationMemoryStore(tmp_path / "conversations.db")
    first_session = store.create_session()
    second_session = store.create_session()
    first_turn = store.save_turn(
        ConversationTurn(
            session_id=first_session.session_id,
            question="zone temperature",
            answer="24 C",
            route="timeseries_query",
        )
    )
    second_turn = store.save_turn(
        ConversationTurn(
            session_id=second_session.session_id,
            question="policy action",
            answer="bounded policy",
            route="policy_recommendation",
        )
    )

    store.save_chunks(
        [
            {
                "session_id": first_session.session_id,
                "turn_id": first_turn.turn_id,
                "chunk_index": 0,
                "text": "Question: zone temperature",
                "metadata": {"route": "timeseries_query"},
            },
            {
                "session_id": second_session.session_id,
                "turn_id": second_turn.turn_id,
                "chunk_index": 0,
                "text": "Question: policy action",
                "metadata": {"route": "policy_recommendation"},
            },
        ]
    )

    first_chunks = store.load_chunks(first_session.session_id)

    assert len(first_chunks) == 1
    assert first_chunks[0].session_id == first_session.session_id
    assert first_chunks[0].text == "Question: zone temperature"
