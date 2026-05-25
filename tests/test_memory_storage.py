from pathlib import Path

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
