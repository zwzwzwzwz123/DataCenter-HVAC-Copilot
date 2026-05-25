from src.memory.indexer import TurnMemoryIndexer
from src.memory.schemas import ConversationTurn


def test_indexer_converts_turn_to_retrieval_friendly_chunk():
    turn = ConversationTurn(
        session_id="session-1",
        turn_id="turn-1",
        turn_index=1,
        question="What did zone_a temperature do?",
        answer="Zone temperature peaked at 30 C based on query_metric.",
        route="timeseries_query",
        tools=["query_metric"],
        citations=[{"source_id": "thermal_note", "title": "Thermal Note"}],
        tool_results=[{"metric_name": "zone_temperature", "summary": {"max": 30.0}}],
        policy_result={"policy_name": "rule_based"},
        data_source={"kind": "processed_csv", "path": "data/bear_processed/bear_rollout.csv"},
    )

    chunks = TurnMemoryIndexer().chunks_from_turn(turn)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.session_id == "session-1"
    assert chunk.turn_id == "turn-1"
    assert "Question: What did zone_a temperature do?" in chunk.text
    assert "Route: timeseries_query" in chunk.text
    assert "Tools: query_metric" in chunk.text
    assert "Citation source ids: thermal_note" in chunk.text
    assert "Data boundary: processed_csv" in chunk.text
    assert chunk.metadata["route"] == "timeseries_query"
    assert chunk.metadata["tools"] == ["query_metric"]
