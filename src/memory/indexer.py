from __future__ import annotations

import json
from typing import Any

from src.memory.schemas import ConversationTurn, MemoryChunk


class TurnMemoryIndexer:
    """Convert a completed turn into concise retrieval-oriented chunks."""

    def chunks_from_turn(self, turn: ConversationTurn) -> list[MemoryChunk]:
        if not turn.turn_id:
            raise ValueError("turn_id is required before indexing a conversation turn")
        source_ids = _source_ids(turn.citations)
        text = "\n".join(
            [
                f"Question: {turn.question}",
                f"Route: {turn.route}",
                f"Tools: {', '.join(turn.tools) if turn.tools else 'none'}",
                f"Answer summary: {_truncate(turn.answer, 700)}",
                f"Policy result: {_compact_json(turn.policy_result)}",
                f"Citation source ids: {', '.join(source_ids) if source_ids else 'none'}",
                f"Tool result summary: {_compact_json(turn.tool_results)}",
                f"Data boundary: {turn.data_source.get('kind', 'unknown')}",
            ]
        )
        return [
            MemoryChunk(
                session_id=turn.session_id,
                turn_id=turn.turn_id,
                chunk_index=0,
                text=text,
                metadata={
                    "route": turn.route,
                    "tools": list(turn.tools),
                    "citation_source_ids": source_ids,
                    "turn_index": turn.turn_index,
                    "data_source_kind": turn.data_source.get("kind"),
                },
            )
        ]


def _source_ids(citations: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for citation in citations:
        source_id = citation.get("source_id")
        if source_id is not None:
            ids.append(str(source_id))
    return ids


def _compact_json(value: Any, max_chars: int = 700) -> str:
    if not value:
        return "none"
    return _truncate(json.dumps(value, ensure_ascii=False, sort_keys=True), max_chars)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + "... [truncated]"
