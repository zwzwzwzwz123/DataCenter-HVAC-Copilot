from __future__ import annotations

from typing import Any


class ContextBudgetManager:
    """Deterministic character-budget policy for conversation context."""

    def __init__(
        self,
        *,
        max_chars: int = 6000,
        summary_chars: int = 1200,
        recent_turn_chars: int = 700,
        memory_chunk_chars: int = 700,
        reusable_evidence_chars: int = 1000,
        recent_turn_limit: int = 3,
        memory_chunk_limit: int = 5,
    ) -> None:
        self.max_chars = max_chars
        self.summary_chars = summary_chars
        self.recent_turn_chars = recent_turn_chars
        self.memory_chunk_chars = memory_chunk_chars
        self.reusable_evidence_chars = reusable_evidence_chars
        self.recent_turn_limit = recent_turn_limit
        self.memory_chunk_limit = memory_chunk_limit

    def apply(
        self,
        *,
        summary: str,
        recent_turns: list[dict[str, Any]],
        relevant_memory: list[dict[str, Any]],
        reusable_evidence_refs: list[dict[str, Any]],
        stable_context: dict[str, Any],
    ) -> dict[str, Any]:
        truncated = False
        summary, was_truncated = _truncate_text(summary, self.summary_chars)
        truncated = truncated or was_truncated

        budgeted_recent = []
        for turn in recent_turns[-self.recent_turn_limit :]:
            item, was_truncated = _truncate_mapping(turn, self.recent_turn_chars)
            truncated = truncated or was_truncated
            budgeted_recent.append(item)

        sorted_memory = sorted(
            relevant_memory,
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )[: self.memory_chunk_limit]
        budgeted_memory = []
        for memory in sorted_memory:
            item, was_truncated = _truncate_mapping(memory, self.memory_chunk_chars)
            truncated = truncated or was_truncated
            budgeted_memory.append(item)

        evidence_refs, was_truncated = _truncate_list_of_mappings(
            reusable_evidence_refs,
            self.reusable_evidence_chars,
        )
        truncated = truncated or was_truncated

        payload = {
            "summary": summary,
            "recent_turns": budgeted_recent,
            "relevant_memory": budgeted_memory,
            "reusable_evidence_refs": evidence_refs,
            "stable_context": stable_context,
        }
        while _char_count(payload) > self.max_chars and budgeted_memory:
            budgeted_memory.pop()
            truncated = True
            payload["relevant_memory"] = budgeted_memory
        while _char_count(payload) > self.max_chars and budgeted_recent:
            budgeted_recent.pop(0)
            truncated = True
            payload["recent_turns"] = budgeted_recent

        return {
            **payload,
            "budget": {
                "max_chars": self.max_chars,
                "used_chars": _char_count(payload),
                "truncated": truncated or _char_count(payload) > self.max_chars,
            },
        }


def _truncate_mapping(item: dict[str, Any], max_chars: int) -> tuple[dict[str, Any], bool]:
    copy = dict(item)
    text = str(copy)
    if len(text) <= max_chars:
        return copy, False
    for key in ("answer", "text"):
        if key in copy:
            copy[key], _ = _truncate_text(str(copy[key]), max(80, max_chars // 2))
    if len(str(copy)) <= max_chars:
        return copy, True
    return {"truncated": True, "text": str(copy)[: max_chars - 15] + "... [truncated]"}, True


def _truncate_list_of_mappings(items: list[dict[str, Any]], max_chars: int) -> tuple[list[dict[str, Any]], bool]:
    kept: list[dict[str, Any]] = []
    truncated = False
    for item in items:
        candidate = [*kept, item]
        if _char_count(candidate) > max_chars:
            truncated = True
            break
        kept.append(item)
    return kept, truncated


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[: max_chars - 15].rstrip() + "... [truncated]", True


def _char_count(value: Any) -> int:
    return len(str(value))
