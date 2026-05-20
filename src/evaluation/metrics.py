from __future__ import annotations

import re

from src.evaluation.dataset import EvalRecord

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\-\u4e00-\u9fff]+")


def citation_hit_rate(records: list[EvalRecord], predictions: dict[str, dict]) -> float:
    required_records = [record for record in records if record.required_documents]
    if not required_records:
        return 0.0

    hits = 0
    for record in required_records:
        predicted = predictions.get(record.id, {})
        citation_ids = {
            citation.get("source_id")
            for citation in predicted.get("citations", [])
            if isinstance(citation, dict)
        }
        if set(record.required_documents).issubset(citation_ids):
            hits += 1
    return hits / len(required_records)


def context_recall(records: list[EvalRecord], predictions: dict[str, dict]) -> float:
    required_records = [record for record in records if record.required_documents]
    if not required_records:
        return 0.0

    hits = 0
    for record in required_records:
        predicted = predictions.get(record.id, {})
        context_source_ids = {
            context.get("citation", {}).get("source_id")
            for context in predicted.get("retrieved_contexts", [])
            if isinstance(context, dict)
        }
        if set(record.required_documents).issubset(context_source_ids):
            hits += 1
    return hits / len(required_records)


def tool_selection_accuracy(records: list[EvalRecord], predictions: dict[str, dict]) -> float:
    required_records = [record for record in records if record.required_tools]
    if not required_records:
        return 0.0

    hits = 0
    for record in required_records:
        predicted = predictions.get(record.id, {})
        tool_names = set(predicted.get("tools", []))
        if set(record.required_tools).issubset(tool_names):
            hits += 1
    return hits / len(required_records)


def lexical_answer_coverage(records: list[EvalRecord], predictions: dict[str, dict]) -> float:
    if not records:
        return 0.0

    scores = []
    for record in records:
        gold_tokens = set(_tokenize(record.gold_answer))
        if not gold_tokens:
            continue
        predicted_tokens = set(_tokenize(str(predictions.get(record.id, {}).get("answer", ""))))
        scores.append(len(gold_tokens & predicted_tokens) / len(gold_tokens))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def expected_keyword_coverage(records: list[EvalRecord], predictions: dict[str, dict]) -> float:
    keyword_records = [record for record in records if record.expected_keywords]
    if not keyword_records:
        return 0.0

    scores = []
    for record in keyword_records:
        answer = str(predictions.get(record.id, {}).get("answer", "")).lower()
        expected_keywords = [keyword.lower() for keyword in record.expected_keywords]
        matches = [keyword for keyword in expected_keywords if keyword in answer]
        scores.append(len(matches) / len(expected_keywords))
    return sum(scores) / len(scores)


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def tool_execution_success_rate(records: list[EvalRecord], predictions: dict[str, dict]) -> float:
    required_records = [record for record in records if record.required_tools]
    if not required_records:
        return 0.0

    hits = 0
    for record in required_records:
        predicted = predictions.get(record.id, {})
        if predicted.get("tool_results"):
            hits += 1
    return hits / len(required_records)


def evidence_coverage(records: list[EvalRecord], predictions: dict[str, dict]) -> float:
    evidence_required_records = [
        record for record in records if record.required_documents or record.required_tools
    ]
    if not evidence_required_records:
        return 0.0

    hits = 0
    for record in evidence_required_records:
        predicted = predictions.get(record.id, {})
        has_citation = bool(predicted.get("citations"))
        has_tool_result = bool(predicted.get("tool_results"))
        if has_citation or has_tool_result:
            hits += 1
    return hits / len(evidence_required_records)
