from __future__ import annotations

from pathlib import Path
from math import log2
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
        citation_aliases = [
            _source_aliases_from_citation(citation)
            for citation in predicted.get("citations", [])
            if isinstance(citation, dict)
        ]
        if _all_required_sources_found(record.required_documents, citation_aliases):
            hits += 1
    return hits / len(required_records)


def context_recall(records: list[EvalRecord], predictions: dict[str, dict]) -> float:
    required_records = [record for record in records if record.required_documents]
    if not required_records:
        return 0.0

    hits = 0
    for record in required_records:
        predicted = predictions.get(record.id, {})
        context_aliases = [
            _source_aliases_from_context(context)
            for context in predicted.get("retrieved_contexts", [])
            if isinstance(context, dict)
        ]
        if _all_required_sources_found(record.required_documents, context_aliases):
            hits += 1
    return hits / len(required_records)


def retrieval_recall_at_k(
    records: list[EvalRecord],
    predictions: dict[str, dict],
    *,
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    required_records = [record for record in records if record.required_documents]
    if not required_records:
        return 0.0

    scores = []
    for record in required_records:
        required = [_normalize_source_alias(source) for source in record.required_documents]
        retrieved = _ranked_context_source_aliases(predictions.get(record.id, {}), k=k)
        matched = [
            required_source
            for required_source in required
            if any(required_source in aliases for aliases in retrieved)
        ]
        scores.append(len(matched) / len(required))
    return sum(scores) / len(scores)


def retrieval_mrr_at_k(
    records: list[EvalRecord],
    predictions: dict[str, dict],
    *,
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    required_records = [record for record in records if record.required_documents]
    if not required_records:
        return 0.0

    scores = []
    for record in required_records:
        required = [_normalize_source_alias(source) for source in record.required_documents]
        reciprocal_rank = 0.0
        for rank, aliases in enumerate(
            _ranked_context_source_aliases(predictions.get(record.id, {}), k=k),
            start=1,
        ):
            if any(required_source in aliases for required_source in required):
                reciprocal_rank = 1 / rank
                break
        scores.append(reciprocal_rank)
    return sum(scores) / len(scores)


def retrieval_ndcg_at_k(
    records: list[EvalRecord],
    predictions: dict[str, dict],
    *,
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    required_records = [record for record in records if record.required_documents]
    if not required_records:
        return 0.0

    scores = []
    for record in required_records:
        required = [_normalize_source_alias(source) for source in record.required_documents]
        gains = [
            1.0 if any(required_source in aliases for required_source in required) else 0.0
            for aliases in _ranked_context_source_aliases(
                predictions.get(record.id, {}),
                k=k,
            )
        ]
        dcg = _discounted_cumulative_gain(gains)
        ideal_hits = min(len(required), k)
        idcg = _discounted_cumulative_gain([1.0] * ideal_hits)
        scores.append(dcg / idcg if idcg else 0.0)
    return sum(scores) / len(scores)


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


def planned_step_accuracy(records: list[EvalRecord], predictions: dict[str, dict]) -> float | None:
    planned_records = [record for record in records if record.expected_steps]
    if not planned_records:
        return None

    hits = 0
    for record in planned_records:
        if set(record.expected_steps) == set(_planned_routes(predictions.get(record.id, {}))):
            hits += 1
    return hits / len(planned_records)


def planned_step_order_accuracy(records: list[EvalRecord], predictions: dict[str, dict]) -> float | None:
    planned_records = [record for record in records if record.expected_steps]
    if not planned_records:
        return None

    hits = 0
    for record in planned_records:
        if record.expected_steps == _planned_routes(predictions.get(record.id, {})):
            hits += 1
    return hits / len(planned_records)


def required_step_recall(records: list[EvalRecord], predictions: dict[str, dict]) -> float | None:
    planned_records = [record for record in records if record.expected_steps]
    if not planned_records:
        return None

    scores = []
    for record in planned_records:
        expected = set(record.expected_steps)
        planned = set(_planned_routes(predictions.get(record.id, {})))
        scores.append(len(expected & planned) / len(expected))
    return sum(scores) / len(scores)


def policy_final_step_rate(records: list[EvalRecord], predictions: dict[str, dict]) -> float | None:
    policy_records = [
        record
        for record in records
        if record.expected_steps and "policy_recommendation" in record.expected_steps
    ]
    if not policy_records:
        return None

    hits = 0
    for record in policy_records:
        planned = _planned_routes(predictions.get(record.id, {}))
        if planned and planned[-1] == "policy_recommendation":
            hits += 1
    return hits / len(policy_records)


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


def answer_correctness_proxy(records: list[EvalRecord], predictions: dict[str, dict]) -> float:
    annotated_records = [record for record in records if record.must_include]
    if not annotated_records:
        return 0.0

    scores = []
    for record in annotated_records:
        answer = str(predictions.get(record.id, {}).get("answer", "")).lower()
        required = [item.lower() for item in record.must_include]
        matches = [item for item in required if item in answer]
        scores.append(len(matches) / len(required))
    return sum(scores) / len(scores)


def faithfulness_proxy(records: list[EvalRecord], predictions: dict[str, dict]) -> float:
    annotated_records = [
        record for record in records if record.must_include or record.must_not_include
    ]
    if not annotated_records:
        return 0.0

    scores = []
    for record in annotated_records:
        predicted = predictions.get(record.id, {})
        answer = str(predicted.get("answer", "")).lower()
        forbidden = [item.lower() for item in record.must_not_include]
        if any(item in answer for item in forbidden):
            scores.append(0.0)
            continue

        score = 1.0
        if record.must_include:
            required = [item.lower() for item in record.must_include]
            matches = [item for item in required if item in answer]
            score *= len(matches) / len(required)

        needs_evidence = bool(record.required_documents or record.required_tools)
        has_evidence = bool(predicted.get("citations") or predicted.get("tool_results"))
        if needs_evidence and not has_evidence:
            score = min(score, 0.5)
        scores.append(score)
    return sum(scores) / len(scores)


def hallucination_proxy_rate(records: list[EvalRecord], predictions: dict[str, dict]) -> float:
    boundary_records = [record for record in records if record.must_not_include]
    if not boundary_records:
        return 0.0

    violations = 0
    for record in boundary_records:
        answer = str(predictions.get(record.id, {}).get("answer", "")).lower()
        forbidden = [item.lower() for item in record.must_not_include]
        if any(item in answer for item in forbidden):
            violations += 1
    return violations / len(boundary_records)


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _ranked_context_source_aliases(prediction: dict, *, k: int) -> list[set[str]]:
    source_aliases = []
    seen = set()
    for context in prediction.get("retrieved_contexts", []):
        if not isinstance(context, dict):
            continue
        aliases = _source_aliases_from_context(context)
        if not aliases:
            continue
        identity_key = next(iter(sorted(aliases)))
        if identity_key not in seen:
            source_aliases.append(aliases)
            seen.add(identity_key)
        if len(source_aliases) >= k:
            break
    return source_aliases


def _all_required_sources_found(
    required_documents: list[str],
    candidate_aliases: list[set[str]],
) -> bool:
    return all(
        any(_normalize_source_alias(required) in aliases for aliases in candidate_aliases)
        for required in required_documents
    )


def _source_aliases_from_context(context: dict) -> set[str]:
    aliases = set()
    citation = context.get("citation", {})
    if isinstance(citation, dict):
        aliases.update(_source_aliases_from_citation(citation))
    aliases.update(_source_aliases_from_mapping(context))
    metadata = context.get("metadata", {})
    if isinstance(metadata, dict):
        aliases.update(_source_aliases_from_mapping(metadata))
    return aliases


def _source_aliases_from_citation(citation: dict) -> set[str]:
    return _source_aliases_from_mapping(citation)


def _source_aliases_from_mapping(mapping: dict) -> set[str]:
    aliases: set[str] = set()
    for key in (
        "source_id",
        "document_id",
        "chunk_id",
        "file_hash",
        "filename",
        "title",
        "source_path",
        "source_url",
        "url",
    ):
        value = mapping.get(key)
        if value is None:
            continue
        aliases.update(_source_alias_variants(str(value)))
    return {alias for alias in aliases if alias}


def _source_alias_variants(value: str) -> set[str]:
    normalized = _normalize_source_alias(value)
    if not normalized:
        return set()
    aliases = {normalized}
    path_name = Path(value.replace("\\", "/")).name
    filename = _normalize_source_alias(path_name)
    if filename:
        aliases.add(filename)
        aliases.add(_strip_document_id_prefix(filename))
        aliases.add(_strip_suffix(filename))
    aliases.add(_strip_suffix(normalized))
    aliases.add(_strip_document_id_prefix(normalized))
    return {alias for alias in aliases if alias}


def _normalize_source_alias(value: str) -> str:
    return value.strip().replace("\\", "/").lower()


def _strip_suffix(value: str) -> str:
    path = Path(value)
    suffix = path.suffix.lower()
    if suffix:
        return value[: -len(suffix)]
    return value


def _strip_document_id_prefix(value: str) -> str:
    return re.sub(r"^doc_[0-9a-f]{32}_", "", value)


def _discounted_cumulative_gain(gains: list[float]) -> float:
    return sum(gain / log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def _planned_routes(prediction: dict) -> list[str]:
    steps = prediction.get("planned_steps", [])
    routes = []
    for step in steps:
        if isinstance(step, dict):
            route = step.get("route")
        else:
            route = step
        if route:
            routes.append(str(route))
    return routes


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


def grounding_rate(records: list[EvalRecord], predictions: dict[str, dict]) -> float:
    grounded_records = [record for record in records if record.required_documents]
    if not grounded_records:
        return 0.0

    hits = 0
    for record in grounded_records:
        predicted = predictions.get(record.id, {})
        retrieved_contexts = [
            context
            for context in predicted.get("retrieved_contexts", [])
            if isinstance(context, dict)
        ]
        citation_ids = {
            context.get("citation", {}).get("source_id")
            for context in retrieved_contexts
            if isinstance(context.get("citation"), dict)
        }
        answer = str(predicted.get("answer", ""))
        answer_citation_ids = _extract_answer_citation_ids(answer)
        if answer_citation_ids and answer_citation_ids.issubset(citation_ids):
            hits += 1
    return hits / len(grounded_records)


def _extract_answer_citation_ids(answer: str) -> set[str]:
    citation_ids: set[str] = set()
    in_citation_section = False
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped in {"引用：", "Citations:"}:
            in_citation_section = True
            continue
        if in_citation_section and stripped.startswith("- ") and ":" in stripped:
            head, _ = stripped[2:].split(":", 1)
            citation_ids.add(head.strip())
    return citation_ids
