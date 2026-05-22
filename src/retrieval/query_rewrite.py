from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.retrieval.rag import RAGAnswer


@dataclass(frozen=True)
class QueryRewriteResult:
    original_query: str
    rewritten_query: str
    added_terms: list[str]
    strategy: str = "rule_based_hvac_rewrite"


@dataclass(frozen=True)
class HyDEResult:
    original_query: str
    hypothetical_document: str
    strategy: str = "template_hyde"


class QueryRewriter(Protocol):
    def rewrite(self, query: str, task_type: str | None = None) -> QueryRewriteResult:
        ...


class HyDEGenerator(Protocol):
    def generate(self, query: str, task_type: str | None = None) -> HyDEResult:
        ...


class RuleBasedHVACQueryRewriter:
    """Deterministic HVAC/BEAR query expansion for reproducible retrieval baselines."""

    def rewrite(self, query: str, task_type: str | None = None) -> QueryRewriteResult:
        terms = ["BEAR HVAC simulation"]
        terms.extend(_terms_for_task(task_type))
        terms.extend(_terms_for_query(query))
        added_terms = _dedupe(terms)
        return QueryRewriteResult(
            original_query=query,
            rewritten_query=" ".join([query, *added_terms]),
            added_terms=added_terms,
        )


class TemplateHyDEGenerator:
    """Template HyDE baseline that avoids network calls and model dependencies."""

    def generate(self, query: str, task_type: str | None = None) -> HyDEResult:
        evidence_terms = _dedupe(
            [
                "BEAR HVAC simulation evidence",
                *_terms_for_task(task_type),
                *_terms_for_query(query),
            ]
        )
        hypothetical = (
            f"Hypothetical evidence document for query: {query}. "
            f"Relevant terms: {'; '.join(evidence_terms)}. "
            "The answer should cite retrieved contexts or tool outputs, preserve data boundaries, "
            "and state that LLM 不直接生成或写回控制动作 for policy questions. "
            "For policy tasks, reference policy_result, policy_name, notes, recommended_action, "
            "estimated_energy, and comfort risk only when they come from a policy tool."
        )
        return HyDEResult(original_query=query, hypothetical_document=hypothetical)


class RewriteRAGPipeline:
    """RAG pipeline that retrieves with a rewritten query and answers the original query."""

    def __init__(
        self,
        retriever,
        *,
        query_rewriter: QueryRewriter | None = None,
        task_type: str | None = None,
    ) -> None:
        self.retriever = retriever
        self.query_rewriter = query_rewriter or RuleBasedHVACQueryRewriter()
        self.task_type = task_type

    def answer(self, question: str, top_k: int = 3) -> RAGAnswer:
        rewritten = self.query_rewriter.rewrite(question, task_type=self.task_type)
        contexts = [
            _with_retrieval_metadata(context, rewritten.rewritten_query, rewritten.strategy)
            for context in self.retriever.search(rewritten.rewritten_query, top_k=top_k)
        ]
        return _answer_from_contexts(question, contexts)


class HyDERAGPipeline:
    """RAG pipeline that retrieves with a deterministic hypothetical document."""

    def __init__(
        self,
        retriever,
        *,
        hyde_generator: HyDEGenerator | None = None,
        task_type: str | None = None,
    ) -> None:
        self.retriever = retriever
        self.hyde_generator = hyde_generator or TemplateHyDEGenerator()
        self.task_type = task_type

    def answer(self, question: str, top_k: int = 3) -> RAGAnswer:
        hyde = self.hyde_generator.generate(question, task_type=self.task_type)
        contexts = [
            _with_retrieval_metadata(context, hyde.hypothetical_document, hyde.strategy)
            for context in self.retriever.search(hyde.hypothetical_document, top_k=top_k)
        ]
        return _answer_from_contexts(question, contexts)


def _answer_from_contexts(question: str, contexts: list[dict]) -> RAGAnswer:
    if not contexts:
        return RAGAnswer(
            question=question,
            answer="未找到足够的检索证据，无法给出可靠回答。",
            citations=[],
            retrieved_contexts=[],
        )
    return RAGAnswer(
        question=question,
        answer=" ".join(context["text"] for context in contexts),
        citations=[context["citation"] for context in contexts],
        retrieved_contexts=contexts,
    )


def _with_retrieval_metadata(context: dict, retrieval_query: str, strategy: str) -> dict:
    updated = dict(context)
    updated["retrieval_query"] = retrieval_query
    updated["retrieval_query_strategy"] = strategy
    return updated


def _terms_for_task(task_type: str | None) -> list[str]:
    if task_type == "timeseries_query":
        return ["query_metric", "zone_temperature", "summary", "timestamp", "zone_id"]
    if task_type == "anomaly_diagnosis":
        return ["detect_anomaly", "zone_temperature", "alarm", "anomaly", "comfort_violation"]
    if task_type == "policy_recommendation":
        return ["policy_result", "policy_name", "rule_based_policy", "recommended_action", "comfort risk"]
    if task_type == "document_qa":
        return ["data boundary", "retrieved context", "citation"]
    return []


def _terms_for_query(query: str) -> list[str]:
    terms: list[str] = []
    lowered = query.lower()
    if any(token in query for token in ["温度", "最大值", "最近", "小时"]) or "temperature" in lowered:
        terms.extend(["query_metric", "zone_temperature", "summary"])
    if any(token in query for token in ["异常", "告警", "升高"]) or "alarm" in lowered:
        terms.extend(["detect_anomaly", "anomaly", "zone_temperature"])
    if any(token in query for token in ["策略", "控制", "调整"]) or "policy" in lowered:
        terms.extend(["policy_result", "policy_name", "recommended_action"])
    if "bear" in lowered or "仿真" in query:
        terms.extend(["BEAR HVAC simulation", "data boundary"])
    return terms


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
