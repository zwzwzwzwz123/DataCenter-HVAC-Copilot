from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Protocol

from src.agent.deepseek_generator import Transport
from src.core.env import load_env_file
from src.retrieval.rag import RAGAnswer
from src.retrieval.retriever import reciprocal_rank_fusion


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


@dataclass(frozen=True)
class MultiQueryRewriteResult:
    original_query: str
    queries: list[str]
    strategy: str
    fallback_used: bool = False
    error: str | None = None


class QueryRewriter(Protocol):
    def rewrite(self, query: str, task_type: str | None = None) -> QueryRewriteResult:
        ...


class MultiQueryRewriter(Protocol):
    def rewrite_queries(self, query: str, task_type: str | None = None) -> MultiQueryRewriteResult:
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


class LLMMultiQueryRewriter:
    """OpenAI-compatible multi-query rewriter with rule-based fallback."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 20.0,
        fallback: QueryRewriter | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.fallback = fallback or RuleBasedHVACQueryRewriter()
        self.transport = transport or _default_transport

    def rewrite_queries(self, query: str, task_type: str | None = None) -> MultiQueryRewriteResult:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _multi_query_system_prompt()},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"query": query, "task_type": task_type},
                            ensure_ascii=False,
                        ),
                    },
                ],
                "temperature": 0.0,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = self.transport(
                f"{self.base_url}/chat/completions",
                headers,
                body,
                self.timeout_seconds,
            )
            content = str(response["choices"][0]["message"]["content"])
            queries = _parse_multi_query_payload(content)
            return MultiQueryRewriteResult(
                original_query=query,
                queries=queries,
                strategy="llm_multi_query_rewrite",
                fallback_used=False,
            )
        except Exception as exc:
            return _fallback_multi_query_result(
                query=query,
                task_type=task_type,
                fallback=self.fallback,
                error=str(exc),
            )


class DeterministicMultiQueryRewriter:
    """Deterministic adapter that exposes rule rewrite through the multi-query API."""

    def __init__(self, fallback: QueryRewriter | None = None) -> None:
        self.fallback = fallback or RuleBasedHVACQueryRewriter()

    def rewrite_queries(self, query: str, task_type: str | None = None) -> MultiQueryRewriteResult:
        rewritten = self.fallback.rewrite(query, task_type=task_type)
        return MultiQueryRewriteResult(
            original_query=query,
            queries=[rewritten.rewritten_query],
            strategy=rewritten.strategy,
            fallback_used=False,
            error=None,
        )


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


class LLMMultiQueryRAGPipeline:
    """RAG pipeline that retrieves LLM query variants and fuses them with RRF."""

    def __init__(
        self,
        retriever,
        *,
        query_rewriter: MultiQueryRewriter | None = None,
        task_type: str | None = None,
        candidate_k: int = 10,
        rrf_k: int = 60,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive.")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")
        self.retriever = retriever
        self.query_rewriter = query_rewriter or DeterministicMultiQueryRewriter()
        self.task_type = task_type
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def answer(self, question: str, top_k: int = 3) -> RAGAnswer:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        rewritten = self.query_rewriter.rewrite_queries(question, task_type=self.task_type)
        ranked_lists = []
        for retrieval_query in rewritten.queries:
            results = [
                _with_retrieval_metadata(context, retrieval_query, rewritten.strategy)
                for context in self.retriever.search(retrieval_query, top_k=self.candidate_k)
            ]
            ranked_lists.append(results)
        contexts = [
            _with_multi_query_metadata(context, rewritten)
            for context in reciprocal_rank_fusion(
                ranked_lists,
                k=self.rrf_k,
                top_k=top_k,
            )
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


def _with_multi_query_metadata(context: dict, rewritten: MultiQueryRewriteResult) -> dict:
    updated = dict(context)
    updated["retrieval_queries"] = rewritten.queries
    updated["retrieval_query_strategy"] = rewritten.strategy
    updated["retrieval_query_fallback_used"] = rewritten.fallback_used
    if rewritten.error:
        updated["retrieval_query_error"] = rewritten.error
    return updated


def build_multi_query_rewriter_from_env(
    project_root: str | Path | None = None,
    transport: Transport | None = None,
) -> MultiQueryRewriter:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    load_env_file(root / ".env")
    provider = os.getenv("QUERY_REWRITE_PROVIDER", "auto").strip().lower()
    if provider in {"", "auto"}:
        provider = "deepseek" if os.getenv("DEEPSEEK_API_KEY", "").strip() else "deterministic"
    if provider in {"deterministic", "rule_based"}:
        return DeterministicMultiQueryRewriter()
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            return DeterministicMultiQueryRewriter()
        return LLMMultiQueryRewriter(
            provider="deepseek",
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("QUERY_REWRITE_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
            timeout_seconds=float(os.getenv("QUERY_REWRITE_TIMEOUT_SECONDS", "20")),
            transport=transport,
        )
    return DeterministicMultiQueryRewriter()


def _fallback_multi_query_result(
    *,
    query: str,
    task_type: str | None,
    fallback: QueryRewriter,
    error: str | None,
) -> MultiQueryRewriteResult:
    rewritten = fallback.rewrite(query, task_type=task_type)
    return MultiQueryRewriteResult(
        original_query=query,
        queries=[rewritten.rewritten_query],
        strategy=rewritten.strategy,
        fallback_used=True,
        error=error,
    )


def _parse_multi_query_payload(content: str) -> list[str]:
    parsed = json.loads(_strip_json_fence(content))
    if not isinstance(parsed, list):
        raise ValueError("multi-query rewrite payload must be a JSON array")
    if not parsed or len(parsed) > 5:
        raise ValueError("multi-query rewrite payload must contain 1 to 5 queries")
    if not all(isinstance(item, str) for item in parsed):
        raise ValueError("multi-query rewrite payload items must be strings")

    queries = _dedupe([item.strip() for item in parsed if item.strip()])
    if not queries:
        raise ValueError("multi-query rewrite payload must not be empty")
    if len(queries) > 5:
        raise ValueError("multi-query rewrite payload must contain at most 5 unique queries")
    return queries


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    return stripped


def _multi_query_system_prompt() -> str:
    return (
        "You rewrite DataCenter-HVAC Copilot retrieval queries. "
        "Return only a JSON array of 1 to 5 strings. "
        "Each string should be a semantic variant of the user query for document retrieval. "
        "Keep domain terms such as BEAR, HVAC, zone_temperature, query_metric, policy_result, "
        "control_action, anomaly, setpoint, fan_power, and cooling_power when relevant. "
        "Do not answer the question, do not call tools, and do not return any object wrapper."
    )


def _default_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> dict[str, Any]:
    from urllib import request

    req = request.Request(url=url, data=body, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


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
