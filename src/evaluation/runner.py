from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from src.agent.orchestrator import BaselineOrchestrator
from src.agent.react_agent import ReActOrchestrator
from src.agent.langgraph_workflow import LangGraphOrchestrator
from src.evaluation.dataset import load_eval_dataset
from src.evaluation.dataset import EvalRecord
from src.evaluation.llm_judge import LLMJudge
from src.evaluation.metrics import (
    answer_correctness_proxy,
    citation_hit_rate,
    context_recall,
    evidence_coverage,
    expected_keyword_coverage,
    grounding_rate,
    faithfulness_proxy,
    lexical_answer_coverage,
    planned_step_accuracy,
    planned_step_order_accuracy,
    policy_final_step_rate,
    required_step_recall,
    tool_execution_success_rate,
    tool_selection_accuracy,
)
from src.retrieval.rag import ExtractiveRAGPipeline
from src.retrieval.rag import GroundedRAGPipeline
from src.retrieval.dense import DenseRetriever
from src.retrieval.embeddings import (
    DeterministicHashEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from src.retrieval.faiss_retriever import FaissDenseRetriever
from src.retrieval.query_rewrite import (
    LLMMultiQueryRAGPipeline,
    HyDERAGPipeline,
    RewriteRAGPipeline,
    RuleBasedHVACQueryRewriter,
    build_multi_query_rewriter_from_env,
)
from src.retrieval.schemas import DocumentChunk
from src.retrieval.retriever import (
    HybridRetriever,
    HybridRRFRetriever,
    KeywordRetriever,
    RerankingRetriever,
)


def run_baseline_eval(
    eval_path: str | Path,
    orchestrator: BaselineOrchestrator,
    llm_judge: LLMJudge | None = None,
) -> dict[str, Any]:
    records = load_eval_dataset(eval_path)
    predictions = []
    for record in records:
        output = orchestrator.run(
            record.question,
            task_type=None if record.expected_steps else record.task_type,
        )
        prediction = {
                "id": record.id,
                "question": record.question,
                "task_type": record.task_type,
                "answer": output.get("answer"),
                "route": output.get("route"),
                "tools": output.get("tools", []),
                "citations": output.get("citations", []),
                "retrieved_contexts": output.get("retrieved_contexts", []),
                "tool_results": output.get("tool_results", []),
                "planned_steps": output.get("planned_steps", [{"route": output.get("route")}]),
                "answer_audit": output.get("answer_audit", {}),
            }
        if llm_judge is not None:
            prediction["llm_judge"] = llm_judge.judge(
                question=record.question,
                answer=str(output.get("answer") or ""),
                gold_answer=record.gold_answer,
                expected_keywords=record.expected_keywords,
                evidence_texts=_evidence_texts(output),
            )
        predictions.append(prediction)

    prediction_map = {prediction["id"]: prediction for prediction in predictions}
    metrics = _compute_metrics(records, prediction_map)
    if llm_judge is not None:
        metrics.update(_compute_llm_judge_metrics(predictions))
    return {
        "predictions": predictions,
        "metrics": metrics,
        "by_task_type": _compute_metrics_by_task_type(records, prediction_map),
    }


def run_baseline_comparison(
    eval_path: str | Path,
    orchestrator: BaselineOrchestrator,
    *,
    dense_provider: str = "deterministic",
    dense_backend: str = "memory",
    dense_model: str | None = None,
) -> dict[str, Any]:
    records = load_eval_dataset(eval_path)
    chunks = getattr(orchestrator.rag_pipeline.retriever, "chunks", [])
    keyword_rag = ExtractiveRAGPipeline(KeywordRetriever(chunks))
    keyword_grounded_rag = GroundedRAGPipeline(KeywordRetriever(chunks))
    dense_rag = build_dense_rag_pipeline(
        chunks,
        provider=dense_provider,
        backend=dense_backend,
        model_name=dense_model,
    )
    dense_grounded_rag = build_grounded_rag_pipeline(
        chunks,
        provider=dense_provider,
        backend=dense_backend,
        model_name=dense_model,
    )
    hybrid_rag = ExtractiveRAGPipeline(HybridRetriever(chunks))
    hybrid_rrf_rag = build_hybrid_rrf_rag_pipeline(
        chunks,
        provider=dense_provider,
        backend=dense_backend,
        model_name=dense_model,
    )
    hybrid_rerank_rag = ExtractiveRAGPipeline(
        RerankingRetriever(HybridRetriever(chunks), candidate_k=10)
    )
    rewrite_rag = RewriteRAGPipeline(HybridRetriever(chunks))
    rewrite_llm_rag = LLMMultiQueryRAGPipeline(
        HybridRetriever(chunks),
        query_rewriter=build_multi_query_rewriter_from_env(),
        candidate_k=10,
    )
    rewrite_grounded_rag = GroundedRAGPipeline(
        RewritingSearcher(HybridRetriever(chunks))
    )
    hyde_rag = HyDERAGPipeline(HybridRetriever(chunks))
    hyde_rerank_rag = HyDERAGPipeline(
        RerankingRetriever(HybridRetriever(chunks), candidate_k=10)
    )
    runs = [
        _evaluate_predictions("llm_only", records, _run_llm_only(records)),
        _evaluate_predictions(
            "rag_keyword",
            records,
            _run_rag_only(records, keyword_rag),
        ),
        _evaluate_predictions(
            "rag_keyword_grounded",
            records,
            _run_rag_only(records, keyword_grounded_rag),
        ),
        _evaluate_predictions(
            "rag_dense",
            records,
            _run_rag_only(records, dense_rag),
        ),
        _evaluate_predictions(
            "rag_dense_grounded",
            records,
            _run_rag_only(records, dense_grounded_rag),
        ),
        _evaluate_predictions(
            "rag_hybrid",
            records,
            _run_rag_only(records, hybrid_rag),
        ),
        _evaluate_predictions(
            "hybrid_rrf",
            records,
            _run_rag_only(records, hybrid_rrf_rag),
        ),
        _evaluate_predictions(
            "rag_hybrid_rerank",
            records,
            _run_rag_only(records, hybrid_rerank_rag),
        ),
        _evaluate_predictions(
            "rag_rewrite",
            records,
            _run_rag_only(records, rewrite_rag),
        ),
        _evaluate_predictions(
            "rewrite_llm",
            records,
            _run_rag_only(records, rewrite_llm_rag, include_latency=True),
        ),
        _evaluate_predictions(
            "rag_rewrite_grounded",
            records,
            _run_rag_only(records, rewrite_grounded_rag),
        ),
        _evaluate_predictions(
            "rag_hyde",
            records,
            _run_rag_only(records, hyde_rag),
        ),
        _evaluate_predictions(
            "rag_hyde_rerank",
            records,
            _run_rag_only(records, hyde_rerank_rag),
        ),
        _evaluate_predictions(
            "rag",
            records,
            _run_rag_only(records, orchestrator.rag_pipeline),
        ),
    ]
    agent_run = run_baseline_eval(eval_path, orchestrator)
    runs.append(
        {
            "mode": "rag_tool_agent",
            "predictions": agent_run["predictions"],
            "metrics": agent_run["metrics"],
            "by_task_type": agent_run["by_task_type"],
        }
    )
    langgraph_run = run_baseline_eval(
        eval_path,
        LangGraphOrchestrator(orchestrator),
    )
    runs.append(
        {
            "mode": "langgraph_tool_agent",
            "predictions": langgraph_run["predictions"],
            "metrics": langgraph_run["metrics"],
            "by_task_type": langgraph_run["by_task_type"],
        }
    )
    react_run = run_baseline_eval(
        eval_path,
        ReActOrchestrator(orchestrator),
    )
    runs.append(
        {
            "mode": "react_agent",
            "predictions": react_run["predictions"],
            "metrics": react_run["metrics"],
            "by_task_type": react_run["by_task_type"],
        }
    )
    return {
        "runs": runs,
        "summary": {run["mode"]: run["metrics"] for run in runs},
        "by_task_type": {run["mode"]: run["by_task_type"] for run in runs},
    }


def _run_llm_only(records: list[EvalRecord]) -> list[dict[str, Any]]:
    predictions = []
    for record in records:
        predictions.append(
            {
                "id": record.id,
                "question": record.question,
                "task_type": record.task_type,
                "answer": "LLM-only baseline: 未使用检索证据或时序工具，无法给出可验证结论。",
                "route": "llm_only",
                "tools": [],
                "citations": [],
                "retrieved_contexts": [],
                "tool_results": [],
                "planned_steps": [],
            }
        )
    return predictions


def build_dense_rag_pipeline(
    chunks: list[DocumentChunk],
    *,
    provider: str = "deterministic",
    backend: str = "memory",
    model_name: str | None = None,
) -> ExtractiveRAGPipeline:
    if provider == "deterministic":
        embedding_provider = DeterministicHashEmbeddingProvider()
    elif provider == "sentence-transformers":
        embedding_provider = SentenceTransformerEmbeddingProvider(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2"
        )
    else:
        raise ValueError(f"Unsupported dense embedding provider: {provider}")

    if backend == "memory":
        retriever = DenseRetriever(chunks, embedding_provider=embedding_provider)
    elif backend == "faiss":
        retriever = FaissDenseRetriever(chunks, embedding_provider=embedding_provider)
    else:
        raise ValueError(f"Unsupported dense retrieval backend: {backend}")
    return ExtractiveRAGPipeline(retriever)


def build_hybrid_rrf_rag_pipeline(
    chunks: list[DocumentChunk],
    *,
    provider: str = "deterministic",
    backend: str = "memory",
    model_name: str | None = None,
    candidate_k: int = 20,
    rrf_k: int = 60,
) -> ExtractiveRAGPipeline:
    dense_rag = build_dense_rag_pipeline(
        chunks,
        provider=provider,
        backend=backend,
        model_name=model_name,
    )
    return ExtractiveRAGPipeline(
        HybridRRFRetriever(
            HybridRetriever(chunks),
            dense_rag.retriever,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
        )
    )


def build_grounded_rag_pipeline(
    chunks: list[DocumentChunk],
    *,
    provider: str = "deterministic",
    backend: str = "memory",
    model_name: str | None = None,
) -> GroundedRAGPipeline:
    if provider == "deterministic":
        embedding_provider = DeterministicHashEmbeddingProvider()
    elif provider == "sentence-transformers":
        embedding_provider = SentenceTransformerEmbeddingProvider(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2"
        )
    else:
        raise ValueError(f"Unsupported dense embedding provider: {provider}")

    if backend == "memory":
        retriever = DenseRetriever(chunks, embedding_provider=embedding_provider)
    elif backend == "faiss":
        retriever = FaissDenseRetriever(chunks, embedding_provider=embedding_provider)
    else:
        raise ValueError(f"Unsupported dense retrieval backend: {backend}")
    return GroundedRAGPipeline(retriever)


class RewritingSearcher:
    def __init__(self, retriever, query_rewriter: RuleBasedHVACQueryRewriter | None = None) -> None:
        self.retriever = retriever
        self.query_rewriter = query_rewriter or RuleBasedHVACQueryRewriter()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        rewritten = self.query_rewriter.rewrite(query)
        return [
            {
                **context,
                "retrieval_query": rewritten.rewritten_query,
                "retrieval_query_strategy": rewritten.strategy,
            }
            for context in self.retriever.search(rewritten.rewritten_query, top_k=top_k)
        ]


def _run_rag_only(
    records: list[EvalRecord],
    rag_pipeline: ExtractiveRAGPipeline,
    *,
    include_latency: bool = False,
) -> list[dict[str, Any]]:
    predictions = []
    for record in records:
        started_at = perf_counter()
        answer = rag_pipeline.answer(record.question, top_k=3)
        prediction = {
            "id": record.id,
            "question": record.question,
            "task_type": record.task_type,
            "answer": answer.answer,
            "route": "rag",
            "tools": [],
            "citations": answer.citations,
            "retrieved_contexts": answer.retrieved_contexts,
            "tool_results": [],
        }
        if include_latency:
            prediction["latency_seconds"] = perf_counter() - started_at
        predictions.append(prediction)
    return predictions


def _evaluate_predictions(
    mode: str,
    records: list[EvalRecord],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    prediction_map = {prediction["id"]: prediction for prediction in predictions}
    return {
        "mode": mode,
        "predictions": predictions,
        "metrics": _compute_metrics(records, prediction_map),
        "by_task_type": _compute_metrics_by_task_type(records, prediction_map),
    }


def _compute_metrics(records: list[EvalRecord], prediction_map: dict[str, dict]) -> dict[str, float]:
    metrics = {
        "citation_hit_rate": citation_hit_rate(records, prediction_map),
        "context_recall": context_recall(records, prediction_map),
        "expected_keyword_coverage": expected_keyword_coverage(records, prediction_map),
        "lexical_answer_coverage": lexical_answer_coverage(records, prediction_map),
        "tool_selection_accuracy": tool_selection_accuracy(records, prediction_map),
        "tool_execution_success_rate": tool_execution_success_rate(records, prediction_map),
        "evidence_coverage": evidence_coverage(records, prediction_map),
        "answer_correctness_proxy": answer_correctness_proxy(records, prediction_map),
        "faithfulness_proxy": faithfulness_proxy(records, prediction_map),
        "grounding_rate": grounding_rate(records, prediction_map),
        "planned_step_accuracy": planned_step_accuracy(records, prediction_map),
        "planned_step_order_accuracy": planned_step_order_accuracy(records, prediction_map),
        "required_step_recall": required_step_recall(records, prediction_map),
        "policy_final_step_rate": policy_final_step_rate(records, prediction_map),
    }
    latencies = [
        float(prediction["latency_seconds"])
        for prediction in prediction_map.values()
        if prediction.get("latency_seconds") is not None
    ]
    if latencies:
        metrics["retrieval_average_latency_seconds"] = sum(latencies) / len(latencies)
    return {
        name: value
        for name, value in metrics.items()
        if value is not None
    }


def _compute_llm_judge_metrics(predictions: list[dict[str, Any]]) -> dict[str, float]:
    judged = [prediction["llm_judge"] for prediction in predictions if "llm_judge" in prediction]
    if not judged:
        return {}
    return {
        "llm_judge_correctness": sum(float(item["correctness"]) for item in judged) / len(judged),
        "llm_judge_faithfulness": sum(float(item["faithfulness"]) for item in judged) / len(judged),
    }


def _evidence_texts(output: dict[str, Any]) -> list[str]:
    texts = []
    for context in output.get("retrieved_contexts", []):
        texts.append(str(context.get("text") or context.get("content") or context))
    for tool_result in output.get("tool_results", []):
        texts.append(str(tool_result))
    return texts


def _compute_metrics_by_task_type(
    records: list[EvalRecord],
    prediction_map: dict[str, dict],
) -> dict[str, dict[str, float]]:
    task_types = sorted({record.task_type for record in records})
    return {
        task_type: _compute_metrics(
            [record for record in records if record.task_type == task_type],
            prediction_map,
        )
        for task_type in task_types
    }


def save_predictions_jsonl(predictions: list[dict], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(prediction, ensure_ascii=False) for prediction in predictions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
