from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agent.orchestrator import BaselineOrchestrator
from src.evaluation.dataset import load_eval_dataset
from src.evaluation.dataset import EvalRecord
from src.evaluation.llm_judge import LLMJudge
from src.evaluation.metrics import (
    answer_correctness_proxy,
    citation_hit_rate,
    context_recall,
    evidence_coverage,
    expected_keyword_coverage,
    faithfulness_proxy,
    lexical_answer_coverage,
    tool_execution_success_rate,
    tool_selection_accuracy,
)
from src.retrieval.rag import ExtractiveRAGPipeline
from src.retrieval.dense import DenseRetriever
from src.retrieval.embeddings import (
    DeterministicHashEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from src.retrieval.faiss_retriever import FaissDenseRetriever
from src.retrieval.schemas import DocumentChunk
from src.retrieval.retriever import HybridRetriever, KeywordRetriever, RerankingRetriever


def run_baseline_eval(
    eval_path: str | Path,
    orchestrator: BaselineOrchestrator,
    llm_judge: LLMJudge | None = None,
) -> dict[str, Any]:
    records = load_eval_dataset(eval_path)
    predictions = []
    for record in records:
        output = orchestrator.run(record.question, task_type=record.task_type)
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
) -> dict[str, Any]:
    records = load_eval_dataset(eval_path)
    chunks = getattr(orchestrator.rag_pipeline.retriever, "chunks", [])
    keyword_rag = ExtractiveRAGPipeline(KeywordRetriever(chunks))
    dense_rag = build_dense_rag_pipeline(
        chunks,
        provider=dense_provider,
        backend=dense_backend,
    )
    hybrid_rag = ExtractiveRAGPipeline(HybridRetriever(chunks))
    hybrid_rerank_rag = ExtractiveRAGPipeline(
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
            "rag_dense",
            records,
            _run_rag_only(records, dense_rag),
        ),
        _evaluate_predictions(
            "rag_hybrid",
            records,
            _run_rag_only(records, hybrid_rag),
        ),
        _evaluate_predictions(
            "rag_hybrid_rerank",
            records,
            _run_rag_only(records, hybrid_rerank_rag),
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
            }
        )
    return predictions


def build_dense_rag_pipeline(
    chunks: list[DocumentChunk],
    *,
    provider: str = "deterministic",
    backend: str = "memory",
) -> ExtractiveRAGPipeline:
    if provider == "deterministic":
        embedding_provider = DeterministicHashEmbeddingProvider()
    elif provider == "sentence-transformers":
        embedding_provider = SentenceTransformerEmbeddingProvider()
    else:
        raise ValueError(f"Unsupported dense embedding provider: {provider}")

    if backend == "memory":
        retriever = DenseRetriever(chunks, embedding_provider=embedding_provider)
    elif backend == "faiss":
        retriever = FaissDenseRetriever(chunks, embedding_provider=embedding_provider)
    else:
        raise ValueError(f"Unsupported dense retrieval backend: {backend}")
    return ExtractiveRAGPipeline(retriever)


def _run_rag_only(
    records: list[EvalRecord],
    rag_pipeline: ExtractiveRAGPipeline,
) -> list[dict[str, Any]]:
    predictions = []
    for record in records:
        answer = rag_pipeline.answer(record.question, top_k=3)
        predictions.append(
            {
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
        )
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
    return {
        "citation_hit_rate": citation_hit_rate(records, prediction_map),
        "context_recall": context_recall(records, prediction_map),
        "expected_keyword_coverage": expected_keyword_coverage(records, prediction_map),
        "lexical_answer_coverage": lexical_answer_coverage(records, prediction_map),
        "tool_selection_accuracy": tool_selection_accuracy(records, prediction_map),
        "tool_execution_success_rate": tool_execution_success_rate(records, prediction_map),
        "evidence_coverage": evidence_coverage(records, prediction_map),
        "answer_correctness_proxy": answer_correctness_proxy(records, prediction_map),
        "faithfulness_proxy": faithfulness_proxy(records, prediction_map),
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
