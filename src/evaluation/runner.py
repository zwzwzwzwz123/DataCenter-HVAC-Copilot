from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agent.orchestrator import BaselineOrchestrator
from src.evaluation.dataset import load_eval_dataset
from src.evaluation.dataset import EvalRecord
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
from src.retrieval.retriever import HybridRetriever, KeywordRetriever, RerankingRetriever


def run_baseline_eval(
    eval_path: str | Path,
    orchestrator: BaselineOrchestrator,
) -> dict[str, Any]:
    records = load_eval_dataset(eval_path)
    predictions = []
    for record in records:
        output = orchestrator.run(record.question, task_type=record.task_type)
        predictions.append(
            {
                "id": record.id,
                "question": record.question,
                "task_type": record.task_type,
                "answer": output.get("answer"),
                "route": output.get("route"),
                "tools": output.get("tools", []),
                "citations": output.get("citations", []),
                "retrieved_contexts": output.get("retrieved_contexts", []),
                "tool_results": output.get("tool_results", []),
            }
        )

    prediction_map = {prediction["id"]: prediction for prediction in predictions}
    return {
        "predictions": predictions,
        "metrics": _compute_metrics(records, prediction_map),
        "by_task_type": _compute_metrics_by_task_type(records, prediction_map),
    }


def run_baseline_comparison(
    eval_path: str | Path,
    orchestrator: BaselineOrchestrator,
) -> dict[str, Any]:
    records = load_eval_dataset(eval_path)
    chunks = getattr(orchestrator.rag_pipeline.retriever, "chunks", [])
    keyword_rag = ExtractiveRAGPipeline(KeywordRetriever(chunks))
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
