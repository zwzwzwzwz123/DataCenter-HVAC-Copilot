from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from src.agent.answer_generator import AnswerGenerator
from src.agent.executor import AgentTaskExecutor
from src.agent.router import route_task
from src.policies.base import PolicyResult
from src.retrieval.rag import ExtractiveRAGPipeline


class BaselineOrchestrator:
    """Deterministic baseline orchestrator."""

    def __init__(
        self,
        rag_pipeline: ExtractiveRAGPipeline | None = None,
        trajectory: pd.DataFrame | None = None,
        data_source: dict[str, str] | None = None,
        answer_generator: AnswerGenerator | None = None,
        policy_runner: Callable[[dict[str, Any]], PolicyResult] | None = None,
        task_executor: AgentTaskExecutor | None = None,
    ) -> None:
        if task_executor is None:
            if rag_pipeline is None or trajectory is None:
                raise ValueError("rag_pipeline and trajectory are required when task_executor is not provided")
            task_executor = AgentTaskExecutor(
                rag_pipeline=rag_pipeline,
                trajectory=trajectory,
                data_source=data_source,
                answer_generator=answer_generator,
                policy_runner=policy_runner,
            )
        self.task_executor = task_executor
        self.rag_pipeline = task_executor.rag_pipeline
        self.trajectory = task_executor.trajectory
        self.answer_generator = task_executor.answer_generator
        self.policy_runner = task_executor.policy_runner
        self.data_source = task_executor.data_source

    def run(self, question: str, task_type: str | None = None) -> dict[str, Any]:
        decision = route_task(question, task_type=task_type)
        if decision.route == "document_qa":
            return self.run_document_qa(question, decision.reason)
        if decision.route == "timeseries_query":
            return self.run_timeseries_query(question, decision.reason)
        if decision.route == "anomaly_diagnosis":
            return self.run_anomaly_diagnosis(question, decision.reason)
        if decision.route == "policy_recommendation":
            return self.run_policy_recommendation(question, decision.reason)
        raise ValueError(f"Unsupported route: {decision.route}")

    def run_document_qa(self, question: str, reason: str) -> dict[str, Any]:
        return self.task_executor.run_document_qa(question, reason)

    def run_timeseries_query(self, question: str, reason: str) -> dict[str, Any]:
        return self.task_executor.run_timeseries_query(question, reason)

    def run_anomaly_diagnosis(self, question: str, reason: str) -> dict[str, Any]:
        return self.task_executor.run_anomaly_diagnosis(question, reason)

    def run_policy_recommendation(self, question: str, reason: str) -> dict[str, Any]:
        return self.task_executor.run_policy_recommendation(question, reason)

    # Backward-compatible aliases for older tests or notebooks.
    def _run_document_qa(self, question: str, reason: str) -> dict[str, Any]:
        return self.run_document_qa(question, reason)

    def _run_timeseries_query(self, question: str, reason: str) -> dict[str, Any]:
        return self.run_timeseries_query(question, reason)

    def _run_anomaly_diagnosis(self, question: str, reason: str) -> dict[str, Any]:
        return self.run_anomaly_diagnosis(question, reason)

    def _run_policy_recommendation(self, question: str, reason: str) -> dict[str, Any]:
        return self.run_policy_recommendation(question, reason)
