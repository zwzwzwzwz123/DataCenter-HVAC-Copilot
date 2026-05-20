from __future__ import annotations

from typing import Any

import pandas as pd

from src.agent.router import route_task
from src.policies.rule_based import run_rule_based_policy
from src.retrieval.rag import ExtractiveRAGPipeline
from src.tools.timeseries import detect_anomaly, query_metric


class BaselineOrchestrator:
    """Deterministic baseline orchestrator before LangGraph is introduced."""

    def __init__(
        self,
        rag_pipeline: ExtractiveRAGPipeline,
        trajectory: pd.DataFrame,
        data_source: dict[str, str] | None = None,
    ) -> None:
        self.rag_pipeline = rag_pipeline
        self.trajectory = trajectory
        self.data_source = data_source or trajectory.attrs.get(
            "data_source",
            {
                "kind": str(trajectory.attrs.get("source", "unknown")),
                "path": "",
            },
        )

    def run(self, question: str, task_type: str | None = None) -> dict[str, Any]:
        decision = route_task(question, task_type=task_type)
        if decision.route == "document_qa":
            return self._run_document_qa(question, decision.reason)
        if decision.route == "timeseries_query":
            return self._run_timeseries_query(question, decision.reason)
        if decision.route == "anomaly_diagnosis":
            return self._run_anomaly_diagnosis(question, decision.reason)
        if decision.route == "policy_recommendation":
            return self._run_policy_recommendation(question, decision.reason)
        raise ValueError(f"Unsupported route: {decision.route}")

    def _run_document_qa(self, question: str, reason: str) -> dict[str, Any]:
        answer = self.rag_pipeline.answer(question, top_k=3)
        return {
            "question": question,
            "route": "document_qa",
            "route_reason": reason,
            "answer": answer.answer,
            "citations": answer.citations,
            "retrieved_contexts": answer.retrieved_contexts,
            "tools": [],
            "tool_results": [],
            "data_source": self.data_source,
        }

    def _run_timeseries_query(self, question: str, reason: str) -> dict[str, Any]:
        start_time, end_time = _trajectory_bounds(self.trajectory)
        result = query_metric(
            self.trajectory,
            metric_name="zone_temperature",
            start_time=start_time,
            end_time=end_time,
            zone_id=_first_zone(self.trajectory),
        )
        return {
            "question": question,
            "route": "timeseries_query",
            "route_reason": reason,
            "answer": f"zone_temperature 最大值为 {result['summary']['max']}。",
            "citations": [],
            "retrieved_contexts": [],
            "tools": ["query_metric"],
            "tool_results": [result],
            "data_source": self.data_source,
        }

    def _run_anomaly_diagnosis(self, question: str, reason: str) -> dict[str, Any]:
        result = detect_anomaly(
            self.trajectory,
            metric_name="zone_temperature",
            window_size=2,
            threshold=2.0,
            zone_id=_first_zone(self.trajectory),
        )
        count = len(result["anomalies"])
        return {
            "question": question,
            "route": "anomaly_diagnosis",
            "route_reason": reason,
            "answer": f"检测到 {count} 个 zone_temperature 异常点。",
            "citations": [],
            "retrieved_contexts": [],
            "tools": ["detect_anomaly"],
            "tool_results": [result],
            "data_source": self.data_source,
        }

    def _run_policy_recommendation(self, question: str, reason: str) -> dict[str, Any]:
        state = _latest_state(self.trajectory)
        policy_result = run_rule_based_policy(state)
        return {
            "question": question,
            "route": "policy_recommendation",
            "route_reason": reason,
            "answer": policy_result.notes,
            "citations": [],
            "retrieved_contexts": [],
            "tools": ["rule_based_policy"],
            "tool_results": [policy_result.model_dump()],
            "policy_result": policy_result.model_dump(),
            "data_source": self.data_source,
        }


def _trajectory_bounds(trajectory: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    timestamps = pd.to_datetime(trajectory["timestamp"], utc=True)
    return timestamps.min(), timestamps.max()


def _first_zone(trajectory: pd.DataFrame) -> str | None:
    if "zone_id" not in trajectory.columns or trajectory.empty:
        return None
    return str(trajectory["zone_id"].iloc[0])


def _latest_state(trajectory: pd.DataFrame) -> dict[str, Any]:
    if trajectory.empty:
        return {
            "state_id": "empty_trajectory",
            "zone_temperature": 0.0,
            "comfort_upper_bound": 26.0,
            "current_action": [0.0],
        }
    row = trajectory.sort_values("timestamp").iloc[-1]
    return {
        "state_id": f"{row.get('scenario_id', 'scenario')}_latest",
        "zone_temperature": float(row.get("zone_temperature", 0.0)),
        "comfort_upper_bound": 26.0,
        "comfort_lower_bound": 22.0,
        "current_action": [0.0, 0.0],
    }
