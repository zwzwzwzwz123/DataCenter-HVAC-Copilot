from __future__ import annotations

from collections.abc import Callable
import re
from typing import Any

import pandas as pd

from src.agent.answer_audit import audit_answer
from src.agent.answer_generator import (
    AnswerGenerator,
    AnswerGeneratorInput,
    DeterministicAnswerGenerator,
)
from src.policies.base import PolicyResult
from src.policies.rule_based import run_rule_based_policy
from src.retrieval.rag import ExtractiveRAGPipeline
from src.tools.timeseries import (
    compare_period,
    compute_energy_breakdown,
    detect_anomaly,
    plot_metric_trend,
    query_metric,
)


class AgentTaskExecutor:
    """Shared route execution logic for baseline and LangGraph orchestrators."""

    def __init__(
        self,
        rag_pipeline: ExtractiveRAGPipeline,
        trajectory: pd.DataFrame,
        data_source: dict[str, str] | None = None,
        answer_generator: AnswerGenerator | None = None,
        policy_runner: Callable[[dict[str, Any]], PolicyResult] | None = None,
    ) -> None:
        self.rag_pipeline = rag_pipeline
        self.trajectory = trajectory
        self.answer_generator = answer_generator or DeterministicAnswerGenerator()
        self.policy_runner = policy_runner or run_rule_based_policy
        self.data_source = data_source or trajectory.attrs.get(
            "data_source",
            {
                "kind": str(trajectory.attrs.get("source", "unknown")),
                "path": "",
            },
        )

    def run_document_qa(self, question: str, reason: str) -> dict[str, Any]:
        evidence = self.collect_document_qa_evidence(question, reason)
        return self.generate_answer_from_evidence(evidence)

    def collect_document_qa_evidence(
        self,
        question: str,
        reason: str,
        step_spec: Any | None = None,
    ) -> dict[str, Any]:
        rag_answer = self.rag_pipeline.answer(question, top_k=3)
        return {
            "question": question,
            "route": "document_qa",
            "route_reason": reason,
            "citations": rag_answer.citations,
            "retrieved_contexts": rag_answer.retrieved_contexts,
            "tools": [],
            "tool_results": [],
            "data_source": self.data_source,
        }

    def run_timeseries_query(self, question: str, reason: str) -> dict[str, Any]:
        evidence = self.collect_timeseries_query_evidence(question, reason)
        return self.generate_answer_from_evidence(evidence)

    def collect_timeseries_query_evidence(
        self,
        question: str,
        reason: str,
        step_spec: Any | None = None,
    ) -> dict[str, Any]:
        start_time, end_time, time_window_metadata = _select_time_window(
            self.trajectory,
            step_spec,
        )
        tool_name = _select_timeseries_tool(question, step_spec)
        metric_name = _select_metric_name(question, self.trajectory, step_spec)
        zone_id = _select_zone_id(self.trajectory, step_spec)

        if tool_name == "compare_period":
            midpoint = start_time + (end_time - start_time) / 2
            result = compare_period(
                self.trajectory,
                metric_name=metric_name,
                period_a=(start_time, midpoint),
                period_b=(midpoint, end_time),
                zone_id=zone_id,
            )
        elif tool_name == "plot_metric_trend":
            result = plot_metric_trend(
                self.trajectory,
                metric_name=metric_name,
                start_time=start_time,
                end_time=end_time,
                zone_id=zone_id,
            )
        elif tool_name == "compute_energy_breakdown":
            result = compute_energy_breakdown(
                self.trajectory,
                start_time=start_time,
                end_time=end_time,
            )
        else:
            result = query_metric(
                self.trajectory,
                metric_name=metric_name,
                start_time=start_time,
                end_time=end_time,
                zone_id=zone_id,
            )
        evidence = {
            "question": question,
            "route": "timeseries_query",
            "route_reason": reason,
            "citations": [],
            "retrieved_contexts": [],
            "tools": [tool_name],
            "tool_results": [result],
            "data_source": self.data_source,
        }
        _annotate_time_window_result(result, time_window_metadata)
        return evidence

    def run_anomaly_diagnosis(self, question: str, reason: str) -> dict[str, Any]:
        evidence = self.collect_anomaly_diagnosis_evidence(question, reason)
        return self.generate_answer_from_evidence(evidence)

    def collect_anomaly_diagnosis_evidence(
        self,
        question: str,
        reason: str,
        step_spec: Any | None = None,
    ) -> dict[str, Any]:
        result = detect_anomaly(
            self.trajectory,
            metric_name=_select_metric_name(question, self.trajectory, step_spec),
            window_size=2,
            threshold=2.0,
            zone_id=_select_zone_id(self.trajectory, step_spec),
        )
        return {
            "question": question,
            "route": "anomaly_diagnosis",
            "route_reason": reason,
            "citations": [],
            "retrieved_contexts": [],
            "tools": ["detect_anomaly"],
            "tool_results": [result],
            "data_source": self.data_source,
        }

    def run_policy_recommendation(self, question: str, reason: str) -> dict[str, Any]:
        evidence = self.collect_policy_recommendation_evidence(question, reason)
        return self.generate_answer_from_evidence(evidence)

    def collect_policy_recommendation_evidence(
        self,
        question: str,
        reason: str,
        step_spec: Any | None = None,
    ) -> dict[str, Any]:
        state = self.latest_policy_state()
        policy_result = self.policy_runner(state)
        policy_dump = policy_result.model_dump()
        tool_name = _policy_tool_name(policy_result)
        return {
            "question": question,
            "route": "policy_recommendation",
            "route_reason": reason,
            "citations": [],
            "retrieved_contexts": [],
            "tools": [tool_name],
            "tool_results": [policy_dump],
            "policy_result": policy_dump,
            "data_source": self.data_source,
        }

    def generate_answer_from_evidence(self, evidence: dict[str, Any]) -> dict[str, Any]:
        policy_result = (
            evidence.get("policy_result")
            if isinstance(evidence.get("policy_result"), dict)
            else None
        )
        generated = self.answer_generator.generate(
            AnswerGeneratorInput(
                question=str(evidence.get("question", "")),
                route=str(evidence.get("route", "")),
                route_reason=str(evidence.get("route_reason", "")),
                retrieved_contexts=list(evidence.get("retrieved_contexts", [])),
                citations=list(evidence.get("citations", [])),
                tools=list(evidence.get("tools", [])),
                tool_results=list(evidence.get("tool_results", [])),
                data_source=evidence.get("data_source"),
                policy_result=policy_result,
                conversation_context=evidence.get("conversation_context"),
            )
        )
        return {
            **evidence,
            "answer": generated.answer,
            "answer_generator": generated.generator,
            "answer_audit": audit_answer(
                generated.answer,
                route=str(evidence.get("route", "")),
                policy_result=policy_result,
            ),
        }

    def latest_policy_state(self) -> dict[str, Any]:
        return _latest_state(self.trajectory)


def _trajectory_bounds(trajectory: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    timestamps = pd.to_datetime(trajectory["timestamp"], utc=True)
    return timestamps.min(), timestamps.max()


def _select_time_window(
    trajectory: pd.DataFrame,
    step_spec: Any | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp, dict[str, Any]]:
    start_time, end_time = _trajectory_bounds(trajectory)
    planned_window = _step_attr(step_spec, "time_window")
    if not planned_window:
        return start_time, end_time, {
            "requested": None,
            "applied": "full_demo_range",
            "notes": [],
        }

    normalized = str(planned_window).strip().lower().replace("-", "_")
    if normalized in {"full_demo_range", "full_range", "all", "all_data"}:
        return start_time, end_time, {
            "requested": str(planned_window),
            "applied": "full_demo_range",
            "notes": [],
        }
    if normalized in {"latest", "recent"}:
        return end_time, end_time, {
            "requested": str(planned_window),
            "applied": normalized,
            "notes": [],
        }

    match = re.fullmatch(r"(?:last|latest|recent)_(\d+)_hours?", normalized)
    if match:
        hours = int(match.group(1))
        return max(start_time, end_time - pd.Timedelta(hours=hours)), end_time, {
            "requested": str(planned_window),
            "applied": normalized,
            "notes": [],
        }

    match = re.fullmatch(r"(?:last|latest|recent)_(\d+)_minutes?", normalized)
    if match:
        minutes = int(match.group(1))
        return max(start_time, end_time - pd.Timedelta(minutes=minutes)), end_time, {
            "requested": str(planned_window),
            "applied": normalized,
            "notes": [],
        }

    return start_time, end_time, {
        "requested": str(planned_window),
        "applied": "full_demo_range",
        "notes": [f"Unsupported time_window '{planned_window}'; used full_demo_range."],
    }


def _annotate_time_window_result(result: dict[str, Any], metadata: dict[str, Any]) -> None:
    requested = metadata.get("requested")
    if requested:
        result["time_window"] = requested
    result["time_window_applied"] = metadata.get("applied", "full_demo_range")
    notes = list(result.get("notes", []))
    notes.extend(metadata.get("notes", []))
    if notes:
        result["notes"] = notes


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
    ordered = trajectory.sort_values("timestamp")
    row = ordered.iloc[-1]
    current_action = _latest_control_action(ordered)
    bear_state_vector = _latest_bear_state_vector(ordered)
    return {
        "state_id": f"{row.get('scenario_id', 'scenario')}_latest",
        "zone_temperature": float(row.get("zone_temperature", 0.0)),
        "comfort_upper_bound": 26.0,
        "comfort_lower_bound": 22.0,
        "current_action": current_action,
        **({"bear_state_vector": bear_state_vector} if bear_state_vector is not None else {}),
    }


def _policy_tool_name(policy_result: PolicyResult) -> str:
    if policy_result.policy_name == "rule_based":
        return "rule_based_policy"
    return policy_result.policy_name


def _latest_bear_state_vector(trajectory: pd.DataFrame) -> list[float] | None:
    required_columns = {
        "timestamp",
        "scenario_id",
        "zone_id",
        "zone_temperature",
        "outdoor_temp",
        "solar_irradiance",
        "ground_temp",
        "internal_load",
    }
    if not required_columns.issubset(set(trajectory.columns)):
        return None

    latest_timestamp = pd.to_datetime(trajectory["timestamp"], utc=True).max()
    latest_rows = trajectory[pd.to_datetime(trajectory["timestamp"], utc=True) == latest_timestamp].copy()
    if latest_rows.empty:
        return None

    latest_scenario = str(latest_rows["scenario_id"].iloc[-1])
    latest_rows = latest_rows[latest_rows["scenario_id"].astype(str) == latest_scenario]
    if len(latest_rows) != 6:
        return None

    latest_rows = latest_rows.sort_values("zone_id")

    def _to_float(value: Any) -> float:
        if pd.isna(value):
            raise ValueError
        return float(value)

    try:
        zone_temperature = [_to_float(value) for value in latest_rows["zone_temperature"].tolist()]
        outdoor_temp = _to_float(latest_rows["outdoor_temp"].iloc[0])
        solar_irradiance = [_to_float(value) for value in latest_rows["solar_irradiance"].tolist()]
        ground_temp = _to_float(latest_rows["ground_temp"].iloc[0])
        internal_load = [_to_float(value) for value in latest_rows["internal_load"].tolist()]
    except (TypeError, ValueError):
        return None

    return zone_temperature + [outdoor_temp] + solar_irradiance + [ground_temp] + internal_load


def _latest_control_action(trajectory: pd.DataFrame) -> list[float]:
    if "control_action" not in trajectory.columns or trajectory.empty:
        return [0.0]
    latest_timestamp = pd.to_datetime(trajectory["timestamp"], utc=True).max()
    latest_rows = trajectory[pd.to_datetime(trajectory["timestamp"], utc=True) == latest_timestamp].copy()
    if latest_rows.empty:
        return [0.0]
    latest_rows = latest_rows.sort_values("zone_id")
    values: list[float] = []
    for value in latest_rows["control_action"].tolist():
        if pd.isna(value):
            continue
        values.append(float(value))
    return values or [0.0 for _ in range(len(latest_rows))]


def _select_timeseries_tool(question: str, step_spec: Any | None = None) -> str:
    planned_tool = _step_attr(step_spec, "tool")
    if planned_tool in {
        "query_metric",
        "compare_period",
        "plot_metric_trend",
        "compute_energy_breakdown",
    }:
        return planned_tool
    normalized = question.lower()
    if any(token in normalized for token in ["构成", "breakdown", "能耗字段", "能耗"]):
        return "compute_energy_breakdown"
    if any(token in normalized for token in ["趋势", "trend", "折线图", "序列", "画"]):
        return "plot_metric_trend"
    if any(token in normalized for token in ["比较", "对比", "前后", "变化", "compare"]):
        return "compare_period"
    return "query_metric"


def _select_metric_name(
    question: str,
    trajectory: pd.DataFrame,
    step_spec: Any | None = None,
) -> str:
    planned_metric = _step_attr(step_spec, "metric_name")
    if planned_metric and planned_metric in trajectory.columns:
        return planned_metric
    normalized = question.lower()
    candidates = [
        "fan_power",
        "control_action",
        "outdoor_temp",
        "internal_load",
        "cooling_power",
        "hvac_power",
        "zone_temperature",
    ]
    for candidate in candidates:
        if candidate in trajectory.columns and candidate.lower() in normalized:
            return candidate
    if "温度" in question and "zone_temperature" in trajectory.columns:
        return "zone_temperature"
    if "风机" in question and "fan_power" in trajectory.columns:
        return "fan_power"
    if "控制" in question and "control_action" in trajectory.columns:
        return "control_action"
    if "室外" in question and "outdoor_temp" in trajectory.columns:
        return "outdoor_temp"
    if "负载" in question and "internal_load" in trajectory.columns:
        return "internal_load"
    return "zone_temperature"


def _select_zone_id(trajectory: pd.DataFrame, step_spec: Any | None = None) -> str | None:
    planned_zone = _step_attr(step_spec, "zone_id")
    if planned_zone:
        return planned_zone
    return _first_zone(trajectory)


def _step_attr(step_spec: Any | None, key: str) -> Any:
    if step_spec is None:
        return None
    if isinstance(step_spec, dict):
        return step_spec.get(key)
    return getattr(step_spec, key, None)
