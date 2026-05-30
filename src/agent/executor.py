from __future__ import annotations

from collections.abc import Callable
import re
from time import perf_counter
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
from src.retrieval.query_rewrite import RuleBasedHVACQueryRewriter
from src.retrieval.rag import ExtractiveRAGPipeline
from src.tools.timeseries import (
    comfort_risk_assessment,
    compare_period,
    control_action_audit,
    cooling_efficiency_summary,
    compute_energy_breakdown,
    data_quality_check,
    detect_anomaly,
    plot_metric_trend,
    query_metric,
    zone_hotspot_rank,
)
from src.tools.registry import TOOL_REGISTRY, validate_tool_input
from src.agent.runtime import AgentRuntimeTrace


class AgentTaskExecutor:
    """Shared route execution logic for baseline and LangGraph orchestrators."""

    def __init__(
        self,
        rag_pipeline: ExtractiveRAGPipeline,
        trajectory: pd.DataFrame,
        data_source: dict[str, str] | None = None,
        answer_generator: AnswerGenerator | None = None,
        policy_runner: Callable[[dict[str, Any]], PolicyResult] | None = None,
        approval_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.rag_pipeline = rag_pipeline
        self.trajectory = trajectory
        self.answer_generator = answer_generator or DeterministicAnswerGenerator()
        self.policy_runner = policy_runner or run_rule_based_policy
        self.approval_handler = approval_handler
        self.default_tool_inputs: dict[str, dict[str, Any]] = {}
        self.runtime_trace: AgentRuntimeTrace | None = None
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
        if not rag_answer.retrieved_contexts:
            rewritten = RuleBasedHVACQueryRewriter().rewrite(question, task_type="document_qa")
            retried_answer = self.rag_pipeline.answer(rewritten.rewritten_query, top_k=3)
            status = "success" if retried_answer.retrieved_contexts else "failed"
            self._record_recovery(
                {
                    "strategy": "query_rewrite_retry",
                    "status": status,
                    "original_query": question,
                    "rewritten_query": rewritten.rewritten_query,
                    "added_terms": rewritten.added_terms,
                }
            )
            if retried_answer.retrieved_contexts:
                rag_answer = retried_answer
                rag_answer.retrieved_contexts = [
                    {
                        **context,
                        "retrieval_recovery": True,
                        "retrieval_query": rewritten.rewritten_query,
                        "retrieval_strategy": rewritten.strategy,
                    }
                    for context in rag_answer.retrieved_contexts
                ]
        return {
            "question": question,
            "route": "document_qa",
            "route_reason": reason,
            "citations": rag_answer.citations,
            "retrieved_contexts": rag_answer.retrieved_contexts,
            "tools": [],
            "tool_results": [],
            "tool_calls": [],
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

        result, tool_call = self._execute_tool_call(
            tool_name,
            self._build_timeseries_tool_input(
                tool_name=tool_name,
                metric_name=metric_name,
                start_time=start_time,
                end_time=end_time,
                zone_id=zone_id,
            ),
            lambda tool_input: self._run_timeseries_tool(
                tool_name=tool_name,
                tool_input=tool_input,
                start_time=start_time,
                end_time=end_time,
                zone_id=zone_id,
            ),
        )
        evidence = {
            "question": question,
            "route": "timeseries_query",
            "route_reason": reason,
            "citations": [],
            "retrieved_contexts": [],
            "tools": [tool_name],
            "tool_results": [result],
            "tool_calls": [tool_call],
            "data_source": self.data_source,
        }
        if result.get("status") not in {"error", "blocked"}:
            _annotate_time_window_result(result, time_window_metadata)
        return evidence

    def _run_timeseries_tool(
        self,
        *,
        tool_name: str,
        tool_input: dict[str, Any],
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        zone_id: str | None,
    ) -> dict[str, Any]:
        if tool_name == "compare_period":
            midpoint = start_time + (end_time - start_time) / 2
            return compare_period(
                self.trajectory,
                metric_name=tool_input["metric_name"],
                period_a=(start_time, midpoint),
                period_b=(midpoint, end_time),
                zone_id=zone_id,
            )
        if tool_name == "plot_metric_trend":
            return plot_metric_trend(
                self.trajectory,
                metric_name=tool_input["metric_name"],
                start_time=start_time,
                end_time=end_time,
                zone_id=zone_id,
            )
        if tool_name == "compute_energy_breakdown":
            return compute_energy_breakdown(
                self.trajectory,
                start_time=start_time,
                end_time=end_time,
            )
        if tool_name == "data_quality_check":
            return data_quality_check(
                self.trajectory,
                required_fields=tool_input["required_fields"],
                expected_frequency=tool_input["expected_frequency"],
            )
        if tool_name == "zone_hotspot_rank":
            return zone_hotspot_rank(
                self.trajectory,
                metric_name=tool_input["metric_name"],
                top_k=tool_input["top_k"],
            )
        if tool_name == "control_action_audit":
            return control_action_audit(
                self.trajectory,
                action_metric=_select_control_action_metric(
                    self.trajectory,
                    tool_input["action_metric"],
                ),
                change_threshold=tool_input["change_threshold"],
            )
        if tool_name == "cooling_efficiency_summary":
            return cooling_efficiency_summary(
                self.trajectory,
                power_metrics=tool_input["power_metrics"],
                temperature_metric=tool_input["temperature_metric"],
                comfort_upper_bound=tool_input["comfort_upper_bound"],
            )
        return query_metric(
            self.trajectory,
            metric_name=tool_input["metric_name"],
            start_time=start_time,
            end_time=end_time,
            zone_id=zone_id,
        )

    def run_anomaly_diagnosis(self, question: str, reason: str) -> dict[str, Any]:
        evidence = self.collect_anomaly_diagnosis_evidence(question, reason)
        return self.generate_answer_from_evidence(evidence)

    def collect_anomaly_diagnosis_evidence(
        self,
        question: str,
        reason: str,
        step_spec: Any | None = None,
    ) -> dict[str, Any]:
        tool_name = _select_anomaly_tool(question, step_spec)
        metric_name = _select_metric_name(question, self.trajectory, step_spec)
        tool_input = self._build_anomaly_tool_input(
            tool_name=tool_name,
            metric_name=metric_name,
            zone_id=_select_zone_id(self.trajectory, step_spec),
        )
        result, tool_call = self._execute_tool_call(
            tool_name,
            tool_input,
            lambda validated: self._run_anomaly_tool(tool_name, validated),
        )
        return {
            "question": question,
            "route": "anomaly_diagnosis",
            "route_reason": reason,
            "citations": [],
            "retrieved_contexts": [],
            "tools": [tool_name],
            "tool_results": [result],
            "tool_calls": [tool_call],
            "data_source": self.data_source,
        }

    def _run_anomaly_tool(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "comfort_risk_assessment":
            return comfort_risk_assessment(
                self.trajectory,
                temperature_metric=tool_input["temperature_metric"],
                comfort_lower_bound=tool_input["comfort_lower_bound"],
                comfort_upper_bound=tool_input["comfort_upper_bound"],
            )
        if tool_name == "data_quality_check":
            return data_quality_check(
                self.trajectory,
                required_fields=tool_input["required_fields"],
                expected_frequency=tool_input["expected_frequency"],
            )
        if tool_name == "zone_hotspot_rank":
            return zone_hotspot_rank(
                self.trajectory,
                metric_name=tool_input["metric_name"],
                top_k=tool_input["top_k"],
            )
        return detect_anomaly(
            self.trajectory,
            metric_name=tool_input["metric_name"],
            window_size=tool_input["window_size"],
            threshold=tool_input["threshold"],
            zone_id=tool_input["zone_id"],
        )

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
        policy_result, tool_call = self._execute_tool_call(
            "policy_runner",
            {"state": state},
            lambda _: self.policy_runner(state).model_dump(),
            fallback_runner=lambda _: self._run_rule_based_policy_fallback(state),
        )
        policy_dump = policy_result
        tool_name = _policy_tool_name_from_dump(policy_dump)
        evidence = {
            "question": question,
            "route": "policy_recommendation",
            "route_reason": reason,
            "citations": [],
            "retrieved_contexts": [],
            "tools": [tool_name],
            "tool_results": [policy_dump],
            "tool_calls": [tool_call],
            "data_source": self.data_source,
        }
        if _is_successful_policy_result(policy_dump, tool_call):
            evidence["policy_result"] = policy_dump
        return evidence

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

    def _build_timeseries_tool_input(
        self,
        *,
        tool_name: str,
        metric_name: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        zone_id: str | None,
    ) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "metric_name": metric_name,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "zone_id": zone_id,
            "period_a": (start_time.isoformat(), end_time.isoformat()),
            "period_b": (start_time.isoformat(), end_time.isoformat()),
            "required_fields": _required_trajectory_fields(self.trajectory),
            "expected_frequency": "1h",
            "top_k": 3,
            "action_metric": "control_action",
            "change_threshold": 0.5,
            "power_metrics": None,
            "temperature_metric": "zone_temperature",
            "comfort_upper_bound": 26.0,
        }
        defaults.update(self.default_tool_inputs.get(tool_name, {}))
        return defaults

    def _build_anomaly_tool_input(
        self,
        *,
        tool_name: str,
        metric_name: str,
        zone_id: str | None,
    ) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "metric_name": metric_name,
            "window_size": 2,
            "threshold": 2.0,
            "zone_id": zone_id,
            "temperature_metric": "zone_temperature",
            "comfort_lower_bound": 22.0,
            "comfort_upper_bound": 26.0,
            "required_fields": _required_trajectory_fields(self.trajectory),
            "expected_frequency": "1h",
            "top_k": 3,
        }
        defaults.update(self.default_tool_inputs.get(tool_name, {}))
        return defaults

    def _run_rule_based_policy_fallback(self, state: dict[str, Any]) -> dict[str, Any]:
        fallback = run_rule_based_policy(state).model_dump()
        fallback.update(
            {
                "fallback_used": True,
                "fallback_from": "policy_runner",
            }
        )
        return fallback

    def _execute_tool_call(
        self,
        tool_name: str,
        raw_input: dict[str, Any],
        runner: Callable[[dict[str, Any]], dict[str, Any]],
        fallback_runner: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        started = perf_counter()
        permission = _permission_decision(tool_name)
        tool_call = {
            "tool_call_id": f"{tool_name}:{id(raw_input)}",
            "tool_name": tool_name,
            "input": raw_input,
            "risk_level": TOOL_REGISTRY[tool_name].risk_level,
            "permission_decision": permission,
            "audit_required": TOOL_REGISTRY[tool_name].risk_level != "read_only",
            "status": "success",
            "duration_ms": 0.0,
            "error": None,
        }
        tool_call["approval"] = _approval_decision(tool_name, permission)
        tool_call["approval"] = self._resolve_approval(tool_name, raw_input, tool_call["approval"])
        self._record_hook(
            {
                "hook": "PreToolUse",
                "tool_name": tool_name,
                "decision": permission,
                "risk_level": TOOL_REGISTRY[tool_name].risk_level,
                "approval": tool_call["approval"],
            }
        )
        if tool_call["approval"].get("required") and not tool_call["approval"].get("approved", True):
            result = {
                "tool_name": tool_name,
                "status": "blocked",
                "error": tool_call["approval"].get("reason", "approval denied"),
            }
            tool_call["status"] = "blocked"
            tool_call["error"] = result["error"]
            tool_call["output"] = result
            tool_call["duration_ms"] = max(0.0, (perf_counter() - started) * 1000.0)
            self._record_hook(
                {
                    "hook": "PostToolUse",
                    "tool_name": tool_name,
                    "status": tool_call["status"],
                    "duration_ms": tool_call["duration_ms"],
                    "error": tool_call.get("error"),
                }
            )
            return result, tool_call
        try:
            validated_input, repaired = self._validate_or_repair_tool_input(tool_name, raw_input)
            tool_call["input"] = validated_input
            if repaired:
                tool_call["recovered"] = True
                tool_call["recovery_strategy"] = "tool_input_repair"
            result, attempts = self._run_tool_with_retry(
                tool_name,
                validated_input,
                runner,
                fallback_runner=fallback_runner,
            )
            tool_call["attempts"] = attempts
            tool_call["output"] = result
            return result, tool_call
        except Exception as exc:
            result = {
                "tool_name": tool_name,
                "status": "error",
                "error": str(exc),
            }
            tool_call["status"] = "error"
            tool_call["error"] = str(exc)
            tool_call["output"] = result
            return result, tool_call
        finally:
            tool_call["duration_ms"] = max(0.0, (perf_counter() - started) * 1000.0)
            self._record_hook(
                {
                    "hook": "PostToolUse",
                    "tool_name": tool_name,
                    "status": tool_call["status"],
                    "duration_ms": tool_call["duration_ms"],
                    "error": tool_call.get("error"),
                }
            )

    def _resolve_approval(
        self,
        tool_name: str,
        raw_input: dict[str, Any],
        approval: dict[str, Any],
    ) -> dict[str, Any]:
        if not approval.get("required"):
            return {**approval, "approved": True, "mode": "automatic"}
        request = {
            "tool_name": tool_name,
            "risk_level": TOOL_REGISTRY[tool_name].risk_level,
            "requires_policy_boundary": TOOL_REGISTRY[tool_name].requires_policy_boundary,
            "input": raw_input,
        }
        if self.approval_handler is None:
            return {
                **approval,
                "approved": True,
                "mode": "policy_boundary",
                "interactive": False,
            }
        decision = self.approval_handler(request)
        return {
            **approval,
            "approved": bool(decision.get("approved", False)),
            "decision": str(decision.get("decision", approval.get("decision", "approval_required"))),
            "reason": str(decision.get("reason", approval.get("reason", ""))),
            "mode": str(decision.get("mode", "human_in_loop")),
            "interactive": True,
        }

    def _validate_or_repair_tool_input(
        self,
        tool_name: str,
        raw_input: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        try:
            return validate_tool_input(tool_name, raw_input), False
        except Exception as exc:
            repaired = self._repair_tool_input(tool_name, raw_input)
            if repaired == raw_input:
                raise
            try:
                validated = validate_tool_input(tool_name, repaired)
            except Exception:
                self._record_recovery(
                    {
                        "strategy": "tool_input_repair",
                        "status": "failed",
                        "tool_name": tool_name,
                        "error": str(exc),
                    }
                )
                raise
            self._record_recovery(
                {
                    "strategy": "tool_input_repair",
                    "status": "success",
                    "tool_name": tool_name,
                    "error": str(exc),
                    "changes": _input_changes(raw_input, repaired),
                }
            )
            return validated, True

    def _repair_tool_input(self, tool_name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
        repaired = dict(raw_input)
        if tool_name in {"zone_hotspot_rank", "rag_retrieval"} and repaired.get("top_k") is None:
            repaired["top_k"] = 3
        if tool_name in {"query_metric", "compare_period", "plot_metric_trend", "detect_anomaly"}:
            if not repaired.get("metric_name"):
                repaired["metric_name"] = _select_metric_name("", self.trajectory, None)
        if tool_name == "data_quality_check" and not repaired.get("required_fields"):
            repaired["required_fields"] = _required_trajectory_fields(self.trajectory)
        return repaired

    def _run_tool_with_retry(
        self,
        tool_name: str,
        validated_input: dict[str, Any],
        runner: Callable[[dict[str, Any]], dict[str, Any]],
        fallback_runner: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], int]:
        try:
            return runner(validated_input), 1
        except Exception as first_exc:
            try:
                result = runner(validated_input)
            except Exception as second_exc:
                self._record_recovery(
                    {
                        "strategy": "tool_retry",
                        "status": "failed",
                        "tool_name": tool_name,
                        "attempts": 2,
                        "error": str(second_exc),
                    }
                )
                if fallback_runner is not None:
                    result = fallback_runner(validated_input)
                    result["fallback_error"] = str(second_exc)
                    self._record_recovery(
                        {
                            "strategy": "policy_fallback",
                            "status": "success",
                            "tool_name": tool_name,
                            "fallback_tool": "rule_based_policy",
                            "error": str(second_exc),
                        }
                    )
                    return result, 2
                raise first_exc from second_exc
            self._record_recovery(
                {
                    "strategy": "tool_retry",
                    "status": "success",
                    "tool_name": tool_name,
                    "attempts": 2,
                    "error": str(first_exc),
                }
            )
            return result, 2

    def _record_hook(self, event: dict[str, Any]) -> None:
        if self.runtime_trace is not None:
            self.runtime_trace.record_hook(event)

    def _record_recovery(self, event: dict[str, Any]) -> None:
        if self.runtime_trace is not None:
            self.runtime_trace.record_recovery(event)


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


def _policy_tool_name_from_dump(policy_result: dict[str, Any]) -> str:
    policy_name = str(policy_result.get("policy_name", "policy_runner"))
    if policy_name == "rule_based":
        return "rule_based_policy"
    return policy_name


def _is_successful_policy_result(
    policy_result: dict[str, Any],
    tool_call: dict[str, Any],
) -> bool:
    return (
        tool_call.get("status") == "success"
        and "policy_name" in policy_result
        and "recommended_action" in policy_result
    )


def _permission_decision(tool_name: str) -> str:
    spec = TOOL_REGISTRY[tool_name]
    if spec.risk_level == "control_boundary" or spec.requires_policy_boundary:
        return "policy_boundary"
    return "allow"


def _approval_decision(tool_name: str, permission: str) -> dict[str, Any]:
    spec = TOOL_REGISTRY[tool_name]
    required = spec.risk_level == "control_boundary" or spec.requires_policy_boundary
    return {
        "required": required,
        "decision": permission if required else "not_required",
        "reason": (
            "control_boundary tool must pass policy boundary approval"
            if required
            else "read_only/advisory tool auto-approved"
        ),
    }


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
        "data_quality_check",
        "zone_hotspot_rank",
        "control_action_audit",
        "cooling_efficiency_summary",
    }:
        return planned_tool
    normalized = question.lower()
    if any(token in normalized for token in ["趋势", "trend", "折线图", "序列", "画"]):
        return "plot_metric_trend"
    if any(token in normalized for token in ["比较", "对比", "前后", "变化", "compare"]):
        return "compare_period"
    if _is_data_quality_request(normalized):
        return "data_quality_check"
    if _is_hotspot_request(normalized):
        return "zone_hotspot_rank"
    if _is_control_action_audit_request(normalized):
        return "control_action_audit"
    if _is_cooling_efficiency_request(normalized):
        return "cooling_efficiency_summary"
    if any(token in normalized for token in ["构成", "breakdown", "能耗字段", "能耗"]):
        return "compute_energy_breakdown"
    return "query_metric"


def _select_anomaly_tool(question: str, step_spec: Any | None = None) -> str:
    planned_tool = _step_attr(step_spec, "tool")
    if planned_tool in {
        "detect_anomaly",
        "comfort_risk_assessment",
        "data_quality_check",
        "zone_hotspot_rank",
    }:
        return planned_tool
    normalized = question.lower()
    if _is_comfort_risk_request(normalized):
        return "comfort_risk_assessment"
    if _is_data_quality_request(normalized):
        return "data_quality_check"
    if _is_hotspot_request(normalized):
        return "zone_hotspot_rank"
    return "detect_anomaly"


def _is_data_quality_request(normalized: str) -> bool:
    return any(token in normalized for token in ["数据质量", "缺失", "quality", "missing", "null", "gap", "schema"])


def _is_hotspot_request(normalized: str) -> bool:
    return any(token in normalized for token in ["热点", "最热", "hotspot", "hottest", "top", "rank"])


def _is_control_action_audit_request(normalized: str) -> bool:
    has_control_action = any(token in normalized for token in ["控制动作", "control_action", "control action"])
    has_audit_intent = any(token in normalized for token in ["震荡", "抖动", "oscillat", "audit", "changing too fast"])
    return has_audit_intent and (has_control_action or "action" in normalized)


def _is_cooling_efficiency_request(normalized: str) -> bool:
    return any(token in normalized for token in ["能效", "效率", "efficiency", "efficient", "power per", "cooling efficiency"])


def _is_comfort_risk_request(normalized: str) -> bool:
    return any(token in normalized for token in ["舒适", "风险", "过热", "comfort", "overheat", "overheating", "thermal risk"])


def _required_trajectory_fields(trajectory: pd.DataFrame) -> list[str]:
    required = ["timestamp", "scenario_id", "zone_id", "zone_temperature"]
    if any(metric in trajectory.columns for metric in ["hvac_power", "cooling_power", "fan_power"]):
        required.append("hvac_power" if "hvac_power" in trajectory.columns else "cooling_power")
    return required


def _select_control_action_metric(trajectory: pd.DataFrame, metric_name: str) -> str:
    if "control_action" in trajectory.columns:
        return "control_action"
    return metric_name


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


def _input_changes(before: dict[str, Any], after: dict[str, Any]) -> dict[str, dict[str, Any]]:
    changes = {}
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            changes[key] = {"before": before.get(key), "after": after.get(key)}
    return changes


def _step_attr(step_spec: Any | None, key: str) -> Any:
    if step_spec is None:
        return None
    if isinstance(step_spec, dict):
        return step_spec.get(key)
    return getattr(step_spec, key, None)
