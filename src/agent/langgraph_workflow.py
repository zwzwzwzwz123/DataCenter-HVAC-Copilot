from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agent.executor import AgentTaskExecutor
from src.agent.intent_classifier import IntentClassifier
from src.agent.orchestrator import BaselineOrchestrator
from src.agent.planner import DeterministicRoutePlanner, PlanDecision, PlanStep, RoutePlanner


class WorkflowState(TypedDict, total=False):
    question: str
    task_type: str | None
    plan: PlanDecision
    merged_evidence: dict[str, Any]
    result: dict[str, Any]
    step_results: list[dict[str, Any]]
    workflow_trace: list[dict[str, Any]]


class LangGraphOrchestrator:
    """LangGraph workflow with controlled multi-step route planning."""

    def __init__(
        self,
        baseline: BaselineOrchestrator,
        route_planner: RoutePlanner | None = None,
        task_executor: AgentTaskExecutor | None = None,
        intent_classifier: IntentClassifier | None = None,
    ) -> None:
        self.baseline = baseline
        self.task_executor = task_executor or baseline.task_executor
        self.route_planner = route_planner or _PlannerFromIntentClassifier(intent_classifier)
        self.graph = self._build_graph()

    def run(self, question: str, task_type: str | None = None) -> dict[str, Any]:
        state = self.graph.invoke(
            {
                "question": question,
                "task_type": task_type,
                "workflow_trace": [],
                "step_results": [],
            }
        )
        result = dict(state["result"])
        result["workflow_engine"] = "langgraph"
        result["workflow_trace"] = state["workflow_trace"]
        return result

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("planner", self._planner)
        graph.add_node("execute_plan_steps", self._execute_plan_steps)
        graph.add_node("evidence_aggregator", self._evidence_aggregator)
        graph.add_node("answer_generator", self._answer_generator)
        graph.add_node("answer_audit", self._answer_audit)

        graph.set_entry_point("planner")
        graph.add_edge("planner", "execute_plan_steps")
        graph.add_edge("execute_plan_steps", "evidence_aggregator")
        graph.add_edge("evidence_aggregator", "answer_generator")
        graph.add_edge("answer_generator", "answer_audit")
        graph.add_edge("answer_audit", END)
        return graph.compile()

    def _planner(self, state: WorkflowState) -> WorkflowState:
        decision = self.route_planner.plan(
            state["question"],
            task_type=state.get("task_type"),
        )
        trace = _append_trace(
            state,
            {
                "node": "planner",
                "planned_steps": [step.route for step in decision.steps],
                "planned_step_specs": [_plan_step_to_dict(step) for step in decision.steps],
                "planner": decision.planner,
                "confidence": decision.confidence,
                "fallback_used": decision.fallback_used,
                "route": decision.steps[-1].route,
            },
        )
        return {
            **state,
            "plan": decision,
            "workflow_trace": trace,
        }

    def _execute_plan_steps(self, state: WorkflowState) -> WorkflowState:
        step_results: list[dict[str, Any]] = []
        trace = state.get("workflow_trace", [])
        for index, step in enumerate(state["plan"].steps, start=1):
            result = self._execute_step(state["question"], step)
            step_results.append(result)
            trace = [
                *trace,
                {
                    "node": "execute_plan_step",
                    "step_index": index,
                    "route": step.route,
                    "reason": step.reason,
                    "tools": result.get("tools", []),
                    "citation_count": len(result.get("citations", [])),
                    "context_count": len(result.get("retrieved_contexts", [])),
                    "tool_result_count": len(result.get("tool_results", [])),
                    "has_policy_result": "policy_result" in result,
                },
            ]
        return {
            **state,
            "step_results": step_results,
            "workflow_trace": trace,
        }

    def _execute_step(self, question: str, step: PlanStep) -> dict[str, Any]:
        if step.route == "document_qa":
            return self.task_executor.collect_document_qa_evidence(question, step.reason, step)
        if step.route == "timeseries_query":
            return self.task_executor.collect_timeseries_query_evidence(question, step.reason, step)
        if step.route == "anomaly_diagnosis":
            return self.task_executor.collect_anomaly_diagnosis_evidence(question, step.reason, step)
        if step.route == "policy_recommendation":
            return self.task_executor.collect_policy_recommendation_evidence(question, step.reason, step)
        raise ValueError(f"Unsupported route: {step.route}")

    def _evidence_aggregator(self, state: WorkflowState) -> WorkflowState:
        merged_evidence = _merge_step_results(
            question=state["question"],
            plan=state["plan"],
            step_results=state["step_results"],
        )
        return {
            **state,
            "merged_evidence": merged_evidence,
            "workflow_trace": _append_trace(
                state,
                {
                    "node": "evidence_aggregator",
                    "route": merged_evidence.get("route"),
                    "citation_count": len(merged_evidence.get("citations", [])),
                    "context_count": len(merged_evidence.get("retrieved_contexts", [])),
                    "tool_result_count": len(merged_evidence.get("tool_results", [])),
                    "has_policy_result": "policy_result" in merged_evidence,
                    "evidence_count": (
                        len(merged_evidence.get("citations", []))
                        + len(merged_evidence.get("retrieved_contexts", []))
                        + len(merged_evidence.get("tool_results", []))
                    ),
                },
            ),
        }

    def _answer_generator(self, state: WorkflowState) -> WorkflowState:
        result = self.task_executor.generate_answer_from_evidence(state["merged_evidence"])
        return {
            **state,
            "result": result,
            "workflow_trace": _append_trace(
                state,
                {
                    "node": "answer_generator",
                    "route": result.get("route"),
                    "answer_generator": result.get("answer_generator"),
                    "citation_count": len(result.get("citations", [])),
                    "context_count": len(result.get("retrieved_contexts", [])),
                    "tool_result_count": len(result.get("tool_results", [])),
                    "has_policy_result": "policy_result" in result,
                },
            ),
        }

    def _answer_audit(self, state: WorkflowState) -> WorkflowState:
        result = state["result"]
        audit = result.get("answer_audit", {})
        return {
            **state,
            "workflow_trace": _append_trace(
                state,
                {
                    "node": "answer_audit",
                    "route": result.get("route"),
                    "passed": bool(audit.get("passed", False)),
                    "audit_passed": bool(audit.get("passed", False)),
                    "violations": audit.get("violations", []),
                    "citation_count": len(result.get("citations", [])),
                    "tool_result_count": len(result.get("tool_results", [])),
                },
            ),
        }


class _PlannerFromIntentClassifier:
    """Compatibility wrapper for callers still injecting an intent classifier."""

    def __init__(self, intent_classifier: IntentClassifier | None = None) -> None:
        self.intent_classifier = intent_classifier
        self.fallback = DeterministicRoutePlanner()

    def plan(self, question: str, task_type: str | None = None) -> PlanDecision:
        if self.intent_classifier is None:
            return self.fallback.plan(question, task_type=task_type)

        decision = self.intent_classifier.classify(question, task_type=task_type)
        return PlanDecision(
            steps=[PlanStep(route=decision.route, reason=decision.reason)],
            planner=decision.classifier,
            confidence=decision.confidence,
            fallback_used=decision.fallback_used,
        )


def _merge_step_results(
    *,
    question: str,
    plan: PlanDecision,
    step_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if not step_results:
        raise ValueError("cannot merge empty plan results")

    final = dict(step_results[-1])
    final["question"] = question
    final["route"] = plan.steps[-1].route
    final["route_reason"] = " | ".join(f"{step.route}: {step.reason}" for step in plan.steps)
    final["planned_steps"] = [
        _plan_step_to_dict(step)
        for step in plan.steps
    ]
    final["planner"] = plan.planner
    final["planner_fallback_used"] = plan.fallback_used
    final["tools"] = _flatten_list(step_results, "tools")
    final["tool_results"] = _flatten_list(step_results, "tool_results")
    final["citations"] = _dedupe_dicts(_flatten_list(step_results, "citations"))
    final["retrieved_contexts"] = _dedupe_dicts(_flatten_list(step_results, "retrieved_contexts"))

    for result in reversed(step_results):
        if "policy_result" in result:
            final["policy_result"] = result["policy_result"]
            break

    return final


def _flatten_list(results: list[dict[str, Any]], key: str) -> list[Any]:
    flattened: list[Any] = []
    for result in results:
        flattened.extend(result.get(key, []))
    return flattened


def _dedupe_dicts(items: list[Any]) -> list[Any]:
    seen: set[str] = set()
    deduped: list[Any] = []
    for item in items:
        marker = repr(item)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(item)
    return deduped


def _plan_step_to_dict(step: PlanStep) -> dict[str, Any]:
    return {
        "route": step.route,
        "reason": step.reason,
        **({"tool": step.tool} if step.tool else {}),
        **({"metric_name": step.metric_name} if step.metric_name else {}),
        **({"zone_id": step.zone_id} if step.zone_id else {}),
        **({"time_window": step.time_window} if step.time_window else {}),
    }


def _append_trace(state: WorkflowState, item: dict[str, Any]) -> list[dict[str, Any]]:
    return [*state.get("workflow_trace", []), item]
