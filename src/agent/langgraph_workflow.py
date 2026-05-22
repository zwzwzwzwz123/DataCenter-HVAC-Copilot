from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agent.orchestrator import BaselineOrchestrator
from src.agent.router import route_task


class WorkflowState(TypedDict, total=False):
    question: str
    task_type: str | None
    route: str
    route_reason: str
    result: dict[str, Any]
    workflow_trace: list[dict[str, Any]]


class LangGraphOrchestrator:
    """LangGraph workflow wrapper that preserves the deterministic baseline outputs."""

    def __init__(self, baseline: BaselineOrchestrator) -> None:
        self.baseline = baseline
        self.graph = self._build_graph()

    def run(self, question: str, task_type: str | None = None) -> dict[str, Any]:
        state = self.graph.invoke(
            {
                "question": question,
                "task_type": task_type,
                "workflow_trace": [],
            }
        )
        result = dict(state["result"])
        result["workflow_engine"] = "langgraph"
        result["workflow_trace"] = state["workflow_trace"]
        return result

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("intent_classifier", self._intent_classifier)
        graph.add_node("retrieval", self._run_document_qa)
        graph.add_node("timeseries_tool", self._run_timeseries_query)
        graph.add_node("anomaly_tool", self._run_anomaly_diagnosis)
        graph.add_node("policy_tool", self._run_policy_recommendation)
        graph.add_node("evidence_aggregator", self._evidence_aggregator)
        graph.add_node("answer_audit", self._answer_audit)

        graph.set_entry_point("intent_classifier")
        graph.add_conditional_edges(
            "intent_classifier",
            self._select_route_node,
            {
                "retrieval": "retrieval",
                "timeseries_tool": "timeseries_tool",
                "anomaly_tool": "anomaly_tool",
                "policy_tool": "policy_tool",
            },
        )
        for node in ["retrieval", "timeseries_tool", "anomaly_tool", "policy_tool"]:
            graph.add_edge(node, "evidence_aggregator")
        graph.add_edge("evidence_aggregator", "answer_audit")
        graph.add_edge("answer_audit", END)
        return graph.compile()

    def _intent_classifier(self, state: WorkflowState) -> WorkflowState:
        decision = route_task(state["question"], task_type=state.get("task_type"))
        trace = _append_trace(
            state,
            {
                "node": "intent_classifier",
                "route": decision.route,
                "reason": decision.reason,
            },
        )
        return {
            **state,
            "route": decision.route,
            "route_reason": decision.reason,
            "workflow_trace": trace,
        }

    def _select_route_node(self, state: WorkflowState) -> str:
        route = state["route"]
        return {
            "document_qa": "retrieval",
            "timeseries_query": "timeseries_tool",
            "anomaly_diagnosis": "anomaly_tool",
            "policy_recommendation": "policy_tool",
        }[route]

    def _run_document_qa(self, state: WorkflowState) -> WorkflowState:
        result = self.baseline._run_document_qa(state["question"], state["route_reason"])
        return _with_result_and_trace(
            state,
            result,
            {
                "node": "retrieval",
                "citation_count": len(result.get("citations", [])),
                "context_count": len(result.get("retrieved_contexts", [])),
            },
        )

    def _run_timeseries_query(self, state: WorkflowState) -> WorkflowState:
        result = self.baseline._run_timeseries_query(state["question"], state["route_reason"])
        return _with_result_and_trace(
            state,
            result,
            {
                "node": "timeseries_tool",
                "tools": result.get("tools", []),
                "tool_result_count": len(result.get("tool_results", [])),
            },
        )

    def _run_anomaly_diagnosis(self, state: WorkflowState) -> WorkflowState:
        result = self.baseline._run_anomaly_diagnosis(state["question"], state["route_reason"])
        return _with_result_and_trace(
            state,
            result,
            {
                "node": "anomaly_tool",
                "tools": result.get("tools", []),
                "tool_result_count": len(result.get("tool_results", [])),
            },
        )

    def _run_policy_recommendation(self, state: WorkflowState) -> WorkflowState:
        result = self.baseline._run_policy_recommendation(
            state["question"],
            state["route_reason"],
        )
        return _with_result_and_trace(
            state,
            result,
            {
                "node": "policy_tool",
                "tools": result.get("tools", []),
                "policy_name": result.get("policy_result", {}).get("policy_name"),
            },
        )

    def _evidence_aggregator(self, state: WorkflowState) -> WorkflowState:
        result = state["result"]
        return {
            **state,
            "workflow_trace": _append_trace(
                state,
                {
                    "node": "evidence_aggregator",
                    "citation_count": len(result.get("citations", [])),
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
                    "passed": bool(audit.get("passed", False)),
                    "violations": audit.get("violations", []),
                },
            ),
        }


def _append_trace(state: WorkflowState, item: dict[str, Any]) -> list[dict[str, Any]]:
    return [*state.get("workflow_trace", []), item]


def _with_result_and_trace(
    state: WorkflowState,
    result: dict[str, Any],
    trace_item: dict[str, Any],
) -> WorkflowState:
    return {
        **state,
        "result": result,
        "workflow_trace": _append_trace(state, trace_item),
    }
