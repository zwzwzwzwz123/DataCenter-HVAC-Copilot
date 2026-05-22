from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agent.executor import AgentTaskExecutor
from src.agent.router import route_task


@dataclass(frozen=True)
class ReActStep:
    step: int
    thought: str
    action: str
    route: str
    observation: dict[str, Any]


class DeterministicReActPlanner:
    """Small deterministic planner that can produce two-step traces."""

    def next_action(
        self,
        question: str,
        task_type: str | None,
        trace: list[ReActStep],
    ) -> tuple[str | None, str]:
        primary = route_task(question, task_type=task_type)
        if not trace:
            if primary.route == "policy_recommendation" and _needs_context_first(question):
                return "timeseries_query", "先收集时序证据，再给出策略建议。"
            return primary.route, f"直接执行 {primary.route} 路由。"

        last_route = trace[-1].route
        if last_route == "timeseries_query" and primary.route == "policy_recommendation":
            return "policy_recommendation", "已经拿到时序上下文，给出策略建议。"
        if last_route == "anomaly_diagnosis" and primary.route == "policy_recommendation":
            return "policy_recommendation", "先确认异常，再转换成策略建议。"
        return None, "证据已经足够，停止多步推理。"


class ReActOrchestrator:
    """Minimal ReAct baseline that can chain one evidence step before answering."""

    def __init__(
        self,
        baseline: Any,
        planner: DeterministicReActPlanner | None = None,
        max_steps: int = 3,
    ) -> None:
        self.baseline = baseline
        self.task_executor: AgentTaskExecutor = baseline.task_executor
        self.planner = planner or DeterministicReActPlanner()
        self.max_steps = max_steps

    def run(self, question: str, task_type: str | None = None) -> dict[str, Any]:
        trace: list[ReActStep] = []
        result: dict[str, Any] | None = None
        route: str | None = None

        for step_index in range(self.max_steps):
            action, thought = self.planner.next_action(question, task_type, trace)
            if action is None:
                break
            route = action
            result = self._execute_action(action, question, trace)
            trace.append(
                ReActStep(
                    step=step_index + 1,
                    thought=thought,
                    action=action,
                    route=route,
                    observation=_summarize_observation(result),
                )
            )
            if not _should_continue(question, task_type, trace):
                break

        if result is None:
            fallback = route_task(question, task_type=task_type)
            route = fallback.route
            result = self._execute_action(fallback.route, question, trace)
            trace.append(
                ReActStep(
                    step=1,
                    thought="Planner fallback executed the default route.",
                    action=fallback.route,
                    route=fallback.route,
                    observation=_summarize_observation(result),
                )
            )

        final = dict(result)
        final = _merge_trace_evidence(final, trace)
        final["workflow_engine"] = "react"
        final["workflow_trace"] = [_public_step(step) for step in trace]
        final["react_trace"] = [_public_step(step) for step in trace]
        if route is not None:
            final["route"] = route
        return final

    def _execute_action(
        self,
        action: str,
        question: str,
        trace: list[ReActStep],
    ) -> dict[str, Any]:
        if action == "document_qa":
            return self.task_executor.run_document_qa(question, "react_planner")
        if action == "timeseries_query":
            if trace and trace[-1].route == "timeseries_query":
                return self.task_executor.run_timeseries_query(
                    _augment_policy_question(question),
                    "react_planner",
                )
            return self.task_executor.run_timeseries_query(question, "react_planner")
        if action == "anomaly_diagnosis":
            return self.task_executor.run_anomaly_diagnosis(question, "react_planner")
        if action == "policy_recommendation":
            return self.task_executor.run_policy_recommendation(question, "react_planner")
        raise ValueError(f"Unsupported ReAct action: {action}")


def _needs_context_first(question: str) -> bool:
    lowered = question.lower()
    keywords = [
        "最近",
        "趋势",
        "trend",
        "温度",
        "功率",
        "最大值",
        "最小值",
        "平均",
        "小时",
        "zone",
        "episode",
        "当前",
        "变化",
    ]
    return any(keyword in lowered or keyword in question for keyword in keywords)


def _should_continue(question: str, task_type: str | None, trace: list[ReActStep]) -> bool:
    if not trace:
        return False
    primary = route_task(question, task_type=task_type).route
    return primary == "policy_recommendation" and trace[-1].route == "timeseries_query"


def _augment_policy_question(question: str) -> str:
    return f"{question} 请结合最新时序上下文再做一次策略建议。"


def _summarize_observation(result: dict[str, Any]) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "route": result.get("route"),
        "answer_generator": result.get("answer_generator"),
        "citation_count": len(result.get("citations", [])),
        "tool_count": len(result.get("tools", [])),
        "tool_names": result.get("tools", []),
    }
    if result.get("tool_results"):
        observation["tool_result_keys"] = [
            sorted(tool_result.keys())[:6]
            for tool_result in result.get("tool_results", [])
            if isinstance(tool_result, dict)
        ]
    if result.get("policy_result"):
        policy_result = result["policy_result"]
        observation["policy_name"] = policy_result.get("policy_name")
        observation["recommended_action"] = policy_result.get("recommended_action")
    observation["_raw_result"] = result
    return observation


def _merge_trace_evidence(final: dict[str, Any], trace: list[ReActStep]) -> dict[str, Any]:
    merged_tools: list[str] = []
    merged_tool_results: list[dict[str, Any]] = []
    for step in trace:
        result = step.observation.get("_raw_result")
        if not isinstance(result, dict):
            continue
        for tool in result.get("tools", []):
            if tool not in merged_tools:
                merged_tools.append(tool)
        for tool_result in result.get("tool_results", []):
            if isinstance(tool_result, dict):
                merged_tool_results.append(tool_result)

    if not merged_tools:
        return final

    final["tools"] = merged_tools
    final["tool_results"] = merged_tool_results
    final["answer"] = _append_trace_evidence_to_answer(str(final.get("answer", "")), trace)
    return final


def _append_trace_evidence_to_answer(answer: str, trace: list[ReActStep]) -> str:
    evidence_lines = ["ReAct trace evidence:"]
    for step in trace:
        tools = step.observation.get("tool_names", [])
        if tools:
            evidence_lines.append(f"- step_{step.step} {step.route}: tools={tools}")
    if len(evidence_lines) == 1:
        return answer
    return "\n".join([answer, *evidence_lines])


def _public_step(step: ReActStep) -> dict[str, Any]:
    observation = {
        key: value
        for key, value in step.observation.items()
        if key != "_raw_result"
    }
    return {
        "step": step.step,
        "thought": step.thought,
        "action": step.action,
        "route": step.route,
        "observation": observation,
    }
