from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import pandas as pd

from src.agent.deepseek_generator import Transport
from src.agent.executor import AgentTaskExecutor
from src.agent.langgraph_workflow import _merge_step_results, _plan_step_to_dict
from src.agent.orchestrator import BaselineOrchestrator
from src.agent.planner import (
    DeterministicRoutePlanner,
    PlanDecision,
    PlanStep,
    RoutePlanner,
    build_route_planner_from_env,
    validate_plan_steps,
)
from src.agent.runtime import AgentRuntimeTrace
from src.core.env import load_env_file
from src.tools.registry import build_planner_tool_prompt

MAX_REACT_STEPS = 5
ReActAction = Literal[
    "continue_next_step",
    "insert_step",
    "replace_next_step",
    "stop_and_answer",
    "stop_blocked",
]
ReActBatchAction = Literal["plan_batch", "stop_and_answer", "stop_blocked"]


@dataclass(frozen=True)
class AgentObservation:
    step_index: int
    route: str
    tool_names: list[str]
    status: str
    citation_count: int
    context_count: int
    tool_result_count: int
    key_findings: list[str]
    has_policy_result: bool
    blocked: bool = False
    step_signature: tuple[Any, ...] = ("", "", "", "", "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "route": self.route,
            "tool_names": self.tool_names,
            "status": self.status,
            "citation_count": self.citation_count,
            "context_count": self.context_count,
            "tool_result_count": self.tool_result_count,
            "key_findings": self.key_findings,
            "has_policy_result": self.has_policy_result,
            "blocked": self.blocked,
            "step_signature": list(self.step_signature),
        }


@dataclass(frozen=True)
class ReActDecision:
    action: ReActAction
    reason: str
    step: PlanStep | None = None
    confidence: float = 0.5
    controller: str = "deterministic"
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "confidence": self.confidence,
            "controller": self.controller,
            "fallback_used": self.fallback_used,
            **({"step": _plan_step_to_dict(self.step)} if self.step else {}),
        }


@dataclass(frozen=True)
class ReActBatchDecision:
    action: ReActBatchAction
    reason: str
    steps: list[PlanStep] | None = None
    confidence: float = 0.5
    controller: str = "deterministic_batch"
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "confidence": self.confidence,
            "controller": self.controller,
            "fallback_used": self.fallback_used,
            "steps": [_plan_step_to_dict(step) for step in self.steps or []],
        }


class ReActController(Protocol):
    def decide(
        self,
        *,
        question: str,
        task_type: str | None,
        original_plan: list[PlanStep],
        pending_steps: list[PlanStep],
        observations: list[AgentObservation],
        remaining_steps: int,
        conversation_context: dict[str, Any] | None = None,
    ) -> ReActDecision:
        """Choose the next bounded ReAct action."""


class BatchReActController(Protocol):
    def decide_batch(
        self,
        *,
        question: str,
        task_type: str | None,
        original_plan: list[PlanStep],
        pending_steps: list[PlanStep],
        observations: list[AgentObservation],
        remaining_steps: int,
        evidence_bundle: dict[str, Any],
        conversation_context: dict[str, Any] | None = None,
    ) -> ReActBatchDecision:
        """Choose the next bounded plan-execute-reflect batch action."""


class DeterministicReActController:
    name = "deterministic_react_guard"

    def decide(
        self,
        *,
        question: str,
        task_type: str | None,
        original_plan: list[PlanStep],
        pending_steps: list[PlanStep],
        observations: list[AgentObservation],
        remaining_steps: int,
        conversation_context: dict[str, Any] | None = None,
    ) -> ReActDecision:
        if any(observation.blocked for observation in observations):
            return ReActDecision(
                action="stop_blocked",
                reason="A tool call was blocked by approval or policy boundary.",
                confidence=1.0,
                controller=self.name,
            )
        if remaining_steps <= 0:
            return ReActDecision(
                action="stop_and_answer",
                reason="Step budget exhausted.",
                confidence=1.0,
                controller=self.name,
            )
        if not observations and pending_steps:
            return ReActDecision(
                action="continue_next_step",
                reason="Execute the next validated plan step.",
                confidence=0.8,
                controller=self.name,
            )
        if pending_steps:
            return ReActDecision(
                action="continue_next_step",
                reason="Continue with the next validated plan step.",
                confidence=0.8,
                controller=self.name,
            )
        return ReActDecision(
            action="stop_and_answer",
            reason="No pending validated plan steps remain.",
            confidence=0.9,
            controller=self.name,
        )


class DeterministicBatchReActController:
    name = "deterministic_batch_react_guard"

    def decide_batch(
        self,
        *,
        question: str,
        task_type: str | None,
        original_plan: list[PlanStep],
        pending_steps: list[PlanStep],
        observations: list[AgentObservation],
        remaining_steps: int,
        evidence_bundle: dict[str, Any],
        conversation_context: dict[str, Any] | None = None,
    ) -> ReActBatchDecision:
        if any(observation.blocked for observation in observations):
            return ReActBatchDecision(
                action="stop_blocked",
                reason="A tool call was blocked by approval or policy boundary.",
                confidence=1.0,
                controller=self.name,
            )
        if observations or not pending_steps or remaining_steps <= 0:
            return ReActBatchDecision(
                action="stop_and_answer",
                reason="No additional evidence batch is required.",
                confidence=0.9,
                controller=self.name,
            )
        return ReActBatchDecision(
            action="plan_batch",
            reason="Execute the validated initial plan as one evidence batch.",
            steps=pending_steps[:remaining_steps],
            confidence=0.8,
            controller=self.name,
        )


class LLMBoundedReActController:
    """LLM controller constrained by local ReAct decision validation."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 20.0,
        fallback: ReActController | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.fallback = fallback or DeterministicReActController()
        self.transport = transport or _default_transport

    def decide(
        self,
        *,
        question: str,
        task_type: str | None,
        original_plan: list[PlanStep],
        pending_steps: list[PlanStep],
        observations: list[AgentObservation],
        remaining_steps: int,
        conversation_context: dict[str, Any] | None = None,
    ) -> ReActDecision:
        try:
            body = json.dumps(
                {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": _controller_system_prompt()},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "question": question,
                                    "task_type": task_type,
                                    "remaining_steps": remaining_steps,
                                    "original_plan": [_plan_step_to_dict(step) for step in original_plan],
                                    "pending_steps": [_plan_step_to_dict(step) for step in pending_steps],
                                    "observations": [item.to_dict() for item in observations],
                                    "conversation_context": conversation_context or {},
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    "temperature": 0.0,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            response = self.transport(
                f"{self.base_url}/chat/completions",
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                body,
                self.timeout_seconds,
            )
            content = str(response["choices"][0]["message"]["content"])
            decision = _decision_from_llm_payload(
                content=content,
                controller=f"llm:{self.provider}:{self.model}",
            )
            return _validated_decision(
                decision,
                pending_steps=pending_steps,
                observations=observations,
                remaining_steps=remaining_steps,
            )
        except Exception as exc:
            fallback = self.fallback.decide(
                question=question,
                task_type=task_type,
                original_plan=original_plan,
                pending_steps=pending_steps,
                observations=observations,
                remaining_steps=remaining_steps,
                conversation_context=conversation_context,
            )
            return ReActDecision(
                action=fallback.action,
                reason=f"LLM ReAct decision failed ({exc}); {fallback.reason}",
                step=fallback.step,
                confidence=fallback.confidence,
                controller=fallback.controller,
                fallback_used=True,
            )


class LLMBatchBoundedReActController:
    """LLM controller that plans evidence batches, then reflects on the merged bundle."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 20.0,
        fallback: BatchReActController | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.fallback = fallback or DeterministicBatchReActController()
        self.transport = transport or _default_transport

    def decide_batch(
        self,
        *,
        question: str,
        task_type: str | None,
        original_plan: list[PlanStep],
        pending_steps: list[PlanStep],
        observations: list[AgentObservation],
        remaining_steps: int,
        evidence_bundle: dict[str, Any],
        conversation_context: dict[str, Any] | None = None,
    ) -> ReActBatchDecision:
        try:
            body = json.dumps(
                {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": _batch_controller_system_prompt()},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "question": question,
                                    "task_type": task_type,
                                    "remaining_steps": remaining_steps,
                                    "original_plan": [_plan_step_to_dict(step) for step in original_plan],
                                    "pending_steps": [_plan_step_to_dict(step) for step in pending_steps],
                                    "observations": [item.to_dict() for item in observations],
                                    "evidence_bundle": _compact_evidence_bundle(evidence_bundle),
                                    "conversation_context": conversation_context or {},
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    "temperature": 0.0,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            response = self.transport(
                f"{self.base_url}/chat/completions",
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                body,
                self.timeout_seconds,
            )
            content = str(response["choices"][0]["message"]["content"])
            decision = _batch_decision_from_llm_payload(
                content=content,
                controller=f"llm_batch:{self.provider}:{self.model}",
            )
            return _validated_batch_decision(
                decision,
                pending_steps=pending_steps,
                observations=observations,
                remaining_steps=remaining_steps,
            )
        except Exception as exc:
            fallback = self.fallback.decide_batch(
                question=question,
                task_type=task_type,
                original_plan=original_plan,
                pending_steps=pending_steps,
                observations=observations,
                remaining_steps=remaining_steps,
                evidence_bundle=evidence_bundle,
                conversation_context=conversation_context,
            )
            return ReActBatchDecision(
                action=fallback.action,
                reason=f"LLM batch ReAct decision failed ({exc}); {fallback.reason}",
                steps=fallback.steps,
                confidence=fallback.confidence,
                controller=fallback.controller,
                fallback_used=True,
            )


class BoundedReActOrchestrator:
    """LLM-driven ReAct loop with hard local execution and safety bounds."""

    def __init__(
        self,
        baseline: BaselineOrchestrator,
        route_planner: RoutePlanner | None = None,
        controller: ReActController | None = None,
        task_executor: AgentTaskExecutor | None = None,
        max_steps: int = MAX_REACT_STEPS,
    ) -> None:
        self.baseline = baseline
        self.task_executor = task_executor or baseline.task_executor
        self.route_planner = route_planner or DeterministicRoutePlanner()
        self.controller = controller or DeterministicReActController()
        self.max_steps = min(max(1, max_steps), MAX_REACT_STEPS)

    def run(
        self,
        question: str,
        task_type: str | None = None,
        conversation_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        runtime_trace = AgentRuntimeTrace()
        plan = self.route_planner.plan(
            question,
            task_type=task_type,
            conversation_context=conversation_context,
        )
        pending_steps = list(plan.steps)
        original_plan = list(plan.steps)
        workflow_trace: list[dict[str, Any]] = [
            {
                "node": "planner",
                "planned_steps": [step.route for step in plan.steps],
                "planned_step_specs": [_plan_step_to_dict(step) for step in plan.steps],
                "planner": plan.planner,
                "confidence": plan.confidence,
                "fallback_used": plan.fallback_used,
                "route": plan.steps[-1].route,
                "memory_context_available": bool(conversation_context),
                "memory_recent_turn_count": len((conversation_context or {}).get("recent_turns", [])),
            }
        ]
        observations: list[AgentObservation] = []
        step_results: list[dict[str, Any]] = []
        executed_steps: list[PlanStep] = []
        required_policy_pending = any(step.route == "policy_recommendation" for step in original_plan)
        previous_runtime_trace = self.task_executor.runtime_trace
        self.task_executor.runtime_trace = runtime_trace

        try:
            while len(executed_steps) < self.max_steps:
                required_policy_pending = _policy_required_but_unmet(
                    required_policy_pending,
                    observations,
                )
                remaining_steps = self.max_steps - len(executed_steps)
                if not pending_steps:
                    if required_policy_pending:
                        pending_steps.append(_required_policy_step(original_plan))
                        runtime_trace.record_recovery(
                            {
                                "strategy": "react_decision_fallback",
                                "status": "success",
                                "error": "required policy step restored before no_pending stop",
                            }
                        )
                    else:
                        workflow_trace.append(
                            {
                                "node": "react_stop",
                                "reason": "no_pending_steps",
                                "executed_step_count": len(executed_steps),
                            }
                        )
                        break

                decision = self._safe_controller_decision(
                    question=question,
                    task_type=task_type,
                    original_plan=original_plan,
                    pending_steps=pending_steps,
                    observations=observations,
                    remaining_steps=remaining_steps,
                    conversation_context=conversation_context,
                    runtime_trace=runtime_trace,
                    required_policy_pending=required_policy_pending,
                )
                workflow_trace.append({"node": "react_controller", **decision.to_dict()})

                if decision.action == "stop_blocked":
                    workflow_trace.append(
                        {
                            "node": "react_stop",
                            "reason": "blocked",
                            "executed_step_count": len(executed_steps),
                        }
                    )
                    break

                if decision.action == "stop_and_answer":
                    if not _has_pending_required_policy(
                        pending_steps,
                        observations,
                        required_policy_pending=required_policy_pending,
                    ):
                        workflow_trace.append(
                            {
                                "node": "react_stop",
                                "reason": "controller_stop",
                                "executed_step_count": len(executed_steps),
                            }
                        )
                        break
                    runtime_trace.record_recovery(
                        {
                            "strategy": "react_decision_fallback",
                            "status": "success",
                            "error": "stop_and_answer rejected with pending policy step",
                        }
                    )
                    decision = ReActDecision(
                        action="continue_next_step",
                        reason=(
                            "Invalid ReAct decision (stop_and_answer rejected with pending "
                            "policy step); Continue with the next validated plan step."
                        ),
                        confidence=1.0,
                        controller="deterministic_react_guard",
                        fallback_used=True,
                    )
                    workflow_trace.append({"node": "react_controller", **decision.to_dict()})

                pending_steps = self._apply_decision(decision, pending_steps)
                if required_policy_pending and not any(
                    step.route == "policy_recommendation"
                    for step in pending_steps
                ):
                    pending_steps.append(_required_policy_step(original_plan))
                    runtime_trace.record_recovery(
                        {
                            "strategy": "react_decision_fallback",
                            "status": "success",
                            "error": "required policy step restored after ReAct decision",
                        }
                    )
                if required_policy_pending:
                    pending_steps = self._promote_required_policy_within_budget(
                        pending_steps=pending_steps,
                        remaining_steps=remaining_steps,
                        runtime_trace=runtime_trace,
                        workflow_trace=workflow_trace,
                        executed_step_count=len(executed_steps),
                    )
                step: PlanStep | None = pending_steps.pop(0)
                while step is not None and self._is_duplicate_step(step, observations):
                    self._record_blocked_guard_step(
                        step=step,
                        reason="duplicate_step_blocked",
                        runtime_trace=runtime_trace,
                        workflow_trace=workflow_trace,
                        executed_step_count=len(executed_steps),
                        append_stop=not pending_steps,
                    )
                    if pending_steps:
                        step = pending_steps.pop(0)
                        continue
                    step = None
                if step is None:
                    break

                self._execute_and_record_step(
                    question=question,
                    step=step,
                    runtime_trace=runtime_trace,
                    workflow_trace=workflow_trace,
                    step_results=step_results,
                    executed_steps=executed_steps,
                    observations=observations,
                )
                if observations[-1].blocked:
                    workflow_trace.append(
                        {
                            "node": "react_stop",
                            "reason": "blocked",
                            "executed_step_count": len(executed_steps),
                        }
                    )
                    break
            else:
                workflow_trace.append(
                    {
                        "node": "react_stop",
                        "reason": "max_steps_exhausted",
                        "executed_step_count": len(executed_steps),
                    }
                )

            if not step_results:
                fallback_step = original_plan[0]
                self._execute_and_record_step(
                    question=question,
                    step=fallback_step,
                    runtime_trace=runtime_trace,
                    workflow_trace=workflow_trace,
                    step_results=step_results,
                    executed_steps=executed_steps,
                    observations=observations,
                )
        finally:
            self.task_executor.runtime_trace = previous_runtime_trace

        merged_plan = PlanDecision(
            steps=executed_steps,
            planner=f"{plan.planner}+bounded_react",
            confidence=plan.confidence,
            fallback_used=plan.fallback_used,
        )
        merged_evidence = _merge_step_results(
            question=question,
            plan=merged_plan,
            step_results=step_results,
        )
        if conversation_context is not None:
            merged_evidence["conversation_context"] = conversation_context
        result = self.task_executor.generate_answer_from_evidence(merged_evidence)
        if any(observation.blocked for observation in observations):
            result["policy_result"] = None
        runtime_dict = runtime_trace.to_dict()
        return {
            **result,
            "workflow_engine": "bounded_react",
            "workflow_trace": [
                *workflow_trace,
                {
                    "node": "evidence_aggregator",
                    "route": result.get("route"),
                    "citation_count": len(result.get("citations", [])),
                    "context_count": len(result.get("retrieved_contexts", [])),
                    "tool_result_count": len(result.get("tool_results", [])),
                    "has_policy_result": isinstance(result.get("policy_result"), dict),
                },
                {
                    "node": "answer_generator",
                    "route": result.get("route"),
                    "answer_generator": result.get("answer_generator"),
                    "citation_count": len(result.get("citations", [])),
                    "context_count": len(result.get("retrieved_contexts", [])),
                    "tool_result_count": len(result.get("tool_results", [])),
                    "has_policy_result": isinstance(result.get("policy_result"), dict),
                },
                {
                    "node": "answer_audit",
                    "route": result.get("route"),
                    "passed": bool(result.get("answer_audit", {}).get("passed", False)),
                    "audit_passed": bool(result.get("answer_audit", {}).get("passed", False)),
                    "violations": result.get("answer_audit", {}).get("violations", []),
                    "citation_count": len(result.get("citations", [])),
                    "tool_result_count": len(result.get("tool_results", [])),
                },
            ],
            "react_trace": [
                {
                    "step": observation.step_index,
                    "action": step.route,
                    "route": step.route,
                    "observation": observation.to_dict(),
                }
                for step, observation in zip(executed_steps, observations)
            ],
            "todos": runtime_dict["todos"],
            "runtime_trace": runtime_dict,
        }

    def _safe_controller_decision(
        self,
        *,
        question: str,
        task_type: str | None,
        original_plan: list[PlanStep],
        pending_steps: list[PlanStep],
        observations: list[AgentObservation],
        remaining_steps: int,
        conversation_context: dict[str, Any] | None,
        runtime_trace: AgentRuntimeTrace,
        required_policy_pending: bool = False,
    ) -> ReActDecision:
        try:
            decision = self.controller.decide(
                question=question,
                task_type=task_type,
                original_plan=original_plan,
                pending_steps=pending_steps,
                observations=observations,
                remaining_steps=remaining_steps,
                conversation_context=conversation_context,
            )
            return _validated_decision(
                decision,
                pending_steps=pending_steps,
                observations=observations,
                remaining_steps=remaining_steps,
                required_policy_pending=required_policy_pending,
                duplicate_checker=self._is_duplicate_step,
            )
        except Exception as exc:
            runtime_trace.record_recovery(
                {
                    "strategy": "react_decision_fallback",
                    "status": "success",
                    "error": str(exc),
                }
            )
            fallback = DeterministicReActController().decide(
                question=question,
                task_type=task_type,
                original_plan=original_plan,
                pending_steps=pending_steps,
                observations=observations,
                remaining_steps=remaining_steps,
                conversation_context=conversation_context,
            )
            return ReActDecision(
                action=fallback.action,
                reason=f"Invalid ReAct decision ({exc}); {fallback.reason}",
                step=fallback.step,
                confidence=fallback.confidence,
                controller=fallback.controller,
                fallback_used=True,
            )

    def _apply_decision(
        self,
        decision: ReActDecision,
        pending_steps: list[PlanStep],
    ) -> list[PlanStep]:
        if decision.action == "continue_next_step":
            return pending_steps
        if decision.action == "insert_step" and decision.step is not None:
            return [decision.step, *pending_steps]
        if decision.action == "replace_next_step" and decision.step is not None:
            return [decision.step, *pending_steps[1:]]
        return pending_steps

    def _execute_and_record_step(
        self,
        *,
        question: str,
        step: PlanStep,
        runtime_trace: AgentRuntimeTrace,
        workflow_trace: list[dict[str, Any]],
        step_results: list[dict[str, Any]],
        executed_steps: list[PlanStep],
        observations: list[AgentObservation],
    ) -> None:
        step_index = len(executed_steps) + 1
        todo = runtime_trace.add_todo(step)
        runtime_trace.mark_todo(todo.step_index, "in_progress")
        result = self._execute_step(question, step)
        step_results.append(result)
        executed_steps.append(step)
        observation = self._observation_from_result(step_index, step, result)
        observations.append(observation)
        runtime_trace.mark_todo(todo.step_index, "blocked" if observation.blocked else "completed")
        workflow_trace.append(
            {
                "node": "execute_react_step",
                "step_index": step_index,
                "route": step.route,
                "reason": step.reason,
                "tools": result.get("tools", []),
                "citation_count": len(result.get("citations", [])),
                "context_count": len(result.get("retrieved_contexts", [])),
                "tool_result_count": len(result.get("tool_results", [])),
                "has_policy_result": isinstance(result.get("policy_result"), dict),
            }
        )
        workflow_trace.append(
            {
                "node": "react_observation",
                "step_index": step_index,
                "observation": observation.to_dict(),
            }
        )

    def _record_blocked_guard_step(
        self,
        *,
        step: PlanStep,
        reason: str,
        runtime_trace: AgentRuntimeTrace,
        workflow_trace: list[dict[str, Any]],
        executed_step_count: int,
        append_stop: bool = True,
    ) -> None:
        todo = runtime_trace.add_todo(step)
        runtime_trace.mark_todo(todo.step_index, "blocked")
        runtime_trace.record_recovery(
            {
                "strategy": "react_duplicate_step_blocked",
                "status": "success",
                "route": step.route,
                "tool": step.tool,
            }
        )
        workflow_trace.append(
            {
                "node": "react_guard_blocked",
                "reason": reason,
                "step": _plan_step_to_dict(step),
            }
        )
        if append_stop:
            workflow_trace.append(
                {
                    "node": "react_stop",
                    "reason": reason,
                    "executed_step_count": executed_step_count,
                }
            )

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

    def _promote_required_policy_within_budget(
        self,
        *,
        pending_steps: list[PlanStep],
        remaining_steps: int,
        runtime_trace: AgentRuntimeTrace,
        workflow_trace: list[dict[str, Any]],
        executed_step_count: int,
    ) -> list[PlanStep]:
        policy_index = _pending_policy_index(pending_steps)
        if policy_index is None or policy_index + 1 <= remaining_steps:
            return pending_steps

        skipped_steps = pending_steps[:policy_index]
        promoted_steps = pending_steps[policy_index:]
        for skipped_step in skipped_steps:
            todo = runtime_trace.add_todo(skipped_step)
            runtime_trace.mark_todo(todo.step_index, "blocked")
        runtime_trace.record_recovery(
            {
                "strategy": "react_policy_budget_guard",
                "status": "success",
                "skipped_step_count": len(skipped_steps),
                "remaining_steps": remaining_steps,
                "error": "required policy step promoted before step budget exhaustion",
            }
        )
        workflow_trace.append(
            {
                "node": "react_policy_budget_guard",
                "reason": "required_policy_would_exceed_budget",
                "skipped_steps": [_plan_step_to_dict(step) for step in skipped_steps],
                "executed_step_count": executed_step_count,
                "remaining_steps": remaining_steps,
            }
        )
        return promoted_steps

    def _is_duplicate_step(
        self,
        step: PlanStep,
        observations: list[AgentObservation],
    ) -> bool:
        candidate = self._step_signature(step)
        return any(observation.step_signature == candidate for observation in observations)

    def _observation_from_result(
        self,
        step_index: int,
        step: PlanStep,
        result: dict[str, Any],
    ) -> AgentObservation:
        return _observation_from_result(
            step_index,
            step,
            result,
            fallback_signature=self._step_signature(step),
        )

    def _step_signature(self, step: PlanStep) -> tuple[Any, ...]:
        return _step_signature(
            step,
            trajectory=getattr(self.task_executor, "trajectory", None),
            default_tool_inputs=getattr(self.task_executor, "default_tool_inputs", None),
        )


class BatchBoundedReActOrchestrator(BoundedReActOrchestrator):
    """Bounded plan-execute-reflect loop that executes tool batches before LLM reflection."""

    def __init__(
        self,
        baseline: BaselineOrchestrator,
        route_planner: RoutePlanner | None = None,
        controller: BatchReActController | None = None,
        task_executor: AgentTaskExecutor | None = None,
        max_steps: int = MAX_REACT_STEPS,
    ) -> None:
        super().__init__(
            baseline,
            route_planner=route_planner,
            controller=DeterministicReActController(),
            task_executor=task_executor,
            max_steps=max_steps,
        )
        self.batch_controller = controller or DeterministicBatchReActController()

    def run(
        self,
        question: str,
        task_type: str | None = None,
        conversation_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        runtime_trace = AgentRuntimeTrace()
        plan = self.route_planner.plan(
            question,
            task_type=task_type,
            conversation_context=conversation_context,
        )
        pending_steps = list(plan.steps)
        original_plan = list(plan.steps)
        workflow_trace: list[dict[str, Any]] = [
            {
                "node": "planner",
                "planned_steps": [step.route for step in plan.steps],
                "planned_step_specs": [_plan_step_to_dict(step) for step in plan.steps],
                "planner": plan.planner,
                "confidence": plan.confidence,
                "fallback_used": plan.fallback_used,
                "route": plan.steps[-1].route,
                "memory_context_available": bool(conversation_context),
                "memory_recent_turn_count": len((conversation_context or {}).get("recent_turns", [])),
            }
        ]
        observations: list[AgentObservation] = []
        step_results: list[dict[str, Any]] = []
        executed_steps: list[PlanStep] = []
        required_policy_pending = any(step.route == "policy_recommendation" for step in original_plan)
        previous_runtime_trace = self.task_executor.runtime_trace
        self.task_executor.runtime_trace = runtime_trace
        stop_reason = "max_steps_exhausted"

        try:
            while len(executed_steps) < self.max_steps:
                required_policy_pending = _policy_required_but_unmet(
                    required_policy_pending,
                    observations,
                )
                remaining_steps = self.max_steps - len(executed_steps)
                evidence_bundle = self._current_evidence_bundle(
                    question=question,
                    plan=plan,
                    step_results=step_results,
                    conversation_context=conversation_context,
                )
                if observations:
                    workflow_trace.append(
                        {
                            "node": "batch_reflection",
                            "observation_count": len(observations),
                            "citation_count": len(evidence_bundle.get("citations", [])),
                            "context_count": len(evidence_bundle.get("retrieved_contexts", [])),
                            "tool_result_count": len(evidence_bundle.get("tool_results", [])),
                            "has_policy_result": isinstance(evidence_bundle.get("policy_result"), dict),
                        }
                    )
                decision = self._safe_batch_controller_decision(
                    question=question,
                    task_type=task_type,
                    original_plan=original_plan,
                    pending_steps=pending_steps,
                    observations=observations,
                    remaining_steps=remaining_steps,
                    evidence_bundle=evidence_bundle,
                    conversation_context=conversation_context,
                    runtime_trace=runtime_trace,
                    required_policy_pending=required_policy_pending,
                )
                workflow_trace.append({"node": "batch_controller", **decision.to_dict()})

                if decision.action == "stop_blocked":
                    stop_reason = "blocked"
                    break
                if decision.action == "stop_and_answer":
                    if not _has_pending_required_policy(
                        pending_steps,
                        observations,
                        required_policy_pending=required_policy_pending,
                    ):
                        stop_reason = "controller_stop"
                        break
                    runtime_trace.record_recovery(
                        {
                            "strategy": "react_decision_fallback",
                            "status": "success",
                            "error": "stop_and_answer rejected with pending policy step",
                        }
                    )
                    runtime_trace.record_recovery(
                        {
                            "strategy": "react_policy_budget_guard",
                            "status": "success",
                            "error": "required policy step promoted after premature stop",
                            "remaining_steps": remaining_steps,
                        }
                    )
                    decision = ReActBatchDecision(
                        action="plan_batch",
                        reason="Pending policy step must be executed before answering.",
                        steps=[_required_policy_step(original_plan)],
                        confidence=1.0,
                        controller="deterministic_batch_react_guard",
                        fallback_used=True,
                    )
                    workflow_trace.append({"node": "batch_controller", **decision.to_dict()})

                batch_steps = list(decision.steps or pending_steps[:remaining_steps])
                batch_steps = self._validated_batch_steps(
                    batch_steps=batch_steps,
                    observations=observations,
                    remaining_steps=remaining_steps,
                    required_policy_pending=required_policy_pending,
                    runtime_trace=runtime_trace,
                )
                if not batch_steps:
                    stop_reason = "no_valid_batch_steps"
                    break
                pending_steps = _remove_batch_steps(pending_steps, batch_steps)
                if required_policy_pending and not any(
                    step.route == "policy_recommendation" for step in [*batch_steps, *pending_steps]
                ):
                    pending_steps.append(_required_policy_step(original_plan))
                    runtime_trace.record_recovery(
                        {
                            "strategy": "react_decision_fallback",
                            "status": "success",
                            "error": "required policy step restored after batch decision",
                        }
                    )
                batch_steps = self._promote_policy_inside_batch_if_needed(
                    batch_steps=batch_steps,
                    pending_steps=pending_steps,
                    remaining_steps=remaining_steps,
                    required_policy_pending=required_policy_pending,
                    runtime_trace=runtime_trace,
                )
                workflow_trace.append(
                    {
                        "node": "execute_batch",
                        "step_count": len(batch_steps),
                        "steps": [_plan_step_to_dict(step) for step in batch_steps],
                    }
                )
                for step in batch_steps:
                    if len(executed_steps) >= self.max_steps:
                        stop_reason = "max_steps_exhausted"
                        break
                    if self._is_duplicate_step(step, observations):
                        self._record_blocked_guard_step(
                            step=step,
                            reason="duplicate_step_blocked",
                            runtime_trace=runtime_trace,
                            workflow_trace=workflow_trace,
                            executed_step_count=len(executed_steps),
                            append_stop=False,
                        )
                        continue
                    self._execute_and_record_step(
                        question=question,
                        step=step,
                        runtime_trace=runtime_trace,
                        workflow_trace=workflow_trace,
                        step_results=step_results,
                        executed_steps=executed_steps,
                        observations=observations,
                    )
                    if observations[-1].blocked:
                        stop_reason = "blocked"
                        break
                if stop_reason == "blocked":
                    break
            if stop_reason == "max_steps_exhausted" and len(executed_steps) < self.max_steps:
                stop_reason = "loop_complete"
            workflow_trace.append(
                {
                    "node": "react_stop",
                    "reason": stop_reason,
                    "executed_step_count": len(executed_steps),
                }
            )
            if not step_results:
                fallback_step = original_plan[0]
                self._execute_and_record_step(
                    question=question,
                    step=fallback_step,
                    runtime_trace=runtime_trace,
                    workflow_trace=workflow_trace,
                    step_results=step_results,
                    executed_steps=executed_steps,
                    observations=observations,
                )
        finally:
            self.task_executor.runtime_trace = previous_runtime_trace

        merged_plan = PlanDecision(
            steps=executed_steps,
            planner=f"{plan.planner}+bounded_react_batch",
            confidence=plan.confidence,
            fallback_used=plan.fallback_used,
        )
        merged_evidence = _merge_step_results(
            question=question,
            plan=merged_plan,
            step_results=step_results,
        )
        if conversation_context is not None:
            merged_evidence["conversation_context"] = conversation_context
        result = self.task_executor.generate_answer_from_evidence(merged_evidence)
        if any(observation.blocked for observation in observations):
            result["policy_result"] = None
        runtime_dict = runtime_trace.to_dict()
        return {
            **result,
            "workflow_engine": "bounded_react_batch",
            "workflow_trace": [
                *workflow_trace,
                {
                    "node": "evidence_aggregator",
                    "route": result.get("route"),
                    "citation_count": len(result.get("citations", [])),
                    "context_count": len(result.get("retrieved_contexts", [])),
                    "tool_result_count": len(result.get("tool_results", [])),
                    "has_policy_result": isinstance(result.get("policy_result"), dict),
                },
                {
                    "node": "answer_generator",
                    "route": result.get("route"),
                    "answer_generator": result.get("answer_generator"),
                    "citation_count": len(result.get("citations", [])),
                    "context_count": len(result.get("retrieved_contexts", [])),
                    "tool_result_count": len(result.get("tool_results", [])),
                    "has_policy_result": isinstance(result.get("policy_result"), dict),
                },
                {
                    "node": "answer_audit",
                    "route": result.get("route"),
                    "passed": bool(result.get("answer_audit", {}).get("passed", False)),
                    "audit_passed": bool(result.get("answer_audit", {}).get("passed", False)),
                    "violations": result.get("answer_audit", {}).get("violations", []),
                    "citation_count": len(result.get("citations", [])),
                    "tool_result_count": len(result.get("tool_results", [])),
                },
            ],
            "react_trace": [
                {
                    "step": observation.step_index,
                    "action": step.route,
                    "route": step.route,
                    "observation": observation.to_dict(),
                }
                for step, observation in zip(executed_steps, observations)
            ],
            "todos": runtime_dict["todos"],
            "runtime_trace": runtime_dict,
        }

    def _safe_batch_controller_decision(
        self,
        *,
        question: str,
        task_type: str | None,
        original_plan: list[PlanStep],
        pending_steps: list[PlanStep],
        observations: list[AgentObservation],
        remaining_steps: int,
        evidence_bundle: dict[str, Any],
        conversation_context: dict[str, Any] | None,
        runtime_trace: AgentRuntimeTrace,
        required_policy_pending: bool = False,
    ) -> ReActBatchDecision:
        try:
            decision = self.batch_controller.decide_batch(
                question=question,
                task_type=task_type,
                original_plan=original_plan,
                pending_steps=pending_steps,
                observations=observations,
                remaining_steps=remaining_steps,
                evidence_bundle=evidence_bundle,
                conversation_context=conversation_context,
            )
            if decision.fallback_used:
                runtime_trace.record_recovery(
                    {
                        "strategy": "react_decision_fallback",
                        "status": "success",
                        "error": decision.reason,
                    }
                )
            return _validated_batch_decision(
                decision,
                pending_steps=pending_steps,
                observations=observations,
                remaining_steps=remaining_steps,
                required_policy_pending=required_policy_pending,
                duplicate_checker=self._is_duplicate_step,
            )
        except Exception as exc:
            runtime_trace.record_recovery(
                {
                    "strategy": "react_decision_fallback",
                    "status": "success",
                    "error": str(exc),
                }
            )
            fallback = DeterministicBatchReActController().decide_batch(
                question=question,
                task_type=task_type,
                original_plan=original_plan,
                pending_steps=pending_steps,
                observations=observations,
                remaining_steps=remaining_steps,
                evidence_bundle=evidence_bundle,
                conversation_context=conversation_context,
            )
            return ReActBatchDecision(
                action=fallback.action,
                reason=f"Invalid batch ReAct decision ({exc}); {fallback.reason}",
                steps=fallback.steps,
                confidence=fallback.confidence,
                controller=fallback.controller,
                fallback_used=True,
            )

    def _validated_batch_steps(
        self,
        *,
        batch_steps: list[PlanStep],
        observations: list[AgentObservation],
        remaining_steps: int,
        required_policy_pending: bool,
        runtime_trace: AgentRuntimeTrace,
    ) -> list[PlanStep]:
        limited_steps = batch_steps[:remaining_steps]
        try:
            validate_plan_steps(limited_steps)
            for step in limited_steps:
                _reject_duplicate_step(
                    step,
                    observations,
                    duplicate_checker=self._is_duplicate_step,
                )
        except Exception as exc:
            runtime_trace.record_recovery(
                {
                    "strategy": "react_decision_fallback",
                    "status": "success",
                    "error": str(exc),
                }
            )
            return []
        return limited_steps

    def _promote_policy_inside_batch_if_needed(
        self,
        *,
        batch_steps: list[PlanStep],
        pending_steps: list[PlanStep],
        remaining_steps: int,
        required_policy_pending: bool,
        runtime_trace: AgentRuntimeTrace,
    ) -> list[PlanStep]:
        if not required_policy_pending:
            return batch_steps
        if any(step.route == "policy_recommendation" for step in batch_steps):
            return batch_steps
        policy_step = _first_policy_step(pending_steps)
        if policy_step is None:
            return batch_steps
        if len(batch_steps) + 1 <= remaining_steps:
            return batch_steps
        runtime_trace.record_recovery(
            {
                "strategy": "react_policy_budget_guard",
                "status": "success",
                "error": "required policy step promoted into batch before budget exhaustion",
            }
        )
        if not batch_steps:
            return [policy_step]
        return [*batch_steps[:-1], policy_step]

    def _current_evidence_bundle(
        self,
        *,
        question: str,
        plan: PlanDecision,
        step_results: list[dict[str, Any]],
        conversation_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not step_results:
            return {
                "question": question,
                "route": plan.steps[-1].route,
                "citations": [],
                "retrieved_contexts": [],
                "tools": [],
                "tool_results": [],
                "tool_calls": [],
            }
        evidence = _merge_step_results(
            question=question,
            plan=PlanDecision(
                steps=plan.steps,
                planner=plan.planner,
                confidence=plan.confidence,
                fallback_used=plan.fallback_used,
            ),
            step_results=step_results,
        )
        if conversation_context is not None:
            evidence["conversation_context"] = conversation_context
        return evidence


def build_react_controller_from_env(
    project_root: str | Path | None = None,
    transport: Transport | None = None,
) -> ReActController:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    load_env_file(root / ".env")
    provider = os.getenv("BOUNDED_REACT_CONTROLLER_PROVIDER", "auto").strip().lower()
    if provider in {"", "auto"}:
        provider = "deepseek" if os.getenv("DEEPSEEK_API_KEY", "").strip() else "deterministic"
    if provider in {"deterministic", "rule_based"}:
        return DeterministicReActController()
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            return DeterministicReActController()
        return LLMBoundedReActController(
            provider="deepseek",
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv(
                "BOUNDED_REACT_CONTROLLER_MODEL",
                os.getenv("LANGGRAPH_PLANNER_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
            ),
            timeout_seconds=float(os.getenv("BOUNDED_REACT_TIMEOUT_SECONDS", "20")),
            transport=transport,
        )
    return DeterministicReActController()


def build_batch_react_controller_from_env(
    project_root: str | Path | None = None,
    transport: Transport | None = None,
) -> BatchReActController:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    load_env_file(root / ".env")
    provider = os.getenv("BOUNDED_REACT_CONTROLLER_PROVIDER", "auto").strip().lower()
    if provider in {"", "auto"}:
        provider = "deepseek" if os.getenv("DEEPSEEK_API_KEY", "").strip() else "deterministic"
    if provider in {"deterministic", "rule_based"}:
        return DeterministicBatchReActController()
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            return DeterministicBatchReActController()
        return LLMBatchBoundedReActController(
            provider="deepseek",
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv(
                "BOUNDED_REACT_CONTROLLER_MODEL",
                os.getenv("LANGGRAPH_PLANNER_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
            ),
            timeout_seconds=float(os.getenv("BOUNDED_REACT_TIMEOUT_SECONDS", "20")),
            transport=transport,
        )
    return DeterministicBatchReActController()


def build_bounded_react_orchestrator_from_env(
    baseline: BaselineOrchestrator,
    *,
    use_env_controller: bool = True,
) -> BoundedReActOrchestrator:
    route_planner = build_route_planner_from_env() if use_env_controller else DeterministicRoutePlanner()
    controller = build_react_controller_from_env() if use_env_controller else DeterministicReActController()
    return BoundedReActOrchestrator(
        baseline,
        route_planner=route_planner,
        controller=controller,
    )


def build_batch_bounded_react_orchestrator_from_env(
    baseline: BaselineOrchestrator,
    *,
    use_env_controller: bool = True,
) -> BatchBoundedReActOrchestrator:
    route_planner = build_route_planner_from_env() if use_env_controller else DeterministicRoutePlanner()
    controller = (
        build_batch_react_controller_from_env()
        if use_env_controller
        else DeterministicBatchReActController()
    )
    return BatchBoundedReActOrchestrator(
        baseline,
        route_planner=route_planner,
        controller=controller,
    )


def _validated_decision(
    decision: ReActDecision,
    *,
    pending_steps: list[PlanStep] | None = None,
    observations: list[AgentObservation],
    remaining_steps: int,
    required_policy_pending: bool = False,
    duplicate_checker: Callable[[PlanStep, list[AgentObservation]], bool] | None = None,
) -> ReActDecision:
    if decision.action not in {
        "continue_next_step",
        "insert_step",
        "replace_next_step",
        "stop_and_answer",
        "stop_blocked",
    }:
        raise ValueError(f"unsupported ReAct action: {decision.action}")
    if remaining_steps <= 0 and decision.action not in {"stop_and_answer", "stop_blocked"}:
        raise ValueError("ReAct step budget exhausted")
    if decision.action in {"insert_step", "replace_next_step"}:
        if decision.step is None:
            raise ValueError(f"{decision.action} requires a step")
        candidate_steps = _candidate_pending_steps(decision, pending_steps or [])
        validate_plan_steps(candidate_steps)
        if required_policy_pending and not any(
            step.route == "policy_recommendation"
            for step in candidate_steps
        ):
            raise ValueError("required policy step cannot be removed before policy execution")
        if (
            required_policy_pending
            and decision.action == "insert_step"
            and _required_policy_would_exceed_budget(candidate_steps, remaining_steps)
        ):
            raise ValueError("required policy budget would be starved by inserted evidence step")
        _reject_duplicate_step(
            decision.step,
            observations,
            duplicate_checker=duplicate_checker,
        )
    return decision


def _validated_batch_decision(
    decision: ReActBatchDecision,
    *,
    pending_steps: list[PlanStep] | None = None,
    observations: list[AgentObservation],
    remaining_steps: int,
    required_policy_pending: bool = False,
    duplicate_checker: Callable[[PlanStep, list[AgentObservation]], bool] | None = None,
) -> ReActBatchDecision:
    if decision.action not in {"plan_batch", "stop_and_answer", "stop_blocked"}:
        raise ValueError(f"unsupported batch ReAct action: {decision.action}")
    if remaining_steps <= 0 and decision.action == "plan_batch":
        raise ValueError("ReAct step budget exhausted")
    if decision.action == "plan_batch":
        steps = list(decision.steps or pending_steps or [])
        if not steps:
            raise ValueError("plan_batch requires at least one step")
        if len(steps) > remaining_steps:
            raise ValueError("plan_batch exceeds remaining step budget")
        validate_plan_steps(steps)
        if required_policy_pending and not any(
            step.route == "policy_recommendation" for step in [*steps, *(pending_steps or [])]
        ):
            raise ValueError("required policy step cannot be removed before policy execution")
        for step in steps:
            _reject_duplicate_step(
                step,
                observations,
                duplicate_checker=duplicate_checker,
            )
    return decision


def _candidate_pending_steps(
    decision: ReActDecision,
    pending_steps: list[PlanStep],
) -> list[PlanStep]:
    if decision.step is None:
        return pending_steps
    if decision.action == "insert_step":
        return [decision.step, *pending_steps]
    if decision.action == "replace_next_step":
        return [decision.step, *pending_steps[1:]]
    return pending_steps


def _remove_batch_steps(
    pending_steps: list[PlanStep],
    batch_steps: list[PlanStep],
) -> list[PlanStep]:
    remaining = list(pending_steps)
    for batch_step in batch_steps:
        for index, pending_step in enumerate(remaining):
            if _steps_equivalent(batch_step, pending_step):
                remaining.pop(index)
                break
    return remaining


def _steps_equivalent(left: PlanStep, right: PlanStep) -> bool:
    return _plan_step_to_dict(left) == _plan_step_to_dict(right)


def _reject_duplicate_step(
    step: PlanStep,
    observations: list[AgentObservation],
    *,
    duplicate_checker: Callable[[PlanStep, list[AgentObservation]], bool] | None = None,
) -> None:
    checker = duplicate_checker or _is_duplicate_step
    if checker(step, observations):
        raise ValueError(f"duplicate ReAct tool call blocked: {step.route}/{step.tool}")


def _is_duplicate_step(step: PlanStep, observations: list[AgentObservation]) -> bool:
    candidate = _step_signature(step)
    return any(observation.step_signature == candidate for observation in observations)


def _step_signature(
    step: PlanStep,
    *,
    trajectory: pd.DataFrame | None = None,
    default_tool_inputs: dict[str, dict[str, Any]] | None = None,
) -> tuple[Any, ...]:
    normalized = _canonical_step(step)
    if normalized.route in {"timeseries_query", "anomaly_diagnosis"}:
        return _executable_step_signature(
            normalized,
            trajectory=trajectory,
            default_tool_inputs=default_tool_inputs or {},
        )
    return (
        normalized.route,
        normalized.tool or "",
        normalized.metric_name or "",
        normalized.zone_id or "",
        normalized.time_window or "",
    )


def _executable_step_signature(
    step: PlanStep,
    *,
    trajectory: pd.DataFrame | None,
    default_tool_inputs: dict[str, dict[str, Any]],
) -> tuple[Any, ...]:
    tool = step.tool or ""
    input_payload = _canonical_expected_tool_input(
        step,
        trajectory=trajectory,
        default_tool_inputs=default_tool_inputs,
    )
    return (
        step.route,
        tool,
        _canonical_json(input_payload),
    )


def _canonical_expected_tool_input(
    step: PlanStep,
    *,
    trajectory: pd.DataFrame | None,
    default_tool_inputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    tool = step.tool or ""
    zone_id = step.zone_id or _first_zone_id(trajectory)
    metric_name = step.metric_name or "zone_temperature"
    payload: dict[str, Any] = {}
    if tool in {"query_metric", "compare_period", "plot_metric_trend"}:
        start_time, end_time = _trajectory_time_bounds(trajectory)
        payload = {
            "metric_name": metric_name,
            "zone_id": zone_id,
            "start_time": start_time,
            "end_time": end_time,
        }
    elif tool == "detect_anomaly":
        payload = {
            "metric_name": metric_name,
            "zone_id": zone_id,
        }
    elif tool == "data_quality_check":
        payload = {
            "required_fields": _required_trajectory_fields_for_signature(trajectory),
            "expected_frequency": "1h",
        }
    elif tool == "zone_hotspot_rank":
        payload = {
            "metric_name": metric_name,
            "top_k": 3,
        }
    elif tool == "comfort_risk_assessment":
        payload = {
            "temperature_metric": "zone_temperature",
            "comfort_lower_bound": 22.0,
            "comfort_upper_bound": 26.0,
        }
    elif tool == "control_action_audit":
        payload = {
            "action_metric": "control_action",
            "change_threshold": 0.5,
        }
    elif tool == "cooling_efficiency_summary":
        payload = {
            "power_metrics": None,
            "temperature_metric": "zone_temperature",
            "comfort_upper_bound": 26.0,
        }
    else:
        payload = {
            "metric_name": metric_name,
            "zone_id": zone_id,
            "time_window": step.time_window or "full_demo_range",
        }
    payload.update(default_tool_inputs.get(tool, {}))
    return payload


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _first_zone_id(trajectory: pd.DataFrame | None) -> str | None:
    if trajectory is None or "zone_id" not in trajectory.columns:
        return None
    values = trajectory["zone_id"].dropna().astype(str).tolist()
    return values[0] if values else None


def _trajectory_time_bounds(trajectory: pd.DataFrame | None) -> tuple[str | None, str | None]:
    if trajectory is None or "timestamp" not in trajectory.columns or trajectory.empty:
        return None, None
    timestamps = pd.to_datetime(trajectory["timestamp"], utc=True)
    return timestamps.min().isoformat(), timestamps.max().isoformat()


def _required_trajectory_fields_for_signature(trajectory: pd.DataFrame | None) -> list[str]:
    if trajectory is None:
        return []
    required = ["timestamp", "scenario_id", "zone_id", "zone_temperature"]
    if any(metric in trajectory.columns for metric in ["hvac_power", "cooling_power", "fan_power"]):
        required.append("hvac_power" if "hvac_power" in trajectory.columns else "cooling_power")
    return sorted(field for field in required if field in trajectory.columns)


def _canonical_step(step: PlanStep) -> PlanStep:
    tool = step.tool
    metric_name = step.metric_name
    time_window = step.time_window
    if step.route == "timeseries_query":
        tool = tool or "query_metric"
        metric_name = metric_name or "zone_temperature"
        time_window = time_window or "full_demo_range"
    elif step.route == "anomaly_diagnosis":
        tool = tool or "detect_anomaly"
        metric_name = metric_name or "zone_temperature"
        time_window = time_window or "full_demo_range"
    elif step.route == "policy_recommendation":
        tool = tool or "policy_runner"
    elif step.route == "document_qa":
        tool = tool or "rag_retrieval"
    return PlanStep(
        route=step.route,
        reason=step.reason,
        tool=tool,
        metric_name=metric_name,
        zone_id=step.zone_id,
        time_window=time_window,
    )


def _has_pending_required_policy(
    pending_steps: list[PlanStep],
    observations: list[AgentObservation],
    *,
    required_policy_pending: bool = False,
) -> bool:
    if not required_policy_pending and not any(step.route == "policy_recommendation" for step in pending_steps):
        return False
    return not any(observation.has_policy_result for observation in observations)


def _required_policy_would_exceed_budget(
    candidate_steps: list[PlanStep],
    remaining_steps: int,
) -> bool:
    policy_index = _pending_policy_index(candidate_steps)
    if policy_index is None:
        return False
    return policy_index + 1 > remaining_steps


def _pending_policy_index(pending_steps: list[PlanStep]) -> int | None:
    for index, step in enumerate(pending_steps):
        if step.route == "policy_recommendation":
            return index
    return None


def _first_policy_step(pending_steps: list[PlanStep]) -> PlanStep | None:
    for step in pending_steps:
        if step.route == "policy_recommendation":
            return step
    return None


def _policy_required_but_unmet(
    required_policy_pending: bool,
    observations: list[AgentObservation],
) -> bool:
    return required_policy_pending and not any(
        observation.has_policy_result
        for observation in observations
    )


def _required_policy_step(original_plan: list[PlanStep]) -> PlanStep:
    for step in reversed(original_plan):
        if step.route == "policy_recommendation":
            return step
    return PlanStep(
        route="policy_recommendation",
        reason="Required policy recommendation step restored by ReAct guard.",
        tool="policy_runner",
    )


def _observation_from_result(
    step_index: int,
    step: PlanStep,
    result: dict[str, Any],
    fallback_signature: tuple[Any, ...] | None = None,
) -> AgentObservation:
    tool_calls = [
        call for call in result.get("tool_calls", [])
        if isinstance(call, dict)
    ]
    blocked = any(call.get("status") == "blocked" for call in tool_calls)
    errored = any(call.get("status") == "error" for call in tool_calls)
    if blocked:
        status = "blocked"
    elif errored:
        status = "error"
    else:
        status = "success"
    signature = fallback_signature or _step_signature(step)
    if tool_calls:
        signature = _signature_from_tool_call(step, tool_calls[0], signature)
    return AgentObservation(
        step_index=step_index,
        route=step.route,
        tool_names=list(result.get("tools", [])),
        status=status,
        citation_count=len(result.get("citations", [])),
        context_count=len(result.get("retrieved_contexts", [])),
        tool_result_count=len(result.get("tool_results", [])),
        key_findings=_key_findings(result),
        has_policy_result=isinstance(result.get("policy_result"), dict),
        blocked=blocked,
        step_signature=signature,
    )


def _signature_from_tool_call(
    step: PlanStep,
    tool_call: dict[str, Any],
    fallback_signature: tuple[Any, ...],
) -> tuple[Any, ...]:
    tool_name = tool_call.get("tool_name")
    tool_input = tool_call.get("input")
    if not isinstance(tool_name, str) or not isinstance(tool_input, dict):
        return fallback_signature
    if step.route in {"timeseries_query", "anomaly_diagnosis"}:
        return (
            step.route,
            tool_name,
            _canonical_json(_semantic_tool_input(tool_name, tool_input)),
        )
    return fallback_signature


def _semantic_tool_input(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    if tool_name in {"query_metric", "compare_period", "plot_metric_trend"}:
        return {
            "metric_name": tool_input.get("metric_name"),
            "zone_id": tool_input.get("zone_id"),
            "start_time": tool_input.get("start_time"),
            "end_time": tool_input.get("end_time"),
        }
    if tool_name == "detect_anomaly":
        return {
            "metric_name": tool_input.get("metric_name"),
            "zone_id": tool_input.get("zone_id"),
            "window_size": tool_input.get("window_size"),
            "threshold": tool_input.get("threshold"),
        }
    if tool_name == "data_quality_check":
        return {
            "required_fields": sorted(tool_input.get("required_fields") or []),
            "expected_frequency": tool_input.get("expected_frequency"),
        }
    if tool_name == "zone_hotspot_rank":
        return {
            "metric_name": tool_input.get("metric_name"),
            "top_k": tool_input.get("top_k"),
        }
    if tool_name == "comfort_risk_assessment":
        return {
            "temperature_metric": tool_input.get("temperature_metric"),
            "comfort_lower_bound": tool_input.get("comfort_lower_bound"),
            "comfort_upper_bound": tool_input.get("comfort_upper_bound"),
        }
    if tool_name == "control_action_audit":
        return {
            "action_metric": tool_input.get("action_metric"),
            "change_threshold": tool_input.get("change_threshold"),
        }
    if tool_name == "cooling_efficiency_summary":
        return {
            "power_metrics": tool_input.get("power_metrics"),
            "temperature_metric": tool_input.get("temperature_metric"),
            "comfort_upper_bound": tool_input.get("comfort_upper_bound"),
        }
    return dict(tool_input)


def _key_findings(result: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    for tool_result in result.get("tool_results", []):
        if not isinstance(tool_result, dict):
            continue
        for key in [
            "status",
            "risk_level",
            "violation_count",
            "quality_score",
            "large_change_count",
            "efficiency_status",
            "policy_name",
        ]:
            if key in tool_result:
                findings.append(f"{key}={tool_result[key]}")
        summary = tool_result.get("summary")
        if isinstance(summary, dict):
            for key in ["min", "max", "mean", "count"]:
                if key in summary:
                    findings.append(f"summary.{key}={summary[key]}")
    if not findings and result.get("citations"):
        findings.append(f"citations={len(result.get('citations', []))}")
    return findings[:8]


def _controller_system_prompt() -> str:
    return (
        "You are a bounded ReAct controller for DataCenter-HVAC Copilot. "
        "Return only JSON with action, reason, optional step, and confidence. "
        "Allowed actions: continue_next_step, insert_step, replace_next_step, stop_and_answer, stop_blocked. "
        "Only choose insert_step or replace_next_step when an additional validated evidence step is necessary. "
        "Never invent tool outputs or control actions. "
        f"{build_planner_tool_prompt()}"
    )


def _batch_controller_system_prompt() -> str:
    return (
        "You are a bounded plan-execute-reflect controller for DataCenter-HVAC Copilot. "
        "Return only JSON with action, reason, optional steps, and confidence. "
        "Allowed actions: plan_batch, stop_and_answer, stop_blocked. "
        "Use plan_batch to request one or more evidence steps that can be executed before "
        "the next reflection. Use stop_and_answer only when the evidence_bundle is enough "
        "to answer the user. Never invent tool outputs or control actions. "
        f"{build_planner_tool_prompt()}"
    )


def _decision_from_llm_payload(*, content: str, controller: str) -> ReActDecision:
    parsed = _parse_json_object(content)
    action = str(parsed.get("action", "")).strip()
    step = parsed.get("step")
    return ReActDecision(
        action=action,  # type: ignore[arg-type]
        reason=str(parsed.get("reason") or "LLM selected bounded ReAct action."),
        step=_step_from_payload(step) if isinstance(step, dict) else None,
        confidence=_bounded_confidence(parsed.get("confidence", 0.5)),
        controller=controller,
    )


def _batch_decision_from_llm_payload(*, content: str, controller: str) -> ReActBatchDecision:
    parsed = _parse_json_object(content)
    action = str(parsed.get("action", "")).strip()
    raw_steps = parsed.get("steps")
    steps = [
        _step_from_payload(item)
        for item in raw_steps
        if isinstance(item, dict)
    ] if isinstance(raw_steps, list) else []
    return ReActBatchDecision(
        action=action,  # type: ignore[arg-type]
        reason=str(parsed.get("reason") or "LLM selected bounded batch ReAct action."),
        steps=steps,
        confidence=_bounded_confidence(parsed.get("confidence", 0.5)),
        controller=controller,
    )


def _step_from_payload(payload: dict[str, Any]) -> PlanStep:
    route = str(payload["route"])
    return PlanStep(
        route=route,
        reason=str(payload.get("reason") or "LLM inserted ReAct step."),
        tool=_optional_string(payload.get("tool")),
        metric_name=_optional_string(payload.get("metric_name")),
        zone_id=_optional_string(payload.get("zone_id")),
        time_window=_optional_string(payload.get("time_window")),
    )


def _parse_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("ReAct controller payload must be a JSON object")
    return parsed


def _compact_evidence_bundle(evidence_bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "route": evidence_bundle.get("route"),
        "tools": evidence_bundle.get("tools", []),
        "citation_count": len(evidence_bundle.get("citations", [])),
        "context_count": len(evidence_bundle.get("retrieved_contexts", [])),
        "tool_result_count": len(evidence_bundle.get("tool_results", [])),
        "has_policy_result": isinstance(evidence_bundle.get("policy_result"), dict),
        "tool_results": [
            _compact_tool_result(result)
            for result in evidence_bundle.get("tool_results", [])[:6]
            if isinstance(result, dict)
        ],
        "retrieved_contexts": [
            {
                "source": context.get("source"),
                "chunk_id": context.get("chunk_id"),
                "text": str(context.get("text", ""))[:400],
            }
            for context in evidence_bundle.get("retrieved_contexts", [])[:4]
            if isinstance(context, dict)
        ],
    }


def _compact_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[key] = value
        elif isinstance(value, list):
            compact[key] = value[:5]
        elif isinstance(value, dict):
            compact[key] = {
                str(nested_key): nested_value
                for nested_key, nested_value in list(value.items())[:8]
                if isinstance(nested_value, (str, int, float, bool)) or nested_value is None
            }
    return compact


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bounded_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, confidence))


def _default_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> dict[str, Any]:
    from urllib import request

    req = request.Request(url=url, data=body, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))
