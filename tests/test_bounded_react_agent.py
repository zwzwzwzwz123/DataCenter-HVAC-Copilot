from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.agent.answer_generator import AnswerGeneratorInput, GeneratedAnswer
from src.agent.bounded_react import (
    BatchBoundedReActOrchestrator,
    BoundedReActOrchestrator,
    ReActBatchDecision,
    ReActDecision,
)
from src.agent.orchestrator import BaselineOrchestrator
from src.agent.planner import PlanStep
from src.retrieval.chunking import chunk_document
from src.retrieval.loader import load_markdown_document
from src.retrieval.rag import ExtractiveRAGPipeline
from src.retrieval.retriever import KeywordRetriever


class SpyAnswerGenerator:
    def __init__(self) -> None:
        self.payloads: list[AnswerGeneratorInput] = []

    def generate(self, payload: AnswerGeneratorInput) -> GeneratedAnswer:
        self.payloads.append(payload)
        return GeneratedAnswer(answer=f"generated:{payload.route}", generator="spy")


class ReplaceWithComfortThenContinueController:
    def decide(self, **kwargs: Any) -> ReActDecision:
        observations = kwargs["observations"]
        if not observations:
            return ReActDecision(
                action="insert_step",
                reason="Collect comfort risk evidence before policy.",
                step=PlanStep(
                    route="anomaly_diagnosis",
                    reason="Check comfort boundary risk.",
                    tool="comfort_risk_assessment",
                    metric_name="zone_temperature",
                    time_window="full_demo_range",
                ),
                confidence=0.91,
                controller="fake_llm",
            )
        if len(observations) == 1:
            return ReActDecision(
                action="continue_next_step",
                reason="Use the original policy step after observing risk.",
                confidence=0.88,
                controller="fake_llm",
            )
        return ReActDecision(
            action="stop_and_answer",
            reason="Enough evidence collected.",
            confidence=0.8,
            controller="fake_llm",
        )


class InvalidToolController:
    def decide(self, **kwargs: Any) -> ReActDecision:
        return ReActDecision(
            action="insert_step",
            reason="Try an invalid route/tool pair.",
            step=PlanStep(
                route="document_qa",
                reason="Invalid tool for document route.",
                tool="policy_runner",
            ),
            confidence=0.99,
            controller="fake_llm",
        )


class AlwaysInsertController:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, **kwargs: Any) -> ReActDecision:
        self.calls += 1
        tool = "query_metric" if self.calls % 2 else "data_quality_check"
        return ReActDecision(
            action="insert_step",
            reason=f"Need more evidence {self.calls}.",
            step=PlanStep(
                route="timeseries_query",
                reason=f"Collect metric slice {self.calls}.",
                tool=tool,
                metric_name="zone_temperature",
                time_window="full_demo_range",
            ),
            confidence=0.7,
            controller="fake_llm",
        )


class InsertPolicyBeforeEvidenceController:
    def decide(self, **kwargs: Any) -> ReActDecision:
        if not kwargs["observations"]:
            return ReActDecision(
                action="insert_step",
                reason="Unsafe attempt to run policy before pending evidence.",
                step=PlanStep(
                    route="policy_recommendation",
                    reason="Run policy too early.",
                    tool="policy_runner",
                ),
                confidence=0.93,
                controller="fake_llm",
            )
        return ReActDecision(
            action="stop_and_answer",
            reason="Stop after fallback.",
            confidence=0.8,
            controller="fake_llm",
        )


class StopImmediatelyController:
    def decide(self, **kwargs: Any) -> ReActDecision:
        return ReActDecision(
            action="stop_and_answer",
            reason="Stop before executing anything.",
            confidence=0.9,
            controller="fake_llm",
        )


class InsertEvidenceThenStopBeforePolicyController:
    def decide(self, **kwargs: Any) -> ReActDecision:
        observations = kwargs["observations"]
        if not observations:
            return ReActDecision(
                action="insert_step",
                reason="Collect evidence before policy.",
                step=PlanStep(
                    route="timeseries_query",
                    reason="Query temperature evidence.",
                    tool="query_metric",
                    metric_name="zone_temperature",
                    time_window="full_demo_range",
                ),
                confidence=0.9,
                controller="fake_llm",
            )
        return ReActDecision(
            action="stop_and_answer",
            reason="Incorrectly tries to stop before policy.",
            confidence=0.9,
            controller="fake_llm",
        )


class RepeatDefaultEquivalentToolController:
    def decide(self, **kwargs: Any) -> ReActDecision:
        observations = kwargs["observations"]
        if not observations:
            return ReActDecision(
                action="insert_step",
                reason="Explicit query metric.",
                step=PlanStep(
                    route="timeseries_query",
                    reason="Explicit metric query.",
                    tool="query_metric",
                    metric_name="zone_temperature",
                    time_window="full_demo_range",
                ),
                confidence=0.8,
                controller="fake_llm",
            )
        return ReActDecision(
            action="insert_step",
            reason="Repeat same effective query with defaults.",
            step=PlanStep(
                route="timeseries_query",
                reason="Default metric query.",
            ),
            confidence=0.8,
            controller="fake_llm",
        )


class RepeatQueryMetricAfterAnotherToolController:
    def decide(self, **kwargs: Any) -> ReActDecision:
        observations = kwargs["observations"]
        if not observations:
            return ReActDecision(
                action="insert_step",
                reason="First query metric.",
                step=PlanStep(
                    route="timeseries_query",
                    reason="Query temperature once.",
                    tool="query_metric",
                    metric_name="zone_temperature",
                    time_window="full_demo_range",
                ),
                confidence=0.8,
                controller="fake_llm",
            )
        if len(observations) == 1:
            return ReActDecision(
                action="insert_step",
                reason="Interleave a different tool.",
                step=PlanStep(
                    route="timeseries_query",
                    reason="Check data quality.",
                    tool="data_quality_check",
                    time_window="full_demo_range",
                ),
                confidence=0.8,
                controller="fake_llm",
            )
        return ReActDecision(
            action="insert_step",
            reason="Try repeating the first query metric.",
            step=PlanStep(
                route="timeseries_query",
                reason="Repeat same temperature query.",
                tool="query_metric",
                metric_name="zone_temperature",
                time_window="full_demo_range",
            ),
            confidence=0.8,
            controller="fake_llm",
        )


class ReplacePolicyWithEvidenceOnlyController:
    def decide(self, **kwargs: Any) -> ReActDecision:
        if not kwargs["observations"]:
            return ReActDecision(
                action="replace_next_step",
                reason="Unsafe attempt to replace the required policy step.",
                step=PlanStep(
                    route="timeseries_query",
                    reason="Collect only temperature evidence.",
                    tool="query_metric",
                    metric_name="zone_temperature",
                    time_window="full_demo_range",
                ),
                confidence=0.91,
                controller="fake_llm",
            )
        return ReActDecision(
            action="stop_and_answer",
            reason="Stop without policy.",
            confidence=0.8,
            controller="fake_llm",
        )


class RepeatDefaultZoneEquivalentToolController:
    def decide(self, **kwargs: Any) -> ReActDecision:
        observations = kwargs["observations"]
        if not observations:
            return ReActDecision(
                action="insert_step",
                reason="Explicitly query the first zone.",
                step=PlanStep(
                    route="timeseries_query",
                    reason="Explicit zone query.",
                    tool="query_metric",
                    metric_name="zone_temperature",
                    zone_id="zone_a",
                    time_window="full_demo_range",
                ),
                confidence=0.8,
                controller="fake_llm",
            )
        return ReActDecision(
            action="insert_step",
            reason="Repeat the same effective query with implicit zone default.",
            step=PlanStep(
                route="timeseries_query",
                reason="Implicit first-zone query.",
                tool="query_metric",
                metric_name="zone_temperature",
                time_window="full_demo_range",
            ),
            confidence=0.8,
            controller="fake_llm",
        )


class StarvePolicyWithEvidenceController:
    def decide(self, **kwargs: Any) -> ReActDecision:
        observations = kwargs["observations"]
        tool = "query_metric" if not observations else "data_quality_check"
        return ReActDecision(
            action="insert_step",
            reason="Keep collecting evidence before the required policy step.",
            step=PlanStep(
                route="timeseries_query",
                reason="Collect another non-policy signal.",
                tool=tool,
                metric_name="zone_temperature",
                time_window="full_demo_range",
            ),
            confidence=0.9,
            controller="fake_llm",
        )


class RepeatDataQualityController:
    def decide(self, **kwargs: Any) -> ReActDecision:
        return ReActDecision(
            action="insert_step",
            reason="Repeat the same data quality check.",
            step=PlanStep(
                route="timeseries_query",
                reason="Check trajectory data quality.",
                tool="data_quality_check",
            ),
            confidence=0.8,
            controller="fake_llm",
        )


class BatchThenReflectController:
    def __init__(self) -> None:
        self.observation_counts: list[int] = []

    def decide_batch(self, **kwargs: Any) -> ReActBatchDecision:
        observations = kwargs["observations"]
        self.observation_counts.append(len(observations))
        if not observations:
            return ReActBatchDecision(
                action="plan_batch",
                reason="Collect temperature and comfort evidence together.",
                steps=[
                    PlanStep(
                        route="timeseries_query",
                        reason="Query the temperature series.",
                        tool="query_metric",
                        metric_name="zone_temperature",
                        time_window="full_demo_range",
                    ),
                    PlanStep(
                        route="anomaly_diagnosis",
                        reason="Assess comfort risk from the same interval.",
                        tool="comfort_risk_assessment",
                        metric_name="zone_temperature",
                        time_window="full_demo_range",
                    ),
                ],
                confidence=0.9,
                controller="fake_batch_llm",
            )
        return ReActBatchDecision(
            action="stop_and_answer",
            reason="The combined evidence is enough.",
            confidence=0.86,
            controller="fake_batch_llm",
        )


class TwoRoundBatchController:
    def decide_batch(self, **kwargs: Any) -> ReActBatchDecision:
        observations = kwargs["observations"]
        if not observations:
            return ReActBatchDecision(
                action="plan_batch",
                reason="Start with the raw temperature metric.",
                steps=[
                    PlanStep(
                        route="timeseries_query",
                        reason="Query temperature first.",
                        tool="query_metric",
                        metric_name="zone_temperature",
                        time_window="full_demo_range",
                    )
                ],
                confidence=0.84,
                controller="fake_batch_llm",
            )
        if len(observations) == 1:
            return ReActBatchDecision(
                action="plan_batch",
                reason="Need an efficiency summary before answering.",
                steps=[
                    PlanStep(
                        route="timeseries_query",
                        reason="Summarize cooling efficiency.",
                        tool="cooling_efficiency_summary",
                        time_window="full_demo_range",
                    )
                ],
                confidence=0.8,
                controller="fake_batch_llm",
            )
        return ReActBatchDecision(
            action="stop_and_answer",
            reason="Metric and efficiency evidence are now enough.",
            confidence=0.88,
            controller="fake_batch_llm",
        )


class BatchEvidenceThenStopBeforePolicyController:
    def decide_batch(self, **kwargs: Any) -> ReActBatchDecision:
        observations = kwargs["observations"]
        if not observations:
            return ReActBatchDecision(
                action="plan_batch",
                reason="Collect evidence before the required policy.",
                steps=[
                    PlanStep(
                        route="timeseries_query",
                        reason="Query temperature before policy.",
                        tool="query_metric",
                        metric_name="zone_temperature",
                        time_window="full_demo_range",
                    )
                ],
                confidence=0.9,
                controller="fake_batch_llm",
            )
        return ReActBatchDecision(
            action="stop_and_answer",
            reason="Incorrectly tries to stop before policy.",
            confidence=0.9,
            controller="fake_batch_llm",
        )


class BatchEvidenceStarvesPolicyController:
    def decide_batch(self, **kwargs: Any) -> ReActBatchDecision:
        observations = kwargs["observations"]
        if not observations:
            return ReActBatchDecision(
                action="plan_batch",
                reason="Collect the first evidence step.",
                steps=[
                    PlanStep(
                        route="timeseries_query",
                        reason="Query temperature before policy.",
                        tool="query_metric",
                        metric_name="zone_temperature",
                        time_window="full_demo_range",
                    )
                ],
                confidence=0.9,
                controller="fake_batch_llm",
            )
        return ReActBatchDecision(
            action="plan_batch",
            reason="Incorrectly spends the last step on non-policy evidence.",
            steps=[
                PlanStep(
                    route="timeseries_query",
                    reason="Check data quality instead of policy.",
                    tool="data_quality_check",
                    time_window="full_demo_range",
                )
            ],
            confidence=0.9,
            controller="fake_batch_llm",
        )


def _baseline(approval_handler=None) -> BaselineOrchestrator:
    document = load_markdown_document(
        Path("data/documents/sample_hvac_guidance.md"),
        source_id="sample_hvac_guidance",
        title="Sample HVAC Guidance",
        published_at="2026",
        category="internal_note",
    )
    chunks = chunk_document(document, chunk_size=45, overlap=5)
    rag = ExtractiveRAGPipeline(KeywordRetriever(chunks))
    trajectory = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            "scenario_id": ["episode_001"] * 4,
            "zone_id": ["zone_a"] * 4,
            "zone_temperature": [23.0, 24.0, 30.0, 25.0],
            "cooling_power": [100.0, 110.0, 180.0, 120.0],
            "fan_power": [20.0, 21.0, 30.0, 24.0],
            "hvac_power": [120.0, 131.0, 210.0, 144.0],
            "control_action": [-500.0, -500.0, -600.0, -600.0],
            "comfort_violation": [False, False, True, False],
        }
    )
    return BaselineOrchestrator(
        rag_pipeline=rag,
        trajectory=trajectory,
        answer_generator=SpyAnswerGenerator(),
        approval_handler=approval_handler,
    )


def test_batch_bounded_react_executes_a_tool_batch_before_reflection() -> None:
    controller = BatchThenReflectController()
    orchestrator = BatchBoundedReActOrchestrator(
        _baseline(),
        controller=controller,
        max_steps=5,
    )

    result = orchestrator.run(
        "Check the zone temperature and comfort risk before answering.",
        task_type="timeseries_query",
    )

    assert result["workflow_engine"] == "bounded_react_batch"
    assert result["tools"] == ["query_metric", "comfort_risk_assessment"]
    assert controller.observation_counts == [0, 2]
    trace_nodes = [node["node"] for node in result["workflow_trace"]]
    first_reflect = trace_nodes.index("batch_reflection")
    executed_before_reflect = trace_nodes[:first_reflect].count("execute_react_step")
    assert executed_before_reflect == 2
    assert result["workflow_trace"][first_reflect]["observation_count"] == 2


def test_batch_bounded_react_can_plan_a_second_batch_after_reflection() -> None:
    orchestrator = BatchBoundedReActOrchestrator(
        _baseline(),
        controller=TwoRoundBatchController(),
        max_steps=5,
    )

    result = orchestrator.run(
        "Check temperature, then decide whether efficiency evidence is enough.",
        task_type="timeseries_query",
    )

    assert result["workflow_engine"] == "bounded_react_batch"
    assert result["tools"] == ["query_metric", "cooling_efficiency_summary"]
    batch_nodes = [
        node for node in result["workflow_trace"] if node["node"] == "batch_controller"
    ]
    assert [node["action"] for node in batch_nodes] == [
        "plan_batch",
        "plan_batch",
        "stop_and_answer",
    ]
    assert result["react_trace"][-1]["observation"]["tool_names"] == [
        "cooling_efficiency_summary"
    ]


def test_batch_bounded_react_promotes_policy_when_controller_stops_after_evidence() -> None:
    orchestrator = BatchBoundedReActOrchestrator(
        _baseline(),
        controller=BatchEvidenceThenStopBeforePolicyController(),
        max_steps=2,
    )

    result = orchestrator.run(
        "Check temperature then recommend a policy.",
        task_type=None,
    )

    assert result["tools"] == ["query_metric", "rule_based_policy"]
    assert result["route"] == "policy_recommendation"
    assert any(
        recovery["strategy"] == "react_policy_budget_guard"
        for recovery in result["runtime_trace"]["recoveries"]
    )
    assert [todo["status"] for todo in result["todos"]] == ["completed", "completed"]


def test_batch_bounded_react_promotes_policy_when_batch_would_starve_budget() -> None:
    orchestrator = BatchBoundedReActOrchestrator(
        _baseline(),
        controller=BatchEvidenceStarvesPolicyController(),
        max_steps=2,
    )

    result = orchestrator.run(
        "Check temperature then recommend a policy.",
        task_type=None,
    )

    assert result["tools"] == ["query_metric", "rule_based_policy"]
    assert result["route"] == "policy_recommendation"
    assert all(tool != "data_quality_check" for tool in result["tools"])
    assert any(
        recovery["strategy"] == "react_policy_budget_guard"
        for recovery in result["runtime_trace"]["recoveries"]
    )


def test_bounded_react_llm_can_replace_next_step_and_continue_original_plan() -> None:
    orchestrator = BoundedReActOrchestrator(
        _baseline(),
        controller=ReplaceWithComfortThenContinueController(),
        max_steps=5,
    )

    result = orchestrator.run(
        "Recommend a policy, but first inspect whether the zone is overheating.",
        task_type="policy_recommendation",
    )

    assert result["workflow_engine"] == "bounded_react"
    assert result["route"] == "policy_recommendation"
    assert result["tools"] == ["comfort_risk_assessment", "rule_based_policy"]
    assert [todo["route"] for todo in result["todos"]] == [
        "anomaly_diagnosis",
        "policy_recommendation",
    ]
    assert [todo["status"] for todo in result["todos"]] == ["completed", "completed"]
    decision_nodes = [
        node for node in result["workflow_trace"] if node["node"] == "react_controller"
    ]
    assert decision_nodes[0]["action"] == "insert_step"
    assert decision_nodes[1]["action"] == "continue_next_step"
    assert result["react_trace"][0]["observation"]["tool_names"] == ["comfort_risk_assessment"]


def test_bounded_react_rejects_invalid_llm_tool_decision_and_falls_back() -> None:
    orchestrator = BoundedReActOrchestrator(
        _baseline(),
        controller=InvalidToolController(),
        max_steps=5,
    )

    result = orchestrator.run("Recommend a policy.", task_type="policy_recommendation")

    assert result["workflow_engine"] == "bounded_react"
    assert result["tools"] == ["rule_based_policy"]
    assert result["runtime_trace"]["recoveries"][0]["strategy"] == "react_decision_fallback"
    assert "unsupported tool" in result["runtime_trace"]["recoveries"][0]["error"]
    decision_nodes = [
        node for node in result["workflow_trace"] if node["node"] == "react_controller"
    ]
    assert decision_nodes[0]["fallback_used"] is True


def test_bounded_react_stops_when_max_steps_budget_is_exhausted() -> None:
    orchestrator = BoundedReActOrchestrator(
        _baseline(),
        controller=AlwaysInsertController(),
        max_steps=2,
    )

    result = orchestrator.run("Keep checking zone_temperature.", task_type="timeseries_query")

    assert result["workflow_engine"] == "bounded_react"
    assert result["runtime_trace"]["summary"]["todo_count"] == 2
    assert len(result["react_trace"]) == 2
    stop_nodes = [node for node in result["workflow_trace"] if node["node"] == "react_stop"]
    assert stop_nodes[-1]["reason"] == "max_steps_exhausted"


def test_bounded_react_stops_blocked_when_policy_approval_is_denied() -> None:
    def deny_policy(request: dict) -> dict:
        return {
            "approved": False,
            "decision": "denied",
            "reason": f"operator denied {request['tool_name']}",
        }

    orchestrator = BoundedReActOrchestrator(_baseline(approval_handler=deny_policy))

    result = orchestrator.run("Recommend a policy.", task_type="policy_recommendation")

    assert result["workflow_engine"] == "bounded_react"
    assert result["policy_result"] is None
    assert result["tool_calls"][0]["status"] == "blocked"
    assert result["todos"][0]["status"] == "blocked"
    assert result["runtime_trace"]["hooks"][0]["approval"]["decision"] == "denied"
    stop_nodes = [node for node in result["workflow_trace"] if node["node"] == "react_stop"]
    assert stop_nodes[-1]["reason"] == "blocked"


def test_bounded_react_rejects_policy_insert_before_pending_evidence() -> None:
    orchestrator = BoundedReActOrchestrator(
        _baseline(),
        controller=InsertPolicyBeforeEvidenceController(),
        max_steps=5,
    )

    result = orchestrator.run(
        "Check temperature then recommend a policy.",
        task_type=None,
    )

    assert result["workflow_engine"] == "bounded_react"
    assert result["tools"][0] != "rule_based_policy"
    assert result["runtime_trace"]["recoveries"][0]["strategy"] == "react_decision_fallback"
    assert "policy_recommendation must be the final step" in result["runtime_trace"]["recoveries"][0]["error"]


def test_bounded_react_fallback_step_records_tool_hooks_when_controller_stops_immediately() -> None:
    orchestrator = BoundedReActOrchestrator(
        _baseline(),
        controller=StopImmediatelyController(),
        max_steps=5,
    )

    result = orchestrator.run("Recommend a policy.", task_type="policy_recommendation")

    hook_names = [hook["hook"] for hook in result["runtime_trace"]["hooks"]]
    assert "PreToolUse" in hook_names
    assert "PostToolUse" in hook_names
    assert result["runtime_trace"]["summary"]["tool_call_count"] == 1
    assert result["tools"] == ["rule_based_policy"]


def test_bounded_react_rejects_non_adjacent_duplicate_tool_calls() -> None:
    orchestrator = BoundedReActOrchestrator(
        _baseline(),
        controller=RepeatQueryMetricAfterAnotherToolController(),
        max_steps=5,
    )

    result = orchestrator.run("Explore zone_temperature repeatedly.", task_type="timeseries_query")

    assert result["tools"] == ["query_metric", "data_quality_check"]
    assert any(
        recovery["strategy"] == "react_decision_fallback"
        and "duplicate ReAct tool call blocked" in recovery["error"]
        for recovery in result["runtime_trace"]["recoveries"]
    )


def test_bounded_react_does_not_allow_stop_before_pending_policy_step() -> None:
    orchestrator = BoundedReActOrchestrator(
        _baseline(),
        controller=InsertEvidenceThenStopBeforePolicyController(),
        max_steps=5,
    )

    result = orchestrator.run("Check temperature then recommend a policy.", task_type=None)

    assert result["tools"] == ["query_metric", "rule_based_policy"]
    assert any(
        recovery["strategy"] == "react_decision_fallback"
        and "pending policy step" in recovery["error"]
        for recovery in result["runtime_trace"]["recoveries"]
    )


def test_bounded_react_blocks_default_equivalent_duplicate_tool_call() -> None:
    orchestrator = BoundedReActOrchestrator(
        _baseline(),
        controller=RepeatDefaultEquivalentToolController(),
        max_steps=5,
    )

    result = orchestrator.run("Explore zone_temperature repeatedly.", task_type="timeseries_query")

    assert result["tools"] == ["query_metric"]
    blocked_todos = [todo for todo in result["todos"] if todo["status"] == "blocked"]
    assert blocked_todos
    assert blocked_todos[-1]["route"] == "timeseries_query"
    assert any(
        recovery["strategy"] == "react_duplicate_step_blocked"
        for recovery in result["runtime_trace"]["recoveries"]
    )


def test_bounded_react_does_not_allow_replace_to_remove_required_policy_step() -> None:
    orchestrator = BoundedReActOrchestrator(
        _baseline(),
        controller=ReplacePolicyWithEvidenceOnlyController(),
        max_steps=5,
    )

    result = orchestrator.run("Recommend a policy.", task_type="policy_recommendation")

    assert result["tools"] == ["rule_based_policy"]
    assert result["route"] == "policy_recommendation"
    assert isinstance(result["policy_result"], dict)
    assert any(
        recovery["strategy"] == "react_decision_fallback"
        and "required policy step" in recovery["error"]
        for recovery in result["runtime_trace"]["recoveries"]
    )


def test_bounded_react_blocks_default_zone_equivalent_duplicate_tool_call() -> None:
    orchestrator = BoundedReActOrchestrator(
        _baseline(),
        controller=RepeatDefaultZoneEquivalentToolController(),
        max_steps=5,
    )

    result = orchestrator.run("Explore zone_temperature repeatedly.", task_type="timeseries_query")

    assert result["tools"] == ["query_metric"]
    blocked_todos = [todo for todo in result["todos"] if todo["status"] == "blocked"]
    assert blocked_todos
    assert blocked_todos[-1]["route"] == "timeseries_query"
    assert any(
        recovery["strategy"] == "react_duplicate_step_blocked"
        for recovery in result["runtime_trace"]["recoveries"]
    )


def test_bounded_react_does_not_allow_evidence_insert_to_starve_required_policy_budget() -> None:
    orchestrator = BoundedReActOrchestrator(
        _baseline(),
        controller=StarvePolicyWithEvidenceController(),
        max_steps=2,
    )

    result = orchestrator.run("Recommend a policy.", task_type="policy_recommendation")

    assert result["tools"] == ["query_metric", "rule_based_policy"]
    assert result["route"] == "policy_recommendation"
    assert isinstance(result["policy_result"], dict)
    assert any(
        recovery["strategy"] == "react_decision_fallback"
        and "required policy budget" in recovery["error"]
        for recovery in result["runtime_trace"]["recoveries"]
    )


def test_bounded_react_promotes_required_policy_when_initial_plan_exceeds_budget() -> None:
    orchestrator = BoundedReActOrchestrator(
        _baseline(),
        max_steps=1,
    )

    result = orchestrator.run("Check temperature then recommend a policy.", task_type=None)

    assert result["tools"] == ["rule_based_policy"]
    assert result["route"] == "policy_recommendation"
    assert isinstance(result["policy_result"], dict)
    blocked_todos = [todo for todo in result["todos"] if todo["status"] == "blocked"]
    assert blocked_todos
    assert blocked_todos[-1]["route"] == "timeseries_query"
    assert any(
        recovery["strategy"] == "react_policy_budget_guard"
        for recovery in result["runtime_trace"]["recoveries"]
    )


def test_bounded_react_blocks_repeated_data_quality_check() -> None:
    orchestrator = BoundedReActOrchestrator(
        _baseline(),
        controller=RepeatDataQualityController(),
        max_steps=3,
    )

    result = orchestrator.run("Check data quality repeatedly.", task_type="timeseries_query")

    assert result["tools"] == ["data_quality_check"]
    blocked_todos = [todo for todo in result["todos"] if todo["status"] == "blocked"]
    assert blocked_todos
    assert blocked_todos[-1]["route"] == "timeseries_query"
    assert any(
        recovery["strategy"] in {"react_decision_fallback", "react_duplicate_step_blocked"}
        and (
            "duplicate ReAct tool call blocked" in recovery.get("error", "")
            or recovery["strategy"] == "react_duplicate_step_blocked"
        )
        for recovery in result["runtime_trace"]["recoveries"]
    )
