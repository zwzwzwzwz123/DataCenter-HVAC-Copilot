from pathlib import Path

import pandas as pd

from src.agent.answer_generator import AnswerGeneratorInput, GeneratedAnswer
import src.agent.executor as executor_module
from src.agent.executor import AgentTaskExecutor
from src.agent.langgraph_workflow import LangGraphOrchestrator
from src.agent.orchestrator import BaselineOrchestrator
from src.agent.planner import PlanDecision, PlanStep
from src.agent.router import route_task
from src.agent.runtime import AgentRuntimeTrace
from src.policies.base import PolicyResult
from src.retrieval.chunking import chunk_document
from src.retrieval.loader import load_markdown_document
from src.retrieval.rag import ExtractiveRAGPipeline, RAGAnswer
from src.retrieval.retriever import KeywordRetriever


def mock_trajectory():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            "scenario_id": ["episode_001"] * 4,
            "zone_id": ["zone_a"] * 4,
            "zone_temperature": [23.0, 24.0, 30.0, 25.0],
            "cooling_power": [100.0, 110.0, 180.0, 120.0],
            "fan_power": [20.0, 21.0, 30.0, 24.0],
            "hvac_power": [120.0, 131.0, 210.0, 144.0],
            "control_action": [0.2, 0.2, 0.9, 0.3],
            "comfort_violation": [False, False, True, False],
        }
    )


def mock_rag():
    document = load_markdown_document(
        Path("data/documents/sample_hvac_guidance.md"),
        source_id="sample_hvac_guidance",
        title="Sample HVAC Guidance",
        published_at="2026",
        category="internal_note",
    )
    chunks = chunk_document(document, chunk_size=45, overlap=5)
    return ExtractiveRAGPipeline(KeywordRetriever(chunks))


class SpyAnswerGenerator:
    def __init__(self) -> None:
        self.payloads: list[AnswerGeneratorInput] = []

    def generate(self, payload: AnswerGeneratorInput) -> GeneratedAnswer:
        self.payloads.append(payload)
        return GeneratedAnswer(answer=f"generated:{payload.route}", generator="spy")


class RewriteOnlyRAGPipeline:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def answer(self, question: str, top_k: int = 3) -> RAGAnswer:
        self.queries.append(question)
        if "data boundary" not in question:
            return RAGAnswer(
                question=question,
                answer="未找到足够的检索证据，无法给出可靠回答。",
                citations=[],
                retrieved_contexts=[],
            )
        context = {
            "text": "BEAR HVAC simulation evidence must preserve the data boundary.",
            "source_id": "boundary_doc",
            "title": "Boundary Guidance",
            "citation": {"source_id": "boundary_doc", "title": "Boundary Guidance"},
        }
        return RAGAnswer(
            question=question,
            answer=context["text"],
            citations=[context["citation"]],
            retrieved_contexts=[context],
        )


class StaticRoutePlanner:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def plan(
        self,
        question: str,
        task_type: str | None = None,
        conversation_context: dict | None = None,
    ) -> PlanDecision:
        call = {"question": question, "task_type": task_type}
        if conversation_context is not None:
            call["conversation_context"] = conversation_context
        self.calls.append(call)
        return PlanDecision(
            steps=[
                PlanStep(route="timeseries_query", reason="Gather metric evidence first."),
                PlanStep(route="policy_recommendation", reason="Then run the bounded policy tool."),
            ],
            planner="llm:deepseek:planner-test",
            confidence=0.88,
            fallback_used=False,
        )


class StructuredStepPlanner:
    def plan(self, question: str, task_type: str | None = None) -> PlanDecision:
        return PlanDecision(
            steps=[
                PlanStep(
                    route="timeseries_query",
                    reason="Planner selected the specific metric query.",
                    tool="compare_period",
                    metric_name="fan_power",
                    zone_id="zone_a",
                    time_window="full_demo_range",
                )
            ],
            planner="structured-test",
            confidence=0.9,
        )


class RecentWindowPlanner:
    def plan(self, question: str, task_type: str | None = None) -> PlanDecision:
        return PlanDecision(
            steps=[
                PlanStep(
                    route="timeseries_query",
                    reason="Planner selected a recent time window.",
                    tool="query_metric",
                    metric_name="zone_temperature",
                    zone_id="zone_a",
                    time_window="last_2_hours",
                )
            ],
            planner="recent-window-test",
            confidence=0.9,
        )


class UnsupportedWindowPlanner:
    def plan(self, question: str, task_type: str | None = None) -> PlanDecision:
        return PlanDecision(
            steps=[
                PlanStep(
                    route="timeseries_query",
                    reason="Planner emitted an unsupported time window.",
                    tool="query_metric",
                    metric_name="zone_temperature",
                    zone_id="zone_a",
                    time_window="last_24",
                )
            ],
            planner="unsupported-window-test",
            confidence=0.9,
        )


class ToolStepPlanner:
    def __init__(self, steps: list[PlanStep]) -> None:
        self.steps = steps

    def plan(self, question: str, task_type: str | None = None) -> PlanDecision:
        return PlanDecision(
            steps=self.steps,
            planner="tool-step-test",
            confidence=0.9,
        )


def test_route_task_uses_eval_task_type_when_available():
    assert route_task("anything", task_type="timeseries_query").route == "timeseries_query"


def test_route_task_infers_document_qa_from_question():
    result = route_task("数据中心冷却系统为什么可能出现高能耗？")

    assert result.route == "document_qa"
    assert result.required_tools == []


def test_orchestrator_handles_document_qa_with_citations():
    generator = SpyAnswerGenerator()
    orchestrator = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=generator,
    )

    result = orchestrator.run("为什么 PUE-like values 不能编造？", task_type="document_qa")

    assert result["route"] == "document_qa"
    assert result["citations"]
    assert result["tools"] == []
    assert result["answer"] == "generated:document_qa"
    assert result["answer_generator"] == "spy"
    assert result["answer_audit"]["passed"] is True
    assert generator.payloads[0].retrieved_contexts


def test_orchestrator_handles_timeseries_query_with_tool_result():
    generator = SpyAnswerGenerator()
    orchestrator = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=generator,
    )

    result = orchestrator.run(
        "episode_001 中 zone_a 在最近 3 小时的温度最大值是多少？",
        task_type="timeseries_query",
    )

    assert result["route"] == "timeseries_query"
    assert result["tools"] == ["query_metric"]
    assert result["tool_results"][0]["summary"]["max"] == 30.0
    assert result["answer"] == "generated:timeseries_query"
    assert result["answer_audit"]["passed"] is True
    assert generator.payloads[0].tool_results


def test_orchestrator_selects_compare_period_for_period_comparison():
    orchestrator = BaselineOrchestrator(rag_pipeline=mock_rag(), trajectory=mock_trajectory())

    result = orchestrator.run(
        "请比较 zone_temperature 在前半段和后半段的平均值变化。",
        task_type="timeseries_query",
    )

    assert result["tools"] == ["compare_period"]
    assert result["tool_results"][0]["tool_name"] == "compare_period"


def test_orchestrator_selects_plot_metric_trend_for_trend_requests():
    orchestrator = BaselineOrchestrator(rag_pipeline=mock_rag(), trajectory=mock_trajectory())

    result = orchestrator.run(
        "请生成 zone_a 温度趋势图所需的数据。",
        task_type="timeseries_query",
    )

    assert result["tools"] == ["plot_metric_trend"]
    assert result["tool_results"][0]["tool_name"] == "plot_metric_trend"


def test_orchestrator_prioritizes_control_action_comparison_and_trend_over_audit():
    orchestrator = BaselineOrchestrator(rag_pipeline=mock_rag(), trajectory=mock_trajectory())

    comparison = orchestrator.run(
        "请比较 control_action 在两个时间窗口中的变化。",
        task_type="timeseries_query",
    )
    trend = orchestrator.run(
        "请给出 control_action 的趋势序列。",
        task_type="timeseries_query",
    )

    assert comparison["tools"] == ["compare_period"]
    assert trend["tools"] == ["plot_metric_trend"]


def test_orchestrator_selects_energy_breakdown_for_energy_breakdown_requests():
    orchestrator = BaselineOrchestrator(rag_pipeline=mock_rag(), trajectory=mock_trajectory())

    result = orchestrator.run(
        "当前轨迹的冷却能耗构成是什么？",
        task_type="timeseries_query",
    )

    assert result["tools"] == ["compute_energy_breakdown"]
    assert result["tool_results"][0]["tool_name"] == "compute_energy_breakdown"


def test_orchestrator_selects_metric_names_from_chinese_questions():
    orchestrator = BaselineOrchestrator(rag_pipeline=mock_rag(), trajectory=mock_trajectory())

    fan_result = orchestrator.run("最近风机功率最大值是多少？", task_type="timeseries_query")
    temp_result = orchestrator.run("最近温度最大值是多少？", task_type="timeseries_query")

    assert fan_result["tool_results"][0]["metric_name"] == "fan_power"
    assert temp_result["tool_results"][0]["metric_name"] == "zone_temperature"


def test_orchestrator_handles_anomaly_diagnosis():
    generator = SpyAnswerGenerator()
    orchestrator = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=generator,
    )

    result = orchestrator.run("zone_a 是否存在温度异常升高？", task_type="anomaly_diagnosis")

    assert result["route"] == "anomaly_diagnosis"
    assert result["tools"] == ["detect_anomaly"]
    assert result["tool_results"][0]["anomalies"]
    assert result["answer"] == "generated:anomaly_diagnosis"
    assert result["answer_audit"]["passed"] is True


def test_orchestrator_handles_policy_recommendation():
    generator = SpyAnswerGenerator()
    orchestrator = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=generator,
    )

    result = orchestrator.run(
        "如果当前温度超过舒适上限，是否应该调整控制策略？",
        task_type="policy_recommendation",
    )

    assert result["route"] == "policy_recommendation"
    assert result["tools"] == ["rule_based_policy"]
    assert result["policy_result"]["policy_name"] == "rule_based"
    assert result["answer"] == "generated:policy_recommendation"
    assert result["answer_audit"]["passed"] is True
    assert generator.payloads[0].policy_result["policy_name"] == "rule_based"


def test_orchestrator_can_use_injected_offline_replay_policy_runner():
    def offline_runner(state: dict) -> PolicyResult:
        return PolicyResult(
            policy_name="guided_diffno_offline_replay",
            input_state_id=state["state_id"],
            recommended_action=[-0.2, -0.1],
            estimated_energy=901.3,
            estimated_comfort_violations=0.1,
            mean_action_change=0.15,
            baseline="rule_based",
            notes="Values come from offline replay, not from LLM generation.",
        )

    generator = SpyAnswerGenerator()
    orchestrator = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=generator,
        policy_runner=offline_runner,
    )

    result = orchestrator.run("请基于 offline replay 给出策略建议。", task_type="policy_recommendation")

    assert result["tools"] == ["guided_diffno_offline_replay"]
    assert result["policy_result"]["policy_name"] == "guided_diffno_offline_replay"
    assert result["policy_result"]["estimated_energy"] == 901.3
    assert result["answer_audit"]["passed"] is True
    assert generator.payloads[0].policy_result["baseline"] == "rule_based"


def test_langgraph_orchestrator_preserves_baseline_result_and_adds_trace():
    baseline = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=SpyAnswerGenerator(),
    )
    orchestrator = LangGraphOrchestrator(baseline)

    result = orchestrator.run(
        "episode_001 中 zone_a 在最近 3 小时的温度最大值是多少？",
        task_type="timeseries_query",
    )

    assert result["route"] == "timeseries_query"
    assert result["tools"] == ["query_metric"]
    assert result["workflow_engine"] == "langgraph"
    assert [step["node"] for step in result["workflow_trace"]] == [
        "planner",
        "execute_plan_step",
        "evidence_aggregator",
        "answer_generator",
        "answer_audit",
    ]
    assert result["workflow_trace"][0]["planned_steps"] == ["timeseries_query"]
    assert result["workflow_trace"][2]["tool_result_count"] == 1
    assert result["workflow_trace"][3]["answer_generator"] == "spy"


def test_langgraph_runtime_trace_tracks_todos_and_hooks_for_tool_steps():
    baseline = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=SpyAnswerGenerator(),
    )
    orchestrator = LangGraphOrchestrator(
        baseline,
        route_planner=ToolStepPlanner(
            [
                PlanStep(
                    route="timeseries_query",
                    reason="Read the temperature metric.",
                    tool="query_metric",
                ),
                PlanStep(
                    route="anomaly_diagnosis",
                    reason="Then detect anomalies.",
                    tool="detect_anomaly",
                ),
            ]
        ),
    )

    result = orchestrator.run("check temperature then diagnose anomaly")

    assert [todo["status"] for todo in result["todos"]] == ["completed", "completed"]
    assert result["todos"][0]["route"] == "timeseries_query"
    assert result["runtime_trace"]["summary"]["todo_count"] == 2
    assert result["runtime_trace"]["summary"]["completed_todo_count"] == 2
    assert result["runtime_trace"]["summary"]["tool_call_count"] == 2
    hook_events = result["runtime_trace"]["hooks"]
    assert [event["hook"] for event in hook_events[:2]] == ["PreToolUse", "PostToolUse"]
    assert hook_events[0]["tool_name"] == "query_metric"
    assert hook_events[0]["decision"] == "allow"
    assert hook_events[1]["status"] == "success"
    assert hook_events[-1]["hook"] == "RunComplete"
    assert hook_events[-1]["status"] == "completed"
    todo_events = [event["event"] for event in result["runtime_trace"]["todo_events"]]
    assert "todo.created" in todo_events
    assert "todo.completed" in todo_events


def test_langgraph_orchestrator_routes_document_qa_through_retrieval_node():
    baseline = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=SpyAnswerGenerator(),
    )
    orchestrator = LangGraphOrchestrator(baseline)

    result = orchestrator.run("为什么 PUE-like values 不能编造？", task_type="document_qa")

    assert result["route"] == "document_qa"
    assert result["workflow_engine"] == "langgraph"
    assert [step["node"] for step in result["workflow_trace"]] == [
        "planner",
        "execute_plan_step",
        "evidence_aggregator",
        "answer_generator",
        "answer_audit",
    ]
    assert result["workflow_trace"][2]["citation_count"] >= 1


def test_langgraph_orchestrator_uses_injected_planner_and_merges_step_evidence():
    generator = SpyAnswerGenerator()
    baseline = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=generator,
    )
    planner = StaticRoutePlanner()
    orchestrator = LangGraphOrchestrator(baseline, route_planner=planner)

    result = orchestrator.run("当前温度超过上限时是否应该调整控制策略？")

    assert result["route"] == "policy_recommendation"
    assert result["tools"] == ["query_metric", "rule_based_policy"]
    assert len(result["tool_results"]) == 2
    assert result["policy_result"]["policy_name"] == "rule_based"
    assert result["answer"] == "generated:policy_recommendation"
    assert generator.payloads[-1].route == "policy_recommendation"
    assert generator.payloads[-1].tools == ["query_metric", "rule_based_policy"]
    assert len(generator.payloads[-1].tool_results) == 2
    assert planner.calls == [
        {"question": "当前温度超过上限时是否应该调整控制策略？", "task_type": None}
    ]
    planner_trace = result["workflow_trace"][0]
    assert planner_trace["node"] == "planner"
    assert planner_trace["planned_steps"] == ["timeseries_query", "policy_recommendation"]
    assert planner_trace["planner"] == "llm:deepseek:planner-test"
    assert planner_trace["confidence"] == 0.88
    assert planner_trace["fallback_used"] is False
    step_traces = [step for step in result["workflow_trace"] if step["node"] == "execute_plan_step"]
    assert [step["route"] for step in step_traces] == ["timeseries_query", "policy_recommendation"]
    assert step_traces[0]["tools"] == ["query_metric"]
    assert step_traces[1]["tools"] == ["rule_based_policy"]


def test_langgraph_multi_step_plan_generates_final_answer_once():
    generator = SpyAnswerGenerator()
    baseline = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=generator,
    )
    orchestrator = LangGraphOrchestrator(baseline, route_planner=StaticRoutePlanner())

    result = orchestrator.run("当前温度超过上限时是否应该调整控制策略？")

    assert result["route"] == "policy_recommendation"
    assert len(generator.payloads) == 1
    assert generator.payloads[0].tools == ["query_metric", "rule_based_policy"]
    assert len(generator.payloads[0].tool_results) == 2
    assert [step["node"] for step in result["workflow_trace"]] == [
        "planner",
        "execute_plan_step",
        "execute_plan_step",
        "evidence_aggregator",
        "answer_generator",
        "answer_audit",
    ]
    assert result["workflow_trace"][4]["answer_generator"] == "spy"


def test_langgraph_executes_structured_planner_step_parameters():
    baseline = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=SpyAnswerGenerator(),
    )
    orchestrator = LangGraphOrchestrator(baseline, route_planner=StructuredStepPlanner())

    result = orchestrator.run("Planner should choose the exact fan power comparison.")

    assert result["tools"] == ["compare_period"]
    assert result["tool_results"][0]["tool_name"] == "compare_period"
    assert result["tool_results"][0]["metric_name"] == "fan_power"
    assert result["tool_results"][0]["zone_id"] == "zone_a"
    assert result["planned_steps"][0]["tool"] == "compare_period"
    assert result["planned_steps"][0]["metric_name"] == "fan_power"


def test_langgraph_executes_structured_planner_time_window():
    baseline = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=SpyAnswerGenerator(),
    )
    orchestrator = LangGraphOrchestrator(baseline, route_planner=RecentWindowPlanner())

    result = orchestrator.run("Use the last two hours only.")

    tool_result = result["tool_results"][0]
    assert tool_result["tool_name"] == "query_metric"
    assert tool_result["start_time"] == "2026-01-01T01:00:00+00:00"
    assert tool_result["end_time"] == "2026-01-01T03:00:00+00:00"
    assert tool_result["summary"]["count"] == 3


def test_langgraph_marks_unsupported_time_window_fallback_in_tool_result():
    baseline = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=SpyAnswerGenerator(),
    )
    orchestrator = LangGraphOrchestrator(baseline, route_planner=UnsupportedWindowPlanner())

    result = orchestrator.run("Use an unsupported window shape.")

    tool_result = result["tool_results"][0]
    assert tool_result["start_time"] == "2026-01-01T00:00:00+00:00"
    assert tool_result["end_time"] == "2026-01-01T03:00:00+00:00"
    assert tool_result["time_window"] == "last_24"
    assert tool_result["time_window_applied"] == "full_demo_range"
    assert "Unsupported time_window 'last_24'; used full_demo_range." in tool_result["notes"]


def test_langgraph_can_execute_data_quality_tool_from_plan_step():
    baseline = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=SpyAnswerGenerator(),
    )
    orchestrator = LangGraphOrchestrator(
        baseline,
        route_planner=ToolStepPlanner(
            [
                PlanStep(
                    route="timeseries_query",
                    reason="Check telemetry quality before analysis.",
                    tool="data_quality_check",
                )
            ]
        ),
    )

    result = orchestrator.run("检查当前数据质量是否可靠")

    assert result["tools"] == ["data_quality_check"]
    assert result["tool_results"][0]["tool_name"] == "data_quality_check"
    assert "quality_score" in result["tool_results"][0]
    assert result["tool_calls"][0]["tool_name"] == "data_quality_check"
    assert result["tool_calls"][0]["status"] == "success"
    assert result["tool_calls"][0]["permission_decision"] == "allow"
    assert result["tool_calls"][0]["duration_ms"] >= 0


def test_langgraph_can_execute_comfort_and_control_risk_tools_from_plan_steps():
    baseline = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=SpyAnswerGenerator(),
    )
    orchestrator = LangGraphOrchestrator(
        baseline,
        route_planner=ToolStepPlanner(
            [
                PlanStep(
                    route="anomaly_diagnosis",
                    reason="Assess thermal comfort risk.",
                    tool="comfort_risk_assessment",
                ),
                PlanStep(
                    route="timeseries_query",
                    reason="Audit control action stability.",
                    tool="control_action_audit",
                ),
            ]
        ),
    )

    result = orchestrator.run("评估过热风险并检查控制动作是否震荡")

    assert result["tools"] == ["comfort_risk_assessment", "control_action_audit"]
    assert result["tool_results"][0]["tool_name"] == "comfort_risk_assessment"
    assert result["tool_results"][1]["tool_name"] == "control_action_audit"
    assert [call["risk_level"] for call in result["tool_calls"]] == ["advisory", "advisory"]
    assert all(call["audit_required"] for call in result["tool_calls"])


def test_executor_rejects_invalid_tool_input_before_execution():
    baseline = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=SpyAnswerGenerator(),
    )
    orchestrator = LangGraphOrchestrator(
        baseline,
        route_planner=ToolStepPlanner(
            [
                PlanStep(
                    route="timeseries_query",
                    reason="Invalid hotspot top_k should be caught by schema validation.",
                    tool="zone_hotspot_rank",
                    metric_name="zone_temperature",
                    time_window="full_demo_range",
                )
            ]
        ),
    )

    baseline.task_executor.default_tool_inputs["zone_hotspot_rank"] = {"top_k": 0}
    result = orchestrator.run("rank hotspots")

    assert result["tools"] == ["zone_hotspot_rank"]
    assert result["tool_results"][0]["status"] == "error"
    assert result["tool_calls"][0]["status"] == "error"
    assert "top_k" in result["tool_calls"][0]["error"]
    assert result["todos"][0]["status"] == "blocked"
    assert result["runtime_trace"]["summary"]["blocked_todo_count"] == 1
    post_tool_events = [
        event for event in result["runtime_trace"]["hooks"] if event["hook"] == "PostToolUse"
    ]
    assert post_tool_events[-1]["status"] == "error"
    assert result["runtime_trace"]["hooks"][-1]["hook"] == "RunComplete"
    assert result["runtime_trace"]["hooks"][-1]["status"] == "blocked"


def test_executor_repairs_missing_tool_input_and_records_recovery():
    baseline = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=SpyAnswerGenerator(),
    )
    orchestrator = LangGraphOrchestrator(
        baseline,
        route_planner=ToolStepPlanner(
            [
                PlanStep(
                    route="timeseries_query",
                    reason="Rank hotspots with a planner-provided incomplete argument set.",
                    tool="zone_hotspot_rank",
                    metric_name="zone_temperature",
                    time_window="full_demo_range",
                )
            ]
        ),
    )

    baseline.task_executor.default_tool_inputs["zone_hotspot_rank"] = {"top_k": None}
    result = orchestrator.run("rank hotspots")

    assert result["tool_calls"][0]["status"] == "success"
    assert result["tool_calls"][0]["input"]["top_k"] == 3
    assert result["tool_calls"][0]["recovered"] is True
    assert result["runtime_trace"]["summary"]["recovery_count"] == 1
    assert result["runtime_trace"]["recoveries"][0]["strategy"] == "tool_input_repair"
    assert result["runtime_trace"]["recoveries"][0]["status"] == "success"


def test_executor_retries_transient_tool_failure_and_records_recovery(monkeypatch):
    attempts = {"count": 0}
    original_query_metric = executor_module.query_metric

    def flaky_query_metric(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("transient metric backend timeout")
        return original_query_metric(*args, **kwargs)

    monkeypatch.setattr(executor_module, "query_metric", flaky_query_metric)
    baseline = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=SpyAnswerGenerator(),
    )
    orchestrator = LangGraphOrchestrator(
        baseline,
        route_planner=ToolStepPlanner(
            [PlanStep(route="timeseries_query", reason="Read metric.", tool="query_metric")]
        ),
    )

    result = orchestrator.run("read temperature")

    assert attempts["count"] == 2
    assert result["tool_calls"][0]["status"] == "success"
    assert result["tool_calls"][0]["attempts"] == 2
    assert result["runtime_trace"]["recoveries"][0]["strategy"] == "tool_retry"
    assert result["runtime_trace"]["recoveries"][0]["status"] == "success"


def test_document_qa_rewrites_query_when_initial_retrieval_has_no_contexts():
    rag = RewriteOnlyRAGPipeline()
    baseline = BaselineOrchestrator(
        rag_pipeline=rag,
        trajectory=mock_trajectory(),
        answer_generator=SpyAnswerGenerator(),
    )
    orchestrator = LangGraphOrchestrator(baseline)

    result = orchestrator.run("为什么不能说是真实生产遥测？", task_type="document_qa")

    assert len(rag.queries) == 2
    assert "data boundary" in rag.queries[1]
    assert result["citations"] == [{"source_id": "boundary_doc", "title": "Boundary Guidance"}]
    assert result["retrieved_contexts"][0]["retrieval_recovery"] is True
    assert result["runtime_trace"]["recoveries"][0]["strategy"] == "query_rewrite_retry"
    assert result["runtime_trace"]["recoveries"][0]["status"] == "success"


def test_policy_runner_falls_back_to_rule_based_policy_when_backend_is_unavailable():
    def unavailable_policy_runner(state: dict) -> PolicyResult:
        raise RuntimeError("DROPT checkpoint unavailable")

    baseline = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=SpyAnswerGenerator(),
        policy_runner=unavailable_policy_runner,
    )
    orchestrator = LangGraphOrchestrator(baseline)

    result = orchestrator.run("recommend a policy", task_type="policy_recommendation")

    assert result["tool_calls"][0]["status"] == "success"
    assert result["policy_result"]["policy_name"] == "rule_based"
    assert result["policy_result"]["fallback_used"] is True
    assert "DROPT checkpoint unavailable" in result["policy_result"]["fallback_error"]
    assert [event["strategy"] for event in result["runtime_trace"]["recoveries"]] == [
        "tool_retry",
        "policy_fallback",
    ]
    assert result["runtime_trace"]["recoveries"][0]["status"] == "failed"
    assert result["runtime_trace"]["recoveries"][1]["status"] == "success"


def test_policy_runner_retries_backend_before_rule_based_fallback():
    attempts = {"count": 0}

    def flaky_policy_runner(state: dict) -> PolicyResult:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("transient policy backend timeout")
        return PolicyResult(
            policy_name="offline_policy_after_retry",
            input_state_id=state["state_id"],
            recommended_action=[-0.3],
            notes="Recovered without fallback.",
        )

    baseline = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=SpyAnswerGenerator(),
        policy_runner=flaky_policy_runner,
    )
    orchestrator = LangGraphOrchestrator(baseline)

    result = orchestrator.run("recommend a policy", task_type="policy_recommendation")

    assert attempts["count"] == 2
    assert result["tools"] == ["offline_policy_after_retry"]
    assert result["policy_result"]["policy_name"] == "offline_policy_after_retry"
    assert "fallback_used" not in result["policy_result"]
    assert [event["strategy"] for event in result["runtime_trace"]["recoveries"]] == ["tool_retry"]


def test_control_boundary_approval_handler_can_block_tool_execution():
    def deny_control_boundary(request: dict) -> dict:
        return {
            "approved": False,
            "decision": "denied",
            "reason": f"operator denied {request['tool_name']}",
        }

    generator = SpyAnswerGenerator()
    executor = AgentTaskExecutor(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=generator,
        approval_handler=deny_control_boundary,
    )
    baseline = BaselineOrchestrator(task_executor=executor)
    orchestrator = LangGraphOrchestrator(baseline)

    result = orchestrator.run("recommend a policy", task_type="policy_recommendation")

    assert result["tool_calls"][0]["status"] == "blocked"
    assert result["tool_calls"][0]["approval"]["approved"] is False
    assert result["tool_results"][0]["status"] == "blocked"
    assert "policy_result" not in result
    assert generator.payloads[0].policy_result is None
    assert result["todos"][0]["status"] == "blocked"
    assert result["runtime_trace"]["hooks"][0]["approval"]["decision"] == "denied"


def test_blocked_tool_result_is_not_annotated_as_success():
    def deny_control_boundary(request: dict) -> dict:
        return {
            "approved": False,
            "decision": "denied",
            "reason": "operator denied control action audit",
        }

    executor = AgentTaskExecutor(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=SpyAnswerGenerator(),
        approval_handler=deny_control_boundary,
    )

    result, tool_call = executor._execute_tool_call(
        "policy_runner",
        {"state": executor.latest_policy_state()},
        lambda _: {"policy_name": "should_not_run", "recommended_action": [1.0]},
    )

    assert result["status"] == "blocked"
    assert tool_call["status"] == "blocked"
    assert "time_window_applied" not in result


def test_runtime_trace_run_id_is_stable_across_serialization():
    trace = AgentRuntimeTrace()
    trace.create_todos([PlanStep(route="timeseries_query", reason="Read metric.")])

    assert trace.to_dict()["run_id"] == trace.to_dict()["run_id"]


def test_control_boundary_tool_call_requires_policy_boundary_permission():
    baseline = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=SpyAnswerGenerator(),
    )

    evidence = baseline.task_executor.collect_policy_recommendation_evidence(
        "recommend a policy",
        "policy boundary test",
    )

    assert evidence["tool_calls"][0]["tool_name"] == "policy_runner"
    assert evidence["tool_calls"][0]["risk_level"] == "control_boundary"
    assert evidence["tool_calls"][0]["permission_decision"] == "policy_boundary"
    assert evidence["tool_calls"][0]["status"] == "success"
    assert evidence["tool_calls"][0]["approval"]["required"] is True
    assert evidence["tool_calls"][0]["approval"]["decision"] == "policy_boundary"


def test_baseline_and_langgraph_can_share_agent_task_executor():
    executor = AgentTaskExecutor(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=SpyAnswerGenerator(),
    )
    baseline = BaselineOrchestrator(task_executor=executor)
    langgraph = LangGraphOrchestrator(baseline, task_executor=executor)

    baseline_result = baseline.run("zone_a 是否存在温度异常升高？", task_type="anomaly_diagnosis")
    langgraph_result = langgraph.run("zone_a 是否存在温度异常升高？", task_type="anomaly_diagnosis")

    assert baseline_result["route"] == "anomaly_diagnosis"
    assert langgraph_result["route"] == "anomaly_diagnosis"
    assert baseline_result["tools"] == ["detect_anomaly"]
    assert langgraph_result["tools"] == ["detect_anomaly"]
    assert langgraph_result["workflow_trace"][1]["node"] == "execute_plan_step"
    assert langgraph_result["workflow_trace"][1]["route"] == "anomaly_diagnosis"


def test_conversation_context_reaches_baseline_answer_generator():
    generator = SpyAnswerGenerator()
    orchestrator = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=generator,
    )
    context = {
        "session_id": "session-1",
        "recent_turns": [{"question": "previous", "answer": "fan power was stable"}],
        "relevant_memory": [{"text": "Fan power was stable."}],
    }

    result = orchestrator.run(
        "What about that result?",
        task_type="document_qa",
        conversation_context=context,
    )

    assert result["conversation_context"] == context
    assert generator.payloads[0].conversation_context == context


def test_conversation_context_reaches_langgraph_planner_and_trace():
    generator = SpyAnswerGenerator()
    baseline = BaselineOrchestrator(
        rag_pipeline=mock_rag(),
        trajectory=mock_trajectory(),
        answer_generator=generator,
    )
    planner = StaticRoutePlanner()
    orchestrator = LangGraphOrchestrator(baseline, route_planner=planner)
    context = {
        "session_id": "session-1",
        "recent_turns": [{"question": "previous", "answer": "zone_a peaked"}],
        "relevant_memory": [{"text": "zone_a peaked at 30 C"}],
        "budget": {"truncated": False},
    }

    result = orchestrator.run("What about that zone?", conversation_context=context)

    assert planner.calls[-1]["conversation_context"] == context
    assert generator.payloads[-1].conversation_context == context
    assert result["conversation_context"] == context
    assert result["workflow_trace"][0]["memory_context_available"] is True
    assert result["workflow_trace"][0]["memory_recent_turn_count"] == 1

