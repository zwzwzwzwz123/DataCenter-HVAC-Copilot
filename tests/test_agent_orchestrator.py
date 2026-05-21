from pathlib import Path

import pandas as pd

from src.agent.answer_generator import AnswerGeneratorInput, GeneratedAnswer
from src.agent.orchestrator import BaselineOrchestrator
from src.agent.router import route_task
from src.policies.base import PolicyResult
from src.retrieval.chunking import chunk_document
from src.retrieval.loader import load_markdown_document
from src.retrieval.rag import ExtractiveRAGPipeline
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


def test_orchestrator_selects_energy_breakdown_for_energy_breakdown_requests():
    orchestrator = BaselineOrchestrator(rag_pipeline=mock_rag(), trajectory=mock_trajectory())

    result = orchestrator.run(
        "当前轨迹的冷却能耗构成是什么？",
        task_type="timeseries_query",
    )

    assert result["tools"] == ["compute_energy_breakdown"]
    assert result["tool_results"][0]["tool_name"] == "compute_energy_breakdown"


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

