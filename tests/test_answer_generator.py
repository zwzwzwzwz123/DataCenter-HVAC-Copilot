from __future__ import annotations

from src.agent.answer_generator import (
    AnswerGeneratorInput,
    DeterministicAnswerGenerator,
)


def test_deterministic_generator_mentions_conversation_memory_without_replacing_fresh_evidence() -> None:
    generator = DeterministicAnswerGenerator()
    result = generator.generate(
        AnswerGeneratorInput(
            question="What about that zone?",
            route="timeseries_query",
            route_reason="follow-up",
            tools=["query_metric"],
            tool_results=[{"metric_name": "zone_temperature", "summary": {"max": 29.0}}],
            conversation_context={
                "recent_turns": [{"question": "previous", "answer": "zone_a peaked at 30 C"}],
                "relevant_memory": [{"text": "Earlier zone_a peaked at 30 C"}],
            },
        )
    )

    assert "Conversation memory" in result.answer
    assert "Earlier zone_a peaked at 30 C" in result.answer
    assert "current evidence remains authoritative" in result.answer
    assert "29.0" in result.answer


def test_document_answer_preserves_citations_and_context_evidence() -> None:
    generator = DeterministicAnswerGenerator()
    result = generator.generate(
        AnswerGeneratorInput(
            question="送风温度升高有什么风险？",
            route="document_qa",
            route_reason="explicit document question",
            retrieved_contexts=[
                {
                    "title": "Supply Air Reset Risk",
                    "source_id": "supply_air_reset_risk",
                    "text": "送风温度上调可能降低冷却能耗，但会增加局部热点和舒适度越界风险。",
                }
            ],
            citations=[
                {
                    "source_id": "supply_air_reset_risk",
                    "title": "Supply Air Reset Risk",
                }
            ],
        )
    )

    assert "Supply Air Reset Risk" in result.answer
    assert "supply_air_reset_risk" in result.answer
    assert "局部热点" in result.answer
    assert result.generator == "deterministic_grounded"


def test_timeseries_answer_mentions_tool_evidence_and_simulation_data_source() -> None:
    generator = DeterministicAnswerGenerator()
    result = generator.generate(
        AnswerGeneratorInput(
            question="最近窗口温度最大值是多少？",
            route="timeseries_query",
            route_reason="metric query",
            tools=["query_metric"],
            tool_results=[
                {
                    "metric_name": "zone_temperature",
                    "summary": {"max": 30.0, "mean": 24.5},
                    "records": [{"timestamp": "2026-01-01T00:00:00Z", "value": 30.0}],
                }
            ],
            data_source={
                "kind": "bear_sample_csv",
                "path": "BEAR/BEAR/Data/Exercise2A-mytest.csv",
            },
        )
    )

    assert "query_metric" in result.answer
    assert "zone_temperature" in result.answer
    assert "30.0" in result.answer
    assert "HVAC 仿真" in result.answer
    assert "不能表述为真实数据中心生产遥测" in result.answer
    assert "来自真实数据中心生产遥测" not in result.answer


def test_policy_answer_uses_policy_result_without_inventing_action() -> None:
    generator = DeterministicAnswerGenerator()
    result = generator.generate(
        AnswerGeneratorInput(
            question="当前是否应该调整控制？",
            route="policy_recommendation",
            route_reason="policy request",
            tools=["rule_based_policy"],
            tool_results=[
                {
                    "policy_name": "rule_based",
                    "recommended_action": [-0.1, -0.1],
                    "estimated_energy": None,
                    "estimated_comfort_violations": 0.0,
                    "notes": "Temperature is within comfort bounds; keep small cooling adjustment.",
                }
            ],
        )
    )

    assert "rule_based" in result.answer
    assert "[-0.1, -0.1]" in result.answer
    assert "控制动作来自策略工具" in result.answer
    assert "-0.2" not in result.answer


def test_no_evidence_answer_states_uncertainty() -> None:
    generator = DeterministicAnswerGenerator()
    result = generator.generate(
        AnswerGeneratorInput(
            question="没有证据的问题",
            route="document_qa",
            route_reason="fallback",
        )
    )

    assert "证据不足" in result.answer
