from __future__ import annotations

from src.agent.answer_audit import audit_answer


def test_audit_flags_production_telemetry_claim() -> None:
    result = audit_answer(
        answer="这些数据来自真实数据中心生产遥测，可以直接反映生产负载。",
        route="document_qa",
        policy_result=None,
    )

    assert result["passed"] is False
    assert "production_telemetry_claim" in result["violations"]


def test_audit_flags_llm_direct_control_claim() -> None:
    result = audit_answer(
        answer="LLM 可以直接生成控制动作并写回 BEAR 环境。",
        route="policy_recommendation",
        policy_result={"recommended_action": [0.0, 0.0]},
    )

    assert result["passed"] is False
    assert "llm_direct_control_claim" in result["violations"]


def test_audit_flags_policy_action_not_present_in_policy_result() -> None:
    result = audit_answer(
        answer="建议采用 recommended_action=[-0.2, -0.2]。",
        route="policy_recommendation",
        policy_result={"recommended_action": [-0.1, -0.1]},
    )

    assert result["passed"] is False
    assert "unverified_policy_action" in result["violations"]


def test_audit_passes_grounded_policy_answer() -> None:
    result = audit_answer(
        answer="策略工具返回 recommended_action=[-0.1, -0.1]，LLM 只解释工具结果。",
        route="policy_recommendation",
        policy_result={"recommended_action": [-0.1, -0.1]},
    )

    assert result["passed"] is True
    assert result["violations"] == []
