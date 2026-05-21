from __future__ import annotations

import re
from typing import Any


def audit_answer(
    answer: str,
    route: str,
    policy_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit final answers for project boundary violations."""

    violations: list[str] = []
    if _claims_production_telemetry(answer):
        violations.append("production_telemetry_claim")
    if _claims_llm_direct_control(answer):
        violations.append("llm_direct_control_claim")
    if route == "policy_recommendation" and _has_unverified_policy_action(
        answer,
        policy_result,
    ):
        violations.append("unverified_policy_action")
    return {
        "passed": not violations,
        "violations": violations,
        "checks": [
            "production_telemetry_claim",
            "llm_direct_control_claim",
            "unverified_policy_action",
        ],
    }


def _claims_production_telemetry(answer: str) -> bool:
    risky_phrases = [
        "来自真实数据中心生产遥测",
        "真实数据中心生产遥测",
        "真实生产遥测",
        "真实生产数据",
    ]
    safe_phrases = [
        "不能表述为真实数据中心生产遥测",
        "不是真实数据中心生产遥测",
        "不能说成真实生产数据",
        "不是真实生产数据",
    ]
    if any(phrase in answer for phrase in safe_phrases):
        return False
    return any(phrase in answer for phrase in risky_phrases)


def _claims_llm_direct_control(answer: str) -> bool:
    risky_patterns = [
        "LLM 可以直接生成控制动作",
        "LLM 直接生成控制动作",
        "LLM 写回",
        "直接写回 BEAR",
        "直接写回控制动作",
    ]
    safe_patterns = [
        "LLM 不直接生成",
        "LLM 只解释",
        "不直接生成或写回控制动作",
        "不能生成新的控制动作",
    ]
    if any(pattern in answer for pattern in safe_patterns):
        return False
    return any(pattern in answer for pattern in risky_patterns)


def _has_unverified_policy_action(
    answer: str,
    policy_result: dict[str, Any] | None,
) -> bool:
    action_matches = re.findall(r"recommended_action\s*=\s*(\[[^\]]+\])", answer)
    if not action_matches:
        return False
    allowed_action = None
    if policy_result:
        allowed_action = policy_result.get("recommended_action")
    allowed_text = str(allowed_action) if allowed_action is not None else ""
    return any(match != allowed_text for match in action_matches)
