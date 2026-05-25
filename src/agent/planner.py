from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.agent.deepseek_generator import Transport
from src.agent.router import SUPPORTED_ROUTES, route_task
from src.core.env import load_env_file

MAX_PLAN_STEPS = 3
ALLOWED_STEP_TOOLS = {
    "document_qa": {None, "rag_retrieval"},
    "timeseries_query": {
        None,
        "query_metric",
        "compare_period",
        "plot_metric_trend",
        "compute_energy_breakdown",
    },
    "anomaly_diagnosis": {None, "detect_anomaly"},
    "policy_recommendation": {None, "policy_runner"},
}
TIME_WINDOW_PATTERN = re.compile(
    r"^(full_demo_range|full_range|all|all_data|latest|recent|(?:last|latest|recent)_\d+_(?:hours?|minutes?))$"
)

TIMESERIES_KEYWORDS = [
    "trend",
    "metric",
    "zone",
    "episode",
    "temperature",
    "temp",
    "温度",
    "功率",
    "能耗",
    "最近",
    "轨迹",
    "趋势",
    "时序",
    "指标",
]
ANOMALY_KEYWORDS = [
    "anomaly",
    "alarm",
    "abnormal",
    "异常",
    "告警",
    "报警",
    "升高",
    "过热",
    "越限",
]
POLICY_KEYWORDS = [
    "policy",
    "control",
    "recommend",
    "adjust",
    "strategy",
    "策略",
    "控制",
    "建议",
    "推荐",
    "调整",
    "怎么办",
]


@dataclass(frozen=True)
class PlanStep:
    route: str
    reason: str
    tool: str | None = None
    metric_name: str | None = None
    zone_id: str | None = None
    time_window: str | None = None


@dataclass(frozen=True)
class PlanDecision:
    steps: list[PlanStep]
    planner: str
    confidence: float
    fallback_used: bool = False


class RoutePlanner(Protocol):
    def plan(
        self,
        question: str,
        task_type: str | None = None,
        conversation_context: dict[str, Any] | None = None,
    ) -> PlanDecision:
        """Plan one to three controlled route steps for a user question."""


class DeterministicRoutePlanner:
    name = "deterministic"

    def plan(
        self,
        question: str,
        task_type: str | None = None,
        conversation_context: dict[str, Any] | None = None,
    ) -> PlanDecision:
        if task_type in SUPPORTED_ROUTES:
            decision = route_task(question, task_type=task_type)
            return PlanDecision(
                steps=[_step_from_route_decision(question, decision.route, decision.reason)],
                planner=self.name,
                confidence=1.0,
                fallback_used=False,
            )

        steps = _infer_steps(question)
        return PlanDecision(
            steps=steps,
            planner=self.name,
            confidence=0.65,
            fallback_used=False,
        )


class LLMRoutePlanner:
    """LLM route planner constrained to a validated route-plan schema."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 20.0,
        fallback: RoutePlanner | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.fallback = fallback or DeterministicRoutePlanner()
        self.transport = transport or _default_transport

    def plan(
        self,
        question: str,
        task_type: str | None = None,
        conversation_context: dict[str, Any] | None = None,
    ) -> PlanDecision:
        if task_type in SUPPORTED_ROUTES:
            return self.fallback.plan(question, task_type=task_type)

        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _system_prompt()},
                    {"role": "user", "content": json.dumps({"question": question}, ensure_ascii=False)},
                ],
                "temperature": 0.0,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = self.transport(
                f"{self.base_url}/chat/completions",
                headers,
                body,
                self.timeout_seconds,
            )
            content = str(response["choices"][0]["message"]["content"])
            return _decision_from_llm_payload(
                content=content,
                planner=f"llm:{self.provider}:{self.model}",
            )
        except Exception as exc:
            fallback_decision = self.fallback.plan(question, task_type=task_type)
            return PlanDecision(
                steps=[
                    PlanStep(
                        route=step.route,
                        reason=f"LLM route planning failed ({exc}); {step.reason}",
                        tool=step.tool,
                        metric_name=step.metric_name,
                        zone_id=step.zone_id,
                        time_window=step.time_window,
                    )
                    for step in fallback_decision.steps
                ],
                planner=fallback_decision.planner,
                confidence=fallback_decision.confidence,
                fallback_used=True,
            )


def build_route_planner_from_env(
    project_root: str | Path | None = None,
    transport: Transport | None = None,
) -> RoutePlanner:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    load_env_file(root / ".env")
    provider = os.getenv("LANGGRAPH_PLANNER_PROVIDER", "auto").strip().lower()
    if provider in {"", "auto"}:
        provider = "deepseek" if os.getenv("DEEPSEEK_API_KEY", "").strip() else "deterministic"
    if provider in {"deterministic", "rule_based"}:
        return DeterministicRoutePlanner()
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            return DeterministicRoutePlanner()
        return LLMRoutePlanner(
            provider="deepseek",
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("LANGGRAPH_PLANNER_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
            timeout_seconds=float(os.getenv("LANGGRAPH_PLANNER_TIMEOUT_SECONDS", "20")),
            transport=transport,
        )
    return DeterministicRoutePlanner()


def _infer_steps(question: str) -> list[PlanStep]:
    normalized = question.lower()
    steps: list[PlanStep] = []

    inferred = _infer_steps_from_keywords(question, normalized)
    if inferred:
        return _validate_steps(inferred)

    if any(
        keyword in normalized
        for keyword in ["trend", "metric", "zone", "episode", "temperature", "娓╁害", "鍔熺巼"]
    ):
        decision = route_task(question, task_type="timeseries_query")
        steps.append(_step_from_route_decision(question, decision.route, decision.reason))
    if any(keyword in normalized for keyword in ["anomaly", "alarm", "abnormal", "寮傚父", "鍛婅"]):
        decision = route_task(question, task_type="anomaly_diagnosis")
        steps.append(_step_from_route_decision(question, decision.route, decision.reason))
    if any(
        keyword in normalized
        for keyword in ["policy", "control", "recommend", "adjust", "strategy", "绛栫暐", "鎺у埗", "璋冩暣"]
    ):
        decision = route_task(question, task_type="policy_recommendation")
        steps.append(_step_from_route_decision(question, decision.route, decision.reason))

    if not steps:
        decision = route_task(question)
        steps.append(_step_from_route_decision(question, decision.route, decision.reason))

    return _validate_steps(steps)


def _infer_steps_from_keywords(question: str, normalized: str) -> list[PlanStep]:
    steps: list[PlanStep] = []
    if _contains_any(normalized, TIMESERIES_KEYWORDS):
        decision = route_task(question, task_type="timeseries_query")
        steps.append(_step_from_route_decision(question, decision.route, decision.reason))
    if _contains_any(normalized, ANOMALY_KEYWORDS):
        decision = route_task(question, task_type="anomaly_diagnosis")
        steps.append(_step_from_route_decision(question, decision.route, decision.reason))
    if _contains_any(normalized, POLICY_KEYWORDS):
        decision = route_task(question, task_type="policy_recommendation")
        steps.append(_step_from_route_decision(question, decision.route, decision.reason))
    return steps


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _system_prompt() -> str:
    return (
        "You are the route planner for DataCenter-HVAC Copilot. "
        "Return only JSON with keys steps and confidence. steps must contain 1 to 3 objects. "
        "Each step object must have route and reason. "
        "Allowed routes are exactly: document_qa, timeseries_query, anomaly_diagnosis, policy_recommendation. "
        "If policy_recommendation is included, it must be the final step. "
        "Do not call tools, write Python, or produce control actions."
    )


def _parse_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    return json.loads(stripped)


def _step_from_llm_item(item: dict[str, Any]) -> PlanStep:
    route = str(item["route"])
    return PlanStep(
        route=route,
        reason=str(item.get("reason") or "LLM planned route step."),
        tool=_optional_string(item.get("tool")) or _default_tool_for_route(route),
        metric_name=_optional_string(item.get("metric_name")) or _default_metric_for_route(route),
        zone_id=_optional_string(item.get("zone_id")),
        time_window=_optional_string(item.get("time_window")) or _default_time_window_for_route(route),
    )


def _step_from_route_decision(question: str, route: str, reason: str) -> PlanStep:
    return PlanStep(
        route=route,
        reason=reason,
        tool=_default_tool_for_route(route, question),
        metric_name=_default_metric_for_route(route, question),
        zone_id=None,
        time_window=_default_time_window_for_route(route),
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _default_tool_for_route(route: str, question: str = "") -> str | None:
    normalized = question.lower()
    if route == "timeseries_query":
        if any(token in normalized for token in ["构成", "breakdown", "能耗字段", "能耗"]):
            return "compute_energy_breakdown"
        if any(token in normalized for token in ["趋势", "trend", "折线图", "序列", "画"]):
            return "plot_metric_trend"
        if any(token in normalized for token in ["比较", "对比", "前后", "变化", "compare"]):
            return "compare_period"
        return "query_metric"
    if route == "anomaly_diagnosis":
        return "detect_anomaly"
    if route == "policy_recommendation":
        return "policy_runner"
    return None


def _default_metric_for_route(route: str, question: str = "") -> str | None:
    if route not in {"timeseries_query", "anomaly_diagnosis"}:
        return None
    normalized = question.lower()
    for metric_name in [
        "fan_power",
        "control_action",
        "outdoor_temp",
        "internal_load",
        "cooling_power",
        "hvac_power",
        "zone_temperature",
    ]:
        if metric_name in normalized:
            return metric_name
    if "风机" in question:
        return "fan_power"
    if "控制" in question:
        return "control_action"
    if "室外" in question:
        return "outdoor_temp"
    if "负载" in question:
        return "internal_load"
    return "zone_temperature"


def _default_time_window_for_route(route: str) -> str | None:
    if route in {"timeseries_query", "anomaly_diagnosis"}:
        return "full_demo_range"
    return None


def _decision_from_llm_payload(*, content: str, planner: str) -> PlanDecision:
    parsed = _parse_json_object(content)
    raw_steps = parsed.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("steps must be a list")
    steps = [
        _step_from_llm_item(item)
        for item in raw_steps
        if isinstance(item, dict)
    ]
    return PlanDecision(
        steps=_validate_steps(steps),
        planner=planner,
        confidence=_bounded_confidence(parsed.get("confidence", 0.5)),
        fallback_used=False,
    )


def _validate_steps(steps: list[PlanStep]) -> list[PlanStep]:
    if not steps:
        raise ValueError("plan must contain at least one step")
    if len(steps) > MAX_PLAN_STEPS:
        raise ValueError("plan must contain at most 3 steps")

    seen: set[str] = set()
    validated: list[PlanStep] = []
    for step in steps:
        if step.route not in SUPPORTED_ROUTES:
            raise ValueError(f"unsupported route: {step.route}")
        if step.tool not in ALLOWED_STEP_TOOLS[step.route]:
            raise ValueError(f"unsupported tool for {step.route}: {step.tool}")
        if step.time_window and not _is_supported_time_window(step.time_window):
            raise ValueError(f"unsupported time_window for {step.route}: {step.time_window}")
        if step.route in seen:
            continue
        seen.add(step.route)
        validated.append(step)

    policy_indexes = [
        index for index, step in enumerate(validated) if step.route == "policy_recommendation"
    ]
    if policy_indexes and policy_indexes[-1] != len(validated) - 1:
        raise ValueError("policy_recommendation must be the final step")
    return validated


def _is_supported_time_window(value: str) -> bool:
    normalized = value.strip().lower().replace("-", "_")
    return bool(TIME_WINDOW_PATTERN.fullmatch(normalized))


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
