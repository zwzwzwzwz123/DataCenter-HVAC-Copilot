from __future__ import annotations

from pydantic import BaseModel, Field


SUPPORTED_ROUTES = {
    "document_qa",
    "timeseries_query",
    "anomaly_diagnosis",
    "policy_recommendation",
}


class RouteDecision(BaseModel):
    route: str
    required_tools: list[str] = Field(default_factory=list)
    reason: str


def route_task(question: str, task_type: str | None = None) -> RouteDecision:
    if task_type in SUPPORTED_ROUTES:
        return _decision_for_route(task_type, reason="Used explicit eval task_type.")

    normalized = question.lower()
    if any(keyword in normalized for keyword in ["异常", "anomaly", "告警", "升高"]):
        return _decision_for_route("anomaly_diagnosis", reason="Question mentions anomaly-like behavior.")
    if any(keyword in normalized for keyword in ["策略", "控制", "policy", "调整"]):
        return _decision_for_route("policy_recommendation", reason="Question asks for control or policy recommendation.")
    if any(keyword in normalized for keyword in ["最近", "最大值", "温度", "功率", "zone", "episode"]):
        return _decision_for_route("timeseries_query", reason="Question asks for trajectory metric analysis.")
    return _decision_for_route("document_qa", reason="Defaulted to document QA.")


def _decision_for_route(route: str, reason: str) -> RouteDecision:
    tool_map = {
        "document_qa": [],
        "timeseries_query": ["query_metric"],
        "anomaly_diagnosis": ["detect_anomaly"],
        "policy_recommendation": ["rule_based_policy"],
    }
    return RouteDecision(route=route, required_tools=tool_map[route], reason=reason)

