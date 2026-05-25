from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class AnswerGeneratorInput:
    question: str
    route: str
    route_reason: str
    retrieved_contexts: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    data_source: dict[str, str] | None = None
    policy_result: dict[str, Any] | None = None
    conversation_context: dict[str, Any] | None = None


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    generator: str


class AnswerGenerator(Protocol):
    def generate(self, payload: AnswerGeneratorInput) -> GeneratedAnswer:
        """Generate an answer using only the evidence in payload."""


class DeterministicAnswerGenerator:
    """Evidence-only fallback generator.

    This class intentionally uses templates instead of open-ended generation so
    the project remains reproducible without API keys.
    """

    name = "deterministic_grounded"

    def generate(self, payload: AnswerGeneratorInput) -> GeneratedAnswer:
        sections = [
            f"问题：{payload.question}",
            f"路由：{payload.route}（{payload.route_reason}）",
        ]
        sections.extend(_data_source_lines(payload.data_source))

        has_evidence = bool(payload.retrieved_contexts or payload.tool_results or payload.citations)
        if not has_evidence:
            sections.append("结论：证据不足，无法基于当前检索上下文或工具结果给出确定回答。")
            return GeneratedAnswer(answer="\n".join(sections), generator=self.name)

        if payload.route == "policy_recommendation":
            sections.append("结论：以下控制建议只解释控制/策略工具返回的结果，LLM 不直接生成或写回控制动作。")
        else:
            sections.append("结论：以下回答仅基于当前检索证据和工具结果。")

        if payload.tool_results:
            sections.append(_format_tool_evidence(payload.tools, payload.tool_results))
        if payload.retrieved_contexts:
            sections.append(_format_context_evidence(payload.retrieved_contexts))
        if payload.citations:
            sections.append(_format_citations(payload.citations))

        return GeneratedAnswer(answer="\n".join(sections), generator=self.name)


def _data_source_lines(data_source: dict[str, str] | None) -> list[str]:
    if not data_source:
        return []
    kind = data_source.get("kind", "unknown")
    path = data_source.get("path", "")
    return [
        f"数据源：{kind}（{path}）。",
        "数据边界：该轨迹用于 HVAC 仿真 / 可控代理场景分析，不能表述为真实数据中心生产遥测。",
    ]


def _format_tool_evidence(tools: list[str], tool_results: list[dict[str, Any]]) -> str:
    lines = ["工具证据："]
    for index, result in enumerate(tool_results):
        tool_name = tools[index] if index < len(tools) else result.get("tool_name", "unknown_tool")
        if "policy_name" in result or "recommended_action" in result:
            lines.extend(_format_policy_result(tool_name, result))
        elif "summary" in result:
            lines.append(f"- {tool_name}: metric={result.get('metric_name', 'unknown')}, summary={result['summary']}")
        elif "total" in result:
            lines.append(f"- {tool_name}: total={result.get('total')}, breakdown={result.get('breakdown', {})}")
        elif "anomalies" in result:
            lines.append(f"- {tool_name}: anomalies={len(result.get('anomalies', []))}")
        else:
            lines.append(f"- {tool_name}: {result}")
    return "\n".join(lines)


def _format_policy_result(tool_name: str, result: dict[str, Any]) -> list[str]:
    lines = [f"- {tool_name}: policy={result.get('policy_name', 'unknown')}"]
    if "recommended_action" in result:
        lines.append(f"  - recommended_action={result['recommended_action']}（控制动作来自策略工具）")
    if result.get("estimated_energy") is not None:
        lines.append(f"  - estimated_energy={result['estimated_energy']}")
    if result.get("estimated_comfort_violations") is not None:
        lines.append(f"  - estimated_comfort_violations={result['estimated_comfort_violations']}")
    if result.get("notes"):
        lines.append(f"  - notes={result['notes']}")
    return lines


def _format_context_evidence(contexts: list[dict[str, Any]]) -> str:
    lines = ["文档证据："]
    for context in contexts[:3]:
        title = context.get("title") or context.get("source_title") or "unknown_title"
        source_id = context.get("source_id", "unknown_source")
        text = str(context.get("text") or context.get("content") or "")
        snippet = text[:180]
        lines.append(f"- [{source_id}] {title}: {snippet}")
    return "\n".join(lines)


def _format_citations(citations: list[dict[str, Any]]) -> str:
    lines = ["引用："]
    for citation in citations:
        source_id = citation.get("source_id", "unknown_source")
        title = citation.get("title") or citation.get("source_title") or "unknown_title"
        lines.append(f"- {source_id}: {title}")
    return "\n".join(lines)
