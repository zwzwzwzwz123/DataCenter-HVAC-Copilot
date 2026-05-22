from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.agent.deepseek_generator import Transport
from src.agent.router import SUPPORTED_ROUTES, RouteDecision, route_task
from src.core.env import load_env_file


@dataclass(frozen=True)
class IntentDecision:
    route: str
    required_tools: list[str]
    reason: str
    classifier: str
    confidence: float
    fallback_used: bool = False


class IntentClassifier(Protocol):
    def classify(self, question: str, task_type: str | None = None) -> IntentDecision:
        """Classify a user question into one supported agent route."""


class RuleBasedIntentClassifier:
    name = "rule_based"

    def classify(self, question: str, task_type: str | None = None) -> IntentDecision:
        decision = route_task(question, task_type=task_type)
        return _from_route_decision(
            decision,
            classifier=self.name,
            confidence=1.0 if task_type else 0.65,
            fallback_used=False,
        )


class LLMIntentClassifier:
    """LLM route classifier with rule-based fallback."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 20.0,
        fallback: IntentClassifier | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.fallback = fallback or RuleBasedIntentClassifier()
        self.transport = transport or _default_transport

    def classify(self, question: str, task_type: str | None = None) -> IntentDecision:
        if task_type in SUPPORTED_ROUTES:
            return self.fallback.classify(question, task_type=task_type)

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
                question=question,
                content=content,
                classifier=f"llm:{self.provider}:{self.model}",
            )
        except Exception as exc:
            fallback_decision = self.fallback.classify(question, task_type=task_type)
            return IntentDecision(
                route=fallback_decision.route,
                required_tools=fallback_decision.required_tools,
                reason=f"LLM intent classification failed ({exc}); {fallback_decision.reason}",
                classifier=fallback_decision.classifier,
                confidence=fallback_decision.confidence,
                fallback_used=True,
            )


class OllamaIntentClassifier:
    """Ollama /api/chat route classifier with rule-based fallback."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 20.0,
        fallback: IntentClassifier | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.fallback = fallback or RuleBasedIntentClassifier()
        self.transport = transport or _default_transport

    def classify(self, question: str, task_type: str | None = None) -> IntentDecision:
        if task_type in SUPPORTED_ROUTES:
            return self.fallback.classify(question, task_type=task_type)

        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _system_prompt()},
                    {"role": "user", "content": json.dumps({"question": question}, ensure_ascii=False)},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        try:
            response = self.transport(
                f"{self.base_url}/api/chat",
                headers,
                body,
                self.timeout_seconds,
            )
            content = str(response["message"]["content"])
            return _decision_from_llm_payload(
                question=question,
                content=content,
                classifier=f"llm:ollama:{self.model}",
            )
        except Exception as exc:
            fallback_decision = self.fallback.classify(question, task_type=task_type)
            return IntentDecision(
                route=fallback_decision.route,
                required_tools=fallback_decision.required_tools,
                reason=f"LLM intent classification failed ({exc}); {fallback_decision.reason}",
                classifier=fallback_decision.classifier,
                confidence=fallback_decision.confidence,
                fallback_used=True,
            )


def _from_route_decision(
    decision: RouteDecision,
    *,
    classifier: str,
    confidence: float,
    fallback_used: bool,
) -> IntentDecision:
    return IntentDecision(
        route=decision.route,
        required_tools=decision.required_tools,
        reason=decision.reason,
        classifier=classifier,
        confidence=confidence,
        fallback_used=fallback_used,
    )


def build_intent_classifier_from_env(
    project_root: str | Path | None = None,
    transport: Transport | None = None,
) -> IntentClassifier:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    load_env_file(root / ".env")
    provider = os.getenv("LANGGRAPH_INTENT_PROVIDER", "auto").strip().lower()
    if provider in {"", "auto"}:
        provider = "deepseek" if os.getenv("DEEPSEEK_API_KEY", "").strip() else "rule_based"
    if provider in {"rule_based", "deterministic"}:
        return RuleBasedIntentClassifier()
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            return RuleBasedIntentClassifier()
        return LLMIntentClassifier(
            provider="deepseek",
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("LANGGRAPH_INTENT_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
            timeout_seconds=float(os.getenv("LANGGRAPH_INTENT_TIMEOUT_SECONDS", "20")),
            transport=transport,
        )
    if provider == "ollama":
        return OllamaIntentClassifier(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("LANGGRAPH_INTENT_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:7b")),
            timeout_seconds=float(os.getenv("LANGGRAPH_INTENT_TIMEOUT_SECONDS", "20")),
            transport=transport,
        )
    return RuleBasedIntentClassifier()


def _system_prompt() -> str:
    return (
        "You are the intent classifier for DataCenter-HVAC Copilot. "
        "Choose exactly one route from: document_qa, timeseries_query, "
        "anomaly_diagnosis, policy_recommendation. "
        "Return only JSON with keys route, confidence, reason. "
        "Use document_qa for conceptual documentation questions, timeseries_query for metric or trajectory analysis, "
        "anomaly_diagnosis for abnormal behavior or alarms, and policy_recommendation for control or strategy requests."
    )


def _parse_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    return json.loads(stripped)


def _decision_from_llm_payload(*, question: str, content: str, classifier: str) -> IntentDecision:
    parsed = _parse_json_object(content)
    route = str(parsed["route"])
    if route not in SUPPORTED_ROUTES:
        raise ValueError(f"unsupported route: {route}")
    base = route_task(question, task_type=route)
    return IntentDecision(
        route=route,
        required_tools=base.required_tools,
        reason=str(parsed.get("reason") or base.reason),
        classifier=classifier,
        confidence=_bounded_confidence(parsed.get("confidence", 0.5)),
        fallback_used=False,
    )


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
