from __future__ import annotations

import json

from src.agent.intent_classifier import (
    LLMIntentClassifier,
    OllamaIntentClassifier,
    RuleBasedIntentClassifier,
    build_intent_classifier_from_env,
)


class FakeIntentTransport:
    def __init__(self, response: dict | None = None, error: Exception | None = None) -> None:
        self.response = response or {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "route": "policy_recommendation",
                                "confidence": 0.91,
                                "reason": "asks for control policy",
                            }
                        )
                    }
                }
            ]
        }
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, url: str, headers: dict[str, str], body: bytes, timeout: float) -> dict:
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "payload": json.loads(body.decode("utf-8")),
                "timeout": timeout,
            }
        )
        if self.error:
            raise self.error
        return self.response


class FakeOllamaIntentTransport(FakeIntentTransport):
    def __init__(self, response: dict | None = None, error: Exception | None = None) -> None:
        super().__init__(
            response=response
            or {
                "message": {
                    "content": json.dumps(
                        {
                            "route": "timeseries_query",
                            "confidence": 0.88,
                            "reason": "asks for metric trend analysis",
                        }
                    )
                }
            },
            error=error,
        )


def test_rule_based_intent_classifier_preserves_explicit_task_type() -> None:
    classifier = RuleBasedIntentClassifier()

    decision = classifier.classify(
        "这句话本身不重要",
        task_type="timeseries_query",
    )

    assert decision.route == "timeseries_query"
    assert decision.classifier == "rule_based"
    assert decision.fallback_used is False
    assert decision.confidence == 1.0


def test_llm_intent_classifier_builds_prompt_and_parses_route() -> None:
    transport = FakeIntentTransport()
    classifier = LLMIntentClassifier(
        provider="deepseek",
        api_key="test-key",
        base_url="https://example.deepseek.test",
        model="intent-test",
        transport=transport,
    )

    decision = classifier.classify("当前温度超过上限时是否应该调整控制策略？")

    assert decision.route == "policy_recommendation"
    assert decision.classifier == "llm:deepseek:intent-test"
    assert decision.confidence == 0.91
    assert decision.fallback_used is False
    assert "control policy" in decision.reason
    assert transport.calls[0]["url"] == "https://example.deepseek.test/chat/completions"
    payload = transport.calls[0]["payload"]
    assert payload["model"] == "intent-test"
    assert "document_qa" in payload["messages"][0]["content"]
    assert "当前温度超过上限" in payload["messages"][1]["content"]


def test_llm_intent_classifier_falls_back_when_response_is_invalid() -> None:
    transport = FakeIntentTransport(
        response={"choices": [{"message": {"content": '{"route": "unsupported"}'}}]}
    )
    classifier = LLMIntentClassifier(
        provider="deepseek",
        api_key="test-key",
        base_url="https://example.deepseek.test",
        model="intent-test",
        transport=transport,
    )

    decision = classifier.classify("zone_a 是否存在温度异常升高？")

    assert decision.route == "anomaly_diagnosis"
    assert decision.classifier == "rule_based"
    assert decision.fallback_used is True
    assert "unsupported" in decision.reason


def test_ollama_intent_classifier_uses_local_chat_api() -> None:
    transport = FakeOllamaIntentTransport()
    classifier = OllamaIntentClassifier(
        base_url="http://localhost:11434",
        model="qwen2.5:7b",
        transport=transport,
    )

    decision = classifier.classify("画一下 zone_temperature 的趋势")

    assert decision.route == "timeseries_query"
    assert decision.classifier == "llm:ollama:qwen2.5:7b"
    assert decision.confidence == 0.88
    assert decision.fallback_used is False
    assert transport.calls[0]["url"] == "http://localhost:11434/api/chat"
    payload = transport.calls[0]["payload"]
    assert payload["model"] == "qwen2.5:7b"
    assert payload["stream"] is False
    assert "document_qa" in payload["messages"][0]["content"]


def test_ollama_intent_classifier_falls_back_when_response_is_invalid() -> None:
    transport = FakeOllamaIntentTransport(response={"message": {"content": "not json"}})
    classifier = OllamaIntentClassifier(
        base_url="http://localhost:11434",
        model="qwen2.5:7b",
        transport=transport,
    )

    decision = classifier.classify("zone_a 是否存在温度异常升高？")

    assert decision.route == "anomaly_diagnosis"
    assert decision.classifier == "rule_based"
    assert decision.fallback_used is True
    assert "LLM intent classification failed" in decision.reason


def test_build_intent_classifier_from_env_defaults_to_rule_based(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LANGGRAPH_INTENT_PROVIDER", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    classifier = build_intent_classifier_from_env(project_root=tmp_path)
    decision = classifier.classify("当前温度超过上限时是否应该调整控制策略？")

    assert decision.classifier == "rule_based"
    assert decision.route == "policy_recommendation"


def test_build_intent_classifier_from_env_uses_deepseek_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_INTENT_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("LANGGRAPH_INTENT_MODEL", "intent-model")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.deepseek.test")

    classifier = build_intent_classifier_from_env(transport=FakeIntentTransport())
    decision = classifier.classify("当前温度超过上限时是否应该调整控制策略？")

    assert decision.classifier == "llm:deepseek:intent-model"
    assert decision.route == "policy_recommendation"
    monkeypatch.delenv("LANGGRAPH_INTENT_PROVIDER", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("LANGGRAPH_INTENT_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)


def test_build_intent_classifier_from_env_uses_ollama_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_INTENT_PROVIDER", "ollama")
    monkeypatch.setenv("LANGGRAPH_INTENT_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test:11434")

    classifier = build_intent_classifier_from_env(transport=FakeOllamaIntentTransport())
    decision = classifier.classify("画一下 zone_temperature 的趋势")

    assert decision.classifier == "llm:ollama:qwen2.5:7b"
    assert decision.route == "timeseries_query"
    monkeypatch.delenv("LANGGRAPH_INTENT_PROVIDER", raising=False)
    monkeypatch.delenv("LANGGRAPH_INTENT_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
