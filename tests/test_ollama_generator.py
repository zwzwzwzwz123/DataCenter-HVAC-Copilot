from __future__ import annotations

import json

from src.agent.answer_generator import AnswerGeneratorInput
from src.agent.ollama_generator import OllamaAnswerGenerator


class FakeOllamaTransport:
    def __init__(self, response: dict | None = None, error: Exception | None = None) -> None:
        self.response = response or {"message": {"content": "基于证据的 Ollama 回答"}}
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


def test_ollama_generator_builds_chat_payload() -> None:
    transport = FakeOllamaTransport()
    generator = OllamaAnswerGenerator(
        base_url="http://ollama.test:11434",
        model="qwen2.5:7b",
        transport=transport,
    )

    result = generator.generate(
        AnswerGeneratorInput(
            question="送风温度升高有什么风险？",
            route="document_qa",
            route_reason="explicit document question",
            retrieved_contexts=[
                {
                    "source_id": "supply_air_reset_risk",
                    "title": "Supply Air Reset Risk",
                    "text": "送风温度上调可能降低能耗，但会增加局部热点风险。",
                }
            ],
            citations=[{"source_id": "supply_air_reset_risk", "title": "Supply Air Reset Risk"}],
        )
    )

    assert result.answer == "基于证据的 Ollama 回答"
    assert result.generator == "ollama:qwen2.5:7b"
    assert transport.calls[0]["url"] == "http://ollama.test:11434/api/chat"
    payload = transport.calls[0]["payload"]
    assert payload["model"] == "qwen2.5:7b"
    assert payload["stream"] is False
    assert "不能生成新的控制动作" in payload["messages"][0]["content"]
    assert "supply_air_reset_risk" in payload["messages"][1]["content"]


def test_ollama_generator_falls_back_when_transport_fails() -> None:
    transport = FakeOllamaTransport(error=RuntimeError("ollama unavailable"))
    generator = OllamaAnswerGenerator(
        base_url="http://ollama.test:11434",
        model="qwen2.5:7b",
        transport=transport,
    )

    result = generator.generate(
        AnswerGeneratorInput(
            question="当前策略如何？",
            route="policy_recommendation",
            route_reason="policy",
            tools=["rule_based_policy"],
            tool_results=[
                {
                    "policy_name": "rule_based",
                    "recommended_action": [0.0, 0.0],
                    "notes": "keep action",
                }
            ],
        )
    )

    assert result.generator == "deterministic_grounded"
    assert "rule_based" in result.answer


def test_ollama_generator_prompt_includes_conversation_context() -> None:
    transport = FakeOllamaTransport()
    generator = OllamaAnswerGenerator(
        base_url="http://ollama.test:11434",
        model="qwen2.5:7b",
        transport=transport,
    )

    generator.generate(
        AnswerGeneratorInput(
            question="What about that zone?",
            route="document_qa",
            route_reason="follow-up",
            conversation_context={
                "recent_turns": [{"question": "previous", "answer": "zone_a peaked"}],
                "relevant_memory": [{"text": "zone_a peaked at 30 C"}],
            },
        )
    )

    user_evidence = json.loads(transport.calls[0]["payload"]["messages"][1]["content"])
    assert user_evidence["conversation_context"]["relevant_memory"][0]["text"] == "zone_a peaked at 30 C"
