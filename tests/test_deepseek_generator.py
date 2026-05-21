from __future__ import annotations

import json

from src.agent.answer_generator import AnswerGeneratorInput
from src.agent.deepseek_generator import DeepSeekAnswerGenerator, build_answer_generator_from_env


class FakeTransport:
    def __init__(self, response: dict | None = None, error: Exception | None = None) -> None:
        self.response = response or {
            "choices": [{"message": {"content": "基于证据的 DeepSeek 回答"}}]
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


def test_deepseek_generator_builds_chat_completion_payload() -> None:
    transport = FakeTransport()
    generator = DeepSeekAnswerGenerator(
        api_key="test-key",
        base_url="https://example.deepseek.test",
        model="deepseek-test",
        transport=transport,
    )

    result = generator.generate(
        AnswerGeneratorInput(
            question="为什么温度升高？",
            route="timeseries_query",
            route_reason="metric query",
            tools=["query_metric"],
            tool_results=[{"metric_name": "zone_temperature", "summary": {"max": 30.0}}],
            data_source={"kind": "mock", "path": "built-in"},
        )
    )

    assert result.answer == "基于证据的 DeepSeek 回答"
    assert result.generator == "deepseek:deepseek-test"
    assert transport.calls[0]["url"] == "https://example.deepseek.test/chat/completions"
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer test-key"
    payload = transport.calls[0]["payload"]
    assert payload["model"] == "deepseek-test"
    assert "不能生成新的控制动作" in payload["messages"][0]["content"]
    assert "zone_temperature" in payload["messages"][1]["content"]


def test_deepseek_generator_falls_back_when_transport_fails() -> None:
    transport = FakeTransport(error=RuntimeError("network down"))
    generator = DeepSeekAnswerGenerator(
        api_key="test-key",
        base_url="https://example.deepseek.test",
        model="deepseek-test",
        transport=transport,
    )

    result = generator.generate(
        AnswerGeneratorInput(
            question="当前控制策略如何？",
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


def test_build_answer_generator_from_env_uses_deterministic_without_key(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    generator = build_answer_generator_from_env(project_root=tmp_path)

    result = generator.generate(
        AnswerGeneratorInput(question="x", route="document_qa", route_reason="test")
    )
    assert result.generator == "deterministic_grounded"


def test_build_answer_generator_from_env_uses_deepseek_with_key(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.deepseek.test")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-test")

    generator = build_answer_generator_from_env(transport=FakeTransport())

    result = generator.generate(
        AnswerGeneratorInput(
            question="x",
            route="document_qa",
            route_reason="test",
            retrieved_contexts=[{"source_id": "doc", "title": "Doc", "text": "evidence"}],
        )
    )
    assert result.generator == "deepseek:deepseek-test"


def test_build_answer_generator_from_env_loads_project_dotenv(
    tmp_path,
    monkeypatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "DEEPSEEK_API_KEY=test-key",
                "DEEPSEEK_BASE_URL=https://example.deepseek.test",
                "DEEPSEEK_MODEL=deepseek-dotenv",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    generator = build_answer_generator_from_env(
        project_root=tmp_path,
        transport=FakeTransport(),
    )

    result = generator.generate(
        AnswerGeneratorInput(
            question="x",
            route="document_qa",
            route_reason="test",
            retrieved_contexts=[{"source_id": "doc", "title": "Doc", "text": "evidence"}],
        )
    )
    assert result.generator == "deepseek:deepseek-dotenv"
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_TIMEOUT_SECONDS", raising=False)
