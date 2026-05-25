from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable
from urllib import request

from src.core.env import load_env_file
from src.agent.answer_generator import (
    AnswerGenerator,
    AnswerGeneratorInput,
    DeterministicAnswerGenerator,
    GeneratedAnswer,
)

Transport = Callable[[str, dict[str, str], bytes, float], dict[str, Any]]


class DeepSeekAnswerGenerator:
    """OpenAI-compatible DeepSeek answer generator with deterministic fallback."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout_seconds: float = 30.0,
        fallback: AnswerGenerator | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.fallback = fallback or DeterministicAnswerGenerator()
        self.transport = transport or _default_transport

    def generate(self, payload: AnswerGeneratorInput) -> GeneratedAnswer:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _system_prompt()},
                    {"role": "user", "content": _payload_to_prompt(payload)},
                ],
                "temperature": 0.1,
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
            content = response["choices"][0]["message"]["content"]
            return GeneratedAnswer(answer=str(content), generator=f"deepseek:{self.model}")
        except Exception:
            return self.fallback.generate(payload)


def build_answer_generator_from_env(
    project_root: str | Path | None = None,
    transport: Transport | None = None,
) -> AnswerGenerator:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    load_env_file(root / ".env")
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider == "deterministic":
        return DeterministicAnswerGenerator()
    if provider == "ollama":
        from src.agent.ollama_generator import OllamaAnswerGenerator

        timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
        return OllamaAnswerGenerator(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            timeout_seconds=timeout,
            transport=transport,
        )
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return DeterministicAnswerGenerator()
    timeout = float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "30"))
    return DeepSeekAnswerGenerator(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        timeout_seconds=timeout,
        transport=transport,
    )


def _default_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> dict[str, Any]:
    req = request.Request(url=url, data=body, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _system_prompt() -> str:
    return (
        "你是 DataCenter-HVAC Copilot 的证据整合解释器。"
        "只能使用用户消息中提供的 retrieved_contexts、citations、tool_results、policy_result 和 data_source。"
        "如果证据不足，必须明确说明证据不足。"
        "不能生成新的控制动作，不能写回环境，不能声称 BEAR 是真实数据中心生产遥测。"
        "策略建议中的动作或风险只能来自策略工具返回结果。"
        "请用中文回答，并保留关键引用和工具证据。"
        "Conversation memory may help resolve references, but current fresh evidence from tools, RAG, and policy is authoritative. "
    )


def _payload_to_prompt(payload: AnswerGeneratorInput) -> str:
    evidence = {
        "question": payload.question,
        "route": payload.route,
        "route_reason": payload.route_reason,
        "retrieved_contexts": payload.retrieved_contexts,
        "citations": payload.citations,
        "tools": payload.tools,
        "tool_results": payload.tool_results,
        "policy_result": payload.policy_result,
        "data_source": payload.data_source,
        "conversation_context": payload.conversation_context or {},
    }
    return json.dumps(evidence, ensure_ascii=False, indent=2, default=str)
