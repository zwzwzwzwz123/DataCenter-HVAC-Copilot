from __future__ import annotations

import json
from typing import Any
from urllib import request

from src.agent.answer_generator import (
    AnswerGenerator,
    AnswerGeneratorInput,
    DeterministicAnswerGenerator,
    GeneratedAnswer,
)
from src.agent.deepseek_generator import Transport, _payload_to_prompt, _system_prompt


class OllamaAnswerGenerator:
    """Local Ollama chat generator with deterministic fallback."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        timeout_seconds: float = 60.0,
        fallback: AnswerGenerator | None = None,
        transport: Transport | None = None,
    ) -> None:
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
                "stream": False,
                "options": {"temperature": 0.1},
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
            content = response["message"]["content"]
            return GeneratedAnswer(answer=str(content), generator=f"ollama:{self.model}")
        except Exception:
            return self.fallback.generate(payload)


def _default_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> dict[str, Any]:
    req = request.Request(url=url, data=body, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))
